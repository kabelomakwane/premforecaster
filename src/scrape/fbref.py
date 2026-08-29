"""FBref team and player match logs, via the ``soccerdata`` package.

FBref (built on Opta/StatsBomb data) carries the detailed match statistics that
Understat does not: passing volume and accuracy, possession, pressures,
tackles, and the disciplinary record - fouls, yellows and reds - that a referee
model needs. We take it from 2017/18, which is as far back as FBref's advanced
stats reliably go for the Premier League.

Two tables are produced:

``fbref_team_match``
    One row per team per match, combining the shooting, possession, passing and
    miscellaneous (cards and fouls) stat groups.
``fbref_player_match``
    The same at player level, for the goalscorer model.

Politeness is not optional here
-------------------------------
This project holds itself to **one request every seven seconds** for FBref, the
limit written into CLAUDE.md. Do not assume ``soccerdata`` enforces that: its
readers ship with ``rate_limit = 0``, so the limit is applied here in
:func:`make_client` and checked before every fetch. A full pull is therefore
slow - several stat groups times several seasons, each a separate request - and
that is the price of scraping someone else's site politely.

Because it is slow, the job is built to never repeat work:

* ``soccerdata`` caches every downloaded page under ``data/raw/fbref/``, so a
  rerun makes **zero** network requests.
* Each (table, season, stat group) is staged to parquet and recorded in a
  checkpoint as soon as it parses, so an interrupted run resumes rather than
  restarting.

Cloudflare
----------
FBref sits behind Cloudflare, which rejects ordinary HTTP clients. ``soccerdata``
works around this by driving a real browser. If no browser is available, or the
network blocks it, the fetch raises :class:`FBrefUnavailableError` with an
explanation rather than failing obscurely - the rest of the pipeline is designed
to carry on without FBref, since Understat supplies the xG that the model
actually depends on.
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

from src.checkpoint import Checkpoint, report_progress
from src.lookups import to_canonical
from src.scrape.footballdata import current_season_start_year, season_label

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "fbref"
CACHE_DIR = RAW_DIR / "soccerdata_cache"
STAGING_DIR = RAW_DIR / "staged"
CHECKPOINT_PATH = RAW_DIR / "checkpoints" / "fbref.json"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

TEAM_MATCH_PARQUET = PROCESSED_DIR / "fbref_team_match.parquet"
PLAYER_MATCH_PARQUET = PROCESSED_DIR / "fbref_player_match.parquet"

SOURCE = "fbref"
LEAGUE = "ENG-Premier League"

#: FBref's advanced match logs start being reliable for the Premier League in
#: 2017/18.
FIRST_SEASON_START_YEAR = 2017

#: The hard limit from CLAUDE.md: one request every seven seconds, no faster.
#: soccerdata does not enforce this itself, so make_client sets it.
REQUEST_INTERVAL_SECONDS = 7.0

#: Random extra delay so requests are not perfectly periodic.
REQUEST_JITTER_SECONDS = 2.0

USER_AGENT = (
    "premforecaster/0.1 (personal, non-commercial football forecasting project)"
)

#: The stat groups we pull. Each one is a separate request per season, so this
#: list is deliberately short: what the model needs, nothing else.
TEAM_STAT_TYPES = ("schedule", "shooting", "possession", "passing", "misc")
PLAYER_STAT_TYPES = ("summary", "passing", "misc")

#: Where soccerdata looks for a browser, in order. FBref needs a real one to get
#: past Cloudflare. The Playwright location is checked first because this
#: project's automated environment ships Chromium there.
BROWSER_CANDIDATES = (
    "/opt/pw-browsers/chromium-*/chrome-linux/chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


class FBrefUnavailableError(RuntimeError):
    """Raised when FBref cannot be reached at all.

    Almost always one of: no browser installed for the Cloudflare challenge, the
    network blocking the browser's traffic, or FBref itself being down. The
    message says which, because the fix is different for each.
    """


class FBrefFormatError(ValueError):
    """Raised when FBref returns data in a shape we do not recognise."""


# ---------------------------------------------------------------------------
# Seasons and browser discovery
# ---------------------------------------------------------------------------


def available_start_years(today: date | None = None) -> list[int]:
    """Every season we pull from FBref: 2017/18 to the current one."""
    return list(range(FIRST_SEASON_START_YEAR, current_season_start_year(today) + 1))


def find_browser(explicit: str | Path | None = None) -> Path | None:
    """Locate a Chrome/Chromium binary for the Cloudflare challenge.

    Checks, in order: the argument, the ``PREMFORECASTER_BROWSER`` environment
    variable, the known install locations, then anything on PATH.
    """
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None

    from_env = os.environ.get("PREMFORECASTER_BROWSER")
    if from_env and Path(from_env).exists():
        return Path(from_env)

    for pattern in BROWSER_CANDIDATES:
        if "*" in pattern:
            matches = sorted(Path("/").glob(pattern.lstrip("/")))
            if matches:
                return matches[-1]
        elif Path(pattern).exists():
            return Path(pattern)

    for name in ("chromium", "chromium-browser", "google-chrome", "chrome"):
        found = shutil.which(name)
        if found:
            return Path(found)

    return None


def make_client(
    start_year: int,
    *,
    cache_dir: Path | str = CACHE_DIR,
    browser: str | Path | None = None,
):
    """Build a soccerdata FBref reader for one season."""
    import soccerdata

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    browser_path = find_browser(browser)

    if browser_path is None:
        raise FBrefUnavailableError(
            "FBref sits behind Cloudflare and soccerdata needs a real Chrome or "
            "Chromium browser to get past it, but none was found. Install "
            "Chromium, or set PREMFORECASTER_BROWSER to the browser binary. "
            "Understat still provides xG without this, so the rest of the "
            "pipeline will run."
        )

    logger.debug("Using browser at %s", browser_path)
    client = soccerdata.FBref(
        leagues=LEAGUE,
        # The four-plus-four form, never a bare year: soccerdata reads "2021" as
        # the 20/21 season rather than the year 2021 and would quietly return a
        # season of the wrong data. See src/scrape/understat.py for the details.
        seasons=f"{start_year}-{start_year + 1}",
        data_dir=Path(cache_dir),
        path_to_browser=browser_path,
        headless=True,
    )

    # soccerdata leaves these at zero, so the seven-second limit is ours to set.
    # Unlike Understat's reader, FBref's downloads go through soccerdata's
    # _download_and_save, which does sleep for rate_limit + max_delay - so
    # setting the attributes is enough here. This was confirmed by reading
    # soccerdata 1.9.1's source, not end to end, because FBref could not be
    # reached from the machine this was written on. If you get FBref working,
    # measure the gap between requests once and confirm it is really seven
    # seconds; do not assume, because Understat looked fine and was not.
    client.rate_limit = REQUEST_INTERVAL_SECONDS
    client.max_delay = REQUEST_JITTER_SECONDS

    session = getattr(client, "_session", None)
    if session is not None and hasattr(session, "headers"):
        try:
            session.headers.update({"User-Agent": USER_AGENT})
        except (AttributeError, TypeError):  # pragma: no cover - session shape varies
            logger.debug("Could not set a User-Agent on the FBref session.")

    return client


def check_politeness(client) -> None:
    """Refuse to scrape FBref faster than one request every seven seconds."""
    rate = getattr(client, "rate_limit", 0)
    if not rate or rate < REQUEST_INTERVAL_SECONDS:
        raise RuntimeError(
            f"FBref client is set to {rate}s between requests, below the "
            f"{REQUEST_INTERVAL_SECONDS}s hard limit in CLAUDE.md. Refusing to "
            "scrape. soccerdata may have renamed the attribute."
        )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """FBref returns multi-level column headers; squash them into flat names.

    ``("Shooting", "Sh")`` becomes ``shooting_sh``. Levels that are blank or
    repeat the group name are dropped, so we get ``sh`` rather than ``sh_sh``.
    """
    flat = frame.copy()
    if isinstance(flat.columns, pd.MultiIndex):
        names = []
        for parts in flat.columns:
            pieces = [
                str(part).strip()
                for part in parts
                if str(part).strip() and not str(part).startswith("Unnamed")
            ]
            deduped: list[str] = []
            for piece in pieces:
                if not deduped or deduped[-1].lower() != piece.lower():
                    deduped.append(piece)
            names.append("_".join(deduped))
        flat.columns = names

    flat.columns = [
        str(column).strip().lower().replace(" ", "_").replace("-", "_").replace("%", "pct")
        for column in flat.columns
    ]
    return flat


def parse_team_match(raw: pd.DataFrame, start_year: int, stat_type: str) -> pd.DataFrame:
    """Tidy one stat group of one season of team match logs."""
    frame = _flatten_columns(raw.reset_index())

    if "team" not in frame.columns:
        raise FBrefFormatError(
            f"FBref team {stat_type} for {season_label(start_year)} has no 'team' "
            f"column. Columns: {list(frame.columns)[:20]}"
        )
    if "date" not in frame.columns:
        raise FBrefFormatError(
            f"FBref team {stat_type} for {season_label(start_year)} has no 'date' "
            f"column. Columns: {list(frame.columns)[:20]}"
        )

    frame["season"] = season_label(start_year)
    frame["team"] = to_canonical(frame["team"], SOURCE)
    if "opponent" in frame.columns:
        frame["opponent"] = to_canonical(frame["opponent"], SOURCE)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)

    if frame["date"].isna().any():
        raise FBrefFormatError(
            f"FBref team {stat_type} for {season_label(start_year)} has "
            f"{int(frame['date'].isna().sum())} unparseable date(s)."
        )

    frame["stat_type"] = stat_type
    return frame


def parse_player_match(raw: pd.DataFrame, start_year: int, stat_type: str) -> pd.DataFrame:
    """Tidy one stat group of one season of player match logs."""
    frame = _flatten_columns(raw.reset_index())

    if "team" not in frame.columns:
        raise FBrefFormatError(
            f"FBref player {stat_type} for {season_label(start_year)} has no 'team' "
            f"column. Columns: {list(frame.columns)[:20]}"
        )

    frame["season"] = season_label(start_year)
    frame["team"] = to_canonical(frame["team"], SOURCE)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame["stat_type"] = stat_type
    return frame


# ---------------------------------------------------------------------------
# The resumable job
# ---------------------------------------------------------------------------


def staged_path(
    table: str, start_year: int, stat_type: str, staging_dir: Path | str = STAGING_DIR
) -> Path:
    return Path(staging_dir) / table / f"{season_label(start_year)}_{stat_type}.parquet"


def fetch_stat_group(
    table: str,
    start_year: int,
    stat_type: str,
    *,
    cache_dir: Path | str = CACHE_DIR,
    staging_dir: Path | str = STAGING_DIR,
    browser: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch and stage one (table, season, stat group). One or more requests."""
    client = make_client(start_year, cache_dir=cache_dir, browser=browser)
    check_politeness(client)

    try:
        if table == "team_match":
            raw = client.read_team_match_stats(stat_type=stat_type)
            parsed = parse_team_match(raw, start_year, stat_type)
        else:
            raw = client.read_player_match_stats(stat_type=stat_type)
            parsed = parse_player_match(raw, start_year, stat_type)
    except (FBrefFormatError, FBrefUnavailableError):
        raise
    except Exception as error:  # soccerdata raises a variety of network errors
        raise FBrefUnavailableError(
            f"Could not fetch FBref {table} {stat_type} for "
            f"{season_label(start_year)}: {type(error).__name__}: {error}. "
            "FBref is behind Cloudflare; this is usually a blocked browser or a "
            "network restriction rather than a bug."
        ) from error

    destination = staged_path(table, start_year, stat_type, staging_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed.to_parquet(destination, index=False)
    return parsed


def collect(
    table: str,
    start_years: list[int],
    stat_types: tuple[str, ...],
    *,
    cache_dir: Path | str = CACHE_DIR,
    staging_dir: Path | str = STAGING_DIR,
    checkpoint_path: Path | str = CHECKPOINT_PATH,
    browser: str | Path | None = None,
    skip_failures: bool = True,
) -> pd.DataFrame:
    """Fetch every (season, stat group) for one table, resuming where it stopped.

    The stat groups are merged into one wide row per team (or player) per match.
    With ``skip_failures`` set, a stat group that cannot be fetched is logged and
    skipped rather than losing the whole run - a partial FBref table is far more
    useful than none.
    """
    checkpoint = Checkpoint(checkpoint_path)
    units = [(year, stat) for year in start_years for stat in stat_types]
    keys = [f"{table}/{season_label(y)}/{s}" for y, s in units]
    report_progress(checkpoint, keys, f"FBref {table}")

    frames: list[pd.DataFrame] = []
    failures: list[str] = []

    for start_year, stat_type in units:
        key = f"{table}/{season_label(start_year)}/{stat_type}"
        staged = staged_path(table, start_year, stat_type, staging_dir)

        if checkpoint.is_done(key) and staged.exists():
            frames.append(pd.read_parquet(staged))
            continue

        logger.info("FBref %s: fetching %s %s", table, season_label(start_year), stat_type)
        try:
            parsed = fetch_stat_group(
                table, start_year, stat_type,
                cache_dir=cache_dir, staging_dir=staging_dir, browser=browser,
            )
        except (FBrefUnavailableError, FBrefFormatError) as error:
            if not skip_failures:
                raise
            logger.warning("Skipping %s: %s", key, error)
            failures.append(key)
            continue

        checkpoint.mark_done(key, rows=len(parsed))
        frames.append(parsed)

    if failures:
        logger.warning(
            "FBref %s: %d of %d unit(s) could not be fetched: %s",
            table, len(failures), len(keys), ", ".join(failures[:6]),
        )

    if not frames:
        return pd.DataFrame()

    return merge_stat_groups(frames, table)


def merge_stat_groups(frames: list[pd.DataFrame], table: str) -> pd.DataFrame:
    """Merge the separate stat groups into one wide table.

    Each group is keyed on the same match identity, so they are joined rather
    than stacked. Columns that appear in more than one group (a stat group
    usually repeats the basics) are kept once, from the first group that has
    them.
    """
    key = ["season", "team", "date"] if table == "team_match" else ["season", "team", "date", "player"]

    merged: pd.DataFrame | None = None
    for frame in frames:
        frame = frame.drop(columns=["stat_type"], errors="ignore")
        available_key = [column for column in key if column in frame.columns]
        if len(available_key) < 2:
            logger.warning("Skipping an FBref frame with no usable join key.")
            continue

        frame = frame.drop_duplicates(subset=available_key)
        if merged is None:
            merged = frame
            continue

        new_columns = [
            column for column in frame.columns
            if column not in merged.columns or column in available_key
        ]
        merged = merged.merge(
            frame[new_columns], on=available_key, how="outer", suffixes=("", "_dup")
        )

    if merged is None:
        return pd.DataFrame()

    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_dup")], errors="ignore")
    sort_columns = [c for c in ("date", "team", "player") if c in merged.columns]
    return merged.sort_values(sort_columns).reset_index(drop=True)


def build_all(
    *,
    start_years: list[int] | None = None,
    cache_dir: Path | str = CACHE_DIR,
    staging_dir: Path | str = STAGING_DIR,
    checkpoint_path: Path | str = CHECKPOINT_PATH,
    processed_dir: Path | str = PROCESSED_DIR,
    browser: str | Path | None = None,
    today: date | None = None,
    skip_failures: bool = True,
) -> dict[str, Path]:
    """Pull both FBref tables and write the processed parquets."""
    start_years = start_years if start_years is not None else available_start_years(today)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for table, stat_types, destination in (
        ("team_match", TEAM_STAT_TYPES, TEAM_MATCH_PARQUET),
        ("player_match", PLAYER_STAT_TYPES, PLAYER_MATCH_PARQUET),
    ):
        frame = collect(
            table, start_years, stat_types,
            cache_dir=cache_dir, staging_dir=staging_dir,
            checkpoint_path=checkpoint_path, browser=browser,
            skip_failures=skip_failures,
        )
        if frame.empty:
            logger.warning("FBref %s produced no rows; not writing a parquet.", table)
            continue

        path = processed_dir / destination.name
        frame.to_parquet(path, index=False)
        written[table] = path
        logger.info("Wrote %d FBref %s rows to %s", len(frame), table, path)

    return written
