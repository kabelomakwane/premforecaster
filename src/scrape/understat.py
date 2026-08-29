"""Understat xG data, via the ``soccerdata`` package.

Understat models every shot in a match and gives it an expected-goals value: the
probability that an average player scores from that position, in that situation,
with that body part. Summing those gives a far better picture of how well a team
actually played than the scoreline does, because goals are rare and lumpy while
shot quality is not. This is the data that upgrades our Dixon-Coles model from
"goals scored" to "chances created", and it is the single biggest accuracy win
available to us.

Four tables are pulled:

``team_match_xg``
    One row per team per match: xG for and against, non-penalty xG, PPDA (a
    pressing measure) and deep completions. This is what the team strength
    ratings are built from.
``player_season``
    Season totals per player: minutes, shots, xG, non-penalty xG, xA.
``player_match``
    The same per match, which the goalscorer model needs to work out who is
    actually taking the shots right now rather than who took them in August.
``shots``
    Every individual shot: its xG, who took it, the situation (open play,
    corner, penalty) and whether it went in.

Politeness and cost
-------------------
**``soccerdata`` does not rate limit Understat.** Its ``Understat`` class leaves
``rate_limit`` at zero and sends no User-Agent, so straight out of the box it
will fetch as fast as the network allows - and the player and shot tables are
one request *per match*, so a single season is nearly 400 requests. This module
therefore sets the limit itself, in :func:`make_client`: at least
``REQUEST_INTERVAL_SECONDS`` between requests plus random jitter, and a
descriptive User-Agent, as CLAUDE.md requires. Do not remove that; it is the
difference between a polite scraper and hammering someone's free website.

Every response is cached on disk under ``data/raw/understat/``, and that cache
is the point: a second run costs zero requests. On top of it, each season of
each table is written to a staging parquet as soon as it is parsed and recorded
in a checkpoint file, so a run interrupted after nine of thirteen seasons
resumes at the tenth rather than starting again.

All team names come back as Understat spells them and are translated to
canonical names before anything is written, so these tables join cleanly to
``results.parquet``.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from src.checkpoint import Checkpoint, report_progress
from src.lookups import to_canonical
from src.scrape.footballdata import current_season_start_year, season_code, season_label

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "understat"
CACHE_DIR = RAW_DIR / "soccerdata_cache"
STAGING_DIR = RAW_DIR / "staged"
CHECKPOINT_PATH = RAW_DIR / "checkpoints" / "understat.json"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

TEAM_MATCH_PARQUET = PROCESSED_DIR / "team_match_xg.parquet"
PLAYER_SEASON_PARQUET = PROCESSED_DIR / "player_season.parquet"
PLAYER_MATCH_PARQUET = PROCESSED_DIR / "understat_player_match.parquet"
SHOTS_PARQUET = PROCESSED_DIR / "shots.parquet"

SOURCE = "understat"
LEAGUE = "ENG-Premier League"

#: Understat's data starts with the 2014/15 season. That is why the whole
#: project uses 2014/15 as its horizon.
FIRST_SEASON_START_YEAR = 2014

#: How many recent seasons the per-match tables cover by default: the three most
#: recently completed, plus the one in progress.
#:
#: This default exists because the per-match tables (shots and player match logs)
#: cost **one request per match**. A season is ~380 requests, so at the six-second
#: minimum a single season takes about 40 minutes and all thirteen would take
#: most of a day of continuous scraping. The models that use these tables - the
#: goalscorer model above all - care about current form, not who was taking shots
#: in 2015, so fetching the lot would be a lot of load on a free site for data we
#: would not use.
#:
#: Both are still a parameter. To backfill the whole history, pass explicit years
#: (``build_all(shot_years=available_start_years())``) and leave it running: the
#: job is resumable and cached, so it can be stopped and restarted freely.
DEFAULT_SHOT_SEASONS = 4

#: Understat timestamps are UK local time, like football-data's.
SOURCE_TIMEZONE = "Europe/London"

#: soccerdata 1.9.1 tidies Understat's shot ``situation`` values through a lookup
#: that has entries for OpenPlay, FromCorner, SetPiece and DirectFreekick - but
#: **not** for Penalty, so every penalty comes back with a blank situation. That
#: matters: penalties are the single most predictable shot in football and the
#: goalscorer model has to treat them separately, since who takes them is a squad
#: decision rather than a matter of form. We restore the label here.
PENALTY_SITUATION = "Penalty"

#: A penalty is converted a bit under 80% of the time, so its xG sits near 0.76.
#: Used to check that the blanks we relabel really are penalties: if soccerdata
#: ever fails to map a *different* situation, these bounds catch it instead of
#: letting us mislabel ordinary shots.
PENALTY_XG_BOUNDS = (0.70, 0.82)

#: Minimum seconds between requests. CLAUDE.md sets this at six for Understat
#: and calls it non-negotiable. soccerdata does not enforce it, so we do.
REQUEST_INTERVAL_SECONDS = 6.0

#: Extra random delay on top, so requests are not perfectly periodic.
REQUEST_JITTER_SECONDS = 2.0

#: Who we are. A scraper with no User-Agent is impolite and unblockable-by-
#: request; this one says what it is and that it is personal and non-commercial.
USER_AGENT = (
    "premforecaster/0.1 (personal, non-commercial football forecasting project)"
)


class UnderstatFormatError(ValueError):
    """Raised when Understat returns something we do not recognise."""


# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------


def available_start_years(today: date | None = None) -> list[int]:
    """Every season Understat covers, from 2014/15 to the current one."""
    return list(range(FIRST_SEASON_START_YEAR, current_season_start_year(today) + 1))


def shot_start_years(
    today: date | None = None, count: int = DEFAULT_SHOT_SEASONS
) -> list[int]:
    """The recent seasons we keep shot-level data for."""
    return available_start_years(today)[-count:]


def _soccerdata_season(start_year: int) -> str:
    """Ask soccerdata for a season in its unambiguous four-plus-four form.

    This must never be shortened to ``str(start_year)``. soccerdata reads a bare
    ``"2021"`` as the *season* 20/21 rather than the year 2021, so asking for
    "2021" silently returns the 2020/21 season - a whole year of the wrong data,
    with only a warning. ``"2021-2022"`` cannot be misread.
    """
    return f"{start_year}-{start_year + 1}"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def make_client(
    start_year: int,
    cache_dir: Path | str = CACHE_DIR,
    *,
    rate_limit: float = REQUEST_INTERVAL_SECONDS,
    jitter: float = REQUEST_JITTER_SECONDS,
):
    """Build a soccerdata Understat reader for one season, politely configured.

    soccerdata is imported lazily so that merely importing this module does not
    pull it in (it is slow) or touch the network.

    The rate limit and User-Agent are applied here because soccerdata's Understat
    class sets neither: it ships with ``rate_limit = 0`` and no headers. The
    attributes below are what its request loop actually reads, so setting them
    after construction is what makes the limit take effect.
    """
    import soccerdata

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    client = soccerdata.Understat(
        leagues=LEAGUE,
        seasons=_soccerdata_season(start_year),
        data_dir=Path(cache_dir),
    )

    # soccerdata's own rate limit, for the code paths that honour it.
    client.rate_limit = rate_limit
    client.max_delay = jitter

    # Understat's data calls do not go through that path, so throttle them too.
    throttle(client, rate_limit=rate_limit, jitter=jitter)

    # Understat's requests pass their own header dict, which overrides anything
    # set on the session, so the User-Agent has to be added to that dict.
    soccerdata.understat.UNDERSTAT_HEADERS.setdefault("User-Agent", USER_AGENT)

    return client


def throttle(client, *, rate_limit: float, jitter: float) -> None:
    """Force a minimum gap between Understat's *network* requests.

    Necessary because ``soccerdata``'s Understat reader fetches through its own
    ``_request_api`` method, which calls the HTTP session directly and never
    sleeps - the rate limit on the class is simply not consulted on that path.
    Setting ``rate_limit`` alone therefore does nothing, which is easy to
    believe you have fixed when you have not. Measure it if you change this.

    The throttle wraps the HTTP session's ``get`` rather than any one reader
    method, so it covers *every* request the client makes - the data calls, and
    the separate call that primes Understat's cookies, which happens outside
    ``_request_api`` and would otherwise slip through.

    Wrapping at this level also keeps cache hits instant for free: a cached read
    returns before it ever reaches the session, so a rerun over a warm cache
    never sleeps.
    """
    import random
    import time

    session = client._session
    original_get = session.get
    last_request = [0.0]

    def polite_get(*args, **kwargs):
        target = rate_limit + random.random() * jitter
        waited = time.monotonic() - last_request[0]
        if waited < target:
            time.sleep(target - waited)
        last_request[0] = time.monotonic()
        return original_get(*args, **kwargs)

    session.get = polite_get
    client._premforecaster_throttled = True


def check_politeness(client) -> None:
    """Assert a client really will wait between requests.

    Checks both the attribute and that the throttle was actually installed. A
    silent regression here - a soccerdata upgrade renaming a method, say - would
    turn this into an impolite scraper without anyone noticing.
    """
    rate = getattr(client, "rate_limit", 0)
    if not rate or rate < REQUEST_INTERVAL_SECONDS:
        raise RuntimeError(
            f"Understat client is set to {rate}s between requests, below the "
            f"{REQUEST_INTERVAL_SECONDS}s minimum this project holds itself to. "
            "Refusing to scrape. soccerdata may have renamed the attribute."
        )
    if not getattr(client, "_premforecaster_throttled", False):
        raise RuntimeError(
            "Understat client has no throttle installed on its request method, "
            "so it would fetch as fast as the network allows. Refusing to "
            "scrape. Build clients with make_client()."
        )


def check_season_matches(raw: pd.DataFrame, start_year: int, table: str) -> None:
    """Check soccerdata gave us the season we actually asked for.

    Everything downstream is labelled from the season we *requested*, so if the
    package ever resolves a season differently we would stamp the wrong label on
    real data and never notice. That happened once already: a bare ``"2021"`` is
    read as the 20/21 season, so 2021/22 quietly came back as 2020/21. This
    check turns that class of mistake into an immediate, obvious failure.
    """
    if "season" not in getattr(raw.index, "names", []) or raw.empty:
        return

    returned = {str(value) for value in raw.index.get_level_values("season").unique()}
    expected = season_code(start_year)

    if returned != {expected}:
        raise UnderstatFormatError(
            f"Asked Understat for {season_label(start_year)} (season code "
            f"{expected!r}) while reading {table}, but it returned season(s) "
            f"{sorted(returned)}. Refusing to label another season's data as "
            f"{season_label(start_year)}."
        )


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], table: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise UnderstatFormatError(
            f"Understat {table} is missing the column(s) {missing}. The package or "
            f"the site has changed format. Columns present: {list(frame.columns)}"
        )


def _to_utc(values: pd.Series) -> pd.Series:
    """Understat gives UK local timestamps; store them in UTC like everything else."""
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        bad = values[parsed.isna()].unique()[:5].tolist()
        raise UnderstatFormatError(f"Could not parse Understat date(s): {bad}")

    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(
            SOURCE_TIMEZONE, ambiguous=True, nonexistent="shift_forward"
        ).dt.tz_convert("UTC")
    return parsed.dt.tz_convert("UTC")


# ---------------------------------------------------------------------------
# Per-season parsing
# ---------------------------------------------------------------------------


def parse_team_match_stats(raw: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """Reshape Understat's one-row-per-match table into one row per team per match.

    Understat returns home and away figures side by side. The model wants a row
    per team, because team strength is estimated per team, so each match becomes
    two rows: one from the home side's point of view and one from the away
    side's. ``home_team`` and ``away_team`` are kept on both rows so the table
    still joins to results.parquet on the fixture.
    """
    frame = raw.reset_index()
    _require_columns(
        frame,
        ["game_id", "date", "home_team", "away_team", "home_goals", "away_goals",
         "home_xg", "away_xg", "home_np_xg", "away_np_xg"],
        "team match stats",
    )

    kickoff = _to_utc(frame["date"])
    home_team = to_canonical(frame["home_team"], SOURCE)
    away_team = to_canonical(frame["away_team"], SOURCE)

    def side(is_home: bool) -> pd.DataFrame:
        us, them = ("home", "away") if is_home else ("away", "home")
        return pd.DataFrame(
            {
                "date": kickoff,
                "season": season_label(start_year),
                "game_id": frame["game_id"].astype(str),
                "home_team": home_team,
                "away_team": away_team,
                "team": home_team if is_home else away_team,
                "opponent": away_team if is_home else home_team,
                "is_home": is_home,
                "goals_for": pd.to_numeric(frame[f"{us}_goals"], errors="coerce"),
                "goals_against": pd.to_numeric(frame[f"{them}_goals"], errors="coerce"),
                "xg_for": pd.to_numeric(frame[f"{us}_xg"], errors="coerce"),
                "xg_against": pd.to_numeric(frame[f"{them}_xg"], errors="coerce"),
                "npxg_for": pd.to_numeric(frame[f"{us}_np_xg"], errors="coerce"),
                "npxg_against": pd.to_numeric(frame[f"{them}_np_xg"], errors="coerce"),
                "ppda": pd.to_numeric(frame.get(f"{us}_ppda"), errors="coerce"),
                "deep_completions": pd.to_numeric(
                    frame.get(f"{us}_deep_completions"), errors="coerce"
                ),
                "points": pd.to_numeric(frame.get(f"{us}_points"), errors="coerce"),
                "expected_points": pd.to_numeric(
                    frame.get(f"{us}_expected_points"), errors="coerce"
                ),
            }
        )

    both = pd.concat([side(True), side(False)], ignore_index=True)
    return both.sort_values(["date", "game_id", "is_home"], ascending=[True, True, False]).reset_index(drop=True)


def parse_player_season_stats(raw: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """Season totals per player, with canonical team names."""
    frame = raw.reset_index()
    _require_columns(
        frame,
        ["team", "player", "minutes", "goals", "xg", "np_xg", "assists", "xa", "shots"],
        "player season stats",
    )

    tidy = pd.DataFrame(
        {
            "season": season_label(start_year),
            "team": to_canonical(frame["team"], SOURCE),
            "player": frame["player"].astype("string").str.strip(),
            "player_id": frame.get("player_id", pd.Series(pd.NA, index=frame.index)).astype("string"),
            "position": frame.get("position", pd.Series(pd.NA, index=frame.index)).astype("string"),
            "matches": pd.to_numeric(frame.get("matches"), errors="coerce"),
            "minutes": pd.to_numeric(frame["minutes"], errors="coerce"),
            "goals": pd.to_numeric(frame["goals"], errors="coerce"),
            "np_goals": pd.to_numeric(frame.get("np_goals"), errors="coerce"),
            "xg": pd.to_numeric(frame["xg"], errors="coerce"),
            "npxg": pd.to_numeric(frame["np_xg"], errors="coerce"),
            "assists": pd.to_numeric(frame["assists"], errors="coerce"),
            "xa": pd.to_numeric(frame["xa"], errors="coerce"),
            "shots": pd.to_numeric(frame["shots"], errors="coerce"),
            "key_passes": pd.to_numeric(frame.get("key_passes"), errors="coerce"),
            "yellow_cards": pd.to_numeric(frame.get("yellow_cards"), errors="coerce"),
            "red_cards": pd.to_numeric(frame.get("red_cards"), errors="coerce"),
            "xg_chain": pd.to_numeric(frame.get("xg_chain"), errors="coerce"),
            "xg_buildup": pd.to_numeric(frame.get("xg_buildup"), errors="coerce"),
        }
    )
    return tidy.sort_values(["team", "player"]).reset_index(drop=True)


def parse_player_match_stats(raw: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """Per-player, per-match rows. Feeds the goalscorer model's minutes signal."""
    frame = raw.reset_index()
    _require_columns(
        frame, ["game_id", "team", "player", "minutes", "goals", "shots", "xg"],
        "player match stats",
    )

    tidy = pd.DataFrame(
        {
            "season": season_label(start_year),
            "game_id": frame["game_id"].astype(str),
            "team": to_canonical(frame["team"], SOURCE),
            "player": frame["player"].astype("string").str.strip(),
            "player_id": frame.get("player_id", pd.Series(pd.NA, index=frame.index)).astype("string"),
            "position": frame.get("position", pd.Series(pd.NA, index=frame.index)).astype("string"),
            "minutes": pd.to_numeric(frame["minutes"], errors="coerce"),
            "goals": pd.to_numeric(frame["goals"], errors="coerce"),
            "own_goals": pd.to_numeric(frame.get("own_goals"), errors="coerce"),
            "shots": pd.to_numeric(frame["shots"], errors="coerce"),
            "xg": pd.to_numeric(frame["xg"], errors="coerce"),
            "assists": pd.to_numeric(frame.get("assists"), errors="coerce"),
            "xa": pd.to_numeric(frame.get("xa"), errors="coerce"),
            "key_passes": pd.to_numeric(frame.get("key_passes"), errors="coerce"),
            "xg_chain": pd.to_numeric(frame.get("xg_chain"), errors="coerce"),
            "xg_buildup": pd.to_numeric(frame.get("xg_buildup"), errors="coerce"),
        }
    )
    return tidy.sort_values(["game_id", "team", "player"]).reset_index(drop=True)


def parse_shot_events(raw: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """Every shot, with its xG, situation and outcome."""
    frame = raw.reset_index()
    _require_columns(
        frame, ["game_id", "team", "player", "xg", "minute", "situation", "result"],
        "shot events",
    )

    tidy = pd.DataFrame(
        {
            "date": _to_utc(frame["date"]) if "date" in frame.columns else pd.NaT,
            "season": season_label(start_year),
            "game_id": frame["game_id"].astype(str),
            "shot_id": frame.get("shot_id", pd.Series(pd.NA, index=frame.index)).astype("string"),
            "team": to_canonical(frame["team"], SOURCE),
            "player": frame["player"].astype("string").str.strip(),
            "player_id": frame.get("player_id", pd.Series(pd.NA, index=frame.index)).astype("string"),
            "assist_player": frame.get("assist_player", pd.Series(pd.NA, index=frame.index)).astype("string"),
            "minute": pd.to_numeric(frame["minute"], errors="coerce"),
            "xg": pd.to_numeric(frame["xg"], errors="coerce"),
            "body_part": frame.get("body_part", pd.Series(pd.NA, index=frame.index)).astype("string"),
            "situation": frame["situation"].astype("string"),
            "result": frame["result"].astype("string"),
            "location_x": pd.to_numeric(frame.get("location_x"), errors="coerce"),
            "location_y": pd.to_numeric(frame.get("location_y"), errors="coerce"),
        }
    )
    tidy["is_goal"] = tidy["result"].str.lower().eq("goal")
    tidy = restore_penalty_situations(tidy, start_year)
    tidy["is_penalty"] = tidy["situation"].eq(PENALTY_SITUATION)
    return tidy.sort_values(["game_id", "minute"]).reset_index(drop=True)


def restore_penalty_situations(shots: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """Put the ``Penalty`` label back on the shots soccerdata leaves blank.

    See :data:`PENALTY_SITUATION`. Before relabelling anything we check the
    blank rows really do look like penalties, by their xG: a penalty is worth
    about 0.76 expected goals and almost nothing else in football is. If the
    blanks look like ordinary shots we raise instead, because that would mean
    soccerdata has stopped mapping some *other* situation and relabelling would
    be inventing data.
    """
    missing = shots["situation"].isna()
    if not missing.any():
        return shots

    mean_xg = shots.loc[missing, "xg"].mean()
    low, high = PENALTY_XG_BOUNDS
    if not (low <= mean_xg <= high):
        raise UnderstatFormatError(
            f"{season_label(start_year)}: {int(missing.sum())} shot(s) have no "
            f"situation, and their mean xG of {mean_xg:.3f} is outside the "
            f"{low}-{high} range expected of penalties. Refusing to guess what "
            "they are - check what situations Understat is now returning."
        )

    shots = shots.copy()
    shots.loc[missing, "situation"] = PENALTY_SITUATION
    logger.info(
        "%s: labelled %d blank shot situation(s) as penalties (mean xG %.3f).",
        season_label(start_year), int(missing.sum()), mean_xg,
    )
    return shots


# ---------------------------------------------------------------------------
# The resumable job
# ---------------------------------------------------------------------------

#: table name -> (soccerdata reader method, parser, output parquet)
TABLES: dict[str, tuple[str, Callable[[pd.DataFrame, int], pd.DataFrame], Path]] = {
    "team_match": ("read_team_match_stats", parse_team_match_stats, TEAM_MATCH_PARQUET),
    "player_season": ("read_player_season_stats", parse_player_season_stats, PLAYER_SEASON_PARQUET),
    "player_match": ("read_player_match_stats", parse_player_match_stats, PLAYER_MATCH_PARQUET),
    "shots": ("read_shot_events", parse_shot_events, SHOTS_PARQUET),
}


def staged_path(table: str, start_year: int, staging_dir: Path | str = STAGING_DIR) -> Path:
    return Path(staging_dir) / table / f"{season_label(start_year)}.parquet"


def fetch_season(
    table: str,
    start_year: int,
    *,
    cache_dir: Path | str = CACHE_DIR,
    staging_dir: Path | str = STAGING_DIR,
) -> pd.DataFrame:
    """Fetch and parse one season of one table, writing it to staging."""
    reader_name, parser, _ = TABLES[table]
    client = make_client(start_year, cache_dir)
    check_politeness(client)
    raw = getattr(client, reader_name)()

    if raw.empty:
        raise UnderstatFormatError(
            f"Understat returned no {table} rows for {season_label(start_year)}."
        )

    check_season_matches(raw, start_year, table)
    parsed = parser(raw, start_year)
    destination = staged_path(table, start_year, staging_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed.to_parquet(destination, index=False)
    return parsed


def collect(
    table: str,
    start_years: list[int],
    *,
    cache_dir: Path | str = CACHE_DIR,
    staging_dir: Path | str = STAGING_DIR,
    checkpoint_path: Path | str = CHECKPOINT_PATH,
) -> pd.DataFrame:
    """Fetch every requested season of one table, resuming where it left off.

    Seasons already recorded in the checkpoint are read from staging instead of
    being re-parsed. Anything else is fetched (from soccerdata's cache when it
    has been fetched before, from the network when it has not).
    """
    checkpoint = Checkpoint(checkpoint_path)
    keys = [f"{table}/{season_label(year)}" for year in start_years]
    report_progress(checkpoint, keys, f"Understat {table}")

    frames = []
    for start_year in start_years:
        key = f"{table}/{season_label(start_year)}"
        staged = staged_path(table, start_year, staging_dir)

        if checkpoint.is_done(key) and staged.exists():
            frames.append(pd.read_parquet(staged))
            continue

        logger.info("Understat %s: fetching %s", table, season_label(start_year))
        parsed = fetch_season(
            table, start_year, cache_dir=cache_dir, staging_dir=staging_dir
        )
        checkpoint.mark_done(key, rows=len(parsed))
        frames.append(parsed)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def already_staged_years(
    table: str,
    *,
    staging_dir: Path | str = STAGING_DIR,
    checkpoint_path: Path | str = CHECKPOINT_PATH,
) -> list[int]:
    """Seasons of a table we have already fetched and parsed.

    Used to widen the per-match tables for free. Those default to a recent
    window because each season costs hundreds of requests - but a season we
    downloaded on an earlier run costs nothing to include, and throwing it away
    would mean having scraped it for nothing.
    """
    checkpoint = Checkpoint(checkpoint_path)
    years = []
    for start_year in available_start_years():
        key = f"{table}/{season_label(start_year)}"
        if checkpoint.is_done(key) and staged_path(table, start_year, staging_dir).exists():
            years.append(start_year)
    return years


def build_all(
    *,
    start_years: list[int] | None = None,
    shot_years: list[int] | None = None,
    player_match_years: list[int] | None = None,
    cache_dir: Path | str = CACHE_DIR,
    staging_dir: Path | str = STAGING_DIR,
    checkpoint_path: Path | str = CHECKPOINT_PATH,
    processed_dir: Path | str = PROCESSED_DIR,
    today: date | None = None,
) -> dict[str, Path]:
    """Pull every Understat table and write the processed parquets.

    ``start_years`` covers the cheap season-level tables (one request each) and
    defaults to the full 2014/15-to-now history. ``shot_years`` and
    ``player_match_years`` cover the expensive per-match tables and default to
    the recent window - see :data:`DEFAULT_SHOT_SEASONS` for why.

    Tables are built cheapest-first, so an interrupted run has already produced
    the season-level tables before it starts the slow per-match ones. Shots and
    player match logs come from the same cached match files, so whichever runs
    first pays for both.

    Returns ``{table name: parquet path}``.
    """
    start_years = start_years if start_years is not None else available_start_years(today)
    shot_years = shot_years if shot_years is not None else shot_start_years(today)
    player_match_years = (
        player_match_years if player_match_years is not None else shot_years
    )
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Seasons of the per-match tables we already hold cost nothing to include,
    # so fold them in rather than discarding data we have already fetched.
    def widen(table: str, requested: list[int]) -> list[int]:
        free = already_staged_years(
            table, staging_dir=staging_dir, checkpoint_path=checkpoint_path
        )
        extra = sorted(set(free) - set(requested))
        if extra:
            logger.info(
                "Understat %s: also including %d already-downloaded season(s) "
                "at no request cost: %s",
                table, len(extra), ", ".join(season_label(y) for y in extra),
            )
        return sorted(set(requested) | set(free))

    years_for_table = {
        "team_match": start_years,
        "player_season": start_years,
        "shots": widen("shots", shot_years),
        "player_match": widen("player_match", player_match_years),
    }
    # Cheapest first: one request per season, then one request per match.
    order = ("team_match", "player_season", "shots", "player_match")

    written: dict[str, Path] = {}
    for table in order:
        _, _, default_path = TABLES[table]
        frame = collect(
            table, years_for_table[table],
            cache_dir=cache_dir, staging_dir=staging_dir, checkpoint_path=checkpoint_path,
        )
        if frame.empty:
            logger.warning("Understat %s produced no rows; not writing.", table)
            continue

        destination = processed_dir / default_path.name
        frame.to_parquet(destination, index=False)
        written[table] = destination
        logger.info("Wrote %d %s rows to %s", len(frame), table, destination)

    return written
