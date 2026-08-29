"""Club Elo ratings, from the free API at api.clubelo.com.

Elo is a single number summarising how strong a club is, updated after every
match: beat someone stronger than you and it goes up, lose to someone weaker and
it drops further. Club Elo (clubelo.com) maintains these for European clubs and
publishes them as plain CSV over HTTP, no key and no scraping needed.

Why we want it: our own team strengths are built from Premier League matches
only, so a promoted club arrives with no history at all and a club that has been
playing in Europe looks the same as one that has not. Elo has seen all of that,
so it gives the model a sensible prior for a team we know little about, and an
independent check on the strengths we estimate ourselves.

Two endpoints matter:

``api.clubelo.com/{Club}``
    One club's entire rating history, one row per period, with the dates each
    rating was valid between. This is what we pull.
``api.clubelo.com/{YYYY-MM-DD}``
    Every club's rating on one date. Useful for spot checks; not used here.

Look-ups go through :func:`get_elo`, which answers "what was this team's rating
on this date" - the question the model actually asks when building features for
a fixture.

Reachability
------------
Club Elo was unreachable from the environment this module was written in: both
HTTP and HTTPS to api.clubelo.com failed to connect, in two separate sessions.
That looks like a network restriction rather than the site being down, since it
serves ordinary CSV over a plain connection.

So this module is written **offline-first**: every parsing, lookup and merging
function works on data you already have, and is tested that way with no network.
:func:`fetch_club_history` is the only function that touches the internet, and
:func:`build_elo_history` will happily rebuild the parquet from the cached raw
CSVs in ``data/raw/clubelo/`` without making a single request. **The download
path itself is unverified against the live API** - see data/lookups/NOTES.md.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from src.lookups import load_team_names, to_canonical

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "clubelo"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ELO_HISTORY_PARQUET = PROCESSED_DIR / "elo_history.parquet"

SOURCE = "clubelo"
BASE_URL = "http://api.clubelo.com/{club}"
USER_AGENT = (
    "premforecaster/0.1 (personal, non-commercial football forecasting project)"
)

#: Club Elo asks for no more than one request a second. We take one every two,
#: with jitter, which is ample for ~36 clubs.
REQUEST_INTERVAL_SECONDS = 2.0
REQUEST_JITTER_SECONDS = 1.0

#: The columns Club Elo's per-club CSV has. ``From``/``To`` bracket the dates a
#: rating was in force, which is what makes an as-of lookup possible.
EXPECTED_RAW_COLUMNS = ("Club", "Country", "Level", "Elo", "From", "To")

#: Club Elo's own far-future end date for a club's current rating.
OPEN_ENDED_TO = pd.Timestamp("2100-01-01")

ELO_COLUMNS = ["team", "clubelo_name", "country", "level", "elo", "valid_from", "valid_to"]

#: A Premier League club sits somewhere in this band. Outside it the data is
#: wrong, not surprising: the very best clubs in Europe peak near 2100, and a
#: newly promoted side is rarely below 1300.
PLAUSIBLE_ELO_RANGE = (1000.0, 2200.0)


class ClubEloUnavailableError(RuntimeError):
    """Raised when api.clubelo.com cannot be reached.

    Known to happen in restricted network environments. The processed parquet
    can still be rebuilt from cached raw CSVs, so this is not fatal to the
    pipeline - it just means no fresh ratings.
    """


class ClubEloFormatError(ValueError):
    """Raised when Club Elo returns something we do not recognise."""


# ---------------------------------------------------------------------------
# Which clubs to pull
# ---------------------------------------------------------------------------


def clubelo_names(path: Path | str | None = None) -> list[str]:
    """Every club we track, spelled the way Club Elo spells it.

    Read straight from the lookup, so adding a promoted club to
    team_names.csv is all that is needed to start pulling its ratings.
    """
    return sorted(load_team_names(path)["clubelo_name"])


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------


def raw_path(clubelo_name: str, downloaded_on: date, raw_dir: Path | str = RAW_DIR) -> Path:
    """Date-stamped, so a later download never overwrites an earlier one."""
    safe = clubelo_name.replace(" ", "_").replace("/", "-")
    return Path(raw_dir) / f"{safe}_downloaded-{downloaded_on:%Y-%m-%d}.csv"


def fetch_club_history(
    clubelo_name: str,
    *,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> str:
    """Download one club's full rating history and return the raw CSV text.

    The only function here that touches the network.
    """
    url = BASE_URL.format(club=clubelo_name.replace(" ", "%20"))
    owns_session = session is None
    session = session or requests.Session()

    try:
        response = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as error:
        raise ClubEloUnavailableError(
            f"Could not reach Club Elo for {clubelo_name!r} at {url}: "
            f"{type(error).__name__}: {error}. api.clubelo.com is a plain HTTP "
            "CSV endpoint; if this fails everywhere, the network is probably "
            "blocking it. The processed parquet can still be rebuilt from any "
            "cached CSVs in data/raw/clubelo/."
        ) from error
    finally:
        if owns_session:
            session.close()

    text = response.text
    if not text.strip():
        raise ClubEloFormatError(f"Club Elo returned an empty file for {clubelo_name!r}.")
    if "Elo" not in text.split("\n", 1)[0]:
        raise ClubEloFormatError(
            f"Club Elo's response for {clubelo_name!r} does not look like its usual "
            f"CSV: the header row has no Elo column. First 200 characters: {text[:200]!r}"
        )
    return text


def download_all(
    clubs: list[str] | None = None,
    raw_dir: Path | str = RAW_DIR,
    *,
    today: date | None = None,
    skip_failures: bool = True,
) -> dict[str, Path]:
    """Download every club's history, saving each untouched to ``raw_dir``.

    Returns ``{clubelo name: raw file path}``. With ``skip_failures`` set, clubs
    that cannot be fetched are logged and skipped, so one unreachable club does
    not cost the whole run.
    """
    clubs = clubs if clubs is not None else clubelo_names()
    today = today or date.today()
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    failures: list[str] = []

    with requests.Session() as session:
        for index, club in enumerate(clubs):
            destination = raw_path(club, today, raw_dir)
            if destination.exists():
                logger.debug("Already downloaded today: %s", destination.name)
                saved[club] = destination
                continue

            if index:
                time.sleep(REQUEST_INTERVAL_SECONDS + random.random() * REQUEST_JITTER_SECONDS)

            try:
                text = fetch_club_history(club, session=session)
            except (ClubEloUnavailableError, ClubEloFormatError) as error:
                if not skip_failures:
                    raise
                logger.warning("Skipping %s: %s", club, error)
                failures.append(club)
                continue

            destination.write_text(text, encoding="utf-8")
            saved[club] = destination
            logger.info("Saved %s (%d bytes)", destination.name, len(text))

    if failures:
        logger.warning(
            "Club Elo: %d of %d club(s) could not be downloaded: %s",
            len(failures), len(clubs), ", ".join(failures[:8]),
        )
    return saved


def find_raw_files(raw_dir: Path | str = RAW_DIR) -> dict[str, Path]:
    """The newest downloaded CSV for each club, keyed by the file's club name."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return {}

    newest: dict[str, Path] = {}
    for path in sorted(raw_dir.glob("*_downloaded-*.csv")):
        club = path.stem.rsplit("_downloaded-", 1)[0].replace("_", " ")
        newest[club] = path
    return newest


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_club_history(csv_text: str) -> pd.DataFrame:
    """Turn one club's raw CSV into tidy rows with canonical team names.

    Club Elo gives one row per period a rating held, bracketed by ``From`` and
    ``To``. We keep that shape rather than expanding to one row per day, because
    it is far smaller and an as-of lookup is just as easy either way.
    """
    frame = pd.read_csv(StringIO(csv_text))
    frame.columns = [str(column).strip() for column in frame.columns]

    missing = [column for column in EXPECTED_RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ClubEloFormatError(
            f"Club Elo CSV is missing the column(s) {missing}. The API may have "
            f"changed format. Columns present: {list(frame.columns)}"
        )

    if frame.empty:
        raise ClubEloFormatError("Club Elo CSV has a header but no rows.")

    elo = pd.to_numeric(frame["Elo"], errors="coerce")
    if elo.isna().any():
        bad = frame.loc[elo.isna(), "Elo"].unique()[:5].tolist()
        raise ClubEloFormatError(f"Non-numeric Elo value(s): {bad}")

    # Club Elo writes plain ISO dates. Pinning the format keeps parsing
    # predictable and turns anything unexpected into a loud failure below,
    # rather than letting pandas guess differently from row to row.
    valid_from = pd.to_datetime(frame["From"], format="%Y-%m-%d", errors="coerce")
    valid_to = pd.to_datetime(frame["To"], format="%Y-%m-%d", errors="coerce")
    if valid_from.isna().any() or valid_to.isna().any():
        raise ClubEloFormatError(
            "Club Elo has unparseable From/To dates; refusing to guess when a "
            "rating was in force."
        )

    tidy = pd.DataFrame(
        {
            "team": to_canonical(frame["Club"], SOURCE),
            "clubelo_name": frame["Club"].astype("string").str.strip(),
            "country": frame["Country"].astype("string").str.strip(),
            "level": pd.to_numeric(frame["Level"], errors="coerce").astype("Int64"),
            "elo": elo.astype("float64"),
            "valid_from": valid_from,
            "valid_to": valid_to,
        }
    )

    backwards = tidy["valid_to"] < tidy["valid_from"]
    if backwards.any():
        raise ClubEloFormatError(
            f"{int(backwards.sum())} Club Elo row(s) end before they start."
        )

    return tidy.sort_values("valid_from").reset_index(drop=True)


def build_elo_history(
    raw_dir: Path | str = RAW_DIR,
    *,
    clubs: dict[str, Path] | None = None,
) -> pd.DataFrame:
    """Combine every cached club CSV into one tidy history table.

    Makes no network requests: it reads whatever is already in ``raw_dir``.
    """
    files = clubs if clubs is not None else find_raw_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No Club Elo CSVs in {raw_dir}. Run the download step first - and "
            "note that api.clubelo.com is unreachable from some networks."
        )

    frames = [parse_club_history(path.read_text(encoding="utf-8")) for path in files.values()]
    history = pd.concat(frames, ignore_index=True)
    history = history.sort_values(["team", "valid_from"]).reset_index(drop=True)

    overlapping = _find_overlaps(history)
    if overlapping:
        raise ClubEloFormatError(
            f"Overlapping rating periods for {overlapping[:5]}. An as-of lookup "
            "would be ambiguous, so this is refused rather than guessed at."
        )

    return history[ELO_COLUMNS]


def _find_overlaps(history: pd.DataFrame) -> list[str]:
    """Teams whose rating periods overlap, which would break as-of lookups."""
    offenders = []
    for team, rows in history.groupby("team"):
        rows = rows.sort_values("valid_from")
        previous_end = rows["valid_to"].shift()
        if (rows["valid_from"] < previous_end).any():
            offenders.append(str(team))
    return offenders


def write_elo_history(history: pd.DataFrame, path: Path | str = ELO_HISTORY_PARQUET) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(path, index=False)
    logger.info("Wrote %d Elo rows to %s", len(history), path)
    return path


# ---------------------------------------------------------------------------
# Looking a rating up
# ---------------------------------------------------------------------------


def load_elo_history(path: Path | str = ELO_HISTORY_PARQUET) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No Elo history at {path}. Build it with pipelines/build_context.py."
        )
    return pd.read_parquet(path)


def _as_timestamp(when: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(when)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize()


def get_elo(
    team: str,
    when: str | date | datetime | pd.Timestamp,
    history: pd.DataFrame | None = None,
    *,
    path: Path | str = ELO_HISTORY_PARQUET,
    strict: bool = False,
) -> float | None:
    """What was ``team`` rated on ``when``?

    ``team`` is a canonical name, ``when`` anything pandas reads as a date.
    Timezone-aware inputs are converted to UTC first, so passing a kick-off
    straight from results.parquet works.

    Returns the rating in force on that date. If the date falls in a gap - Club
    Elo does not rate a club while it is outside the leagues it covers - the
    most recent earlier rating is returned instead, which is the honest answer
    to "how good were they": their last known strength. Returns ``None`` when
    nothing is known yet (a date before the club's first rating), or raises if
    ``strict``.

    A note on using this for features: always look up the rating **before** the
    match, never on the day it was updated, or the feature leaks the result of
    the match you are trying to predict.
    """
    history = history if history is not None else load_elo_history(path)
    stamp = _as_timestamp(when)

    rows = history[history["team"] == team]
    if rows.empty:
        message = (
            f"No Elo history for {team!r}. Canonical names only - check "
            "data/lookups/team_names.csv."
        )
        if strict:
            raise KeyError(message)
        logger.warning(message)
        return None

    current = rows[(rows["valid_from"] <= stamp) & (rows["valid_to"] >= stamp)]
    if not current.empty:
        return float(current.iloc[-1]["elo"])

    earlier = rows[rows["valid_from"] <= stamp]
    if not earlier.empty:
        latest = earlier.sort_values("valid_from").iloc[-1]
        logger.debug(
            "No Elo period covers %s for %s; using their rating from %s.",
            stamp.date(), team, latest["valid_from"].date(),
        )
        return float(latest["elo"])

    message = (
        f"No Elo for {team!r} on or before {stamp.date()}; their first rating is "
        f"{rows['valid_from'].min().date()}."
    )
    if strict:
        raise KeyError(message)
    logger.debug(message)
    return None


def add_elo_to_fixtures(
    fixtures: pd.DataFrame,
    history: pd.DataFrame | None = None,
    *,
    path: Path | str = ELO_HISTORY_PARQUET,
    date_column: str = "date",
) -> pd.DataFrame:
    """Attach home and away Elo (and the difference) to a table of fixtures.

    The difference is usually the more useful feature: Elo is on an arbitrary
    scale, and what predicts a result is the gap between the two sides.
    """
    history = history if history is not None else load_elo_history(path)
    result = fixtures.copy()

    for side in ("home", "away"):
        column = f"{side}_team"
        if column not in result.columns:
            raise ValueError(f"Fixtures table has no {column!r} column.")
        result[f"{side}_elo"] = [
            get_elo(team, when, history)
            for team, when in zip(result[column], result[date_column])
        ]

    result["elo_difference"] = result["home_elo"] - result["away_elo"]
    return result
