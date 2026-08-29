"""Premier League results and closing odds from football-data.co.uk.

football-data.co.uk publishes one CSV per season per division. The Premier
League is division ``E0``, and each season lives at a URL built from a
four-digit season code, so 2014/15 is ``.../mmz4281/1415/E0.csv``. These are
plain file downloads, not a scrape - there is no HTML to parse and no crawling
involved - but we still send a descriptive User-Agent and pause between files.

This module does two jobs:

1. **Download** each season CSV and drop it, byte for byte untouched, into
   ``data/raw/footballdata/`` with the download date in the filename. Files are
   never overwritten, so we always keep the raw record of what we fetched.
2. **Process** those raw files into ``data/processed/results.parquet``: one tidy
   row per match, canonical team names, UTC kick-off times, and the best
   available closing odds converted into de-margined market probabilities.

Quirks of this source that the code below deliberately handles
---------------------------------------------------------------

*The date format changes between seasons, and not in a tidy way.* 2014/15 uses
``16/08/14``, 2015/16 uses ``09/08/2015``, 2016/17 goes back to two digits, and
from 2017/18 onward it is four. Both formats are day-first. We try both.

*Kick-off times only exist from 2019/20 onwards.* The five seasons before that
have a date and no time at all. Where the time is known it is UK local time, so
a 20:00 August kick-off is 19:00 UTC - it must be converted, not just labelled.
Where it is unknown we store midnight UTC on the match date and set
``kickoff_time_known`` to False, so nothing downstream mistakes a placeholder
for a real kick-off time.

*The available bookmakers change over time, mid-season.* Pinnacle closing odds
(``PSC*``) cover 2014/15 to 2025/26, but disappear partway through January 2026
and are absent from 2026/27 entirely. Bet365 and market-average closing odds
(``B365C*``, ``AvgC*``) only start in 2019/20. So odds are chosen **per row**,
not per season, and every row records which bookmaker it used in
``odds_source``.

*Older files have a trailing blank row.* The 2014/15 file has 381 rows, one of
which is entirely empty. Blank rows are dropped before anything else happens.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.lookups import to_canonical

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "footballdata"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_PARQUET = PROCESSED_DIR / "results.parquet"

SOURCE = "footballdata"
DIVISION = "E0"  # the Premier League
BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/{division}.csv"
USER_AGENT = (
    "premforecaster/0.1 (personal, non-commercial football forecasting project)"
)

#: Understat's xG data starts in 2014/15, so that is our horizon for everything.
FIRST_SEASON_START_YEAR = 2014

#: A Premier League season is named for the calendar year it starts in. From
#: July onwards we consider the new season to have begun, which is comfortably
#: before the first fixture in mid-August.
SEASON_START_MONTH = 7

#: Both spellings the Date column has used. Day first in both cases.
DATE_FORMATS = ("%d/%m/%Y", "%d/%m/%y")

#: Kick-off times in the file are UK wall-clock time, not UTC.
SOURCE_TIMEZONE = "Europe/London"

#: Closing odds in order of preference. Pinnacle is a sharp book with a thin
#: margin, so it is the best available read on true probability; Bet365 is the
#: next best; the market average is the fallback. All three are *closing* prices
#: (the "C" in the column name), which is what we want - closing odds embody
#: everything the market learned before kick-off, including team news.
CLOSING_ODDS_SOURCES: tuple[tuple[str, tuple[str, str, str]], ...] = (
    ("pinnacle_closing", ("PSCH", "PSCD", "PSCA")),
    ("bet365_closing", ("B365CH", "B365CD", "B365CA")),
    ("market_average_closing", ("AvgCH", "AvgCD", "AvgCA")),
)

#: What odds_source says when a row has no usable closing prices at all.
NO_ODDS = "none"

#: Decimal odds must pay out more than the stake. Anything at or below this is
#: a placeholder or a corrupt value, not a real price.
MIN_VALID_ODDS = 1.01

#: The overround (sum of implied probabilities before de-margining) of a real
#: three-way football market. Pinnacle sits near 1.02, a high-street book nearer
#: 1.08. Anything outside this range means the prices are wrong.
OVERROUND_BOUNDS = (0.90, 1.50)

REQUIRED_RAW_COLUMNS = (
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
)

RESULTS_COLUMNS = [
    "date",
    "kickoff_time_known",
    "season",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "referee",
    "odds_home",
    "odds_draw",
    "odds_away",
    "odds_source",
    "market_p_home",
    "market_p_draw",
    "market_p_away",
    "market_overround",
]


# ---------------------------------------------------------------------------
# Season naming
# ---------------------------------------------------------------------------
# A season is identified internally by the calendar year it starts in (2014),
# labelled for humans as "2014-15", and requested from the website as "1415".


def season_label(start_year: int) -> str:
    """2014 -> ``"2014-15"``. This is what goes in the ``season`` column."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_code(start_year: int) -> str:
    """2014 -> ``"1415"``, the code football-data.co.uk uses in its URLs."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_start_year(label: str) -> int:
    """``"2014-15"`` -> 2014. The inverse of :func:`season_label`."""
    try:
        start_year = int(label.split("-")[0])
    except (ValueError, IndexError) as error:
        raise ValueError(
            f"Could not read a season from {label!r}. Expected 'YYYY-YY', e.g. '2014-15'."
        ) from error

    if season_label(start_year) != label:
        raise ValueError(
            f"Malformed season label {label!r}. Expected {season_label(start_year)!r}."
        )
    return start_year


def current_season_start_year(today: date | None = None) -> int:
    """Which season are we in right now?

    Seasons run August to May, so anything from July onwards belongs to the
    season starting this year, and anything before July to the one that started
    last year.
    """
    today = today or date.today()
    return today.year if today.month >= SEASON_START_MONTH else today.year - 1


def season_start_years(
    through: int | None = None,
    first: int = FIRST_SEASON_START_YEAR,
    today: date | None = None,
) -> list[int]:
    """Every season from 2014/15 up to and including the current one."""
    through = through if through is not None else current_season_start_year(today)
    if through < first:
        raise ValueError(
            f"Season {season_label(through)} is before the first season we cover "
            f"({season_label(first)})."
        )
    return list(range(first, through + 1))


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------


def raw_filename(start_year: int, downloaded_on: date) -> str:
    """e.g. ``E0_2014-15_downloaded-2026-08-29.csv``.

    The download date is in the name so a later download never overwrites an
    earlier one, which is the rule for everything in data/raw/.
    """
    return f"{DIVISION}_{season_label(start_year)}_downloaded-{downloaded_on:%Y-%m-%d}.csv"


def season_url(start_year: int) -> str:
    return BASE_URL.format(code=season_code(start_year), division=DIVISION)


def download_season(
    start_year: int,
    raw_dir: Path | str = RAW_DIR,
    *,
    session: requests.Session | None = None,
    today: date | None = None,
    timeout: int = 60,
    retries: int = 3,
) -> Path:
    """Download one season CSV and save it untouched to ``raw_dir``.

    If we already downloaded this season today, the existing file is returned
    and nothing is re-fetched. Returns the path to the raw file.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / raw_filename(start_year, today or date.today())

    if destination.exists():
        logger.info("Already downloaded today, skipping: %s", destination.name)
        return destination

    url = season_url(start_year)
    owns_session = session is None
    session = session or requests.Session()

    try:
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = session.get(
                    url, headers={"User-Agent": USER_AGENT}, timeout=timeout
                )
                response.raise_for_status()
                break
            except requests.RequestException as error:
                last_error = error
                if attempt == retries:
                    raise RuntimeError(
                        f"Could not download {season_label(start_year)} from {url} "
                        f"after {retries} attempts: {error}"
                    ) from error
                backoff = 2**attempt
                logger.warning(
                    "Download of %s failed (%s). Retrying in %ss.",
                    season_label(start_year),
                    error,
                    backoff,
                )
                time.sleep(backoff)
        else:  # pragma: no cover - the loop always breaks or raises
            raise RuntimeError(str(last_error))
    finally:
        if owns_session:
            session.close()

    content = response.content
    if not content.strip():
        raise ValueError(
            f"{url} returned an empty file. The season CSV may not be published yet."
        )
    if b"HomeTeam" not in content.split(b"\n", 1)[0]:
        raise ValueError(
            f"{url} does not look like a football-data.co.uk match CSV: the header "
            f"row has no HomeTeam column. First 200 bytes: {content[:200]!r}"
        )

    destination.write_bytes(content)
    logger.info("Saved %s (%d bytes)", destination.name, len(content))
    return destination


def download_all(
    raw_dir: Path | str = RAW_DIR,
    *,
    start_years: list[int] | None = None,
    today: date | None = None,
    pause_seconds: float = 1.0,
) -> dict[int, Path]:
    """Download every season from 2014/15 to the current one.

    A short pause with a little jitter separates the requests. These are static
    files, so this is politeness rather than a rate limit.

    The current season is allowed to be missing (in July the new file may not be
    published yet); any *historic* season failing to download is an error,
    because those files never change and should always be there.
    """
    start_years = start_years if start_years is not None else season_start_years(today=today)
    latest = max(start_years)
    downloaded: dict[int, Path] = {}

    with requests.Session() as session:
        for index, start_year in enumerate(start_years):
            if index:
                time.sleep(pause_seconds + random.uniform(0, 0.5))
            try:
                downloaded[start_year] = download_season(
                    start_year, raw_dir, session=session, today=today
                )
            except (RuntimeError, ValueError):
                if start_year == latest:
                    logger.warning(
                        "Could not download the current season (%s) - it may not be "
                        "published yet. Carrying on with earlier seasons.",
                        season_label(start_year),
                    )
                    continue
                raise

    return downloaded


# ---------------------------------------------------------------------------
# Reading raw files back
# ---------------------------------------------------------------------------


def find_raw_files(raw_dir: Path | str = RAW_DIR) -> dict[int, Path]:
    """Find the most recent raw download for each season.

    Filenames sort correctly by download date because the date is written
    ``YYYY-MM-DD``, so the last one alphabetically is the newest.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"No raw football-data directory at {raw_dir}. Run the download step first."
        )

    newest: dict[int, Path] = {}
    for path in sorted(raw_dir.glob(f"{DIVISION}_*_downloaded-*.csv")):
        label = path.stem.split("_")[1]
        try:
            start_year = season_start_year(label)
        except ValueError:
            logger.warning("Ignoring unrecognised raw filename: %s", path.name)
            continue
        newest[start_year] = path

    if not newest:
        raise FileNotFoundError(
            f"No football-data season CSVs found in {raw_dir}. Run the download step first."
        )
    return dict(sorted(newest.items()))


def read_raw_csv(path: Path | str) -> pd.DataFrame:
    """Read one raw season CSV, dropping the blank rows some files end with.

    Newer files are saved with a byte-order mark, hence ``utf-8-sig``: read as
    plain utf-8 the first column would be named ``\\ufeffDiv`` and every lookup
    of ``Div`` would fail.
    """
    path = Path(path)
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"Date": str, "Time": str})
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.dropna(how="all").reset_index(drop=True)

    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing the column(s) {missing}. football-data.co.uk "
            f"may have changed its format. Columns present: {list(frame.columns)}"
        )

    return frame


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_dates(dates: pd.Series) -> pd.Series:
    """Parse the Date column, coping with both year spellings it has used.

    Tries ``dd/mm/yyyy`` first and ``dd/mm/yy`` for whatever is left, so a file
    mixing the two would still parse. Raises listing the offending values if
    anything is left unparsed, rather than quietly producing NaT.
    """
    text = dates.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")

    for fmt in DATE_FORMATS:
        remaining = parsed.isna() & text.notna()
        if not remaining.any():
            break
        parsed.loc[remaining] = pd.to_datetime(
            text[remaining], format=fmt, errors="coerce"
        )

    unparsed = text[parsed.isna()].dropna().unique().tolist()
    if unparsed:
        raise ValueError(
            f"Could not parse {len(unparsed)} date value(s) using {list(DATE_FORMATS)}: "
            f"{unparsed[:10]}. The Date column format may have changed again."
        )
    if text.isna().any():
        raise ValueError(
            f"{int(text.isna().sum())} row(s) have no date at all. Refusing to guess."
        )

    return parsed


def build_kickoff_utc(
    dates: pd.Series, times: pd.Series | None
) -> tuple[pd.Series, pd.Series]:
    """Turn the Date (and Time, when present) columns into UTC timestamps.

    Returns the UTC timestamps and a boolean flag saying whether the kick-off
    time was actually known.

    The conversion matters. The file writes UK wall-clock time, so an 20:00
    kick-off in August is 19:00 UTC, and in January it is 20:00 UTC. Treating
    the column as if it were already UTC would put matches in the wrong hour for
    two thirds of the season, which would then feed straight into any weather or
    rest-days feature we build later.

    Where the time is unknown we use midnight *UTC* on the match date rather
    than midnight UK time, because localising midnight UK in summer would push
    the timestamp back to 23:00 on the previous day and change the match date.
    """
    day = parse_dates(dates)

    if times is None:
        known = pd.Series(False, index=day.index)
    else:
        known = times.astype("string").str.strip().replace("", pd.NA).notna()

    kickoff = pd.Series(pd.NaT, index=day.index, dtype="datetime64[ns, UTC]")

    if known.any():
        clock = times.astype("string").str.strip()[known]
        local = pd.to_datetime(
            day[known].dt.strftime("%Y-%m-%d") + " " + clock,
            format="%Y-%m-%d %H:%M",
            errors="coerce",
        )
        if local.isna().any():
            bad = clock[local.isna()].unique().tolist()
            raise ValueError(
                f"Could not parse kick-off time(s) {bad[:10]}. Expected 24-hour HH:MM."
            )
        kickoff.loc[known] = local.dt.tz_localize(SOURCE_TIMEZONE).dt.tz_convert("UTC")

    if (~known).any():
        kickoff.loc[~known] = day[~known].dt.tz_localize("UTC")

    return kickoff, known


def select_closing_odds(frame: pd.DataFrame) -> pd.DataFrame:
    """Pick the best available closing odds for each row.

    Preference order is Pinnacle, then Bet365, then the market average, and the
    choice is made **per row**: Pinnacle vanished partway through January 2026,
    so the same season legitimately uses different books for different matches.

    A source is only used if all three prices are present and above 1.01 - a
    partial row is no use, because de-margining needs the whole market. Rows
    with nothing usable get NaN odds and ``odds_source == "none"``.
    """
    index = frame.index
    odds = pd.DataFrame(
        {
            "odds_home": pd.Series(np.nan, index=index, dtype="float64"),
            "odds_draw": pd.Series(np.nan, index=index, dtype="float64"),
            "odds_away": pd.Series(np.nan, index=index, dtype="float64"),
            "odds_source": pd.Series(NO_ODDS, index=index, dtype="object"),
        }
    )

    for name, columns in CLOSING_ODDS_SOURCES:
        if not all(column in frame.columns for column in columns):
            continue

        prices = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
        usable = prices.notna().all(axis=1) & (prices > MIN_VALID_ODDS).all(axis=1)
        fill = usable & (odds["odds_source"] == NO_ODDS)
        if not fill.any():
            continue

        odds.loc[fill, "odds_home"] = prices.loc[fill, columns[0]].to_numpy()
        odds.loc[fill, "odds_draw"] = prices.loc[fill, columns[1]].to_numpy()
        odds.loc[fill, "odds_away"] = prices.loc[fill, columns[2]].to_numpy()
        odds.loc[fill, "odds_source"] = name

    return odds


def add_market_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert closing odds into de-margined probabilities.

    A bookmaker's prices do not sum to a probability of 1 - they sum to slightly
    more, and the excess is the margin (the "overround") they make. Inverting
    each price gives the implied probability including that margin; dividing by
    their total strips it out proportionally, so the three numbers sum to
    exactly 1 and can be compared with our model's forecasts.

    This is the simplest de-margining method and it slightly overstates the
    favourite, because margin is not really spread evenly across outcomes. It is
    the standard benchmark, and good enough to judge the model against. Rows
    with no odds get NaN, never a guess.

    Adds ``market_p_home``, ``market_p_draw``, ``market_p_away`` and
    ``market_overround`` (the pre-de-margining total, kept as a diagnostic).
    """
    result = frame.copy()
    price_columns = ["odds_home", "odds_draw", "odds_away"]

    missing = [column for column in price_columns if column not in result.columns]
    if missing:
        raise ValueError(
            f"Cannot compute market probabilities: missing column(s) {missing}. "
            "Run select_closing_odds first."
        )

    prices = result[price_columns].astype("float64")
    have_odds = prices.notna().all(axis=1)

    implied = 1.0 / prices
    overround = implied.sum(axis=1).where(have_odds)
    probabilities = implied.div(overround, axis=0)

    strange = have_odds & (
        (overround < OVERROUND_BOUNDS[0]) | (overround > OVERROUND_BOUNDS[1])
    )
    if strange.any():
        examples = overround[strange].head(5).round(3).to_dict()
        raise ValueError(
            f"{int(strange.sum())} row(s) have an implausible bookmaker overround "
            f"(expected {OVERROUND_BOUNDS[0]}-{OVERROUND_BOUNDS[1]}): {examples}. "
            "The odds columns are probably corrupt or misaligned."
        )

    result["market_p_home"] = probabilities["odds_home"]
    result["market_p_draw"] = probabilities["odds_draw"]
    result["market_p_away"] = probabilities["odds_away"]
    result["market_overround"] = overround
    return result


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


def process_season(raw: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """Turn one raw season CSV into tidy rows, without the odds probabilities.

    Validates as it goes: goals must be whole non-negative numbers, the result
    letter must agree with the score, and every team name must be in the lookup.
    """
    frame = raw.copy()

    goals = {}
    for column, name in (("FTHG", "home_goals"), ("FTAG", "away_goals")):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any():
            bad = frame.loc[numeric.isna(), column].unique().tolist()
            raise ValueError(
                f"{season_label(start_year)}: non-numeric goal value(s) in {column}: "
                f"{bad[:10]}."
            )
        if (numeric < 0).any() or (numeric % 1 != 0).any():
            raise ValueError(
                f"{season_label(start_year)}: {column} has negative or fractional goals."
            )
        goals[name] = numeric.astype("int64")

    result = frame["FTR"].astype("string").str.strip().str.upper()
    allowed = {"H", "D", "A"}
    unexpected = sorted(set(result.dropna()) - allowed)
    if unexpected:
        raise ValueError(
            f"{season_label(start_year)}: unexpected result code(s) {unexpected}. "
            f"Expected one of {sorted(allowed)}."
        )

    implied = np.sign(goals["home_goals"] - goals["away_goals"])
    expected = pd.Series(implied, index=frame.index).map({1: "H", 0: "D", -1: "A"})
    disagrees = expected != result
    if disagrees.any():
        rows = frame.loc[disagrees, ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]]
        raise ValueError(
            f"{season_label(start_year)}: the result column disagrees with the score "
            f"on {int(disagrees.sum())} row(s):\n{rows.head().to_string()}"
        )

    kickoff, time_known = build_kickoff_utc(
        frame["Date"], frame["Time"] if "Time" in frame.columns else None
    )

    referee = (
        frame["Referee"].astype("string").str.strip().replace("", pd.NA)
        if "Referee" in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="string")
    )

    tidy = pd.DataFrame(
        {
            "date": kickoff,
            "kickoff_time_known": time_known.astype(bool),
            "season": season_label(start_year),
            "home_team": to_canonical(frame["HomeTeam"], SOURCE),
            "away_team": to_canonical(frame["AwayTeam"], SOURCE),
            "home_goals": goals["home_goals"],
            "away_goals": goals["away_goals"],
            "result": result,
            "referee": referee,
        }
    )

    same_team = tidy["home_team"] == tidy["away_team"]
    if same_team.any():
        raise ValueError(
            f"{season_label(start_year)}: {int(same_team.sum())} row(s) have the same "
            "team at home and away. The team name mapping is probably wrong."
        )

    return pd.concat([tidy, select_closing_odds(frame)], axis=1)


def build_results(
    raw_dir: Path | str = RAW_DIR,
    *,
    start_years: list[int] | None = None,
) -> pd.DataFrame:
    """Process every raw season file into one tidy results table."""
    available = find_raw_files(raw_dir)

    if start_years is not None:
        missing = sorted(set(start_years) - set(available))
        if missing:
            raise FileNotFoundError(
                f"No raw file for season(s) {[season_label(y) for y in missing]} in "
                f"{raw_dir}. Download them first."
            )
        available = {year: available[year] for year in sorted(start_years)}

    seasons = [
        process_season(read_raw_csv(path), start_year)
        for start_year, path in available.items()
    ]

    results = pd.concat(seasons, ignore_index=True)
    results = results.sort_values(["date", "home_team"], kind="stable").reset_index(drop=True)
    results = add_market_probabilities(results)

    fixture = ["season", "home_team", "away_team"]
    duplicated = results.duplicated(subset=fixture, keep=False)
    if duplicated.any():
        raise ValueError(
            f"{int(duplicated.sum())} duplicate fixture row(s) found - each pairing "
            f"should appear once per season:\n{results.loc[duplicated, fixture].head().to_string()}"
        )

    return results[RESULTS_COLUMNS]


def write_results(
    results: pd.DataFrame, path: Path | str = RESULTS_PARQUET
) -> Path:
    """Write the tidy results table to parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(path, index=False)
    logger.info("Wrote %d matches to %s", len(results), path)
    return path


def season_summary(results: pd.DataFrame) -> pd.DataFrame:
    """A quick per-season sanity table: matches, dates covered, odds coverage.

    Handy after a run to see at a glance that nothing is missing.
    """
    summary = results.groupby("season").agg(
        matches=("season", "size"),
        first_match=("date", "min"),
        last_match=("date", "max"),
        with_odds=("odds_source", lambda s: int((s != NO_ODDS).sum())),
        kickoff_times=("kickoff_time_known", "sum"),
    )
    summary["complete"] = summary["matches"] == 380
    return summary
