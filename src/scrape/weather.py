"""Match-day weather from Open-Meteo, for every fixture past and future.

Weather is a modest but real effect in football: heavy rain and strong wind both
suppress goals and push games towards scrappier, lower-scoring outcomes. It is
never going to be the biggest term in the model, but it is free, it is knowable
in advance, and it costs nothing to carry.

Open-Meteo is free for non-commercial use and needs no key. Three endpoints are
used, because no single one covers the whole period we care about:

``archive-api.open-meteo.com/v1/archive``
    ERA5 reanalysis, 1940 to a few days ago. The right source for history and
    the one tried first, because it covers our whole horizon back to 2014/15.
``historical-forecast-api.open-meteo.com/v1/forecast``
    What the forecast models actually said, 2016 onward. Used as a fallback when
    the archive cannot be reached.
``api.open-meteo.com/v1/forecast``
    The next couple of weeks, for fixtures that have not been played yet.

Staying well under the limits
-----------------------------
The naive approach - one request per match - would be 4,570 requests. Instead we
group by **stadium and year**: every club plays 19 home games a season at the
same ground, so one request covers all of them. That is about 36 stadiums times
13 years, so roughly 400 requests for the entire history, comfortably inside
Open-Meteo's 10,000-a-day allowance. Every response is cached to
``data/raw/weather/``, so a rerun costs nothing at all.

Kick-off times we do not have
-----------------------------
football-data.co.uk only started publishing kick-off times in 2019/20. For
earlier matches results.parquet stores midnight UTC as a placeholder, and
midnight weather is not match weather - it is the middle of the night. For those
matches we use :data:`ASSUMED_KICKOFF_HOUR_UTC` instead, a typical afternoon
kick-off, and set ``kickoff_time_known`` to False on the row so nothing
downstream mistakes the two for equally reliable.
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from src.lookups import load_stadiums

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "weather"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MATCH_WEATHER_PARQUET = PROCESSED_DIR / "match_weather.parquet"

USER_AGENT = (
    "premforecaster/0.1 (personal, non-commercial football forecasting project)"
)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

#: The hourly variables we ask for. Wind and rain are the ones with a plausible
#: effect on play; temperature is cheap to carry and may matter at the extremes.
HOURLY_VARIABLES = ("temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m")

#: Sources tried in order for a past date. The archive is the real answer; the
#: historical-forecast API is a fallback for networks that cannot reach it, and
#: only goes back to 2016.
HISTORICAL_SOURCES = (
    ("archive", ARCHIVE_URL, None),
    ("historical_forecast", HISTORICAL_FORECAST_URL, date(2016, 1, 1)),
)

#: Used when football-data did not publish a kick-off time (before 2019/20).
#: 15:00 UTC is the traditional Saturday three o'clock, near enough.
ASSUMED_KICKOFF_HOUR_UTC = 15

#: Open-Meteo allows 10,000 calls a day on the free tier. We use ~400 for the
#: full history, so this delay is politeness rather than necessity.
REQUEST_INTERVAL_SECONDS = 0.5
REQUEST_JITTER_SECONDS = 0.3

#: A forecast is only meaningful this far ahead.
FORECAST_HORIZON_DAYS = 14

WEATHER_COLUMNS = [
    "date", "season", "home_team", "away_team", "stadium",
    "latitude", "longitude", "kickoff_time_known",
    "temperature_c", "precipitation_mm", "wind_speed_kmh", "humidity_pct",
    "weather_source",
]

#: Sanity bounds for UK match-day weather. Outside these the data is wrong.
PLAUSIBLE_TEMPERATURE_C = (-20.0, 45.0)
PLAUSIBLE_WIND_KMH = (0.0, 200.0)
PLAUSIBLE_PRECIPITATION_MM = (0.0, 100.0)


class WeatherUnavailableError(RuntimeError):
    """Raised when no Open-Meteo endpoint could serve a request."""


class WeatherFormatError(ValueError):
    """Raised when Open-Meteo returns something we do not recognise."""


# ---------------------------------------------------------------------------
# Stadium coordinates
# ---------------------------------------------------------------------------


def stadium_coordinates(path: Path | str | None = None) -> pd.DataFrame:
    """Canonical team -> stadium name and coordinates, from the lookup.

    Note this holds one ground per club, so a club that has moved (Everton,
    Tottenham, West Ham, Brentford all have within our window) gets its current
    ground for historical matches too. See data/lookups/NOTES.md - making this
    date-effective is the known next step for weather in the back-test.
    """
    stadiums = load_stadiums(path)
    return stadiums.set_index("canonical_name")[["stadium", "latitude", "longitude"]]


# ---------------------------------------------------------------------------
# Working out what to request
# ---------------------------------------------------------------------------


def kickoff_hours(matches: pd.DataFrame) -> pd.Series:
    """The UTC hour to read weather at, filling in a sensible one where unknown."""
    stamps = pd.to_datetime(matches["date"], utc=True)
    known = (
        matches["kickoff_time_known"].astype(bool)
        if "kickoff_time_known" in matches.columns
        else pd.Series(True, index=matches.index)
    )
    return stamps.dt.floor("h").where(
        known, stamps.dt.normalize() + pd.Timedelta(hours=ASSUMED_KICKOFF_HOUR_UTC)
    )


def request_plan(matches: pd.DataFrame, stadiums: pd.DataFrame) -> pd.DataFrame:
    """One row per (stadium, year) we need to fetch - the batching that keeps calls low.

    Every club plays 19 home matches a season at the same ground, so a single
    request covers all of them.
    """
    unknown = sorted(set(matches["home_team"]) - set(stadiums.index))
    if unknown:
        raise WeatherFormatError(
            f"No stadium in the lookup for {unknown}. Add them to "
            "data/lookups/stadiums.csv before fetching weather."
        )

    plan = pd.DataFrame(
        {
            "home_team": matches["home_team"].to_numpy(),
            "year": pd.to_datetime(matches["date"], utc=True).dt.year.to_numpy(),
        }
    ).drop_duplicates()

    plan = plan.join(stadiums, on="home_team")
    return plan.sort_values(["home_team", "year"]).reset_index(drop=True)


def _cache_path(home_team: str, year: int, raw_dir: Path | str) -> Path:
    safe = str(home_team).replace(" ", "_").replace("/", "-").replace("&", "and")
    return Path(raw_dir) / f"{safe}_{year}.json"


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _call(
    url: str, parameters: dict, *, session: requests.Session | None = None, timeout: int = 60
) -> dict:
    owns_session = session is None
    session = session or requests.Session()
    try:
        response = session.get(
            url, params=parameters, headers={"User-Agent": USER_AGENT}, timeout=timeout
        )
        if response.status_code == 400:
            # Open-Meteo explains itself in the body; surface that rather than a bare 400.
            try:
                reason = response.json().get("reason", response.text[:200])
            except json.JSONDecodeError:
                reason = response.text[:200]
            raise WeatherFormatError(f"Open-Meteo rejected the request: {reason}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        raise WeatherUnavailableError(
            f"Could not reach {url}: {type(error).__name__}: {error}"
        ) from error
    finally:
        if owns_session:
            session.close()


def fetch_hourly(
    latitude: float,
    longitude: float,
    start: date,
    end: date,
    *,
    session: requests.Session | None = None,
    today: date | None = None,
) -> tuple[dict, str]:
    """Fetch hourly weather for one location over one date range.

    Picks the endpoint by date: the forecast API for anything in the future,
    otherwise the archive with the historical-forecast API as a fallback.
    Returns the payload and the name of the source that served it, so every row
    can record where its weather came from.
    """
    today = today or date.today()
    parameters = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
    }

    if start > today:
        payload = _call(FORECAST_URL, parameters, session=session)
        return payload, "forecast"

    problems = []
    for name, url, earliest in HISTORICAL_SOURCES:
        if earliest is not None and start < earliest:
            problems.append(f"{name}: only covers {earliest} onwards")
            continue
        try:
            payload = _call(url, parameters, session=session)
        except (WeatherUnavailableError, WeatherFormatError) as error:
            problems.append(f"{name}: {error}")
            continue
        if _has_any_values(payload):
            return payload, name
        problems.append(f"{name}: responded but with no values for this range")

    raise WeatherUnavailableError(
        f"No Open-Meteo source could supply {start} to {end} at "
        f"({latitude}, {longitude}). Tried:\n  " + "\n  ".join(problems)
    )


def _has_any_values(payload: dict) -> bool:
    hourly = payload.get("hourly") or {}
    values = hourly.get("temperature_2m") or []
    return any(value is not None for value in values)


def hourly_frame(payload: dict) -> pd.DataFrame:
    """Turn one Open-Meteo response into a tidy hourly table indexed by UTC time."""
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise WeatherFormatError(
            f"Open-Meteo response has no hourly data. Keys: {list(payload)[:10]}"
        )

    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame.set_index("time").sort_index()


# ---------------------------------------------------------------------------
# Building the table
# ---------------------------------------------------------------------------


def collect_hourly(
    plan: pd.DataFrame,
    *,
    raw_dir: Path | str = RAW_DIR,
    today: date | None = None,
    skip_failures: bool = True,
) -> dict[tuple[str, int], pd.DataFrame]:
    """Fetch (or read from cache) the hourly weather for every planned request.

    Cached responses are reused without a request, so a rerun is free.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    today = today or date.today()

    collected: dict[tuple[str, int], pd.DataFrame] = {}
    sources: dict[tuple[str, int], str] = {}
    failures: list[str] = []
    fetched = 0

    with requests.Session() as session:
        for row in plan.itertuples(index=False):
            key = (row.home_team, int(row.year))
            cache = _cache_path(row.home_team, int(row.year), raw_dir)

            if cache.exists():
                payload = json.loads(cache.read_text(encoding="utf-8"))
                source = payload.get("_premforecaster_source", "cache")
            else:
                if fetched:
                    time.sleep(
                        REQUEST_INTERVAL_SECONDS + random.random() * REQUEST_JITTER_SECONDS
                    )
                start = date(int(row.year), 1, 1)
                end = date(int(row.year), 12, 31)
                # Do not ask the forecast API for further ahead than it can see.
                horizon = today + timedelta(days=FORECAST_HORIZON_DAYS)
                if end > horizon:
                    end = max(horizon, start)

                try:
                    payload, source = fetch_hourly(
                        row.latitude, row.longitude, start, end,
                        session=session, today=today,
                    )
                except (WeatherUnavailableError, WeatherFormatError) as error:
                    if not skip_failures:
                        raise
                    logger.warning("No weather for %s %s: %s", row.home_team, row.year, error)
                    failures.append(f"{row.home_team} {row.year}")
                    continue

                payload["_premforecaster_source"] = source
                cache.write_text(json.dumps(payload), encoding="utf-8")
                fetched += 1

            try:
                collected[key] = hourly_frame(payload)
                sources[key] = source
            except WeatherFormatError as error:
                if not skip_failures:
                    raise
                logger.warning("Unreadable weather for %s: %s", key, error)
                failures.append(f"{row.home_team} {row.year}")

    logger.info(
        "Weather: %d location-year(s) available (%d newly fetched, %d failed).",
        len(collected), fetched, len(failures),
    )
    if failures:
        logger.warning("Weather gaps: %s", ", ".join(failures[:10]))

    for key, source in sources.items():
        collected[key].attrs["source"] = source
    return collected


def build_match_weather(
    matches: pd.DataFrame,
    *,
    raw_dir: Path | str = RAW_DIR,
    stadiums: pd.DataFrame | None = None,
    today: date | None = None,
    skip_failures: bool = True,
) -> pd.DataFrame:
    """Attach match-day weather to every match in ``matches``.

    Matches whose weather could not be fetched are still returned, with blank
    weather columns, so the table always has one row per match and the coverage
    gap is visible rather than hidden by a shorter table.
    """
    stadiums = stadiums if stadiums is not None else stadium_coordinates()
    plan = request_plan(matches, stadiums)
    hourly = collect_hourly(plan, raw_dir=raw_dir, today=today, skip_failures=skip_failures)

    frame = matches.copy()
    hours = kickoff_hours(frame)
    years = pd.to_datetime(frame["date"], utc=True).dt.year
    frame = frame.join(stadiums, on="home_team")

    blank = dict.fromkeys(
        ("temperature_c", "precipitation_mm", "wind_speed_kmh", "humidity_pct")
    )
    readings = []
    for team, year, hour in zip(frame["home_team"], years, hours):
        table = hourly.get((team, int(year)))
        if table is None or hour not in table.index:
            readings.append({**blank, "weather_source": None})
            continue

        values = table.loc[hour]
        readings.append(
            {
                "temperature_c": values.get("temperature_2m"),
                "precipitation_mm": values.get("precipitation"),
                "wind_speed_kmh": values.get("wind_speed_10m"),
                "humidity_pct": values.get("relative_humidity_2m"),
                "weather_source": table.attrs.get("source"),
            }
        )

    weather = pd.DataFrame(readings, index=frame.index)
    result = pd.concat(
        [
            frame[["date", "season", "home_team", "away_team", "stadium",
                   "latitude", "longitude"]],
            frame.get("kickoff_time_known", pd.Series(True, index=frame.index)).rename(
                "kickoff_time_known"
            ),
            weather,
        ],
        axis=1,
    )

    for column in ("temperature_c", "precipitation_mm", "wind_speed_kmh", "humidity_pct"):
        result[column] = pd.to_numeric(result[column], errors="coerce")

    check_plausible(result)
    return result[WEATHER_COLUMNS].sort_values("date").reset_index(drop=True)


def check_plausible(weather: pd.DataFrame) -> None:
    """Fail loudly on physically implausible readings rather than modelling them."""
    checks = (
        ("temperature_c", PLAUSIBLE_TEMPERATURE_C),
        ("wind_speed_kmh", PLAUSIBLE_WIND_KMH),
        ("precipitation_mm", PLAUSIBLE_PRECIPITATION_MM),
    )
    for column, (low, high) in checks:
        values = pd.to_numeric(weather[column], errors="coerce").dropna()
        if values.empty:
            continue
        if values.min() < low or values.max() > high:
            raise WeatherFormatError(
                f"{column} has implausible value(s): range "
                f"{values.min():.1f} to {values.max():.1f}, expected {low} to {high}. "
                "Check the units Open-Meteo returned."
            )


def write_match_weather(
    weather: pd.DataFrame, path: Path | str = MATCH_WEATHER_PARQUET
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(path, index=False)
    covered = int(weather["temperature_c"].notna().sum())
    logger.info(
        "Wrote %d matches to %s (%d with weather, %.1f%%)",
        len(weather), path, covered, 100 * covered / max(len(weather), 1),
    )
    return path


def coverage_by_season(weather: pd.DataFrame) -> pd.DataFrame:
    """How much of each season actually got weather, for the acceptance check."""
    summary = weather.groupby("season").agg(
        matches=("date", "size"),
        with_weather=("temperature_c", "count"),
    )
    summary["coverage_pct"] = (100 * summary["with_weather"] / summary["matches"]).round(2)
    return summary.reset_index()
