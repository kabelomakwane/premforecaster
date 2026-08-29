"""Tests for the Open-Meteo weather scraper.

No network: the fetching is stubbed and the tests focus on the parts that would
quietly go wrong - reading weather at the wrong hour, batching that silently
explodes into thousands of requests, and unit mix-ups.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest
import requests

from src.scrape import weather as w


def matches(rows: list[tuple[str, str, str, bool]]) -> pd.DataFrame:
    """(date, season, home, away, kickoff_known) rows in results.parquet's shape."""
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(when, tz="UTC"),
                "season": season,
                "home_team": home,
                "away_team": away,
                "kickoff_time_known": known,
            }
            for when, season, home, away, known in rows
        ]
    )


STADIUMS = pd.DataFrame(
    {
        "stadium": ["Emirates Stadium", "Anfield"],
        "latitude": [51.5549, 53.4308],
        "longitude": [-0.1084, -2.9608],
    },
    index=pd.Index(["Arsenal", "Liverpool"], name="canonical_name"),
)


def payload(start: str = "2024-08-17 00:00", hours: int = 24) -> dict:
    times = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    return {
        "hourly": {
            "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
            "temperature_2m": [15.0 + i * 0.1 for i in range(hours)],
            "precipitation": [0.0] * hours,
            "wind_speed_10m": [12.0] * hours,
            "relative_humidity_2m": [70.0] * hours,
        }
    }


# ---------------------------------------------------------------------------
# Which hour to read
# ---------------------------------------------------------------------------


def test_a_known_kickoff_is_read_at_its_own_hour():
    frame = matches([("2024-08-17 14:00", "2024-25", "Arsenal", "Liverpool", True)])
    assert w.kickoff_hours(frame).iloc[0] == pd.Timestamp("2024-08-17 14:00", tz="UTC")


def test_an_unknown_kickoff_falls_back_to_an_afternoon_hour():
    """Before 2019/20 there is no kick-off time, and midnight weather is useless."""
    frame = matches([("2015-08-08 00:00", "2015-16", "Arsenal", "Liverpool", False)])
    assumed = w.kickoff_hours(frame).iloc[0]
    assert assumed == pd.Timestamp("2015-08-08 15:00", tz="UTC")
    assert assumed.hour == w.ASSUMED_KICKOFF_HOUR_UTC


def test_a_kickoff_on_the_half_hour_is_floored_to_the_hour():
    """Open-Meteo is hourly, so 11:30 reads the 11:00 value."""
    frame = matches([("2024-08-17 11:30", "2024-25", "Arsenal", "Liverpool", True)])
    assert w.kickoff_hours(frame).iloc[0] == pd.Timestamp("2024-08-17 11:00", tz="UTC")


def test_a_table_with_no_kickoff_flag_is_treated_as_known():
    frame = matches([("2024-08-17 14:00", "2024-25", "Arsenal", "Liverpool", True)])
    assert w.kickoff_hours(frame.drop(columns="kickoff_time_known")).iloc[0].hour == 14


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def test_one_request_covers_a_whole_season_at_one_ground():
    """The batching that keeps this to hundreds of calls rather than thousands."""
    season = matches(
        [(f"2024-09-{day:02d} 14:00", "2024-25", "Arsenal", "Liverpool", True)
         for day in range(1, 20)]
    )
    plan = w.request_plan(season, STADIUMS)
    assert len(plan) == 1, "19 home matches in a year must be one request"


def test_the_plan_is_one_row_per_ground_per_year():
    frame = matches(
        [
            ("2024-08-17 14:00", "2024-25", "Arsenal", "Liverpool", True),
            ("2024-09-14 14:00", "2024-25", "Arsenal", "Liverpool", True),
            ("2025-01-14 20:00", "2024-25", "Arsenal", "Liverpool", True),
            ("2024-08-24 14:00", "2024-25", "Liverpool", "Arsenal", True),
        ]
    )
    plan = w.request_plan(frame, STADIUMS)
    assert len(plan) == 3  # Arsenal 2024, Arsenal 2025, Liverpool 2024


def test_the_plan_carries_the_stadium_coordinates():
    frame = matches([("2024-08-17 14:00", "2024-25", "Arsenal", "Liverpool", True)])
    plan = w.request_plan(frame, STADIUMS)
    assert plan.loc[0, "stadium"] == "Emirates Stadium"
    assert plan.loc[0, "latitude"] == pytest.approx(51.5549)


def test_a_club_with_no_stadium_in_the_lookup_stops_the_build():
    frame = matches([("2024-08-17 14:00", "2024-25", "Everton", "Liverpool", True)])
    with pytest.raises(w.WeatherFormatError, match="No stadium"):
        w.request_plan(frame, STADIUMS)


def test_the_full_history_stays_well_under_the_daily_limit():
    """Sanity on the batching: ~36 grounds over ~13 years, not ~4,600 matches."""
    frame = matches(
        [(f"{year}-09-{day:02d} 14:00", f"{year}-{(year+1)%100:02d}", team, "Liverpool", True)
         for year in range(2014, 2027) for day in (1, 8, 15) for team in ("Arsenal",)]
    )
    assert len(w.request_plan(frame, STADIUMS)) == 13  # one per year, not 39


# ---------------------------------------------------------------------------
# Reading a response
# ---------------------------------------------------------------------------


def test_an_hourly_response_becomes_a_utc_indexed_table():
    frame = w.hourly_frame(payload())
    assert str(frame.index.tz) == "UTC"
    assert len(frame) == 24
    assert "temperature_2m" in frame.columns


def test_a_response_with_no_hourly_block_is_rejected():
    with pytest.raises(w.WeatherFormatError, match="no hourly data"):
        w.hourly_frame({"error": True, "reason": "bad range"})


def test_an_all_null_response_counts_as_no_data():
    """Open-Meteo answers 200 with nulls for dates outside a source's coverage."""
    empty = payload()
    empty["hourly"]["temperature_2m"] = [None] * 24
    assert not w._has_any_values(empty)
    assert w._has_any_values(payload())


# ---------------------------------------------------------------------------
# Assembling the table
# ---------------------------------------------------------------------------


def test_weather_is_attached_at_the_right_hour(monkeypatch, tmp_path):
    monkeypatch.setattr(w, "fetch_hourly", lambda *a, **k: (payload(), "archive"))

    frame = matches([("2024-08-17 14:00", "2024-25", "Arsenal", "Liverpool", True)])
    built = w.build_match_weather(frame, raw_dir=tmp_path, stadiums=STADIUMS)

    # 15.0 at midnight, +0.1 an hour, so 14:00 is 16.4.
    assert built.loc[0, "temperature_c"] == pytest.approx(16.4)
    assert built.loc[0, "weather_source"] == "archive"
    assert built.loc[0, "stadium"] == "Emirates Stadium"


def test_a_match_with_no_weather_still_gets_a_row(monkeypatch, tmp_path):
    """One row per match always, so a coverage gap is visible not hidden."""
    def unavailable(*args, **kwargs):
        raise w.WeatherUnavailableError("nothing could serve this")

    monkeypatch.setattr(w, "fetch_hourly", unavailable)

    frame = matches([("2014-08-16 15:00", "2014-15", "Arsenal", "Liverpool", True)])
    built = w.build_match_weather(frame, raw_dir=tmp_path, stadiums=STADIUMS)

    assert len(built) == 1
    assert pd.isna(built.loc[0, "temperature_c"])
    assert pd.isna(built.loc[0, "weather_source"])


def test_a_cached_response_is_reused_without_fetching(monkeypatch, tmp_path):
    cache = tmp_path / "Arsenal_2024.json"
    cached = payload()
    cached["_premforecaster_source"] = "archive"
    cache.write_text(json.dumps(cached))

    def must_not_fetch(*args, **kwargs):
        raise AssertionError("a cached location-year must not be fetched again")

    monkeypatch.setattr(w, "fetch_hourly", must_not_fetch)

    frame = matches([("2024-08-17 14:00", "2024-25", "Arsenal", "Liverpool", True)])
    built = w.build_match_weather(frame, raw_dir=tmp_path, stadiums=STADIUMS)
    assert built.loc[0, "temperature_c"] == pytest.approx(16.4)


def test_the_output_has_the_expected_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(w, "fetch_hourly", lambda *a, **k: (payload(), "archive"))
    frame = matches([("2024-08-17 14:00", "2024-25", "Arsenal", "Liverpool", True)])
    built = w.build_match_weather(frame, raw_dir=tmp_path, stadiums=STADIUMS)
    assert list(built.columns) == w.WEATHER_COLUMNS


# ---------------------------------------------------------------------------
# Plausibility
# ---------------------------------------------------------------------------


def test_plausible_uk_weather_passes():
    w.check_plausible(
        pd.DataFrame(
            {"temperature_c": [12.0], "wind_speed_kmh": [20.0], "precipitation_mm": [1.5]}
        )
    )


def test_an_impossible_temperature_is_rejected():
    with pytest.raises(w.WeatherFormatError, match="temperature_c"):
        w.check_plausible(
            pd.DataFrame(
                {"temperature_c": [500.0], "wind_speed_kmh": [20.0], "precipitation_mm": [0.0]}
            )
        )


def test_wind_in_the_wrong_units_is_caught():
    """Metres per second read as km/h would look fine; the reverse would not."""
    with pytest.raises(w.WeatherFormatError, match="wind_speed_kmh"):
        w.check_plausible(
            pd.DataFrame(
                {"temperature_c": [12.0], "wind_speed_kmh": [900.0], "precipitation_mm": [0.0]}
            )
        )


def test_all_missing_weather_is_not_treated_as_implausible():
    w.check_plausible(
        pd.DataFrame(
            {"temperature_c": [None], "wind_speed_kmh": [None], "precipitation_mm": [None]}
        )
    )


# ---------------------------------------------------------------------------
# Choosing an endpoint
# ---------------------------------------------------------------------------


def test_a_future_date_uses_the_forecast_endpoint(monkeypatch):
    seen = {}

    def record(url, parameters, **kwargs):
        seen["url"] = url
        return payload()

    monkeypatch.setattr(w, "_call", record)
    _, source = w.fetch_hourly(51.5, -0.1, date(2026, 9, 5), date(2026, 9, 5),
                               today=date(2026, 8, 29))
    assert seen["url"] == w.FORECAST_URL
    assert source == "forecast"


def test_a_past_date_prefers_the_archive(monkeypatch):
    seen = []

    def record(url, parameters, **kwargs):
        seen.append(url)
        return payload()

    monkeypatch.setattr(w, "_call", record)
    _, source = w.fetch_hourly(51.5, -0.1, date(2015, 8, 8), date(2015, 8, 8),
                               today=date(2026, 8, 29))
    assert seen == [w.ARCHIVE_URL]
    assert source == "archive"


def test_the_historical_forecast_api_is_the_fallback(monkeypatch):
    """Used when the archive host is unreachable, as on some restricted networks."""
    seen = []

    def flaky(url, parameters, **kwargs):
        seen.append(url)
        if url == w.ARCHIVE_URL:
            raise w.WeatherUnavailableError("archive host blocked")
        return payload()

    monkeypatch.setattr(w, "_call", flaky)
    _, source = w.fetch_hourly(51.5, -0.1, date(2020, 8, 8), date(2020, 8, 8),
                               today=date(2026, 8, 29))
    assert seen == [w.ARCHIVE_URL, w.HISTORICAL_FORECAST_URL]
    assert source == "historical_forecast"


def test_when_nothing_can_serve_the_range_the_error_says_what_was_tried(monkeypatch):
    def refuse(url, parameters, **kwargs):
        raise w.WeatherUnavailableError("blocked")

    monkeypatch.setattr(w, "_call", refuse)
    with pytest.raises(w.WeatherUnavailableError, match="Tried"):
        w.fetch_hourly(51.5, -0.1, date(2015, 8, 8), date(2015, 8, 8), today=date(2026, 8, 29))


def test_coverage_is_reported_per_season():
    built = pd.DataFrame(
        {
            "season": ["2024-25", "2024-25", "2014-15"],
            "date": pd.to_datetime(["2024-08-17", "2024-08-18", "2014-08-16"], utc=True),
            "temperature_c": [15.0, None, 20.0],
        }
    )
    coverage = w.coverage_by_season(built).set_index("season")
    assert coverage.loc["2024-25", "coverage_pct"] == pytest.approx(50.0)
    assert coverage.loc["2014-15", "coverage_pct"] == pytest.approx(100.0)
