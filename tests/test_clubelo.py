"""Tests for the Club Elo scraper.

Deliberately offline. api.clubelo.com was unreachable from the environment this
was written in, so every parsing and lookup function is tested against sample
CSV text rather than the live API - which is how it should be anyway, since a
test that needs the internet is not a test worth having.
"""

from __future__ import annotations

import pandas as pd
import pytest
import requests

from src.lookups import UnknownTeamError
from src.scrape import clubelo


# The shape api.clubelo.com/{Club} returns: one row per period a rating held.
ARSENAL_CSV = """Rank,Club,Country,Level,Elo,From,To
1,Arsenal,ENG,1,1834.5,2022-12-27,2023-01-02
1,Arsenal,ENG,1,1852.1,2023-01-03,2023-01-14
1,Arsenal,ENG,1,1845.9,2023-01-15,2023-01-21
"""

# A club with a gap: Club Elo stops rating clubs outside the leagues it covers.
LUTON_CSV = """Rank,Club,Country,Level,Elo,From,To
80,Luton,ENG,2,1490.0,2022-08-01,2022-08-31
80,Luton,ENG,1,1520.0,2023-08-01,2023-08-31
"""


def history() -> pd.DataFrame:
    return pd.concat(
        [clubelo.parse_club_history(ARSENAL_CSV), clubelo.parse_club_history(LUTON_CSV)],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parsing_produces_canonical_names():
    """Club Elo writes "Luton"; the model works in "Luton Town"."""
    parsed = clubelo.parse_club_history(LUTON_CSV)
    assert set(parsed["team"]) == {"Luton Town"}
    assert set(parsed["clubelo_name"]) == {"Luton"}


def test_parsing_keeps_the_validity_window():
    parsed = clubelo.parse_club_history(ARSENAL_CSV)
    assert parsed.loc[0, "valid_from"] == pd.Timestamp("2022-12-27")
    assert parsed.loc[0, "valid_to"] == pd.Timestamp("2023-01-02")
    assert parsed.loc[0, "elo"] == pytest.approx(1834.5)


def test_rows_come_back_in_date_order():
    parsed = clubelo.parse_club_history(ARSENAL_CSV)
    assert parsed["valid_from"].is_monotonic_increasing


def test_an_unknown_club_stops_the_build():
    csv = ARSENAL_CSV.replace("Arsenal", "Real Madrid")
    with pytest.raises(UnknownTeamError, match="Real Madrid"):
        clubelo.parse_club_history(csv)


def test_a_missing_column_is_reported_clearly():
    with pytest.raises(clubelo.ClubEloFormatError, match="Elo"):
        clubelo.parse_club_history("Rank,Club,Country,Level,From,To\n1,Arsenal,ENG,1,a,b\n")


def test_a_non_numeric_rating_is_rejected():
    csv = ARSENAL_CSV.replace("1834.5", "very good")
    with pytest.raises(clubelo.ClubEloFormatError, match="Non-numeric"):
        clubelo.parse_club_history(csv)


def test_an_unparseable_date_is_rejected():
    csv = ARSENAL_CSV.replace("2022-12-27", "Boxing Day")
    with pytest.raises(clubelo.ClubEloFormatError, match="From/To"):
        clubelo.parse_club_history(csv)


def test_a_period_ending_before_it_starts_is_rejected():
    csv = ARSENAL_CSV.replace("2023-01-02", "2022-01-02")
    with pytest.raises(clubelo.ClubEloFormatError, match="end before they start"):
        clubelo.parse_club_history(csv)


def test_an_empty_csv_is_rejected():
    with pytest.raises(clubelo.ClubEloFormatError):
        clubelo.parse_club_history("Rank,Club,Country,Level,Elo,From,To\n")


# ---------------------------------------------------------------------------
# get_elo
# ---------------------------------------------------------------------------


def test_get_elo_returns_the_rating_in_force_on_the_day():
    assert clubelo.get_elo("Arsenal", "2023-01-05", history()) == pytest.approx(1852.1)


def test_get_elo_is_inclusive_of_both_ends_of_a_period():
    assert clubelo.get_elo("Arsenal", "2023-01-03", history()) == pytest.approx(1852.1)
    assert clubelo.get_elo("Arsenal", "2023-01-14", history()) == pytest.approx(1852.1)


def test_get_elo_accepts_dates_in_several_forms():
    import datetime

    for when in ("2023-01-05", datetime.date(2023, 1, 5), pd.Timestamp("2023-01-05")):
        assert clubelo.get_elo("Arsenal", when, history()) == pytest.approx(1852.1)


def test_get_elo_handles_a_timezone_aware_kickoff():
    """Dates come straight out of results.parquet, which is tz-aware UTC."""
    kickoff = pd.Timestamp("2023-01-05 19:30", tz="UTC")
    assert clubelo.get_elo("Arsenal", kickoff, history()) == pytest.approx(1852.1)


def test_get_elo_falls_back_to_the_last_known_rating_in_a_gap():
    """Club Elo does not rate a club while it is outside its covered leagues."""
    assert clubelo.get_elo("Luton Town", "2023-03-01", history()) == pytest.approx(1490.0)


def test_get_elo_returns_none_before_a_clubs_first_rating():
    assert clubelo.get_elo("Luton Town", "2020-01-01", history()) is None


def test_get_elo_returns_none_for_a_club_we_have_no_history_for():
    assert clubelo.get_elo("Chelsea", "2023-01-05", history()) is None


def test_strict_mode_raises_instead_of_returning_none():
    with pytest.raises(KeyError, match="No Elo history"):
        clubelo.get_elo("Chelsea", "2023-01-05", history(), strict=True)
    with pytest.raises(KeyError, match="first rating"):
        clubelo.get_elo("Luton Town", "2020-01-01", history(), strict=True)


def test_ratings_are_in_a_plausible_range():
    """A sanity band, so a units change or a parsing slip would be obvious."""
    low, high = clubelo.PLAUSIBLE_ELO_RANGE
    assert low < clubelo.get_elo("Arsenal", "2023-01-05", history()) < high


# ---------------------------------------------------------------------------
# Attaching Elo to fixtures
# ---------------------------------------------------------------------------


def test_elo_can_be_attached_to_a_fixture_list():
    fixtures = pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-01-05", tz="UTC")],
            "home_team": ["Arsenal"],
            "away_team": ["Luton Town"],
        }
    )
    enriched = clubelo.add_elo_to_fixtures(fixtures, history())
    assert enriched.loc[0, "home_elo"] == pytest.approx(1852.1)
    assert enriched.loc[0, "away_elo"] == pytest.approx(1490.0)
    assert enriched.loc[0, "elo_difference"] == pytest.approx(1852.1 - 1490.0)


def test_attaching_elo_needs_the_team_columns():
    with pytest.raises(ValueError, match="home_team"):
        clubelo.add_elo_to_fixtures(pd.DataFrame({"date": []}), history())


# ---------------------------------------------------------------------------
# Combining and overlaps
# ---------------------------------------------------------------------------


def test_overlapping_periods_are_refused(tmp_path):
    """Two ratings covering the same day would make an as-of lookup ambiguous."""
    overlapping = """Rank,Club,Country,Level,Elo,From,To
1,Arsenal,ENG,1,1834.5,2023-01-01,2023-01-10
1,Arsenal,ENG,1,1852.1,2023-01-05,2023-01-14
"""
    path = tmp_path / "Arsenal_downloaded-2026-08-29.csv"
    path.write_text(overlapping)
    with pytest.raises(clubelo.ClubEloFormatError, match="Overlapping"):
        clubelo.build_elo_history(tmp_path)


def test_building_from_cached_files_makes_no_requests(tmp_path, monkeypatch):
    """The offline path: rebuild the parquet from what is already on disk."""
    (tmp_path / "Arsenal_downloaded-2026-08-29.csv").write_text(ARSENAL_CSV)
    (tmp_path / "Luton_downloaded-2026-08-29.csv").write_text(LUTON_CSV)

    def no_network(*args, **kwargs):
        raise AssertionError("build_elo_history must not make requests")

    monkeypatch.setattr(requests.Session, "get", no_network)

    built = clubelo.build_elo_history(tmp_path)
    assert set(built["team"]) == {"Arsenal", "Luton Town"}
    assert list(built.columns) == clubelo.ELO_COLUMNS


def test_building_with_no_cached_files_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="No Club Elo CSVs"):
        clubelo.build_elo_history(tmp_path)


def test_the_newest_download_per_club_is_the_one_used(tmp_path):
    (tmp_path / "Arsenal_downloaded-2026-01-01.csv").write_text(ARSENAL_CSV)
    (tmp_path / "Arsenal_downloaded-2026-08-29.csv").write_text(ARSENAL_CSV)
    found = clubelo.find_raw_files(tmp_path)
    assert found["Arsenal"].name == "Arsenal_downloaded-2026-08-29.csv"


# ---------------------------------------------------------------------------
# When the API cannot be reached
# ---------------------------------------------------------------------------


def test_an_unreachable_api_gives_an_explanatory_error(monkeypatch):
    """This is the case that actually happens on a restricted network."""
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests.Session, "get", refuse)

    with pytest.raises(clubelo.ClubEloUnavailableError, match="blocking"):
        clubelo.fetch_club_history("Arsenal")


def test_download_all_skips_unreachable_clubs_rather_than_giving_up(monkeypatch, tmp_path):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests.Session, "get", refuse)

    saved = clubelo.download_all(["Arsenal", "Chelsea"], tmp_path, skip_failures=True)
    assert saved == {}


def test_the_club_list_comes_from_the_lookup():
    names = clubelo.clubelo_names()
    assert "Man City" in names  # Club Elo's spelling, verified against the live site
    assert "Forest" in names
    assert len(names) == 36
