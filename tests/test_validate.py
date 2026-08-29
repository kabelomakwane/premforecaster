"""Tests for the season-axis guard.

The guard exists because of a real hole: the player match table once held
2014/15 to 2020/21 and 2023/24 to 2026/27, missing the two seasons in between,
and nothing noticed - every row was valid, the table loaded fine, and only a
model walking forward through time would have quietly skipped two years.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import validate
from src.validate import SeasonAxisError, SeasonAxisRule


def table(seasons: list[str], rows_per_season: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {"season": [s for s in seasons for _ in range(rows_per_season)], "value": 1}
    )


def contiguous(first: int, last: int) -> list[str]:
    return [validate.season_label(year) for year in range(first, last + 1)]


# ---------------------------------------------------------------------------
# Season labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "year"), [("2014-15", 2014), ("2026-27", 2026), ("1999-00", 1999)]
)
def test_season_labels_round_trip(label, year):
    assert validate.start_year(label) == year
    assert validate.season_label(year) == label


@pytest.mark.parametrize("bad", ["2014", "14-15", "2014/15", "last year", "", "2014-2015"])
def test_things_that_are_not_season_labels_are_rejected(bad):
    with pytest.raises(ValueError, match="not a season label"):
        validate.start_year(bad)


# ---------------------------------------------------------------------------
# Detecting the axis
# ---------------------------------------------------------------------------


def test_a_season_column_is_recognised_as_a_time_axis():
    assert validate.has_season_axis(table(["2024-25", "2025-26"]))


def test_a_table_without_a_season_column_is_not_guarded():
    assert not validate.has_season_axis(pd.DataFrame({"team": ["Arsenal"]}))


def test_a_column_of_non_season_values_is_not_a_season_axis():
    """The lookup tables have no time axis and must not be dragged into this."""
    assert not validate.has_season_axis(pd.DataFrame({"season": ["winter", "summer"]}))


def test_an_empty_season_column_is_not_a_season_axis():
    assert not validate.has_season_axis(pd.DataFrame({"season": []}))


# ---------------------------------------------------------------------------
# Finding gaps
# ---------------------------------------------------------------------------


def test_a_contiguous_run_has_no_gaps():
    assert validate.find_season_gaps(contiguous(2014, 2026)) == []


def test_a_hole_in_the_middle_is_found():
    seasons = contiguous(2014, 2020) + contiguous(2023, 2026)
    assert validate.find_season_gaps(seasons) == ["2021-22", "2022-23"]


def test_starting_late_is_not_a_gap():
    """Only interior holes count. Covering less history is coverage, not a hole."""
    assert validate.find_season_gaps(contiguous(2023, 2026)) == []


def test_a_single_season_has_no_gaps():
    assert validate.find_season_gaps(["2024-25"]) == []


def test_no_seasons_at_all_has_no_gaps():
    assert validate.find_season_gaps([]) == []


def test_gaps_come_back_in_chronological_order():
    seasons = ["2014-15", "2017-18", "2016-17"]
    assert validate.find_season_gaps(seasons) == ["2015-16"]


def test_duplicate_seasons_do_not_confuse_the_check():
    """Real tables have thousands of rows per season."""
    assert validate.find_season_gaps(["2024-25"] * 500 + ["2025-26"] * 500) == []


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_a_contiguous_table_passes():
    assert validate.check_season_contiguity(table(contiguous(2014, 2026)), "team_match") == []


def test_a_table_with_a_hole_fails_loudly():
    holed = table(contiguous(2014, 2020) + contiguous(2023, 2026))
    with pytest.raises(SeasonAxisError, match="2021-22"):
        validate.check_season_contiguity(holed, "understat_player_match")


def test_the_error_says_what_is_missing_and_what_to_do():
    holed = table(contiguous(2014, 2020) + contiguous(2023, 2026))
    with pytest.raises(SeasonAxisError) as caught:
        validate.check_season_contiguity(holed, "understat_player_match")

    message = str(caught.value)
    assert "2021-22" in message and "2022-23" in message
    assert "2014-15" in message and "2026-27" in message  # the span it holds
    assert "TABLE_RULES" in message  # how to declare it if intended


def test_a_table_with_no_season_column_is_an_error_not_a_silent_pass():
    with pytest.raises(SeasonAxisError, match="no 'season' column"):
        validate.check_season_contiguity(pd.DataFrame({"team": ["Arsenal"]}), "mystery")


# ---------------------------------------------------------------------------
# The whitelist
# ---------------------------------------------------------------------------


def test_an_explicitly_whitelisted_gap_is_allowed():
    holed = table(contiguous(2014, 2020) + contiguous(2023, 2026))
    rules = {
        "understat_player_match": SeasonAxisRule(
            allowed_gaps=frozenset({"2021-22", "2022-23"}), reason="known, accepted"
        )
    }
    allowed = validate.check_season_contiguity(holed, "understat_player_match", rules=rules)
    assert allowed == ["2021-22", "2022-23"]


def test_a_partially_whitelisted_gap_still_fails():
    """Whitelisting one season must not excuse the other."""
    holed = table(contiguous(2014, 2020) + contiguous(2023, 2026))
    rules = {
        "t": SeasonAxisRule(allowed_gaps=frozenset({"2021-22"}), reason="only this one")
    }
    with pytest.raises(SeasonAxisError, match="2022-23"):
        validate.check_season_contiguity(holed, "t", rules=rules)


def test_documented_from_allows_older_seasons_to_be_absent():
    """A partial backfill of the shot table must not trip the guard."""
    backfilled = table(["2014-15"] + contiguous(2023, 2026))
    allowed = validate.check_season_contiguity(backfilled, "shots")
    assert "2015-16" in allowed and "2022-23" in allowed


def test_documented_from_does_not_excuse_a_gap_inside_the_window():
    """Missing 2024/25 from the documented window is a real hole."""
    broken = table(["2023-24", "2025-26", "2026-27"])
    with pytest.raises(SeasonAxisError, match="2024-25"):
        validate.check_season_contiguity(broken, "shots")


def test_the_shots_rule_is_registered_and_explained():
    rule = validate.TABLE_RULES["shots"]
    assert rule.documented_from == "2023-24"
    assert rule.reason, "a whitelist entry must say why"


def test_a_table_with_no_rule_gets_no_leeway():
    holed = table(contiguous(2014, 2020) + contiguous(2023, 2026))
    assert "understat_player_match" not in validate.TABLE_RULES
    with pytest.raises(SeasonAxisError):
        validate.check_season_contiguity(holed, "understat_player_match")


# ---------------------------------------------------------------------------
# Checking several tables at once
# ---------------------------------------------------------------------------


def test_check_tables_skips_tables_with_no_time_axis():
    tables = {
        "team_match": table(contiguous(2014, 2026)),
        "team_names": pd.DataFrame({"canonical_name": ["Arsenal"]}),
    }
    assert validate.check_tables(tables) == {"team_match": []}


def test_check_tables_reports_every_broken_table_not_just_the_first():
    tables = {
        "a": table(contiguous(2014, 2016) + ["2018-19"]),
        "b": table(contiguous(2020, 2021) + ["2024-25"]),
    }
    with pytest.raises(SeasonAxisError) as caught:
        validate.check_tables(tables)

    message = str(caught.value)
    assert "a is missing" in message
    assert "b is missing" in message


def test_check_tables_ignores_empty_tables():
    assert validate.check_tables({"empty": pd.DataFrame()}) == {}


def test_season_span_reads_nicely():
    assert validate.season_span(table(contiguous(2014, 2026))) == "2014-15 to 2026-27 (13 seasons)"


# ---------------------------------------------------------------------------
# Against the real processed tables
# ---------------------------------------------------------------------------


def test_the_real_processed_tables_have_no_unexplained_holes():
    """The guard's whole point, run against what is actually on disk."""
    processed = Path(__file__).resolve().parents[1] / "data" / "processed"
    if not any(processed.glob("*.parquet")):
        pytest.skip("no processed tables built yet")

    validate.validate_processed_tables(processed)
