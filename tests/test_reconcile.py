"""Tests for the reconciliation between results.parquet and the xG tables.

The whole point of this report is to notice matches that quietly have no xG, so
the tests are mostly about making sure nothing goes unnoticed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import reconcile


def results_frame(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """(season, date, home, away) -> a minimal results.parquet-shaped frame."""
    return pd.DataFrame(
        [
            {"season": season, "date": pd.Timestamp(date, tz="UTC"),
             "home_team": home, "away_team": away}
            for season, date, home, away in rows
        ]
    )


BASE = [
    ("2024-25", "2024-08-16 19:00", "Manchester United", "Fulham"),
    ("2024-25", "2024-08-17 14:00", "Ipswich Town", "Liverpool"),
    ("2024-25", "2024-08-17 14:00", "Arsenal", "Wolverhampton Wanderers"),
]


def test_identical_tables_produce_no_exceptions():
    results = results_frame(BASE)
    problems = reconcile.compare_fixtures(results, results.copy(), "understat")
    assert problems.empty
    assert list(problems.columns) == reconcile.REPORT_COLUMNS


def test_a_match_missing_from_the_xg_source_is_reported():
    results = results_frame(BASE)
    partial = results_frame(BASE[:2])

    problems = reconcile.compare_fixtures(results, partial, "understat")
    assert len(problems) == 1
    assert problems.iloc[0]["issue"] == "missing_from_understat"
    assert problems.iloc[0]["home_team"] == "Arsenal"


def test_a_match_the_source_has_but_results_does_not_is_reported():
    results = results_frame(BASE[:2])
    extra = results_frame(BASE)

    problems = reconcile.compare_fixtures(results, extra, "understat")
    assert len(problems) == 1
    assert problems.iloc[0]["issue"] == "missing_from_results"


def test_a_small_date_difference_is_tolerated():
    """Sources disagree by hours over timezones; that is not a missing match."""
    results = results_frame(BASE)
    shifted = results_frame(
        [(s, d.replace("19:00", "23:30").replace("14:00", "23:30"), h, a)
         for s, d, h, a in BASE]
    )
    assert reconcile.compare_fixtures(results, shifted, "understat").empty


def test_a_large_date_difference_is_reported():
    """A postponed match rescheduled weeks later would break a date join."""
    results = results_frame(BASE)
    moved = results_frame(
        [("2024-25", "2024-12-01 15:00", "Arsenal", "Wolverhampton Wanderers")] + [
            (s, d, h, a) for s, d, h, a in BASE[:2]
        ]
    )
    problems = reconcile.compare_fixtures(results, moved, "understat")
    assert len(problems) == 1
    assert problems.iloc[0]["issue"] == "date_mismatch"
    assert problems.iloc[0]["home_team"] == "Arsenal"


def test_the_long_team_match_shape_reconciles_correctly():
    """team_match_xg has two rows per match; that must not look like a duplicate."""
    results = results_frame(BASE)
    long_form = pd.concat([results, results]).reset_index(drop=True)
    assert reconcile.compare_fixtures(results, long_form, "understat").empty


def test_coverage_is_reported_per_season():
    results = results_frame(
        BASE + [("2023-24", "2023-08-11 19:00", "Burnley", "Manchester City")]
    )
    partial = results_frame(BASE[:2])

    coverage = reconcile.coverage_by_season(results, partial, "understat")
    by_season = coverage.set_index("season")

    assert by_season.loc["2024-25", "covered"] == 2
    assert by_season.loc["2024-25", "missing"] == 1
    assert by_season.loc["2024-25", "coverage_pct"] == pytest.approx(66.67, abs=0.01)
    assert by_season.loc["2023-24", "covered"] == 0


def test_full_coverage_reads_as_100_percent():
    results = results_frame(BASE)
    coverage = reconcile.coverage_by_season(results, results.copy(), "understat")
    assert (coverage["coverage_pct"] == 100.0).all()


def test_build_report_covers_every_source():
    results = results_frame(BASE)
    problems, coverage = reconcile.build_report(
        results,
        {"understat": results_frame(BASE[:2]), "fbref": results_frame(BASE[:1])},
    )
    assert set(problems["source"]) == {"understat", "fbref"}
    assert set(coverage["source"]) == {"understat", "fbref"}
    assert len(problems) == 1 + 2


def test_build_report_skips_a_source_with_no_data():
    """FBref being unavailable must not break the report for Understat."""
    results = results_frame(BASE)
    problems, coverage = reconcile.build_report(
        results, {"understat": results_frame(BASE), "fbref": pd.DataFrame()}
    )
    assert problems.empty
    assert set(coverage["source"]) == {"understat"}


def test_a_table_without_the_join_keys_is_rejected():
    results = results_frame(BASE)
    with pytest.raises(ValueError, match="missing the column"):
        reconcile.compare_fixtures(results, pd.DataFrame({"xg": [1.0]}), "understat")


def test_the_report_writes_a_csv_even_when_it_is_empty(tmp_path):
    """An empty report with headers is the signal that everything reconciled."""
    path = reconcile.write_report(
        pd.DataFrame(columns=reconcile.REPORT_COLUMNS), tmp_path / "report.csv"
    )
    assert path.exists()
    assert list(pd.read_csv(path).columns) == reconcile.REPORT_COLUMNS
