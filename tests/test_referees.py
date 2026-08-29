"""Tests for the referee profile aggregation.

Pure computation over results.parquet, so no network anywhere. The subtle part
is the era adjustment: league-wide card rates move a lot between seasons, so a
raw cards-per-match ranking mostly ranks eras rather than referees.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.scrape import referees as ref


def match(referee: str, season: str, *, yellows: int = 3, reds: int = 0,
          result: str = "H", fouls: int = 20, date: str = "2024-08-17") -> dict:
    return {
        "date": pd.Timestamp(date, tz="UTC"),
        "season": season,
        "referee": referee,
        "result": result,
        "home_goals": 2,
        "away_goals": 1,
        "home_yellows": yellows, "away_yellows": 0,
        "home_reds": reds, "away_reds": 0,
        "home_fouls": fouls, "away_fouls": 0,
    }


def results(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_rates_are_per_match_not_totals():
    frame = results([match("M Dean", "2024-25", yellows=4) for _ in range(3)])
    profiles = ref.build_profiles(frame)
    assert profiles.loc[0, "matches"] == 3
    assert profiles.loc[0, "yellows_per_match"] == pytest.approx(4.0)


def test_home_and_away_cards_are_added_together():
    frame = pd.DataFrame([match("M Dean", "2024-25", yellows=2) | {"away_yellows": 3}])
    assert ref.build_profiles(frame).loc[0, "yellows_per_match"] == pytest.approx(5.0)


def test_cards_per_match_combines_yellows_and_reds():
    frame = results([match("M Dean", "2024-25", yellows=3, reds=1)])
    row = ref.build_profiles(frame).iloc[0]
    assert row["cards_per_match"] == pytest.approx(4.0)
    assert row["reds_per_match"] == pytest.approx(1.0)


def test_result_rates_add_up_to_one():
    frame = results(
        [match("M Dean", "2024-25", result=r) for r in ("H", "H", "D", "A")]
    )
    row = ref.build_profiles(frame).iloc[0]
    assert row["home_win_rate"] == pytest.approx(0.5)
    assert row["draw_rate"] == pytest.approx(0.25)
    assert row["away_win_rate"] == pytest.approx(0.25)
    assert row["home_win_rate"] + row["draw_rate"] + row["away_win_rate"] == pytest.approx(1.0)


def test_profiles_are_split_by_season_by_default():
    frame = results([match("M Dean", "2023-24"), match("M Dean", "2024-25")])
    profiles = ref.build_profiles(frame)
    assert len(profiles) == 2
    assert set(profiles["season"]) == {"2023-24", "2024-25"}


def test_career_totals_can_be_asked_for_instead():
    frame = results([match("M Dean", "2023-24"), match("M Dean", "2024-25")])
    profiles = ref.build_profiles(frame, by_season=False)
    assert len(profiles) == 1
    assert profiles.loc[0, "matches"] == 2
    assert profiles.loc[0, "season"] == "all"


def test_yellows_per_foul_measures_readiness_to_book():
    frame = results([match("M Dean", "2024-25", yellows=4, fouls=20)])
    assert ref.build_profiles(frame).loc[0, "yellows_per_foul"] == pytest.approx(0.2)


def test_a_referee_with_no_fouls_recorded_gets_no_ratio_rather_than_a_divide_by_zero():
    frame = results([match("M Dean", "2024-25", fouls=0) | {"away_fouls": 0}])
    assert pd.isna(ref.build_profiles(frame).loc[0, "yellows_per_foul"])


# ---------------------------------------------------------------------------
# The era adjustment
# ---------------------------------------------------------------------------


def test_a_referee_at_the_league_average_scores_one():
    frame = results([match("A", "2024-25", yellows=4), match("B", "2024-25", yellows=4)])
    profiles = ref.build_profiles(frame)
    assert profiles["yellows_vs_season_average"].tolist() == pytest.approx([1.0, 1.0])


def test_a_strict_referee_scores_above_one_and_a_lenient_one_below():
    frame = results(
        [match("Strict", "2024-25", yellows=6), match("Lenient", "2024-25", yellows=2)]
    )
    profiles = ref.build_profiles(frame).set_index("referee")
    assert profiles.loc["Strict", "yellows_vs_season_average"] > 1
    assert profiles.loc["Lenient", "yellows_vs_season_average"] < 1


def test_the_adjustment_separates_the_referee_from_the_era():
    """League card rates rose sharply after 2021. A referee who was strict in a
    lenient era must not be ranked below an average referee in a strict one."""
    frame = results(
        # 2020/21: league averages 2 yellows. This referee shows 3 - strict.
        [match("Old School", "2020-21", yellows=3)]
        + [match("Average Then", "2020-21", yellows=1)]
        # 2023/24: league averages 4. This referee shows 4 - exactly average.
        + [match("Modern", "2023-24", yellows=4)]
        + [match("Average Now", "2023-24", yellows=4)]
    )
    profiles = ref.build_profiles(frame).set_index("referee")

    # On raw rates the modern average referee looks stricter than the old strict one.
    assert profiles.loc["Modern", "yellows_per_match"] > profiles.loc["Old School", "yellows_per_match"]
    # Adjusted for era, the truth comes out.
    assert (
        profiles.loc["Old School", "yellows_vs_season_average"]
        > profiles.loc["Modern", "yellows_vs_season_average"]
    )


def test_the_strictest_ranking_uses_the_era_adjustment():
    frame = results(
        [match("Old School", "2020-21", yellows=3)] * 40
        + [match("Average Then", "2020-21", yellows=1)] * 40
        + [match("Modern", "2023-24", yellows=4)] * 40
        + [match("Average Now", "2023-24", yellows=4)] * 40
    )
    ranked = ref.strictest(ref.build_profiles(frame), top=1, min_matches=30)
    assert ranked.index[0] == "Old School"


# ---------------------------------------------------------------------------
# Reliability and missing data
# ---------------------------------------------------------------------------


def test_a_small_sample_is_flagged_as_unreliable():
    frame = results([match("Debutant", "2024-25") for _ in range(3)])
    assert not ref.build_profiles(frame).loc[0, "reliable"]


def test_enough_matches_counts_as_reliable():
    frame = results([match("Regular", "2024-25") for _ in range(20)])
    assert ref.build_profiles(frame).loc[0, "reliable"]


def test_matches_with_no_referee_are_ignored_not_grouped_together():
    frame = results([match("M Dean", "2024-25"), match(None, "2024-25")])
    profiles = ref.build_profiles(frame)
    assert list(profiles["referee"]) == ["M Dean"]


def test_no_referee_anywhere_is_an_error():
    frame = results([match(None, "2024-25")])
    with pytest.raises(ref.RefereeDataError, match="No match"):
        ref.build_profiles(frame)


def test_missing_card_columns_are_reported_clearly(tmp_path):
    path = tmp_path / "results.parquet"
    pd.DataFrame({"season": ["2024-25"], "referee": ["M Dean"], "result": ["H"]}).to_parquet(path)
    with pytest.raises(ref.RefereeDataError, match="missing the column"):
        ref.load_results(path)


def test_penalties_are_blank_when_there_is_no_shot_data():
    frame = results([match("M Dean", "2024-25")])
    profiles = ref.build_profiles(frame)
    assert pd.isna(profiles.loc[0, "penalties_per_match"])


def test_penalties_stay_blank_rather_than_becoming_zero():
    """A referee who awarded none and one we have no data for are different."""
    frame = results([match("M Dean", "2024-25")])
    profiles = ref.add_penalty_rates(ref.build_profiles(frame), frame, pd.DataFrame())
    assert pd.isna(profiles.loc[0, "penalties_per_match"])


def test_penalty_rates_are_filled_where_shot_data_overlaps():
    frame = results([match("M Dean", "2024-25", date="2024-08-17")])
    shots = pd.DataFrame(
        {
            "season": ["2024-25"] * 3,
            "game_id": ["1", "1", "1"],
            "date": [pd.Timestamp("2024-08-17 14:00", tz="UTC")] * 3,
            "is_penalty": [True, False, False],
        }
    )
    profiles = ref.add_penalty_rates(ref.build_profiles(frame), frame, shots)
    assert profiles.loc[0, "penalties_per_match"] == pytest.approx(1.0)


def test_shot_data_without_the_penalty_flag_is_an_error():
    frame = results([match("M Dean", "2024-25")])
    shots = pd.DataFrame({"season": ["2024-25"], "game_id": ["1"], "date": [pd.Timestamp("2024-08-17", tz="UTC")]})
    with pytest.raises(ref.RefereeDataError, match="is_penalty"):
        ref.add_penalty_rates(ref.build_profiles(frame), frame, shots)


def test_the_output_has_the_expected_columns():
    frame = results([match("M Dean", "2024-25")])
    assert list(ref.build_profiles(frame).columns) == ref.PROFILE_COLUMNS
