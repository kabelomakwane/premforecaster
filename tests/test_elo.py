"""Tests for the internally computed Elo ratings.

These are the default source the model reads, so they need to be right without
any help from a third party. Everything here runs on synthetic matches - no
network, no data files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ratings import elo


def match(home, away, hg, ag, *, season="2024-25", date="2024-08-17") -> dict:
    return {
        "date": pd.Timestamp(date, tz="UTC"),
        "season": season,
        "home_team": home,
        "away_team": away,
        "home_goals": hg,
        "away_goals": ag,
        "result": "H" if hg > ag else ("D" if hg == ag else "A"),
    }


def season_of(teams: list[str], season: str, start_day: int = 1) -> list[dict]:
    """A round-robin so every club in a season plays, for promotion tests."""
    rows = []
    day = start_day
    for i, home in enumerate(teams):
        for away in teams[i + 1 :]:
            rows.append(match(home, away, 1, 1, season=season,
                              date=f"{season[:4]}-09-{day:02d}"))
            day = day % 28 + 1
    return rows


# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------


def test_evenly_matched_sides_expect_half_the_points():
    assert elo.expected_score(0) == pytest.approx(0.5)


def test_a_400_point_lead_means_about_91_percent():
    """The defining property of the Elo scale."""
    assert elo.expected_score(400) == pytest.approx(0.909, abs=0.001)
    assert elo.expected_score(-400) == pytest.approx(0.091, abs=0.001)


def test_expectations_are_symmetric():
    for difference in (0, 50, 200, 700):
        assert elo.expected_score(difference) + elo.expected_score(-difference) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("goal_difference", "weight"),
    [(0, 1.0), (1, 1.0), (-1, 1.0), (2, 1.5), (-2, 1.5), (3, 1.75), (4, 1.875), (5, 2.0)],
)
def test_margin_weighting_matches_the_published_scale(goal_difference, weight):
    assert elo.margin_weight(goal_difference) == pytest.approx(weight)


def test_a_bigger_win_moves_the_ratings_further():
    assert elo.margin_weight(5) > elo.margin_weight(2) > elo.margin_weight(1)


def test_margin_weighting_has_diminishing_returns():
    """A rout should count for more, but not proportionally more."""
    step_small = elo.margin_weight(3) - elo.margin_weight(2)
    step_large = elo.margin_weight(7) - elo.margin_weight(6)
    assert step_large < step_small


@pytest.mark.parametrize(
    ("hg", "ag", "score"), [(2, 0, 1.0), (1, 1, 0.5), (0, 3, 0.0)]
)
def test_match_result_score(hg, ag, score):
    assert elo.match_result_score(hg, ag) == score


# ---------------------------------------------------------------------------
# Updating
# ---------------------------------------------------------------------------


def test_ratings_are_zero_sum():
    """What one side gains the other must lose, or the league inflates."""
    results = pd.DataFrame([match("Arsenal", "Chelsea", 3, 0)])
    history, _ = elo.compute_ratings(results)
    final = history.groupby("team")["elo"].last()
    assert final.sum() == pytest.approx(2 * elo.INITIAL_RATING)


def test_winning_raises_your_rating_and_losing_lowers_it():
    results = pd.DataFrame([match("Arsenal", "Chelsea", 3, 0)])
    history, _ = elo.compute_ratings(results)
    final = history.groupby("team")["elo"].last()
    assert final["Arsenal"] > elo.INITIAL_RATING
    assert final["Chelsea"] < elo.INITIAL_RATING


def test_a_home_draw_between_equals_slightly_favours_the_away_side():
    """The home team was expected to win, so a draw is a small disappointment."""
    results = pd.DataFrame([match("Arsenal", "Chelsea", 1, 1)])
    history, _ = elo.compute_ratings(results)
    final = history.groupby("team")["elo"].last()
    assert final["Arsenal"] < elo.INITIAL_RATING
    assert final["Chelsea"] > elo.INITIAL_RATING


def test_beating_a_stronger_side_is_worth_more():
    strong = pd.DataFrame(
        [match("A", "B", 1, 0, date="2024-08-17"), match("A", "B", 1, 0, date="2024-08-24")]
    )
    history, _ = elo.compute_ratings(strong)
    gains = history[history["team"] == "A"].sort_values("valid_from")["elo"].tolist()
    # The second win is against a now-weaker B, so it is worth less.
    assert (gains[1] - gains[0]) > (gains[2] - gains[1])


def test_a_bigger_win_moves_the_rating_further_end_to_end():
    narrow, _ = elo.compute_ratings(pd.DataFrame([match("A", "B", 1, 0)]))
    rout, _ = elo.compute_ratings(pd.DataFrame([match("A", "B", 5, 0)]))
    assert rout.groupby("team")["elo"].last()["A"] > narrow.groupby("team")["elo"].last()["A"]


def test_margin_weighting_can_be_turned_off():
    with_margin, _ = elo.compute_ratings(pd.DataFrame([match("A", "B", 5, 0)]))
    without, _ = elo.compute_ratings(pd.DataFrame([match("A", "B", 5, 0)]), use_margin=False)
    assert with_margin.groupby("team")["elo"].last()["A"] > without.groupby("team")["elo"].last()["A"]


def test_a_larger_k_moves_ratings_faster():
    slow, _ = elo.compute_ratings(pd.DataFrame([match("A", "B", 1, 0)]), k=10)
    fast, _ = elo.compute_ratings(pd.DataFrame([match("A", "B", 1, 0)]), k=40)
    assert fast.groupby("team")["elo"].last()["A"] > slow.groupby("team")["elo"].last()["A"]


def test_home_advantage_changes_what_a_result_is_worth():
    """With a big home advantage, a home win is expected and so earns less."""
    none, _ = elo.compute_ratings(pd.DataFrame([match("A", "B", 1, 0)]), home_advantage=0)
    lots, _ = elo.compute_ratings(pd.DataFrame([match("A", "B", 1, 0)]), home_advantage=300)
    assert lots.groupby("team")["elo"].last()["A"] < none.groupby("team")["elo"].last()["A"]


# ---------------------------------------------------------------------------
# No leakage
# ---------------------------------------------------------------------------


def test_the_rating_attached_to_a_match_is_the_one_before_it_was_played():
    """The whole point: a feature must not see the result it predicts."""
    results = pd.DataFrame(
        [match("A", "B", 5, 0, date="2024-08-17"), match("A", "B", 5, 0, date="2024-08-24")]
    )
    _, per_match = elo.compute_ratings(results)

    assert per_match.loc[0, "home_elo_before"] == pytest.approx(elo.INITIAL_RATING)
    # By the second match A's first win is reflected, but not the second's.
    assert per_match.loc[1, "home_elo_before"] > elo.INITIAL_RATING


def test_a_rating_period_starts_the_day_after_the_match():
    results = pd.DataFrame([match("A", "B", 1, 0, date="2024-08-17")])
    history, _ = elo.compute_ratings(results)
    after = history[(history["team"] == "A") & (history["matches_played"] == 1)]
    assert after.iloc[0]["valid_from"] == pd.Timestamp("2024-08-18")


def test_matches_are_processed_in_date_order_whatever_order_they_arrive():
    ordered = pd.DataFrame(
        [match("A", "B", 3, 0, date="2024-08-17"), match("A", "B", 0, 3, date="2024-08-24")]
    )
    shuffled = ordered.iloc[::-1].reset_index(drop=True)
    assert elo.compute_ratings(ordered)[0].equals(elo.compute_ratings(shuffled)[0])


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


def test_a_promoted_club_starts_at_the_level_of_who_went_down():
    """Rather than at league average, which would flatter them."""
    rows = season_of(["A", "B", "C"], "2023-24")
    # C is relegated after losing heavily; D comes up in its place.
    rows.append(match("A", "C", 5, 0, season="2023-24", date="2023-09-20"))
    rows += season_of(["A", "B", "D"], "2024-25")

    history, _ = elo.compute_ratings(pd.DataFrame(rows))
    d_start = history[(history["team"] == "D")].sort_values("valid_from").iloc[0]["elo"]

    # C's rating when it went down, which D should inherit.
    c_final = history[history["team"] == "C"].sort_values("valid_from").iloc[-1]["elo"]
    assert d_start == pytest.approx(c_final)
    assert d_start < elo.INITIAL_RATING


def test_everyone_starts_level_in_the_first_season():
    results = pd.DataFrame(season_of(["A", "B", "C"], "2024-25"))
    history, per_match = elo.compute_ratings(results)
    assert per_match.iloc[0]["home_elo_before"] == pytest.approx(elo.INITIAL_RATING)


def test_the_fallback_rating_is_used_when_nobody_was_relegated():
    """A league that only ever grows has no relegated clubs to learn from."""
    rows = season_of(["A", "B"], "2023-24") + season_of(["A", "B", "C"], "2024-25")
    history, _ = elo.compute_ratings(pd.DataFrame(rows))
    c_start = history[history["team"] == "C"].sort_values("valid_from").iloc[0]["elo"]
    assert c_start == pytest.approx(elo.PROMOTED_FALLBACK_RATING)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_the_home_advantage_constant_is_the_one_the_data_chose():
    """Guards against quietly reverting to an inherited number.

    65 came from general club football and was measurably too high here.
    """
    assert elo.DEFAULT_HOME_ADVANTAGE == pytest.approx(55.0)
    assert elo.DEFAULT_K == pytest.approx(20.0)


def test_fitting_recovers_a_sensible_home_advantage():
    """Build a league where home sides win far more than they should."""
    rows = []
    for day in range(1, 29):
        rows.append(match("A", "B", 2, 0, date=f"2024-09-{day:02d}"))
        rows.append(match("B", "A", 2, 0, date=f"2024-10-{day:02d}"))
    fitted = elo.fit_home_advantage(pd.DataFrame(rows), candidates=(0, 50, 100, 200, 400))
    assert fitted > 0, "a home-dominated league must fit a positive home advantage"


def test_prediction_error_reports_bias_in_the_right_direction():
    """Home sides winning more than expected should show a positive bias."""
    rows = [match("A", "B", 1, 0, date=f"2024-09-{day:02d}") for day in range(1, 20)]
    _, bias = elo.prediction_error(pd.DataFrame(rows), home_advantage=0)
    assert bias > 0


def test_calibration_groups_matches_by_rating_gap():
    rows = [
        match("A", "B", 3, 0, date=f"2024-09-{day:02d}") for day in range(1, 15)
    ] + [match("B", "A", 0, 3, date=f"2024-10-{day:02d}") for day in range(1, 15)]
    _, per_match = elo.compute_ratings(pd.DataFrame(rows))
    table = elo.calibration_by_rating_gap(per_match, bins=2)
    assert len(table) == 2
    assert set(["matches", "expected_home_score", "actual_home_score", "error"]) <= set(table.columns)


def test_calibration_needs_results_to_compare_against():
    empty = pd.DataFrame({"elo_difference": [], "expected_home_score": [], "result": []})
    with pytest.raises(elo.EloError, match="No matches"):
        elo.calibration_by_rating_gap(empty)


# ---------------------------------------------------------------------------
# Comparing with Club Elo
# ---------------------------------------------------------------------------


def elo_history(values: dict[str, float], source: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": team,
                "elo": value,
                "valid_from": pd.Timestamp("2024-01-01"),
                "valid_to": pd.Timestamp("2100-01-01"),
                "source": source,
                "matches_played": 10,
            }
            for team, value in values.items()
        ]
    )


def test_two_sources_that_rank_clubs_identically_agree_perfectly():
    ours = elo_history({"Arsenal": 1800, "Chelsea": 1700, "Everton": 1500}, "internal")
    theirs = elo_history({"Arsenal": 1950, "Chelsea": 1850, "Everton": 1650}, "clubelo")

    comparison = elo.compare_with_clubelo(ours, theirs, when="2024-06-01")
    summary = elo.agreement_summary(comparison)

    assert summary["spearman"] == pytest.approx(1.0)
    assert summary["max_rank_difference"] == 0


def test_disagreement_on_order_shows_up_in_the_rank_difference():
    ours = elo_history({"Arsenal": 1800, "Chelsea": 1700, "Everton": 1500}, "internal")
    theirs = elo_history({"Arsenal": 1500, "Chelsea": 1700, "Everton": 1900}, "clubelo")

    summary = elo.agreement_summary(elo.compare_with_clubelo(ours, theirs, when="2024-06-01"))
    assert summary["spearman"] < 0
    assert summary["max_rank_difference"] == 2


def test_a_club_missing_from_one_source_is_left_out_of_the_comparison():
    ours = elo_history({"Arsenal": 1800, "Chelsea": 1700}, "internal")
    theirs = elo_history({"Arsenal": 1950}, "clubelo")
    comparison = elo.compare_with_clubelo(ours, theirs, when="2024-06-01")
    assert list(comparison["team"]) == ["Arsenal"]


def test_the_comparison_survives_having_nothing_in_common():
    ours = elo_history({"Arsenal": 1800}, "internal")
    theirs = elo_history({"Chelsea": 1700}, "clubelo")
    comparison = elo.compare_with_clubelo(ours, theirs, when="2024-06-01")
    assert comparison.empty
    assert elo.agreement_summary(comparison) == {"clubs": 0.0}


# ---------------------------------------------------------------------------
# get_elo reads this table
# ---------------------------------------------------------------------------


def test_get_elo_reads_the_internal_history():
    """The existing interface must work unchanged on internally built ratings."""
    from src.scrape.clubelo import get_elo

    results = pd.DataFrame([match("Arsenal", "Chelsea", 3, 0, date="2024-08-17")])
    history, _ = elo.compute_ratings(results)

    before = get_elo("Arsenal", "2024-08-17", history)
    after = get_elo("Arsenal", "2024-08-19", history)
    assert before == pytest.approx(elo.INITIAL_RATING)
    assert after > before


def test_missing_columns_are_reported_clearly():
    with pytest.raises(elo.EloError, match="missing"):
        elo.compute_ratings(pd.DataFrame({"date": [], "season": []}))


def test_no_matches_is_an_error():
    empty = pd.DataFrame(
        columns=["date", "season", "home_team", "away_team", "home_goals", "away_goals"]
    )
    with pytest.raises(elo.EloError, match="No matches"):
        elo.compute_ratings(empty)
