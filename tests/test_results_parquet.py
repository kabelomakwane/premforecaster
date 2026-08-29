"""Checks on the real data/processed/results.parquet.

These tests are about the actual downloaded data rather than the code, so they
skip themselves when the parquet has not been built yet (it is not in git -
run ``python -m pipelines.build_results`` first). That keeps a fresh checkout
green while still checking the real thing whenever it is present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.lookups import load_team_names
from src.scrape.footballdata import NO_ODDS, RESULTS_COLUMNS, RESULTS_PARQUET

MATCHES_PER_SEASON = 380  # 20 clubs playing each other home and away
TEAMS_PER_SEASON = 20
PROBABILITY_TOLERANCE = 1e-6

pytestmark = pytest.mark.skipif(
    not RESULTS_PARQUET.exists(),
    reason=(
        "data/processed/results.parquet has not been built. "
        "Run: python -m pipelines.build_results"
    ),
)


@pytest.fixture(scope="module")
def results() -> pd.DataFrame:
    return pd.read_parquet(RESULTS_PARQUET)


@pytest.fixture(scope="module")
def completed_seasons(results) -> list[str]:
    """Every season except the one currently being played."""
    return sorted(results["season"].unique())[:-1]


def test_parquet_loads_with_the_expected_columns(results):
    assert list(results.columns) == RESULTS_COLUMNS
    assert len(results) > 0


def test_every_completed_season_has_380_matches(results, completed_seasons):
    counts = results.groupby("season").size()
    wrong = {
        season: int(counts[season])
        for season in completed_seasons
        if counts[season] != MATCHES_PER_SEASON
    }
    assert wrong == {}, f"seasons without {MATCHES_PER_SEASON} matches: {wrong}"


def test_the_current_season_is_not_over_full(results):
    counts = results.groupby("season").size()
    assert counts.iloc[-1] <= MATCHES_PER_SEASON


def test_the_horizon_starts_at_2014_15(results):
    assert results["season"].min() == "2014-15"


def test_every_completed_season_has_20_teams_playing_38_games(results, completed_seasons):
    for season in completed_seasons:
        season_rows = results[results["season"] == season]
        appearances = pd.concat(
            [season_rows["home_team"], season_rows["away_team"]]
        ).value_counts()
        assert len(appearances) == TEAMS_PER_SEASON, f"{season} has {len(appearances)} teams"
        assert (appearances == 38).all(), f"{season}: {appearances[appearances != 38].to_dict()}"


def test_every_team_name_is_canonical(results):
    canonical = set(load_team_names()["canonical_name"])
    used = set(results["home_team"]) | set(results["away_team"])
    assert used <= canonical, f"non-canonical names in the output: {sorted(used - canonical)}"
    assert not results["home_team"].isna().any()
    assert not results["away_team"].isna().any()


def test_no_duplicate_matches(results):
    duplicated = results.duplicated(subset=["season", "home_team", "away_team"])
    assert not duplicated.any(), (
        f"{int(duplicated.sum())} duplicate fixtures: "
        f"{results.loc[duplicated, ['season', 'home_team', 'away_team']].to_dict('records')}"
    )


def test_no_team_plays_itself(results):
    assert not (results["home_team"] == results["away_team"]).any()


def test_market_probabilities_sum_to_one_wherever_odds_exist(results):
    priced = results[results["odds_source"] != NO_ODDS]
    totals = priced[["market_p_home", "market_p_draw", "market_p_away"]].sum(axis=1)
    assert np.allclose(totals, 1.0, atol=PROBABILITY_TOLERANCE), (
        f"worst deviation from 1.0: {float((totals - 1.0).abs().max())}"
    )


def test_market_probabilities_are_between_zero_and_one(results):
    probabilities = results[["market_p_home", "market_p_draw", "market_p_away"]].dropna()
    assert (probabilities > 0).all().all()
    assert (probabilities < 1).all().all()


def test_rows_without_odds_have_no_probabilities(results):
    unpriced = results[results["odds_source"] == NO_ODDS]
    if unpriced.empty:
        pytest.skip("every row has odds")
    assert unpriced[["market_p_home", "market_p_draw", "market_p_away"]].isna().all().all()


def test_the_bookmakers_margin_is_plausible(results):
    """A three-way football market runs roughly 2% to 12% over. Nothing wild."""
    overround = results["market_overround"].dropna()
    assert overround.min() >= 0.99
    assert overround.max() <= 1.30


def test_the_market_favours_the_home_side_on_average(results):
    """Home advantage is the most reliable effect in football. If this fails,
    home and away have almost certainly been swapped somewhere."""
    assert results["market_p_home"].mean() > results["market_p_away"].mean()


def test_the_result_column_agrees_with_the_score(results):
    expected = np.sign(results["home_goals"] - results["away_goals"])
    expected = pd.Series(expected).map({1: "H", 0: "D", -1: "A"})
    assert (expected == results["result"]).all()


def test_home_teams_win_more_often_than_away_teams(results):
    shares = results["result"].value_counts(normalize=True)
    assert shares["H"] > shares["A"]
    assert shares["H"] > shares["D"]


def test_dates_are_utc_and_ordered(results):
    assert str(results["date"].dt.tz) == "UTC"
    assert results["date"].is_monotonic_increasing
    assert not results["date"].isna().any()


def test_matches_fall_inside_their_season(results):
    """A season runs August to May. 2019/20 is the exception - COVID pushed it
    into July 2020 - so the window is deliberately generous."""
    for season, rows in results.groupby("season"):
        start_year = int(season.split("-")[0])
        assert rows["date"].min() >= pd.Timestamp(f"{start_year}-06-01", tz="UTC"), season
        assert rows["date"].max() <= pd.Timestamp(f"{start_year + 1}-08-31", tz="UTC"), season


def test_kickoff_times_are_known_from_2019_20_onwards(results):
    """football-data.co.uk only added the Time column in 2019/20."""
    by_season = results.groupby("season")["kickoff_time_known"].mean()
    assert (by_season[by_season.index < "2019-20"] == 0).all()
    assert (by_season[by_season.index >= "2019-20"] == 1).all()


def test_kickoff_hours_look_like_football_matches(results):
    """UK kick-offs run roughly 11:00 to 21:00 local, so 10:00-21:00 UTC."""
    known = results[results["kickoff_time_known"]]
    hours = known["date"].dt.hour
    assert hours.between(10, 21).all(), f"odd kick-off hours: {sorted(hours.unique())}"


def test_every_match_has_a_referee(results):
    missing = results["referee"].isna().sum()
    assert missing == 0, f"{missing} matches have no referee"


def test_goals_are_sane(results):
    for column in ("home_goals", "away_goals"):
        assert (results[column] >= 0).all()
        assert (results[column] <= 12).all()


# ---------------------------------------------------------------------------
# Spot checks against reality
# ---------------------------------------------------------------------------
# If the joins, dates or home/away orientation ever break, these well known
# results are the fastest way to notice.


def one_match(results, season, home, away):
    match = results[
        (results["season"] == season)
        & (results["home_team"] == home)
        & (results["away_team"] == away)
    ]
    assert len(match) == 1, f"expected exactly one {home} v {away} in {season}"
    return match.iloc[0]


def test_spot_check_battle_of_the_bridge(results):
    """Chelsea 2-2 Tottenham, 2 May 2016: the draw that gave Leicester the title."""
    match = one_match(results, "2015-16", "Chelsea", "Tottenham Hotspur")
    assert (match["home_goals"], match["away_goals"]) == (2, 2)
    assert match["result"] == "D"
    assert match["date"].date() == pd.Timestamp("2016-05-02").date()
    assert match["referee"] == "M Clattenburg"


def test_spot_check_southampton_0_9_leicester(results):
    """25 October 2019: the record Premier League away win, 8pm on a Friday."""
    match = one_match(results, "2019-20", "Southampton", "Leicester City")
    assert (match["home_goals"], match["away_goals"]) == (0, 9)
    assert match["result"] == "A"
    # 20:00 UK in October is still BST, so 19:00 UTC.
    assert match["date"] == pd.Timestamp("2019-10-25 19:00", tz="UTC")


def test_spot_check_opening_day_2014_15(results):
    """Arsenal 2-1 Crystal Palace, 16 August 2014: the first match we hold."""
    match = one_match(results, "2014-15", "Arsenal", "Crystal Palace")
    assert (match["home_goals"], match["away_goals"]) == (2, 1)
    assert match["result"] == "H"
    assert match["date"].date() == pd.Timestamp("2014-08-16").date()
    assert not match["kickoff_time_known"]


def league_table(results, season):
    """Rebuild a final league table so we can check it against the record books."""
    rows = results[results["season"] == season]
    home = rows.rename(
        columns={"home_team": "team", "home_goals": "for", "away_goals": "against"}
    )[["team", "for", "against"]]
    away = rows.rename(
        columns={"away_team": "team", "away_goals": "for", "home_goals": "against"}
    )[["team", "for", "against"]]
    both = pd.concat([home, away])
    both["points"] = (both["for"] > both["against"]) * 3 + (both["for"] == both["against"])
    table = both.groupby("team").agg(
        played=("points", "size"),
        won=("points", lambda s: int((s == 3).sum())),
        drawn=("points", lambda s: int((s == 1).sum())),
        lost=("points", lambda s: int((s == 0).sum())),
        goals_for=("for", "sum"),
        goals_against=("against", "sum"),
        points=("points", "sum"),
    )
    return table.sort_values("points", ascending=False)


def test_spot_check_leicester_win_the_2015_16_title(results):
    """23 wins, 12 draws, 3 defeats, 81 points - the 5000-1 season."""
    table = league_table(results, "2015-16")
    champion = table.iloc[0]
    assert champion.name == "Leicester City"
    assert champion["points"] == 81
    assert (champion["won"], champion["drawn"], champion["lost"]) == (23, 12, 3)
    assert (champion["goals_for"], champion["goals_against"]) == (68, 36)


def test_spot_check_the_centurions(results):
    """Manchester City, 2017/18: 100 points and 106 goals, both records."""
    champion = league_table(results, "2017-18").iloc[0]
    assert champion.name == "Manchester City"
    assert champion["points"] == 100
    assert champion["goals_for"] == 106
    assert (champion["won"], champion["drawn"], champion["lost"]) == (32, 4, 2)


def test_spot_check_liverpool_2019_20(results):
    """99 points, and the season that finished in July because of COVID."""
    champion = league_table(results, "2019-20").iloc[0]
    assert champion.name == "Liverpool"
    assert champion["points"] == 99
    assert (champion["won"], champion["drawn"], champion["lost"]) == (32, 3, 3)
    assert results[results["season"] == "2019-20"]["date"].max().month == 7
