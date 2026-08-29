"""Checks on the real xG parquets and the reconciliation report.

Like test_results_parquet.py, these skip themselves when the data has not been
built (it is not in git). Build it with:

    python -m pipelines.build_results
    python -m pipelines.build_xg

The FBref checks skip separately from the Understat ones, because FBref needs a
browser to get past Cloudflare and is not always reachable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import reconcile
from src.lookups import load_team_names
from src.scrape import fbref, understat
from src.scrape.footballdata import RESULTS_PARQUET

MATCHES_PER_SEASON = 380
TEAMS_PER_SEASON = 20
MIN_COVERAGE_PCT = 99.0


def _load(path):
    if not path.exists():
        pytest.skip(f"{path.name} has not been built. Run: python -m pipelines.build_xg")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def canonical_names() -> set[str]:
    return set(load_team_names()["canonical_name"])


@pytest.fixture(scope="module")
def results() -> pd.DataFrame:
    if not RESULTS_PARQUET.exists():
        pytest.skip("results.parquet has not been built.")
    return pd.read_parquet(RESULTS_PARQUET)


@pytest.fixture(scope="module")
def team_match() -> pd.DataFrame:
    return _load(understat.TEAM_MATCH_PARQUET)


@pytest.fixture(scope="module")
def player_season() -> pd.DataFrame:
    return _load(understat.PLAYER_SEASON_PARQUET)


@pytest.fixture(scope="module")
def shots() -> pd.DataFrame:
    return _load(understat.SHOTS_PARQUET)


@pytest.fixture(scope="module")
def completed_seasons(team_match) -> list[str]:
    """Seasons with a full 380 matches, i.e. everything but the current one."""
    counts = team_match.groupby("season").size()
    return sorted(counts[counts == MATCHES_PER_SEASON * 2].index)


# ---------------------------------------------------------------------------
# team_match_xg
# ---------------------------------------------------------------------------


def test_team_match_has_two_rows_per_match(team_match, completed_seasons):
    counts = team_match.groupby("season").size()
    for season in completed_seasons:
        assert counts[season] == MATCHES_PER_SEASON * 2


def test_every_completed_season_has_20_teams_playing_38_each(team_match, completed_seasons):
    for season in completed_seasons:
        rows = team_match[team_match["season"] == season]
        appearances = rows["team"].value_counts()
        assert len(appearances) == TEAMS_PER_SEASON, f"{season}: {len(appearances)} teams"
        assert (appearances == 38).all(), f"{season}: {appearances[appearances != 38].to_dict()}"


def test_each_team_plays_half_its_matches_at_home(team_match, completed_seasons):
    for season in completed_seasons:
        rows = team_match[team_match["season"] == season]
        home_games = rows[rows["is_home"]]["team"].value_counts()
        assert (home_games == 19).all(), f"{season}: {home_games[home_games != 19].to_dict()}"


def test_no_unmapped_team_names(team_match, canonical_names):
    for column in ("team", "opponent", "home_team", "away_team"):
        used = set(team_match[column].dropna())
        assert used <= canonical_names, f"non-canonical in {column}: {sorted(used - canonical_names)}"
        assert not team_match[column].isna().any(), f"{column} has blanks"


def test_a_team_is_never_its_own_opponent(team_match):
    assert not (team_match["team"] == team_match["opponent"]).any()


def test_xg_values_are_non_negative(team_match):
    for column in ("xg_for", "xg_against", "npxg_for", "npxg_against"):
        values = team_match[column].dropna()
        assert (values >= 0).all(), f"{column} has negative values"


def test_xg_values_are_plausible(team_match):
    """A team almost never generates more than about 7 xG in a match."""
    assert team_match["xg_for"].max() < 8.0
    assert team_match["xg_for"].mean() == pytest.approx(1.4, abs=0.4)


def test_non_penalty_xg_never_exceeds_total_xg(team_match):
    """npxG is xG with penalties removed, so it cannot be larger."""
    both = team_match[["xg_for", "npxg_for"]].dropna()
    assert (both["npxg_for"] <= both["xg_for"] + 1e-9).all()


def test_goals_and_xg_agree_on_average(team_match):
    """Across thousands of matches these must be close, or the model is broken."""
    assert team_match["goals_for"].mean() == pytest.approx(
        team_match["xg_for"].mean(), abs=0.2
    )


def test_home_teams_out_create_away_teams(team_match):
    home = team_match[team_match["is_home"]]["xg_for"].mean()
    away = team_match[~team_match["is_home"]]["xg_for"].mean()
    assert home > away, "home advantage should show up in xG"


def test_the_two_rows_of_a_match_mirror_each_other(team_match):
    """One team's xG for must be the other's xG against."""
    sample = team_match[team_match["season"] == team_match["season"].max()]
    paired = sample.groupby("game_id").agg(
        rows=("team", "size"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
    )
    assert (paired["rows"] == 2).all()
    assert (paired["goals_for"] == paired["goals_against"]).all()


def test_dates_are_utc(team_match):
    assert str(team_match["date"].dt.tz) == "UTC"


def test_the_horizon_starts_at_2014_15(team_match):
    assert team_match["season"].min() == "2014-15"


# ---------------------------------------------------------------------------
# player_season
# ---------------------------------------------------------------------------


def test_player_season_names_are_canonical(player_season, canonical_names):
    used = set(player_season["team"].dropna())
    assert used <= canonical_names, f"non-canonical: {sorted(used - canonical_names)}"


def test_player_season_counts_are_sane(player_season, completed_seasons):
    """Roughly 500-700 players appear across a full Premier League season.

    Only completed seasons are checked: a season a few matches old legitimately
    has far fewer players, because most squads have not been rotated yet.
    """
    per_season = player_season.groupby("season").size()
    finished = per_season[per_season.index.isin(completed_seasons)]
    assert finished.between(350, 900).all(), finished.to_dict()


def test_player_xg_and_minutes_are_non_negative(player_season):
    for column in ("minutes", "goals", "xg", "npxg", "assists", "xa", "shots"):
        values = player_season[column].dropna()
        assert (values >= 0).all(), f"{column} has negative values"


def test_no_player_plays_more_than_a_seasons_worth_of_minutes(player_season):
    """38 matches of 90 minutes is 3420; a little stoppage time is fine."""
    assert player_season["minutes"].max() <= 3600


def test_player_non_penalty_xg_never_exceeds_total(player_season):
    both = player_season[["xg", "npxg"]].dropna()
    assert (both["npxg"] <= both["xg"] + 1e-9).all()


def test_a_player_cannot_score_more_goals_than_shots(player_season):
    both = player_season[["goals", "shots"]].dropna()
    assert (both["goals"] <= both["shots"]).all()


# ---------------------------------------------------------------------------
# shots
# ---------------------------------------------------------------------------


def test_shot_teams_are_canonical(shots, canonical_names):
    used = set(shots["team"].dropna())
    assert used <= canonical_names, f"non-canonical: {sorted(used - canonical_names)}"


def test_every_shot_xg_is_a_probability(shots):
    values = shots["xg"].dropna()
    assert (values >= 0).all()
    assert (values <= 1).all()


def test_shot_minutes_are_within_a_match(shots):
    assert shots["minute"].min() >= 0
    assert shots["minute"].max() <= 130  # stoppage time in extra-long matches


def test_goals_are_a_believable_share_of_shots(shots):
    """Roughly one shot in ten is a goal."""
    assert 0.05 < shots["is_goal"].mean() < 0.20


def test_penalties_are_labelled(shots):
    """soccerdata drops Understat's Penalty label; we restore it. Check it stuck."""
    assert shots["is_penalty"].any(), "no penalties found - the label was lost again"
    assert not shots["situation"].isna().any(), "some shots still have no situation"


def test_penalties_have_high_xg_and_convert_often(shots):
    """A penalty is worth about 0.76 xG and goes in a bit under 80% of the time."""
    penalties = shots[shots["is_penalty"]]
    assert penalties["xg"].mean() == pytest.approx(0.76, abs=0.05)
    assert 0.70 < penalties["is_goal"].mean() < 0.90


def test_penalties_are_a_small_share_of_shots(shots):
    """Roughly one shot in a hundred is a penalty."""
    assert 0.003 < shots["is_penalty"].mean() < 0.03


def test_open_play_shots_are_worth_far_less_than_penalties(shots):
    open_play = shots[shots["situation"] == "Open Play"]
    assert open_play["xg"].mean() < 0.2


def test_shots_per_match_are_sane(shots):
    per_match = shots.groupby("game_id").size()
    assert per_match.mean() == pytest.approx(26, abs=8)


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def test_understat_covers_over_99_percent_of_completed_seasons(results, team_match):
    coverage = reconcile.coverage_by_season(results, team_match, "understat")
    full = coverage[coverage["matches"] == MATCHES_PER_SEASON]
    if full.empty:
        pytest.skip("no completed seasons to check")

    poor = full[full["coverage_pct"] < MIN_COVERAGE_PCT]
    assert poor.empty, f"seasons below {MIN_COVERAGE_PCT}% coverage:\n{poor.to_string(index=False)}"


def test_the_reconciliation_report_exists_and_is_readable(results, team_match):
    path = reconcile.RECONCILIATION_CSV
    if not path.exists():
        pytest.skip("reconciliation report has not been built")
    report = pd.read_csv(path)
    assert list(report.columns) == reconcile.REPORT_COLUMNS


# ---------------------------------------------------------------------------
# FBref (skips separately - it needs a browser to get past Cloudflare)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fbref_team_match() -> pd.DataFrame:
    if not fbref.TEAM_MATCH_PARQUET.exists():
        pytest.skip(
            "fbref_team_match.parquet not built. FBref needs a browser for its "
            "Cloudflare challenge; see src/scrape/fbref.py."
        )
    return pd.read_parquet(fbref.TEAM_MATCH_PARQUET)


def test_fbref_team_names_are_canonical(fbref_team_match, canonical_names):
    used = set(fbref_team_match["team"].dropna())
    assert used <= canonical_names, f"non-canonical: {sorted(used - canonical_names)}"


def test_fbref_seasons_start_at_2017_18(fbref_team_match):
    assert fbref_team_match["season"].min() >= "2017-18"


def test_fbref_match_counts_are_sane(fbref_team_match):
    counts = fbref_team_match.groupby("season").size()
    complete = counts[counts >= MATCHES_PER_SEASON * 2]
    for season, count in complete.items():
        assert count == MATCHES_PER_SEASON * 2, f"{season} has {count} rows"


def test_fbref_has_the_disciplinary_columns_the_referee_model_needs(fbref_team_match):
    columns = set(fbref_team_match.columns)
    assert any("crdy" in c or "yellow" in c for c in columns), (
        f"no yellow-card column found in {sorted(columns)[:40]}"
    )
