"""Elo ratings computed from our own match results.

This is the guaranteed default source of Elo for the model. It needs nothing but
``results.parquet``, so it always works - no third-party API, no network, no
waiting on a slow server. The live Club Elo pull in ``src/scrape/clubelo.py`` is
kept as a cross-check on these numbers rather than as the thing the model
depends on.

The method
----------
Standard Elo, as used by Club Elo and the World Football Elo Ratings it is built
on:

    expected  = 1 / (10 ** (-difference / 400) + 1)
    new       = old + K * weight * (result - expected)

where ``difference`` is the home team's rating minus the away team's, plus a
home advantage; ``result`` is 1 for a win, 0.5 for a draw, 0 for a loss; and
``weight`` scales K by the margin of victory, so a 4-0 moves the ratings further
than a 1-0. Every point one team gains, the other loses, so the league's total
rating never drifts.

**On the constants.** clubelo.com/System could not be read while this was
written - the site is reachable but too slow to serve the page, which is the
same problem that made the API look blocked. The values below are the standard
published Elo constants for club league matches; ``K = 20`` and the margin
weights are the World Football Elo convention that Club Elo documents itself as
following. They are named and adjustable rather than buried, and
:func:`calibration_by_rating_gap` checks the output actually predicts results.
If you can read clubelo.com/System, confirm them and note it in
data/lookups/NOTES.md.

Promoted clubs
--------------
We only see Premier League matches, so a promoted club arrives with no history.
Rather than guess a number, a newly promoted club starts at the average final
rating of the clubs it replaced - the ones relegated the season before. That is
self-calibrating, and closer to the truth than dropping them in at league
average, which would treat a promoted side as mid-table on day one.

No leakage
----------
Ratings are built strictly forward in time. The rating attached to a match is
always the one **before** it was played, so a feature built from this can never
see the result it is trying to predict.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_PARQUET = PROCESSED_DIR / "results.parquet"
ELO_HISTORY_PARQUET = PROCESSED_DIR / "elo_history.parquet"
MATCH_ELO_PARQUET = PROCESSED_DIR / "match_elo.parquet"

SOURCE = "internal"

#: Everyone starts level at the beginning of the history. The scale is
#: arbitrary; only differences between clubs mean anything.
INITIAL_RATING = 1500.0

#: K controls how fast ratings move. 20 is the published value for club league
#: matches - high enough to track form, low enough not to overreact to one game.
DEFAULT_K = 20.0

#: Home advantage in rating points, added to the home side before working out
#: what was expected of them.
#:
#: **Fitted to our own data, not inherited.** The first draft used 65, a figure
#: carried over from general club-football Elo. Sweeping this against all 4,570
#: Premier League matches put the minimum squared error at 55, and 65 left a
#: visible bias: home favourites were over-predicted by up to five points of
#: expected score. :func:`fit_home_advantage` reproduces the sweep, and there is
#: a test that this constant stays near the fitted optimum.
DEFAULT_HOME_ADVANTAGE = 55.0

#: Used only if a promoted club arrives and we cannot work out who went down.
PROMOTED_FALLBACK_RATING = 1425.0

#: The far-future date marking a club's current rating, matching Club Elo's own
#: convention so the two tables can be read the same way.
OPEN_ENDED_TO = pd.Timestamp("2100-01-01")

ELO_HISTORY_COLUMNS = ["team", "elo", "valid_from", "valid_to", "source", "matches_played"]

MATCH_ELO_COLUMNS = [
    "date", "season", "home_team", "away_team",
    "home_elo_before", "away_elo_before", "elo_difference",
    "expected_home_score", "result",
]


class EloError(ValueError):
    """Raised when results.parquet cannot support an Elo calculation."""


# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------


def expected_score(rating_difference: float | np.ndarray) -> float | np.ndarray:
    """The share of the points the stronger side is expected to take.

    Returns a number between 0 and 1: 0.5 for evenly matched sides, rising
    towards 1 as the difference grows. A 400-point gap means the favourite is
    expected to take about 91% of the points on offer.
    """
    return 1.0 / (10.0 ** (-np.asarray(rating_difference, dtype="float64") / 400.0) + 1.0)


def margin_weight(goal_difference: int) -> float:
    """How much a win by this margin counts, relative to a one-goal win.

    The published weighting: a two-goal win counts half as much again, three
    goals counts 1.75x, and each further goal adds an eighth. Beating someone 5-0
    is stronger evidence than edging them 1-0, but with diminishing returns, so a
    rout does not swamp the rating.
    """
    margin = abs(int(goal_difference))
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return 1.75 + (margin - 3) / 8.0


def match_result_score(home_goals: int, away_goals: int) -> float:
    """1.0 if the home side won, 0.5 for a draw, 0.0 if they lost."""
    if home_goals > away_goals:
        return 1.0
    if home_goals == away_goals:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Building the ratings
# ---------------------------------------------------------------------------


def _starting_rating_for_newcomers(
    ratings: dict[str, float],
    previous_season_teams: set[str],
    current_season_teams: set[str],
) -> float:
    """What a promoted club should start on: the average of who went down.

    A promoted club replaces a relegated one, so the relegated clubs' final
    ratings are the best available estimate of the level the newcomers enter at.
    Falls back to a constant only when there is nothing to work from.
    """
    relegated = previous_season_teams - current_season_teams
    rated = [ratings[team] for team in relegated if team in ratings]
    if not rated:
        return PROMOTED_FALLBACK_RATING
    return float(np.mean(rated))


def compute_ratings(
    results: pd.DataFrame,
    *,
    k: float = DEFAULT_K,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
    initial_rating: float = INITIAL_RATING,
    use_margin: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk through every match in order, updating ratings as it goes.

    Returns ``(history, match_ratings)``:

    ``history``
        One row per club per rating period, with the dates it was in force -
        the same shape Club Elo publishes, so :func:`get_elo` reads either.
    ``match_ratings``
        One row per match with both sides' ratings **before** it was played,
        which is what a feature should use.
    """
    required = {"date", "season", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(results.columns)
    if missing:
        raise EloError(
            f"results.parquet is missing {sorted(missing)}, so Elo cannot be computed."
        )
    if results.empty:
        raise EloError("No matches to compute Elo from.")

    matches = results.sort_values("date", kind="stable").reset_index(drop=True)
    if matches["date"].isna().any():
        raise EloError("Some matches have no date; Elo must be built in time order.")

    ratings: dict[str, float] = {}
    played: dict[str, int] = {}
    history: list[dict] = []
    per_match: list[dict] = []

    season_teams = (
        matches.groupby("season")
        .apply(lambda g: set(g["home_team"]) | set(g["away_team"]), include_groups=False)
        .to_dict()
    )
    seasons_in_order = list(dict.fromkeys(matches["season"]))
    previous_season: str | None = None

    for season, group in matches.groupby("season", sort=False):
        # Anyone new this season is promoted (or it is the first season).
        newcomers = {
            team
            for team in season_teams[season]
            if team not in ratings
        }
        if newcomers:
            start = (
                initial_rating
                if previous_season is None
                else _starting_rating_for_newcomers(
                    ratings, season_teams[previous_season], season_teams[season]
                )
            )
            for team in sorted(newcomers):
                ratings[team] = start
                played[team] = 0
                history.append(
                    {
                        "team": team,
                        "elo": start,
                        "valid_from": group["date"].min().tz_convert("UTC").normalize().tz_localize(None),
                        "source": SOURCE,
                        "matches_played": 0,
                    }
                )
            if previous_season is not None:
                logger.debug(
                    "%s: %d club(s) promoted in at %.0f", season, len(newcomers), start
                )

        for row in group.itertuples(index=False):
            home, away = row.home_team, row.away_team
            home_rating, away_rating = ratings[home], ratings[away]

            difference = home_rating + home_advantage - away_rating
            expected = float(expected_score(difference))
            actual = match_result_score(row.home_goals, row.away_goals)
            weight = margin_weight(row.home_goals - row.away_goals) if use_margin else 1.0

            change = k * weight * (actual - expected)
            day = pd.Timestamp(row.date).tz_convert("UTC").normalize().tz_localize(None)

            per_match.append(
                {
                    "date": row.date,
                    "season": row.season,
                    "home_team": home,
                    "away_team": away,
                    "home_elo_before": home_rating,
                    "away_elo_before": away_rating,
                    "elo_difference": home_rating - away_rating,
                    "expected_home_score": expected,
                    "result": getattr(row, "result", None),
                }
            )

            # Zero sum: what one side gains, the other loses.
            ratings[home] = home_rating + change
            ratings[away] = away_rating - change
            played[home] = played.get(home, 0) + 1
            played[away] = played.get(away, 0) + 1

            for team in (home, away):
                history.append(
                    {
                        "team": team,
                        "elo": ratings[team],
                        # The new rating applies from the day after the match.
                        "valid_from": day + pd.Timedelta(days=1),
                        "source": SOURCE,
                        "matches_played": played[team],
                    }
                )

        previous_season = season

    return _close_periods(pd.DataFrame(history)), pd.DataFrame(per_match)[MATCH_ELO_COLUMNS]


def _close_periods(history: pd.DataFrame) -> pd.DataFrame:
    """Turn a list of rating changes into periods with a start and an end.

    Each rating holds until the club's next change; the newest holds open-ended,
    exactly as Club Elo publishes it.
    """
    if history.empty:
        return pd.DataFrame(columns=ELO_HISTORY_COLUMNS)

    history = history.sort_values(["team", "valid_from"], kind="stable").reset_index(drop=True)
    # Two matches on the same day would otherwise produce a zero-length period.
    history = history.drop_duplicates(subset=["team", "valid_from"], keep="last")

    history["valid_to"] = (
        history.groupby("team")["valid_from"].shift(-1) - pd.Timedelta(days=1)
    ).fillna(OPEN_ENDED_TO)

    return history[ELO_HISTORY_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Checking it works
# ---------------------------------------------------------------------------


def prediction_error(
    results: pd.DataFrame, *, k: float = DEFAULT_K, home_advantage: float = DEFAULT_HOME_ADVANTAGE
) -> tuple[float, float]:
    """How well a given set of parameters predicts the matches we have.

    Returns ``(mean squared error, bias)`` between the expected home score and
    what actually happened. Bias near zero means the home advantage is right:
    positive means home sides do better than the ratings expect, negative worse.
    """
    _, per_match = compute_ratings(results, k=k, home_advantage=home_advantage)
    scored = per_match.dropna(subset=["result"])
    if scored.empty:
        raise EloError("No matches with a result to score against.")

    actual = scored["result"].map({"H": 1.0, "D": 0.5, "A": 0.0})
    error = actual - scored["expected_home_score"]
    return float((error**2).mean()), float(error.mean())


def fit_home_advantage(
    results: pd.DataFrame,
    candidates: tuple[float, ...] = (0, 20, 30, 40, 50, 55, 60, 65, 70, 80),
    *,
    k: float = DEFAULT_K,
) -> float:
    """Find the home advantage that predicts our own matches best.

    Worth doing rather than inheriting a number: the general club-football
    figure of 65 was measurably too high here, leaving home favourites
    over-predicted. This is what set :data:`DEFAULT_HOME_ADVANTAGE`.
    """
    scored = {
        candidate: prediction_error(results, k=k, home_advantage=candidate)[0]
        for candidate in candidates
    }
    best = min(scored, key=scored.get)
    logger.info(
        "Fitted home advantage: %.0f (squared error %.5f)", best, scored[best]
    )
    return float(best)


def calibration_by_rating_gap(
    match_ratings: pd.DataFrame, bins: int = 6
) -> pd.DataFrame:
    """Do bigger rating gaps actually mean more home wins?

    The honest test of whether these ratings mean anything: group matches by how
    far apart the two sides were rated and compare what the model expected with
    what happened. The two columns should track each other closely.
    """
    frame = match_ratings.dropna(subset=["result"]).copy()
    if frame.empty:
        raise EloError("No matches with a result to calibrate against.")

    frame["actual"] = frame["result"].map({"H": 1.0, "D": 0.5, "A": 0.0})
    frame["bucket"] = pd.qcut(frame["elo_difference"], bins, duplicates="drop")

    summary = frame.groupby("bucket", observed=True).agg(
        matches=("actual", "size"),
        mean_elo_difference=("elo_difference", "mean"),
        expected_home_score=("expected_home_score", "mean"),
        actual_home_score=("actual", "mean"),
        home_win_rate=("result", lambda s: float((s == "H").mean())),
    )
    summary["error"] = (summary["actual_home_score"] - summary["expected_home_score"]).round(4)
    return summary.round(4).reset_index()


def compare_with_clubelo(
    internal: pd.DataFrame,
    clubelo: pd.DataFrame,
    when: pd.Timestamp | str | None = None,
    teams: list[str] | None = None,
) -> pd.DataFrame:
    """Line our ratings up against Club Elo's for the same clubs on the same day.

    The two are not expected to match in level - Club Elo has seen European and
    cup matches we never see, and its scale is its own - so the number that
    matters is whether they **agree on the ordering** and on the gaps between
    clubs, not whether they print the same figure.
    """
    from src.scrape.clubelo import get_elo

    when = pd.Timestamp(when) if when is not None else pd.Timestamp.utcnow().normalize()
    teams = teams if teams is not None else sorted(set(internal["team"]))

    rows = []
    for team in teams:
        ours = get_elo(team, when, internal)
        theirs = get_elo(team, when, clubelo)
        if ours is None or theirs is None:
            continue
        rows.append({"team": team, "internal": round(ours, 1), "clubelo": round(theirs, 1)})

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison

    comparison["internal_rank"] = comparison["internal"].rank(ascending=False).astype(int)
    comparison["clubelo_rank"] = comparison["clubelo"].rank(ascending=False).astype(int)
    comparison["rank_difference"] = (
        comparison["internal_rank"] - comparison["clubelo_rank"]
    ).abs()
    return comparison.sort_values("clubelo", ascending=False).reset_index(drop=True)


def agreement_summary(comparison: pd.DataFrame) -> dict[str, float]:
    """One-line verdict on how closely the two sources agree."""
    if comparison.empty or len(comparison) < 3:
        return {"clubs": float(len(comparison))}

    return {
        "clubs": float(len(comparison)),
        "pearson": round(float(comparison["internal"].corr(comparison["clubelo"])), 4),
        "spearman": round(
            float(comparison["internal"].corr(comparison["clubelo"], method="spearman")), 4
        ),
        "mean_rank_difference": round(float(comparison["rank_difference"].mean()), 2),
        "max_rank_difference": float(comparison["rank_difference"].max()),
    }


# ---------------------------------------------------------------------------
# Building and writing
# ---------------------------------------------------------------------------


def build_all(
    *,
    results_path: Path | str = RESULTS_PARQUET,
    history_path: Path | str = ELO_HISTORY_PARQUET,
    match_path: Path | str = MATCH_ELO_PARQUET,
    k: float = DEFAULT_K,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
) -> dict[str, Path]:
    """Compute Elo from results.parquet and write both output tables."""
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(
            f"No results at {results_path}. Run 'python -m pipelines.build_results' first."
        )

    results = pd.read_parquet(results_path)
    history, per_match = compute_ratings(results, k=k, home_advantage=home_advantage)

    for path, frame, label in (
        (Path(history_path), history, "Elo history"),
        (Path(match_path), per_match, "per-match Elo"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        logger.info("Wrote %d %s rows to %s", len(frame), label, path)

    return {"history": Path(history_path), "match": Path(match_path)}
