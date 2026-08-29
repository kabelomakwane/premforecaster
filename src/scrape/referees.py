"""Referee profiles, computed from match results we already hold.

Referees differ, and consistently so: some show twice as many yellow cards per
match as others, and those differences persist season after season. That matters
for any card or booking market, and it is a small but real signal for the match
itself, since cards and penalties change games.

No scraping is needed for this. football-data.co.uk already publishes the
referee's name for every match alongside the cards and fouls, and
``results.parquet`` now carries all of it. So this module is pure computation
over data on disk - which also means it is fast, deterministic and needs no
network at all.

What is and is not derivable here
---------------------------------
From football-data we get, per referee per season: matches officiated, yellow
and red cards per match, fouls per match, and the home win rate in their games.

**Penalties are not in football-data.** There is no penalty column in the E0
files, so a penalty rate cannot be computed from that source alone. Where
Understat shot data is available it can be - a shot with situation "Penalty" is
exactly a penalty awarded - so :func:`add_penalty_rates` will fill that in for
the seasons the shot table covers, and leave it blank elsewhere rather than
inventing a number.

A word of warning about reading these numbers
---------------------------------------------
The home win rate in a referee's matches is **not** evidence of bias. Referees
are assigned to matches, often the bigger ones to the more senior officials, so
the figure mostly reflects which fixtures they were given. It is included
because it is a useful feature, not because it says anything about the referee.
Likewise a high card rate partly reflects being assigned to fierce fixtures.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_PARQUET = PROCESSED_DIR / "results.parquet"
SHOTS_PARQUET = PROCESSED_DIR / "shots.parquet"
REFEREE_PROFILES_PARQUET = PROCESSED_DIR / "referee_profiles.parquet"

#: Columns results.parquet must have for this to mean anything.
REQUIRED_COLUMNS = (
    "season", "referee", "result",
    "home_yellows", "away_yellows", "home_reds", "away_reds",
)

#: Below this many matches a referee's rates are mostly noise. They are still
#: reported - dropping them would hide who officiated - but flagged so nothing
#: downstream treats a three-match sample as a real tendency.
MIN_MATCHES_FOR_RELIABLE_RATE = 10

PROFILE_COLUMNS = [
    "referee", "season", "matches",
    "yellows_per_match", "reds_per_match", "cards_per_match", "fouls_per_match",
    "yellows_per_foul", "yellows_vs_season_average",
    "home_win_rate", "draw_rate", "away_win_rate",
    "goals_per_match", "penalties_per_match", "reliable",
]


class RefereeDataError(ValueError):
    """Raised when results.parquet cannot support referee profiles."""


def load_results(path: Path | str = RESULTS_PARQUET) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No results at {path}. Run 'python -m pipelines.build_results' first."
        )
    results = pd.read_parquet(path)

    missing = [column for column in REQUIRED_COLUMNS if column not in results.columns]
    if missing:
        raise RefereeDataError(
            f"results.parquet is missing the column(s) {missing}. The card columns "
            "were added to the football-data processor for exactly this; rebuild "
            "with 'python -m pipelines.build_results --no-download'."
        )
    return results


def build_profiles(
    results: pd.DataFrame,
    *,
    by_season: bool = True,
    min_matches: int = MIN_MATCHES_FOR_RELIABLE_RATE,
) -> pd.DataFrame:
    """Aggregate per referee (and by default per season).

    Set ``by_season=False`` for career totals across the whole history, which
    are steadier but blur genuine changes in how strictly a referee officiates.
    """
    frame = results.copy()

    named = frame["referee"].notna() & (frame["referee"].astype("string").str.strip() != "")
    if not named.any():
        raise RefereeDataError("No match in results.parquet has a referee name.")
    dropped = int((~named).sum())
    if dropped:
        logger.warning("Ignoring %d match(es) with no referee recorded.", dropped)
    frame = frame[named].copy()

    frame["referee"] = frame["referee"].astype("string").str.strip()
    frame["yellows"] = frame["home_yellows"].fillna(0) + frame["away_yellows"].fillna(0)
    frame["reds"] = frame["home_reds"].fillna(0) + frame["away_reds"].fillna(0)
    frame["cards"] = frame["yellows"] + frame["reds"]
    frame["fouls"] = frame.get("home_fouls", 0).fillna(0) + frame.get("away_fouls", 0).fillna(0)
    frame["goals"] = frame["home_goals"] + frame["away_goals"]
    frame["home_win"] = frame["result"].eq("H")
    frame["draw"] = frame["result"].eq("D")
    frame["away_win"] = frame["result"].eq("A")

    keys = ["referee", "season"] if by_season else ["referee"]
    grouped = frame.groupby(keys, dropna=False)

    profiles = grouped.agg(
        matches=("referee", "size"),
        yellows_per_match=("yellows", "mean"),
        reds_per_match=("reds", "mean"),
        cards_per_match=("cards", "mean"),
        fouls_per_match=("fouls", "mean"),
        home_win_rate=("home_win", "mean"),
        draw_rate=("draw", "mean"),
        away_win_rate=("away_win", "mean"),
        goals_per_match=("goals", "mean"),
    ).reset_index()

    # How readily a referee reaches for a card given the fouls in front of them:
    # a stricter measure of temperament than cards alone, which partly reflect
    # how fractious their fixtures were.
    profiles["yellows_per_foul"] = (
        profiles["yellows_per_match"] / profiles["fouls_per_match"]
    ).where(profiles["fouls_per_match"] > 0)

    # How strict this referee was *relative to their era*. League-wide card rates
    # move a lot - yellows per match went from 2.87 in 2020/21 to 4.17 in 2023/24
    # after the rules on dissent and time-wasting were tightened - so a raw rate
    # conflates the referee with the season they worked in. A ratio of 1.2 means
    # 20% more cards than their contemporaries, which is the bit that is actually
    # about them, and the better feature for a model.
    if by_season:
        season_average = frame.groupby("season")["yellows"].mean().rename("_league")
        profiles = profiles.merge(
            season_average, left_on="season", right_index=True, how="left"
        )
    else:
        profiles["_league"] = frame["yellows"].mean()

    profiles["yellows_vs_season_average"] = (
        profiles["yellows_per_match"] / profiles["_league"]
    ).where(profiles["_league"] > 0)
    profiles = profiles.drop(columns="_league")

    profiles["penalties_per_match"] = pd.NA
    profiles["reliable"] = profiles["matches"] >= min_matches

    if not by_season:
        profiles["season"] = "all"

    rounded = [
        "yellows_per_match", "reds_per_match", "cards_per_match", "fouls_per_match",
        "yellows_per_foul", "yellows_vs_season_average",
        "home_win_rate", "draw_rate", "away_win_rate", "goals_per_match",
    ]
    profiles[rounded] = profiles[rounded].astype("float64").round(4)

    return profiles[PROFILE_COLUMNS].sort_values(
        ["season", "cards_per_match"], ascending=[True, False]
    ).reset_index(drop=True)


def add_penalty_rates(
    profiles: pd.DataFrame,
    results: pd.DataFrame,
    shots: pd.DataFrame,
    *,
    by_season: bool = True,
) -> pd.DataFrame:
    """Fill in penalties per match, for the seasons shot data covers.

    football-data has no penalty column, so this is the only way to derive one.
    A shot with situation "Penalty" in the Understat data is a penalty awarded,
    so counting those per match gives the rate. Seasons with no shot data keep a
    blank rather than a made-up zero - a referee who awarded no penalties and a
    referee we have no data for must not look the same.
    """
    if shots is None or shots.empty:
        logger.info("No shot data available; penalties_per_match left blank.")
        return profiles

    if "is_penalty" not in shots.columns:
        raise RefereeDataError(
            "The shot table has no is_penalty column. It is added by "
            "src/scrape/understat.py; rebuild the shot data."
        )

    penalties = (
        shots[shots["is_penalty"]]
        .groupby(["season", "game_id"])
        .size()
        .rename("penalties")
        .reset_index()
    )

    # Match the shot table's per-match penalty counts onto referees via the
    # fixture date, which is the join key both tables share.
    match_days = results.assign(
        match_day=pd.to_datetime(results["date"], utc=True).dt.normalize()
    )
    shot_days = (
        shots.groupby(["season", "game_id"])["date"]
        .min()
        .reset_index()
        .assign(match_day=lambda d: pd.to_datetime(d["date"], utc=True).dt.normalize())
        .drop(columns="date")
        .merge(penalties, on=["season", "game_id"], how="left")
    )
    shot_days["penalties"] = shot_days["penalties"].fillna(0)

    per_day = shot_days.groupby(["season", "match_day"])["penalties"].sum().reset_index()
    matches_per_day = (
        shot_days.groupby(["season", "match_day"]).size().rename("shot_matches").reset_index()
    )
    per_day = per_day.merge(matches_per_day, on=["season", "match_day"])

    covered = match_days.merge(per_day, on=["season", "match_day"], how="inner")
    if covered.empty:
        logger.info("Shot data does not overlap the results by date; penalties left blank.")
        return profiles

    # Penalties per match on a given day, attributed to the referees working it.
    covered["penalties_per_match"] = covered["penalties"] / covered["shot_matches"]

    keys = ["referee", "season"] if by_season else ["referee"]
    rates = covered.groupby(keys)["penalties_per_match"].mean().round(4).reset_index()

    updated = profiles.drop(columns=["penalties_per_match"]).merge(rates, on=keys, how="left")
    logger.info(
        "Penalty rates filled for %d of %d profile row(s).",
        int(updated["penalties_per_match"].notna().sum()), len(updated),
    )
    return updated[PROFILE_COLUMNS]


def write_profiles(
    profiles: pd.DataFrame, path: Path | str = REFEREE_PROFILES_PARQUET
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_parquet(path, index=False)
    logger.info("Wrote %d referee profile rows to %s", len(profiles), path)
    return path


def build_all(
    *,
    results_path: Path | str = RESULTS_PARQUET,
    shots_path: Path | str = SHOTS_PARQUET,
    output_path: Path | str = REFEREE_PROFILES_PARQUET,
    by_season: bool = True,
) -> Path:
    """Build referee profiles from the tables on disk. No network at all."""
    results = load_results(results_path)
    profiles = build_profiles(results, by_season=by_season)

    shots_path = Path(shots_path)
    if shots_path.exists():
        profiles = add_penalty_rates(
            profiles, results, pd.read_parquet(shots_path), by_season=by_season
        )
    else:
        logger.info("No shot table at %s; penalties_per_match left blank.", shots_path)

    return write_profiles(profiles, output_path)


def strictest(profiles: pd.DataFrame, top: int = 10, min_matches: int = 30) -> pd.DataFrame:
    """The strictest referees, ranked against their own era rather than raw rates.

    Ranking on raw cards per match mostly ranks *eras*: a referee working in
    2023/24 shows more yellows than one working in 2020/21 whatever their
    temperament. Ranking on ``yellows_vs_season_average`` compares each referee
    with their contemporaries, which is the comparison worth making.

    ``min_matches`` is a career total: a referee with two dozen matches can top
    a rate table by chance.
    """
    reliable = profiles[profiles["reliable"]]
    if reliable.empty:
        return reliable

    weighted = reliable.assign(
        _relative=reliable["yellows_vs_season_average"] * reliable["matches"],
        _yellows=reliable["yellows_per_match"] * reliable["matches"],
        _reds=reliable["reds_per_match"] * reliable["matches"],
    )
    totals = weighted.groupby("referee").agg(
        matches=("matches", "sum"),
        _relative=("_relative", "sum"),
        _yellows=("_yellows", "sum"),
        _reds=("_reds", "sum"),
    )
    career = pd.DataFrame(
        {
            "matches": totals["matches"],
            "yellows_per_match": totals["_yellows"] / totals["matches"],
            "reds_per_match": totals["_reds"] / totals["matches"],
            "vs_era_average": totals["_relative"] / totals["matches"],
        }
    )
    career = career[career["matches"] >= min_matches]
    return career.sort_values("vs_era_average", ascending=False).head(top).round(3)
