"""Cross-checking the xG tables against the match results.

football-data.co.uk is our spine: it has every Premier League match, and
``results.parquet`` is built from it. Understat and FBref are then joined on to
that spine to add xG and advanced stats.

Any match that exists on one side but not the other is a silent hole in the
model - a fixture that quietly gets no xG, or an xG row that never finds its
match and is dropped. This module finds those holes and writes them to
``data/processed/reconciliation_report.csv`` so they can be looked at rather
than discovered later as a mysterious dip in accuracy.

Matches are joined on canonical team name plus match date. Dates need a little
care: football-data records the kick-off in UTC, while Understat and FBref
record the local match date, so a 20:00 UK kick-off is the same calendar day but
a late kick-off elsewhere might not be. We therefore join on the calendar date
and, if that fails, allow one day either side before calling a match missing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_PARQUET = PROCESSED_DIR / "results.parquet"
RECONCILIATION_CSV = PROCESSED_DIR / "reconciliation_report.csv"

#: How many days apart two records of the same fixture may be and still be
#: considered the same match. One day absorbs timezone and late-kick-off edges.
DATE_TOLERANCE_DAYS = 1

REPORT_COLUMNS = [
    "source",
    "issue",
    "season",
    "date",
    "home_team",
    "away_team",
    "detail",
]


def _fixture_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Reduce any match table to season / date / home / away, one row per match."""
    required = {"season", "date", "home_team", "away_team"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{label} is missing the column(s) {sorted(missing)}, so it cannot be "
            f"reconciled. Columns present: {list(frame.columns)}"
        )

    fixtures = frame[["season", "date", "home_team", "away_team"]].copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"], utc=True).dt.tz_convert("UTC")
    fixtures["match_day"] = fixtures["date"].dt.normalize()
    return fixtures.drop_duplicates(subset=["season", "home_team", "away_team"])


def compare_fixtures(
    results: pd.DataFrame,
    other: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    """List fixtures in one table but not the other.

    Returns a frame of problems, empty when the two sides agree. Three kinds of
    problem are reported:

    ``missing_from_<source>``
        The match is in results.parquet but the source has no row for it. This
        is the one that matters: those matches will have no xG.
    ``missing_from_results``
        The source has a match football-data does not. Usually a sign the season
        or team mapping is off.
    ``date_mismatch``
        Both sides have the fixture but disagree on the date by more than a day,
        which means a join on date would fail even though the match is present.
    """
    left = _fixture_frame(results, "results.parquet")
    right = _fixture_frame(other, source)

    key = ["season", "home_team", "away_team"]
    merged = left.merge(right, on=key, how="outer", suffixes=("_results", f"_{source}"), indicator=True)

    problems: list[pd.DataFrame] = []

    only_results = merged[merged["_merge"] == "left_only"]
    if not only_results.empty:
        problems.append(
            pd.DataFrame(
                {
                    "source": source,
                    "issue": f"missing_from_{source}",
                    "season": only_results["season"],
                    "date": only_results["date_results"],
                    "home_team": only_results["home_team"],
                    "away_team": only_results["away_team"],
                    "detail": f"in results.parquet but not in {source}",
                }
            )
        )

    only_other = merged[merged["_merge"] == "right_only"]
    if not only_other.empty:
        problems.append(
            pd.DataFrame(
                {
                    "source": source,
                    "issue": "missing_from_results",
                    "season": only_other["season"],
                    "date": only_other[f"date_{source}"],
                    "home_team": only_other["home_team"],
                    "away_team": only_other["away_team"],
                    "detail": f"in {source} but not in results.parquet",
                }
            )
        )

    both = merged[merged["_merge"] == "both"].copy()
    if not both.empty:
        gap = (both["match_day_results"] - both[f"match_day_{source}"]).abs()
        drifted = both[gap > pd.Timedelta(days=DATE_TOLERANCE_DAYS)]
        if not drifted.empty:
            drift_days = (
                (drifted["match_day_results"] - drifted[f"match_day_{source}"])
                .dt.days.astype(str)
            )
            problems.append(
                pd.DataFrame(
                    {
                        "source": source,
                        "issue": "date_mismatch",
                        "season": drifted["season"],
                        "date": drifted["date_results"],
                        "home_team": drifted["home_team"],
                        "away_team": drifted["away_team"],
                        "detail": "results date differs by " + drift_days + " day(s)",
                    }
                )
            )

    if not problems:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    return pd.concat(problems, ignore_index=True)[REPORT_COLUMNS]


def coverage_by_season(
    results: pd.DataFrame, other: pd.DataFrame, source: str
) -> pd.DataFrame:
    """What fraction of each season's matches the source actually covers."""
    left = _fixture_frame(results, "results.parquet")
    right = _fixture_frame(other, source)

    key = ["season", "home_team", "away_team"]
    merged = left.merge(right[key].assign(_present=True), on=key, how="left")
    merged["_present"] = merged["_present"].notna()

    coverage = merged.groupby("season").agg(
        matches=("_present", "size"),
        covered=("_present", "sum"),
    )
    coverage["missing"] = coverage["matches"] - coverage["covered"]
    coverage["coverage_pct"] = (100 * coverage["covered"] / coverage["matches"]).round(2)
    coverage.insert(0, "source", source)
    return coverage.reset_index()


def build_report(
    results: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile every available source against the results spine.

    Returns ``(problems, coverage)``: the row-by-row exceptions, and the
    per-season coverage summary.
    """
    problems = []
    coverage = []
    for source, frame in sources.items():
        if frame is None or frame.empty:
            logger.warning("Skipping %s in reconciliation: no data.", source)
            continue
        problems.append(compare_fixtures(results, frame, source))
        coverage.append(coverage_by_season(results, frame, source))

    problem_report = (
        pd.concat(problems, ignore_index=True)
        if problems
        else pd.DataFrame(columns=REPORT_COLUMNS)
    )
    if not problem_report.empty:
        problem_report = problem_report.sort_values(
            ["source", "season", "date", "home_team"]
        ).reset_index(drop=True)

    coverage_report = (
        pd.concat(coverage, ignore_index=True) if coverage else pd.DataFrame()
    )
    return problem_report, coverage_report


def write_report(
    problems: pd.DataFrame, path: Path | str = RECONCILIATION_CSV
) -> Path:
    """Write the exceptions to CSV. An empty file with headers means all clear."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    problems.to_csv(path, index=False)
    logger.info("Wrote %d reconciliation exception(s) to %s", len(problems), path)
    return path
