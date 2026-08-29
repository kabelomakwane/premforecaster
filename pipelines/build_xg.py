"""Pull the xG and advanced-stat tables, then reconcile them against results.

Run from the repo root with the virtual environment active:

    python -m pipelines.build_xg                  # everything
    python -m pipelines.build_xg --skip-fbref     # Understat only (much quicker)
    python -m pipelines.build_xg --seasons 2025-26 2026-27

Both scrapers are cache-first and resumable, so running this twice in a row
makes no network requests the second time, and an interrupted run picks up where
it stopped. FBref is slow by design (one request every seven seconds) and needs
a browser to get past Cloudflare; if it is unavailable the run carries on with
Understat and says so, because Understat is what supplies the xG the model
actually depends on.

At the end it writes data/processed/reconciliation_report.csv listing any match
that is in results.parquet but missing from an xG table, or vice versa.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src import reconcile
from src.scrape import fbref, understat
from src.scrape.footballdata import season_start_year


def _load_if_present(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-understat", action="store_true")
    parser.add_argument("--skip-fbref", action="store_true")
    parser.add_argument(
        "--seasons", nargs="+", metavar="YYYY-YY",
        help="Only these seasons, e.g. 2025-26. Defaults to every season each source covers.",
    )
    parser.add_argument(
        "--shot-seasons", nargs="+", metavar="YYYY-YY",
        help="Seasons to pull shot-level data for. Defaults to the last four.",
    )
    parser.add_argument(
        "--browser",
        help="Path to a Chrome/Chromium binary for FBref's Cloudflare challenge.",
    )
    parser.add_argument(
        "--reconcile-only", action="store_true",
        help="Skip all scraping and just rebuild the reconciliation report.",
    )
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    log = logging.getLogger("build_xg")

    requested = (
        [season_start_year(label) for label in arguments.seasons]
        if arguments.seasons else None
    )
    shot_years = (
        [season_start_year(label) for label in arguments.shot_seasons]
        if arguments.shot_seasons else None
    )

    if not arguments.reconcile_only and not arguments.skip_understat:
        understat.build_all(start_years=requested, shot_years=shot_years)

    if not arguments.reconcile_only and not arguments.skip_fbref:
        try:
            fbref.build_all(start_years=requested, browser=arguments.browser)
        except fbref.FBrefUnavailableError as error:
            log.warning("FBref unavailable, carrying on without it: %s", error)

    # --- Reconciliation ----------------------------------------------------
    results_path = reconcile.RESULTS_PARQUET
    if not results_path.exists():
        log.error(
            "%s not found. Run 'python -m pipelines.build_results' first - it is "
            "the spine everything else is checked against.", results_path,
        )
        return 1

    results = pd.read_parquet(results_path)
    sources = {
        "understat": _load_if_present(understat.TEAM_MATCH_PARQUET),
        "fbref": _load_if_present(fbref.TEAM_MATCH_PARQUET),
    }
    available = {name: frame for name, frame in sources.items() if frame is not None}
    if not available:
        log.error("No xG tables found to reconcile against.")
        return 1

    problems, coverage = reconcile.build_report(results, available)
    reconcile.write_report(problems)

    with pd.option_context("display.width", 200, "display.max_rows", 60):
        print("\n=== Coverage by season ===")
        print(coverage.to_string(index=False))

        completed = coverage[coverage["matches"] == 380]
        if not completed.empty:
            worst = completed["coverage_pct"].min()
            print(f"\nWorst completed-season coverage: {worst:.2f}%")

        print(f"\n=== Exceptions: {len(problems)} ===")
        if problems.empty:
            print("None - every match reconciles.")
        else:
            print(problems.head(50).to_string(index=False))
            print(f"\nFull list written to {reconcile.RECONCILIATION_CSV}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
