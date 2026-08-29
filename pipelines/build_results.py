"""Download football-data.co.uk season CSVs and build data/processed/results.parquet.

Run this from the repo root with the virtual environment active:

    python -m pipelines.build_results              # download, then rebuild
    python -m pipelines.build_results --no-download  # rebuild from existing raw files
    python -m pipelines.build_results --seasons 2025-26 2026-27

It prints a per-season summary at the end so you can see at a glance that every
finished season has its 380 matches and that the odds came through.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from src.scrape import footballdata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip downloading and rebuild from the raw files already on disk.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        metavar="YYYY-YY",
        help="Only handle these seasons, e.g. 2025-26. Defaults to all of them.",
    )
    parser.add_argument(
        "--raw-dir",
        default=footballdata.RAW_DIR,
        help="Where raw downloads live (default: data/raw/footballdata).",
    )
    parser.add_argument(
        "--output",
        default=footballdata.RESULTS_PARQUET,
        help="Where to write the parquet (default: data/processed/results.parquet).",
    )
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )

    start_years = (
        [footballdata.season_start_year(label) for label in arguments.seasons]
        if arguments.seasons
        else None
    )

    if not arguments.no_download:
        footballdata.download_all(arguments.raw_dir, start_years=start_years)

    results = footballdata.build_results(arguments.raw_dir, start_years=start_years)
    footballdata.write_results(results, arguments.output)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print()
        print(footballdata.season_summary(results).to_string())
        print()
        print(f"{len(results)} matches written to {arguments.output}")
        print("Closing odds used:")
        print(results["odds_source"].value_counts().to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
