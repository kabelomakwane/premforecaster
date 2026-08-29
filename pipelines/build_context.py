"""Build the context tables: Elo, FPL availability, weather and referee profiles.

These are the smaller sources that surround the core match and xG data. Run from
the repo root with the virtual environment active:

    python -m pipelines.build_context               # everything
    python -m pipelines.build_context --only fpl weather
    python -m pipelines.build_context --skip clubelo

Each source is independent, so one being unavailable does not stop the others -
the run reports what worked and what did not, and exits non-zero only if
everything failed.

A note on scheduling: the FPL step is the one that must run **often**. It takes
a snapshot of who is injured or doubtful right now, and that is the only way
that history ever gets recorded - the API cannot tell you who was injured last
month. Weather and referees can be rebuilt at any time from data already held.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.scrape import clubelo, fpl, referees, weather

SOURCES = ("elo", "fpl", "weather", "referees")


def run_elo(log: logging.Logger) -> str:
    """Build Elo ratings, and cross-check them against Club Elo where possible.

    Two steps, in this order deliberately:

    1. **Always** compute ratings from results.parquet. This needs no network
       and cannot fail for external reasons, which is why it is the default
       source the model reads.
    2. **Try** a handful of dated Club Elo snapshots. If they arrive they are
       stored separately and compared with ours, so we learn whether the two
       agree without ever depending on a slow third-party server.
    """
    from src.ratings import elo

    written = elo.build_all()
    history = pd.read_parquet(written["history"])
    per_match = pd.read_parquet(written["match"])
    outcome = (
        f"internal: {len(history)} rows, {history['team'].nunique()} clubs "
        f"(from {len(per_match)} matches)"
    )

    try:
        clubelo.download_snapshots()
        live = clubelo.build_history_from_snapshots()
    except (clubelo.ClubEloUnavailableError, clubelo.ClubEloFormatError, FileNotFoundError) as error:
        log.warning("Club Elo cross-check unavailable: %s", error)
        return outcome + "; Club Elo cross-check unavailable"

    clubelo.write_elo_history(live, clubelo.CLUBELO_HISTORY_PARQUET)

    comparison = elo.compare_with_clubelo(history, live)
    summary = elo.agreement_summary(comparison)
    if not comparison.empty:
        log.info("Internal vs Club Elo:\n%s", comparison.to_string(index=False))
    log.info("Agreement: %s", summary)

    return outcome + f"; Club Elo cross-check: {summary}"


def run_fpl(log: logging.Logger) -> str:
    written = fpl.build_all()
    players = pd.read_parquet(written["players"])
    fixtures = pd.read_parquet(written["fixtures"])
    return (
        f"{len(players)} player rows across {players['snapshot_date'].nunique()} "
        f"snapshot(s); {len(fixtures)} fixtures"
    )


def run_weather(log: logging.Logger) -> str:
    results_path = referees.RESULTS_PARQUET
    if not results_path.exists():
        return "skipped - results.parquet not built"

    matches = pd.read_parquet(results_path)
    built = weather.build_match_weather(matches)
    weather.write_match_weather(built)

    covered = int(built["temperature_c"].notna().sum())
    return f"{covered}/{len(built)} matches with weather ({100 * covered / len(built):.1f}%)"


def run_referees(log: logging.Logger) -> str:
    path = referees.build_all()
    profiles = pd.read_parquet(path)
    return f"{len(profiles)} rows, {profiles['referee'].nunique()} referees"


RUNNERS = {
    "elo": run_elo,
    "fpl": run_fpl,
    "weather": run_weather,
    "referees": run_referees,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=SOURCES, help="Run only these sources.")
    parser.add_argument("--skip", nargs="+", choices=SOURCES, default=[], help="Skip these.")
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    log = logging.getLogger("build_context")

    chosen = [
        source for source in (arguments.only or SOURCES) if source not in arguments.skip
    ]

    outcomes: dict[str, str] = {}
    failures: dict[str, str] = {}

    for source in chosen:
        log.info("--- %s ---", source)
        try:
            outcomes[source] = RUNNERS[source](log)
        except Exception as error:  # one broken source must not stop the rest
            log.warning("%s failed: %s: %s", source, type(error).__name__, error)
            failures[source] = f"{type(error).__name__}: {error}"

    print("\n=== Context tables ===")
    for source in chosen:
        if source in outcomes:
            print(f"  {source:<10} ok       {outcomes[source]}")
        else:
            print(f"  {source:<10} FAILED   {failures[source][:110]}")

    if failures and not outcomes:
        print("\nEverything failed.")
        return 1
    if failures:
        print(f"\n{len(failures)} of {len(chosen)} source(s) failed; the rest are built.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
