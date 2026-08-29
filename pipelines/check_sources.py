"""Probe the two data sources that a restricted network can block.

Club Elo and FBref were both unreachable from the environment this project was
developed in - see data/lookups/NOTES.md. Neither failure looked like a fault in
the scrapers: Club Elo serves plain CSV and never answered at all, and FBref
returned a Cloudflare challenge to every kind of client. But "it is probably the
network" is a guess until someone tries from somewhere else.

This script is that test. It makes **a handful of requests** - one club from
Club Elo, one season's schedule from FBref - and reports plainly whether each
one worked. It is meant to be run from a GitHub Actions runner, which sits on a
completely different network, but it works just as well from a laptop:

    python -m pipelines.check_sources --output-dir /tmp/source-check

While it is at it, each probe checks the thing that could not be checked
offline: whether the club names in data/lookups/team_names.csv actually match
what the source sends. Those two columns (``clubelo_name`` and ``fbref_name``)
are the last unverified part of the lookup, and a single successful request is
enough to settle them.

Nothing here writes to the repository. Results go to ``--output-dir`` as files
to be uploaded as artifacts, plus a Markdown summary for the job page.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The one club we ask Club Elo for. Arsenal have been in the Premier League for
#: the whole period we care about, so their history is a good smoke test.
PROBE_CLUB = "Arsenal"

#: The one season we ask FBref for. A completed season, so the schedule is full.
PROBE_SEASON_START_YEAR = 2024

#: The date used for the get_elo spot check, and a plausible range for the
#: answer. Arsenal were top of the league in January 2023, so a correct lookup
#: should land near the upper end of a Premier League club's range.
PROBE_ELO_DATE = "2023-01-01"
PLAUSIBLE_ELO = (1300.0, 2100.0)

PASS = "PASS"
FAIL = "FAIL"


def _result(source: str, status: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"source": source, "status": status, "detail": detail, **extra}


def _exception_detail(error: BaseException) -> str:
    """A one-line description, kept short enough for a summary table."""
    text = f"{type(error).__name__}: {error}"
    return " ".join(text.split())[:400]


# ---------------------------------------------------------------------------
# Club Elo
# ---------------------------------------------------------------------------


def check_clubelo(output_dir: Path, club: str = PROBE_CLUB) -> dict[str, Any]:
    """One request: fetch a single club's rating history and use it.

    Beyond "did it connect", this runs the whole path the pipeline depends on -
    parse the CSV, map the club name through the lookup, and answer a
    get_elo query - so a pass here means the module genuinely works, not just
    that the host was up.
    """
    from src.scrape import clubelo

    try:
        csv_text = clubelo.fetch_club_history(club)
    except Exception as error:
        return _result(
            "clubelo", FAIL,
            f"Could not fetch {club} from api.clubelo.com. {_exception_detail(error)}",
            requests_made=1,
        )

    raw_file = output_dir / f"clubelo_{club}.csv"
    raw_file.write_text(csv_text, encoding="utf-8")

    try:
        history = clubelo.parse_club_history(csv_text)
    except Exception as error:
        return _result(
            "clubelo", FAIL,
            f"Downloaded {len(csv_text)} characters but could not parse them. "
            f"{_exception_detail(error)}",
            requests_made=1, artifact=raw_file.name,
        )

    rating = clubelo.get_elo(club, PROBE_ELO_DATE, history)
    history.to_csv(output_dir / f"clubelo_{club}_parsed.csv", index=False)

    span = (
        f"{history['valid_from'].min().date()} to {history['valid_to'].max().date()}"
    )
    fetched = f"Fetched {len(history)} rating periods for {club} ({span})."
    lookup = f"get_elo({club!r}, {PROBE_ELO_DATE!r}) = " + (
        "None" if rating is None else f"{rating:.1f}"
    )

    if rating is None:
        return _result(
            "clubelo", FAIL,
            f"{fetched} {lookup} - the download worked but the lookup found no "
            "rating for that date, which suggests a parsing problem.",
            requests_made=1, artifact=raw_file.name,
        )

    low, high = PLAUSIBLE_ELO
    if not low <= rating <= high:
        return _result(
            "clubelo", FAIL,
            f"{fetched} {lookup}, which is outside the plausible range "
            f"{low}-{high} for a Premier League club.",
            requests_made=1, artifact=raw_file.name, elo=round(float(rating), 1),
        )

    return _result(
        "clubelo", PASS, f"{fetched} {lookup}, which is plausible.",
        requests_made=1, artifact=raw_file.name, elo=round(float(rating), 1),
        canonical_name=str(history["team"].iloc[0]),
    )


# ---------------------------------------------------------------------------
# FBref
# ---------------------------------------------------------------------------


def check_fbref(
    output_dir: Path, start_year: int = PROBE_SEASON_START_YEAR
) -> dict[str, Any]:
    """A few requests: fetch one season's schedule from behind Cloudflare.

    Also reports whether every club name FBref sends is in the lookup, which is
    the part that could never be checked offline. Unmapped names are listed
    rather than raised, because learning the real spellings is the whole point
    of getting a response at all.
    """
    from src.lookups import load_team_names
    from src.scrape import fbref

    browser = fbref.find_browser()
    if browser is None:
        return _result(
            "fbref", FAIL,
            "No Chrome or Chromium found. FBref needs a real browser for its "
            "Cloudflare challenge; install one or set PREMFORECASTER_BROWSER.",
            requests_made=0,
        )

    try:
        client = fbref.make_client(start_year, cache_dir=output_dir / "fbref_cache")
        fbref.check_politeness(client)
        schedule = client.read_schedule()
    except Exception as error:
        (output_dir / "fbref_traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        return _result(
            "fbref", FAIL,
            f"Could not fetch the {fbref.season_label(start_year)} schedule. "
            f"{_exception_detail(error)}",
            browser=str(browser), artifact="fbref_traceback.txt",
        )

    if schedule is None or schedule.empty:
        return _result(
            "fbref", FAIL,
            "FBref answered but returned an empty schedule.",
            browser=str(browser),
        )

    flat = schedule.reset_index()
    flat.to_csv(output_dir / f"fbref_schedule_{fbref.season_label(start_year)}.csv", index=False)

    names: set[str] = set()
    for column in ("home_team", "away_team"):
        if column in flat.columns:
            names |= {str(value).strip() for value in flat[column].dropna()}

    known = set(load_team_names()["fbref_name"])
    unmapped = sorted(names - known)
    unused = sorted(known - names) if names else []

    detail = (
        f"Fetched {len(flat)} fixtures for {fbref.season_label(start_year)} "
        f"using {Path(browser).name}. Saw {len(names)} club names."
    )
    if unmapped:
        return _result(
            "fbref", FAIL,
            detail + f" Names NOT in team_names.csv: {unmapped}. Add or correct "
            "these in the fbref_name column.",
            browser=str(browser), unmapped=unmapped, fixtures=len(flat),
        )

    return _result(
        "fbref", PASS,
        detail + " Every name maps to a canonical club, so the fbref_name column "
        "is verified for this season.",
        browser=str(browser), fixtures=len(flat), teams_seen=len(names),
        unused_lookup_names=len(unused),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_summary(results: list[dict[str, Any]], *, when: datetime | None = None) -> str:
    """A Markdown report for the GitHub Actions job summary page."""
    when = when or datetime.now(timezone.utc)
    passed = [r for r in results if r["status"] == PASS]

    lines = [
        "# Source reachability check",
        "",
        f"Run at {when:%Y-%m-%d %H:%M} UTC. "
        "Both of these sources were unreachable from the development environment; "
        "this checks whether they are reachable from here.",
        "",
        "| Source | Result | What happened |",
        "| --- | --- | --- |",
    ]
    for result in results:
        icon = "✅ PASS" if result["status"] == PASS else "❌ FAIL"
        detail = result["detail"].replace("|", "\\|")
        lines.append(f"| `{result['source']}` | {icon} | {detail} |")

    lines += ["", f"**{len(passed)} of {len(results)} source(s) reachable.**", ""]

    for result in results:
        if result["status"] != PASS:
            continue
        if result["source"] == "clubelo":
            lines.append(
                "### Club Elo works from here\n\n"
                "Run `python -m pipelines.build_context --only clubelo` on this "
                "network to build `elo_history.parquet` in full."
            )
        if result["source"] == "fbref":
            lines.append(
                "### FBref works from here\n\n"
                "Run `python -m pipelines.build_xg --skip-understat` on this "
                "network to build the FBref tables. Note it is slow: one request "
                "every seven seconds, by design."
            )

    failed = [r for r in results if r["status"] == FAIL]
    if failed:
        lines.append(
            "### Still blocked\n\n"
            + "\n".join(f"- `{r['source']}`: {r['detail']}" for r in failed)
            + "\n\nIf a source fails here too, it is worth checking whether the "
            "site has changed rather than assuming it is the network."
        )

    lines += [
        "",
        "---",
        "",
        "Full output, including any raw downloads, is attached to this run as an "
        "artifact.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="source-check",
        help="Where to write downloads and the report (uploaded as an artifact).",
    )
    parser.add_argument(
        "--only", nargs="+", choices=("clubelo", "fbref"),
        help="Probe only these sources.",
    )
    parser.add_argument(
        "--season", type=int, default=PROBE_SEASON_START_YEAR,
        help="Season start year for the FBref probe (default 2024, i.e. 2024/25).",
    )
    parser.add_argument(
        "--fail-on-blocked", action="store_true",
        help=(
            "Exit non-zero if a source is unreachable. Off by default: a blocked "
            "source is the expected finding, not a broken job."
        ),
    )
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chosen = arguments.only or ("clubelo", "fbref")
    results: list[dict[str, Any]] = []

    for source in chosen:
        logger.info("--- probing %s ---", source)
        try:
            if source == "clubelo":
                results.append(check_clubelo(output_dir))
            else:
                results.append(check_fbref(output_dir, arguments.season))
        except Exception as error:  # a probe must never take the job down
            logger.exception("Probe for %s raised unexpectedly", source)
            (output_dir / f"{source}_unexpected.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            results.append(
                _result(source, FAIL, f"The probe itself failed: {_exception_detail(error)}")
            )

    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    summary = render_summary(results)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(summary + "\n")

    print()
    print(summary)

    blocked = [r for r in results if r["status"] == FAIL]
    if blocked and arguments.fail_on_blocked:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
