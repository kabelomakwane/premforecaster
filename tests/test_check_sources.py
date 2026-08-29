"""Tests for the source reachability probe.

The probe exists to be run somewhere the developer cannot reach - a GitHub
runner - so its **success** path is the one nobody here can exercise for real.
These tests stub the network so both outcomes are covered, because shipping a
diagnostic where only the failure path has ever run would be a poor diagnostic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from pipelines import check_sources as cs
from src.scrape import clubelo, fbref


def full_table(**overrides) -> str:
    """Every club we track, in the shape the dated endpoint returns.

    The probe now asks for one date rather than one club, so a stub has to
    contain the whole lookup - which is exactly what lets the probe verify the
    clubelo_name column.
    """
    from src.scrape.clubelo import clubelo_names

    ratings = {name: 1500.0 for name in clubelo_names()}
    ratings["Arsenal"] = 1834.5
    ratings.update(overrides)

    rows = ["Rank,Club,Country,Level,Elo,From,To"]
    for rank, (club, elo) in enumerate(sorted(ratings.items()), start=1):
        rows.append(f"{rank},{club},ENG,1,{elo},2026-08-20,2026-09-01")
    # A club outside our lookup, as the real response is full of.
    rows.append("999,Real Madrid,ESP,1,1975.0,2026-08-20,2026-09-01")
    return "\n".join(rows) + "\n"


def schedule_frame(home: list[str], away: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"home_team": home, "away_team": away, "score": ["1–0"] * len(home)})


class FakeFBrefClient:
    def __init__(self, schedule: pd.DataFrame):
        self._schedule = schedule
        self.rate_limit = fbref.REQUEST_INTERVAL_SECONDS
        self.max_delay = 1.0

    def read_schedule(self):
        return self._schedule


# ---------------------------------------------------------------------------
# Club Elo probe
# ---------------------------------------------------------------------------


def test_clubelo_passes_when_the_api_answers(tmp_path, monkeypatch):
    """The path a working network takes - never exercisable where it was written."""
    monkeypatch.setattr(clubelo, "fetch_table_on", lambda day, **kw: full_table())

    result = cs.check_clubelo(tmp_path)

    assert result["status"] == cs.PASS
    assert result["elo"] == pytest.approx(1834.5)
    # A pass means every clubelo_name in the lookup was found in the table,
    # which is what verifies that column.
    assert result["clubs_matched"] == 36
    assert "verified" in result["detail"]


def test_a_passing_clubelo_probe_saves_the_raw_download(tmp_path, monkeypatch):
    monkeypatch.setattr(clubelo, "fetch_table_on", lambda day, **kw: full_table())
    cs.check_clubelo(tmp_path)
    saved = list(tmp_path.glob("clubelo_table_*.csv"))
    assert saved, "the raw download should be kept as an artifact"
    assert (tmp_path / "clubelo_table_parsed.csv").exists()


def test_clubelo_fails_when_it_cannot_connect(tmp_path, monkeypatch):
    def blocked(day, **kwargs):
        raise clubelo.ClubEloUnavailableError("connection refused")

    monkeypatch.setattr(clubelo, "fetch_table_on", blocked)

    result = cs.check_clubelo(tmp_path)
    assert result["status"] == cs.FAIL
    assert "connection refused" in result["detail"]


def test_clubelo_fails_when_the_response_will_not_parse(tmp_path, monkeypatch):
    """Reached the host but got something unexpected - a different problem."""
    monkeypatch.setattr(clubelo, "fetch_table_on", lambda day, **kw: "Elo\nnonsense\n")

    result = cs.check_clubelo(tmp_path)
    assert result["status"] == cs.FAIL
    assert "could not parse" in result["detail"]


def test_clubelo_fails_on_an_implausible_rating(tmp_path, monkeypatch):
    """A number in the wrong units would otherwise look like a pass."""
    monkeypatch.setattr(
        clubelo, "fetch_table_on", lambda day, **kw: full_table(Arsenal=12.3)
    )
    result = cs.check_clubelo(tmp_path)
    assert result["status"] == cs.FAIL
    assert "outside the plausible range" in result["detail"]


def test_the_clubelo_probe_makes_exactly_one_request(tmp_path, monkeypatch):
    calls = []

    def counted(day, **kwargs):
        calls.append(day)
        return full_table()

    monkeypatch.setattr(clubelo, "fetch_table_on", counted)
    cs.check_clubelo(tmp_path)
    assert len(calls) == 1, "one dated request covers every club"


# ---------------------------------------------------------------------------
# FBref probe
# ---------------------------------------------------------------------------


def test_fbref_fails_clearly_when_there_is_no_browser(tmp_path, monkeypatch):
    monkeypatch.setattr(fbref, "find_browser", lambda explicit=None: None)
    result = cs.check_fbref(tmp_path)
    assert result["status"] == cs.FAIL
    assert "Cloudflare" in result["detail"]
    assert result["requests_made"] == 0


def test_fbref_passes_and_verifies_the_lookup_column(tmp_path, monkeypatch):
    """A pass settles the fbref_name column, which could never be checked offline."""
    monkeypatch.setattr(fbref, "find_browser", lambda explicit=None: tmp_path / "chrome")
    monkeypatch.setattr(
        fbref, "make_client",
        lambda *a, **k: FakeFBrefClient(
            schedule_frame(["Manchester Utd", "Nott'ham Forest"], ["Fulham", "Wolves"])
        ),
    )

    result = cs.check_fbref(tmp_path)

    assert result["status"] == cs.PASS
    assert result["fixtures"] == 2
    assert "verified" in result["detail"]


def test_fbref_reports_names_that_are_not_in_the_lookup(tmp_path, monkeypatch):
    """The point of a response: learning the spellings we guessed wrong."""
    monkeypatch.setattr(fbref, "find_browser", lambda explicit=None: tmp_path / "chrome")
    monkeypatch.setattr(
        fbref, "make_client",
        lambda *a, **k: FakeFBrefClient(
            schedule_frame(["Manchester United", "Forest FC"], ["Fulham", "Wolves"])
        ),
    )

    result = cs.check_fbref(tmp_path)

    assert result["status"] == cs.FAIL
    assert result["unmapped"] == ["Forest FC", "Manchester United"]
    assert "fbref_name" in result["detail"]


def test_a_passing_fbref_probe_saves_the_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(fbref, "find_browser", lambda explicit=None: tmp_path / "chrome")
    monkeypatch.setattr(
        fbref, "make_client",
        lambda *a, **k: FakeFBrefClient(schedule_frame(["Manchester Utd"], ["Fulham"])),
    )
    cs.check_fbref(tmp_path, 2024)
    assert (tmp_path / "fbref_schedule_2024-25.csv").exists()


def test_fbref_fails_and_saves_a_traceback_when_cloudflare_blocks_it(tmp_path, monkeypatch):
    monkeypatch.setattr(fbref, "find_browser", lambda explicit=None: tmp_path / "chrome")

    def blocked(*args, **kwargs):
        raise fbref.FBrefUnavailableError("403 from Cloudflare")

    monkeypatch.setattr(fbref, "make_client", blocked)

    result = cs.check_fbref(tmp_path)
    assert result["status"] == cs.FAIL
    assert "403" in result["detail"]
    assert (tmp_path / "fbref_traceback.txt").exists()


def test_fbref_fails_on_an_empty_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(fbref, "find_browser", lambda explicit=None: tmp_path / "chrome")
    monkeypatch.setattr(
        fbref, "make_client", lambda *a, **k: FakeFBrefClient(pd.DataFrame())
    )
    result = cs.check_fbref(tmp_path)
    assert result["status"] == cs.FAIL
    assert "empty schedule" in result["detail"]


def test_the_fbref_probe_refuses_an_unthrottled_client(tmp_path, monkeypatch):
    """The seven-second limit applies to a diagnostic as much as to a real run."""
    monkeypatch.setattr(fbref, "find_browser", lambda explicit=None: tmp_path / "chrome")

    class Unthrottled(FakeFBrefClient):
        def __init__(self):
            super().__init__(schedule_frame(["Manchester Utd"], ["Fulham"]))
            self.rate_limit = 0

    monkeypatch.setattr(fbref, "make_client", lambda *a, **k: Unthrottled())

    result = cs.check_fbref(tmp_path)
    assert result["status"] == cs.FAIL
    assert "Refusing to scrape" in result["detail"]


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------


WHEN = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)


def test_the_summary_shows_a_row_per_source():
    summary = cs.render_summary(
        [
            cs._result("clubelo", cs.PASS, "worked"),
            cs._result("fbref", cs.FAIL, "blocked"),
        ],
        when=WHEN,
    )
    assert "`clubelo` | ✅ PASS" in summary
    assert "`fbref` | ❌ FAIL" in summary
    assert "**1 of 2 source(s) reachable.**" in summary


def test_the_summary_says_what_to_run_next_when_a_source_works():
    summary = cs.render_summary([cs._result("clubelo", cs.PASS, "worked")], when=WHEN)
    assert "build_context --only clubelo" in summary


def test_the_summary_lists_what_is_still_blocked():
    summary = cs.render_summary([cs._result("fbref", cs.FAIL, "Cloudflare 403")], when=WHEN)
    assert "Still blocked" in summary
    assert "Cloudflare 403" in summary


def test_a_pipe_in_a_message_does_not_break_the_table():
    """Error text containing a pipe would otherwise split the Markdown row.

    Tracebacks and URLs contain pipes often enough that this matters: an
    unescaped one turns a three-column row into a four-column one and the
    summary renders as nonsense.
    """
    import re

    summary = cs.render_summary([cs._result("fbref", cs.FAIL, "a | b")], when=WHEN)
    table_row = [line for line in summary.splitlines() if line.startswith("| `fbref`")][0]

    assert r"a \| b" in table_row, "the pipe from the message must be escaped"
    # Count only pipes that are not escaped: those are the real cell delimiters.
    delimiters = len(re.findall(r"(?<!\\)\|", table_row))
    assert delimiters == 4, f"expected 3 cells, got row: {table_row}"


def test_an_all_clear_summary_reads_as_such():
    summary = cs.render_summary(
        [cs._result("clubelo", cs.PASS, "ok"), cs._result("fbref", cs.PASS, "ok")], when=WHEN
    )
    assert "**2 of 2 source(s) reachable.**" in summary
    assert "Still blocked" not in summary


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_main_writes_the_artifacts_and_succeeds_even_when_blocked(tmp_path, monkeypatch):
    """A blocked source is the finding, not a broken job - so exit 0."""
    def blocked(day, **kwargs):
        raise clubelo.ClubEloUnavailableError("refused")

    monkeypatch.setattr(clubelo, "fetch_table_on", blocked)

    code = cs.main(["--output-dir", str(tmp_path), "--only", "clubelo"])

    assert code == 0
    results = json.loads((tmp_path / "results.json").read_text())
    assert results[0]["status"] == cs.FAIL
    assert (tmp_path / "summary.md").exists()


def test_fail_on_blocked_makes_it_exit_non_zero(tmp_path, monkeypatch):
    def blocked(day, **kwargs):
        raise clubelo.ClubEloUnavailableError("refused")

    monkeypatch.setattr(clubelo, "fetch_table_on", blocked)

    code = cs.main(["--output-dir", str(tmp_path), "--only", "clubelo", "--fail-on-blocked"])
    assert code == 1


def test_main_succeeds_when_a_source_works(tmp_path, monkeypatch):
    monkeypatch.setattr(clubelo, "fetch_table_on", lambda day, **kw: full_table())
    code = cs.main(["--output-dir", str(tmp_path), "--only", "clubelo", "--fail-on-blocked"])
    assert code == 0


def test_the_summary_is_appended_to_the_github_job_page(tmp_path, monkeypatch):
    """This is how the PASS/FAIL actually reaches the person who triggered it."""
    monkeypatch.setattr(clubelo, "fetch_table_on", lambda day, **kw: full_table())
    summary_file = tmp_path / "step-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))

    cs.main(["--output-dir", str(tmp_path), "--only", "clubelo"])

    assert "Source reachability check" in summary_file.read_text()


def test_a_probe_that_crashes_unexpectedly_does_not_take_the_job_down(tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise MemoryError("something very unexpected")

    monkeypatch.setattr(cs, "check_clubelo", explode)

    code = cs.main(["--output-dir", str(tmp_path), "--only", "clubelo"])

    assert code == 0
    results = json.loads((tmp_path / "results.json").read_text())
    assert "The probe itself failed" in results[0]["detail"]
