"""Tests for the FBref scraper's parsing and its behaviour when unavailable.

FBref sits behind Cloudflare and needs a real browser, which is not always
available - in a locked-down CI environment it often is not. The pipeline is
designed to carry on without it, so as well as the parsing these tests check
that an unavailable FBref produces a clear, catchable error rather than a crash.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.lookups import UnknownTeamError
from src.scrape import fbref as fb


def multi_column_frame() -> pd.DataFrame:
    """FBref returns two-level column headers, which we have to flatten."""
    columns = pd.MultiIndex.from_tuples(
        [
            ("Unnamed: 0_level_0", "date"),
            ("Unnamed: 1_level_0", "team"),
            ("Unnamed: 2_level_0", "opponent"),
            ("Shooting", "Sh"),
            ("Shooting", "SoT%"),
            ("Standard", "Standard"),
        ]
    )
    return pd.DataFrame(
        [["2024-08-16", "Manchester Utd", "Fulham", 14, 42.9, 1]], columns=columns
    )


# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------


def test_fbref_horizon_starts_at_2017_18():
    years = fb.available_start_years(date(2026, 8, 29))
    assert years[0] == 2017
    assert years[-1] == 2026


# ---------------------------------------------------------------------------
# Column flattening
# ---------------------------------------------------------------------------


def test_two_level_headers_are_flattened_to_readable_names():
    flat = fb._flatten_columns(multi_column_frame())
    assert "date" in flat.columns
    assert "shooting_sh" in flat.columns


def test_percent_signs_become_words_so_columns_are_usable():
    flat = fb._flatten_columns(multi_column_frame())
    assert "shooting_sotpct" in flat.columns
    assert not any("%" in column for column in flat.columns)


def test_a_repeated_header_level_is_not_doubled_up():
    """("Standard", "Standard") should become "standard", not "standard_standard"."""
    flat = fb._flatten_columns(multi_column_frame())
    assert "standard" in flat.columns


def test_flattening_leaves_a_flat_frame_alone():
    frame = pd.DataFrame({"Date": ["2024-08-16"], "Team": ["Arsenal"]})
    flat = fb._flatten_columns(frame)
    assert list(flat.columns) == ["date", "team"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_fbref_names_become_canonical_names():
    """FBref writes "Manchester Utd", not "Manchester United"."""
    tidy = fb.parse_team_match(multi_column_frame(), 2024, "shooting")
    assert tidy.loc[0, "team"] == "Manchester United"
    assert tidy.loc[0, "opponent"] == "Fulham"


def test_parsing_stamps_the_season_and_stat_group():
    tidy = fb.parse_team_match(multi_column_frame(), 2024, "shooting")
    assert tidy.loc[0, "season"] == "2024-25"
    assert tidy.loc[0, "stat_type"] == "shooting"


def test_dates_come_out_as_utc():
    tidy = fb.parse_team_match(multi_column_frame(), 2024, "shooting")
    assert str(tidy["date"].dt.tz) == "UTC"


def test_an_unknown_fbref_team_stops_the_build():
    frame = multi_column_frame()
    frame.iloc[0, 1] = "Real Madrid"
    with pytest.raises(UnknownTeamError, match="Real Madrid"):
        fb.parse_team_match(frame, 2024, "shooting")


def test_a_missing_team_column_is_reported_clearly():
    frame = pd.DataFrame({"date": ["2024-08-16"], "sh": [14]})
    with pytest.raises(fb.FBrefFormatError, match="team"):
        fb.parse_team_match(frame, 2024, "shooting")


def test_an_unparseable_date_is_reported_clearly():
    frame = pd.DataFrame({"date": ["not a date"], "team": ["Arsenal"]})
    with pytest.raises(fb.FBrefFormatError, match="date"):
        fb.parse_team_match(frame, 2024, "shooting")


# ---------------------------------------------------------------------------
# Merging stat groups
# ---------------------------------------------------------------------------


def test_stat_groups_are_joined_side_by_side_not_stacked():
    """Shooting and possession for the same match are one row, not two."""
    shooting = pd.DataFrame({
        "season": ["2024-25"], "team": ["Arsenal"],
        "date": [pd.Timestamp("2024-08-17", tz="UTC")],
        "shooting_sh": [14], "stat_type": ["shooting"],
    })
    possession = pd.DataFrame({
        "season": ["2024-25"], "team": ["Arsenal"],
        "date": [pd.Timestamp("2024-08-17", tz="UTC")],
        "poss": [61.2], "stat_type": ["possession"],
    })

    merged = fb.merge_stat_groups([shooting, possession], "team_match")
    assert len(merged) == 1
    assert merged.loc[0, "shooting_sh"] == 14
    assert merged.loc[0, "poss"] == pytest.approx(61.2)
    assert "stat_type" not in merged.columns


def test_columns_repeated_across_groups_are_kept_once():
    common = {
        "season": ["2024-25"], "team": ["Arsenal"],
        "date": [pd.Timestamp("2024-08-17", tz="UTC")], "venue": ["Home"],
    }
    first = pd.DataFrame(common | {"shooting_sh": [14]})
    second = pd.DataFrame(common | {"poss": [61.2]})

    merged = fb.merge_stat_groups([first, second], "team_match")
    assert list(merged.columns).count("venue") == 1
    assert not any(column.endswith("_dup") for column in merged.columns)


# ---------------------------------------------------------------------------
# Behaviour when FBref cannot be reached
# ---------------------------------------------------------------------------


def test_no_browser_gives_an_explanatory_error(monkeypatch):
    monkeypatch.setattr(fb, "find_browser", lambda explicit=None: None)
    with pytest.raises(fb.FBrefUnavailableError, match="Cloudflare"):
        fb.make_client(2024)


def test_find_browser_honours_an_explicit_path(tmp_path):
    browser = tmp_path / "chrome"
    browser.write_text("")
    assert fb.find_browser(browser) == browser


def test_find_browser_returns_none_for_a_path_that_does_not_exist(tmp_path):
    assert fb.find_browser(tmp_path / "nope") is None


def test_find_browser_reads_the_environment_variable(tmp_path, monkeypatch):
    browser = tmp_path / "chrome"
    browser.write_text("")
    monkeypatch.setenv("PREMFORECASTER_BROWSER", str(browser))
    assert fb.find_browser() == browser


def test_an_unreachable_season_is_skipped_rather_than_losing_the_run(tmp_path, monkeypatch):
    """A partial FBref table is much more useful than none at all."""
    def unavailable(*args, **kwargs):
        raise fb.FBrefUnavailableError("Cloudflare blocked the request")

    monkeypatch.setattr(fb, "fetch_stat_group", unavailable)

    frame = fb.collect(
        "team_match", [2024], ("shooting",),
        staging_dir=tmp_path / "staged", checkpoint_path=tmp_path / "cp.json",
        skip_failures=True,
    )
    assert frame.empty


def test_skip_failures_off_lets_the_error_through(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise fb.FBrefUnavailableError("Cloudflare blocked the request")

    monkeypatch.setattr(fb, "fetch_stat_group", unavailable)

    with pytest.raises(fb.FBrefUnavailableError):
        fb.collect(
            "team_match", [2024], ("shooting",),
            staging_dir=tmp_path / "staged", checkpoint_path=tmp_path / "cp.json",
            skip_failures=False,
        )


def test_a_failed_unit_is_not_checkpointed(tmp_path, monkeypatch):
    """It must be retried on the next run, not silently treated as done."""
    from src.checkpoint import Checkpoint

    monkeypatch.setattr(
        fb, "fetch_stat_group",
        lambda *a, **k: (_ for _ in ()).throw(fb.FBrefUnavailableError("blocked")),
    )
    checkpoint_path = tmp_path / "cp.json"
    fb.collect(
        "team_match", [2024], ("shooting",),
        staging_dir=tmp_path / "staged", checkpoint_path=checkpoint_path,
    )
    assert len(Checkpoint(checkpoint_path)) == 0


def test_staged_work_is_reused_without_refetching(tmp_path, monkeypatch):
    from src.checkpoint import Checkpoint

    staging = tmp_path / "staged"
    checkpoint_path = tmp_path / "cp.json"

    staged = fb.staged_path("team_match", 2024, "shooting", staging)
    staged.parent.mkdir(parents=True, exist_ok=True)
    fb.parse_team_match(multi_column_frame(), 2024, "shooting").to_parquet(staged, index=False)
    Checkpoint(checkpoint_path).mark_done("team_match/2024-25/shooting")

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("fetch was called for already-staged work")

    monkeypatch.setattr(fb, "fetch_stat_group", must_not_be_called)

    frame = fb.collect(
        "team_match", [2024], ("shooting",),
        staging_dir=staging, checkpoint_path=checkpoint_path,
    )
    assert len(frame) == 1
