"""Tests for the Fantasy Premier League scraper.

No network: these build bootstrap-static-shaped payloads by hand. The important
behaviour is the append-only snapshot history, because the API can only ever
tell us about today - who was injured last October is unknowable, so a snapshot
we fail to keep is gone for good.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest
import requests

from src.lookups import UnknownTeamError
from src.scrape import fpl


def bootstrap(players: list[dict] | None = None, teams: list[dict] | None = None) -> dict:
    """A minimal bootstrap-static payload in the shape the real API returns."""
    return {
        "teams": teams
        or [
            {"id": 1, "name": "Arsenal", "short_name": "ARS"},
            {"id": 12, "name": "Spurs", "short_name": "TOT"},
            {"id": 13, "name": "Man Utd", "short_name": "MUN"},
        ],
        "elements": players or [player()],
    }


def player(**overrides) -> dict:
    base = {
        "id": 1, "team": 1, "element_type": 4,
        "first_name": "Bukayo", "second_name": "Saka", "web_name": "Saka",
        "status": "a", "minutes": 2700, "news": "",
        "chance_of_playing_next_round": None, "chance_of_playing_this_round": None,
        "starts": 30, "form": "6.2", "points_per_game": "5.8", "total_points": 180,
        "goals_scored": 12, "assists": 10, "expected_goals": "10.4",
        "expected_assists": "8.1", "selected_by_percent": "35.2", "now_cost": 100,
    }
    return base | overrides


TAKEN_AT = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Team mapping
# ---------------------------------------------------------------------------


def test_team_ids_map_to_canonical_names():
    """FPL writes "Spurs" and "Man Utd"; the model works in full names."""
    mapping = fpl.team_id_map(bootstrap())
    assert mapping[1] == "Arsenal"
    assert mapping[12] == "Tottenham Hotspur"
    assert mapping[13] == "Manchester United"


def test_mapping_is_on_the_numeric_id_not_the_display_name():
    """Ids are stable and unambiguous; display names are what change."""
    mapping = fpl.team_id_map(bootstrap())
    assert set(mapping) == {1, 12, 13}


def test_an_unknown_fpl_team_stops_the_build():
    payload = bootstrap(teams=[{"id": 1, "name": "Real Madrid", "short_name": "RMA"}])
    with pytest.raises(UnknownTeamError, match="Real Madrid"):
        fpl.team_id_map(payload)


def test_missing_team_columns_are_reported_clearly():
    with pytest.raises(fpl.FPLFormatError, match="name"):
        fpl.team_id_map({"teams": [{"id": 1}], "elements": []})


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------


def test_players_are_parsed_with_canonical_teams_and_positions():
    tidy = fpl.parse_players(bootstrap(), TAKEN_AT)
    row = tidy.iloc[0]
    assert row["team"] == "Arsenal"
    assert row["position"] == "Forward"
    assert row["full_name"] == "Bukayo Saka"
    assert row["web_name"] == "Saka"


def test_availability_flags_are_translated_into_english():
    """"i" means nothing to a human reading the sheet; "injured" does."""
    players = [
        player(id=1, status="a"),
        player(id=2, status="i", news="Hamstring injury"),
        player(id=3, status="d", chance_of_playing_next_round=75),
        player(id=4, status="s"),
        player(id=5, status="u"),
    ]
    tidy = fpl.parse_players(bootstrap(players), TAKEN_AT).sort_values("player_id")
    assert list(tidy["status_meaning"]) == [
        "available", "injured", "doubtful", "suspended", "unavailable",
    ]


def test_chance_of_playing_is_kept_as_a_number():
    players = [player(id=2, status="d", chance_of_playing_next_round=75)]
    tidy = fpl.parse_players(bootstrap(players), TAKEN_AT)
    assert tidy.loc[0, "chance_of_playing_next_round"] == 75


def test_an_empty_news_string_becomes_missing_not_blank():
    tidy = fpl.parse_players(bootstrap(), TAKEN_AT)
    assert pd.isna(tidy.loc[0, "news"])


def test_the_snapshot_is_stamped_with_the_date_it_was_taken():
    tidy = fpl.parse_players(bootstrap(), TAKEN_AT)
    assert tidy.loc[0, "snapshot_date"] == pd.Timestamp("2026-08-29", tz="UTC")


def test_an_unfamiliar_status_flag_is_kept_rather_than_dropped(caplog):
    """A new flag should warn, not lose the player."""
    tidy = fpl.parse_players(bootstrap([player(status="x")]), TAKEN_AT)
    assert tidy.loc[0, "status"] == "x"
    assert pd.isna(tidy.loc[0, "status_meaning"])


def test_a_player_on_an_unknown_team_id_stops_the_build():
    with pytest.raises(fpl.FPLFormatError, match="unknown FPL team id"):
        fpl.parse_players(bootstrap([player(team=99)]), TAKEN_AT)


def test_missing_player_columns_are_reported_clearly():
    with pytest.raises(fpl.FPLFormatError, match="missing the column"):
        fpl.parse_players({"teams": bootstrap()["teams"], "elements": [{"id": 1}]}, TAKEN_AT)


# ---------------------------------------------------------------------------
# Append-only history
# ---------------------------------------------------------------------------


def test_snapshots_accumulate_rather_than_overwrite(tmp_path):
    """The whole point: the API cannot tell us about the past, so we keep it."""
    path = tmp_path / "fpl_players.parquet"

    monday = fpl.parse_players(bootstrap(), datetime(2026, 8, 24, tzinfo=timezone.utc))
    fpl.append_snapshot(monday, path)

    friday = fpl.parse_players(
        bootstrap([player(status="i")]), datetime(2026, 8, 28, tzinfo=timezone.utc)
    )
    combined = fpl.append_snapshot(friday, path)

    assert len(combined) == 2
    assert combined["snapshot_date"].nunique() == 2
    # The history now shows the player becoming injured, which is the signal.
    assert list(combined.sort_values("snapshot_date")["status_meaning"]) == [
        "available", "injured",
    ]


def test_rerunning_on_the_same_day_replaces_rather_than_duplicates(tmp_path):
    path = tmp_path / "fpl_players.parquet"
    snapshot = fpl.parse_players(bootstrap(), TAKEN_AT)

    fpl.append_snapshot(snapshot, path)
    combined = fpl.append_snapshot(snapshot, path)

    assert len(combined) == 1, "running twice in a day must not double the rows"


def test_the_history_survives_a_reload(tmp_path):
    path = tmp_path / "fpl_players.parquet"
    fpl.append_snapshot(fpl.parse_players(bootstrap(), TAKEN_AT), path)
    assert len(pd.read_parquet(path)) == 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def fixture(**overrides) -> dict:
    base = {
        "id": 1, "event": 1, "team_h": 1, "team_a": 12,
        "kickoff_time": "2026-08-21T19:00:00Z", "finished": True,
        "team_h_score": 3, "team_a_score": 0,
        "team_h_difficulty": 2, "team_a_difficulty": 4,
    }
    return base | overrides


def test_fixtures_are_parsed_with_canonical_teams_and_utc_kickoffs():
    tidy = fpl.parse_fixtures([fixture()], bootstrap())
    row = tidy.iloc[0]
    assert row["home_team"] == "Arsenal"
    assert row["away_team"] == "Tottenham Hotspur"
    assert row["kickoff_utc"] == pd.Timestamp("2026-08-21 19:00", tz="UTC")
    assert row["finished"]


def test_an_unscheduled_fixture_keeps_a_blank_kickoff():
    """Later gameweeks have no kick-off time until the TV picks are made."""
    tidy = fpl.parse_fixtures([fixture(kickoff_time=None, finished=False)], bootstrap())
    assert pd.isna(tidy.loc[0, "kickoff_utc"])


def test_a_team_playing_itself_stops_the_build():
    with pytest.raises(fpl.FPLFormatError, match="playing itself"):
        fpl.parse_fixtures([fixture(team_a=1)], bootstrap())


def test_an_unknown_team_in_a_fixture_stops_the_build():
    with pytest.raises(fpl.FPLFormatError, match="unknown FPL team id"):
        fpl.parse_fixtures([fixture(team_a=99)], bootstrap())


def test_no_fixtures_at_all_is_an_error():
    with pytest.raises(fpl.FPLFormatError, match="no fixtures"):
        fpl.parse_fixtures([], bootstrap())


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_a_suspiciously_small_snapshot_is_refused(tmp_path, monkeypatch):
    """A partial response must not be recorded as the day's truth."""
    monkeypatch.setattr(fpl, "fetch_bootstrap", lambda **kw: bootstrap())
    monkeypatch.setattr(fpl, "fetch_fixtures", lambda **kw: [fixture()])

    with pytest.raises(fpl.FPLFormatError, match="expected at least"):
        fpl.build_all(
            raw_dir=tmp_path,
            players_path=tmp_path / "p.parquet",
            fixtures_path=tmp_path / "f.parquet",
        )


def test_an_unreachable_api_gives_an_explanatory_error(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(requests.Session, "get", refuse)
    with pytest.raises(fpl.FPLUnavailableError, match="Could not reach"):
        fpl.fetch_bootstrap(save_to=None)


def test_a_response_missing_the_expected_keys_is_rejected(monkeypatch):
    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"unexpected": True}

    monkeypatch.setattr(requests.Session, "get", lambda *a, **k: Response())
    with pytest.raises(fpl.FPLFormatError, match="elements"):
        fpl.fetch_bootstrap(save_to=None)


def test_raw_snapshots_are_date_stamped(tmp_path):
    path = fpl.raw_path("bootstrap-static", TAKEN_AT, tmp_path)
    assert path.name == "bootstrap-static_2026-08-29.json"


def test_availability_summary_counts_each_state():
    players = [player(id=1, status="a"), player(id=2, status="i"), player(id=3, status="i")]
    tidy = fpl.parse_players(bootstrap(players), TAKEN_AT)
    summary = fpl.availability_summary(tidy)
    assert summary.loc[pd.Timestamp("2026-08-29", tz="UTC"), "injured"] == 2
