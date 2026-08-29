"""Tests for the Understat parsing logic.

No network: these build small frames in the shape ``soccerdata`` returns and
check the reshaping, the team name mapping and the defensive checks. The tests
against the real downloaded parquets live in test_xg_tables.py.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.lookups import UnknownTeamError
from src.scrape import understat as us


def raw_team_match(rows: list[dict]) -> pd.DataFrame:
    """Mimic soccerdata's read_team_match_stats output (a MultiIndex frame)."""
    frame = pd.DataFrame(rows)
    frame.index = pd.MultiIndex.from_arrays(
        [
            ["ENG-Premier League"] * len(frame),
            ["2024"] * len(frame),
            [f"g{i}" for i in range(len(frame))],
        ],
        names=["league", "season", "game"],
    )
    return frame


TEAM_MATCH_ROW = {
    "game_id": "26100",
    "date": "2024-08-16 20:00:00",
    "home_team": "Manchester United",
    "away_team": "Fulham",
    "home_goals": 1, "away_goals": 0,
    "home_xg": 1.42, "away_xg": 0.91,
    "home_np_xg": 1.42, "away_np_xg": 0.91,
    "home_ppda": 9.5, "away_ppda": 12.1,
    "home_deep_completions": 6, "away_deep_completions": 3,
    "home_points": 3, "away_points": 0,
    "home_expected_points": 1.9, "away_expected_points": 0.7,
}


# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------


def test_understat_horizon_starts_at_2014_15():
    years = us.available_start_years(date(2026, 8, 29))
    assert years[0] == 2014
    assert years[-1] == 2026
    assert len(years) == 13


def test_shot_seasons_are_the_most_recent_few():
    years = us.shot_start_years(date(2026, 8, 29), count=4)
    assert years == [2023, 2024, 2025, 2026]


def test_seasons_are_requested_in_the_unambiguous_form():
    """Regression test for a bug that silently returned the wrong season.

    soccerdata reads a bare ``"2021"`` as the season 20/21, not the year 2021,
    so asking for "2021" returned all 380 matches of 2020/21 - correct-looking
    data under the wrong label, with nothing but a warning. Only the four-plus-
    four form is safe.
    """
    assert us._soccerdata_season(2021) == "2021-2022"
    assert us._soccerdata_season(2014) == "2014-2015"
    for start_year in us.available_start_years(date(2026, 8, 29)):
        assert us._soccerdata_season(start_year) != str(start_year)


def test_the_wrong_season_coming_back_is_caught():
    """The second line of defence: check what arrived is what we asked for."""
    raw = raw_team_match([TEAM_MATCH_ROW])
    # soccerdata labels 2020/21 as "2021"; we asked for 2021/22, i.e. "2122".
    raw.index = pd.MultiIndex.from_arrays(
        [["ENG-Premier League"], ["2021"], ["g0"]], names=["league", "season", "game"]
    )
    with pytest.raises(us.UnderstatFormatError, match="returned season"):
        us.check_season_matches(raw, 2021, "team match stats")


def test_the_right_season_passes_the_check():
    raw = raw_team_match([TEAM_MATCH_ROW])
    raw.index = pd.MultiIndex.from_arrays(
        [["ENG-Premier League"], ["2122"], ["g0"]], names=["league", "season", "game"]
    )
    us.check_season_matches(raw, 2021, "team match stats")  # must not raise


# ---------------------------------------------------------------------------
# Scraping politeness (CLAUDE.md calls these non-negotiable)
# ---------------------------------------------------------------------------


class FakeClient:
    """Stands in for a soccerdata reader, with the attributes its loop reads."""

    def __init__(self, rate_limit=0, max_delay=0):
        self.rate_limit = rate_limit
        self.max_delay = max_delay


def test_the_understat_interval_is_at_least_six_seconds():
    assert us.REQUEST_INTERVAL_SECONDS >= 6.0


def test_requests_have_jitter_so_they_are_not_perfectly_periodic():
    assert us.REQUEST_JITTER_SECONDS > 0


def test_the_user_agent_says_who_we_are():
    assert "premforecaster" in us.USER_AGENT.lower()
    assert "non-commercial" in us.USER_AGENT.lower()


def test_an_unthrottled_client_is_refused():
    """soccerdata ships Understat with rate_limit = 0, which we must not accept."""
    with pytest.raises(RuntimeError, match="Refusing to scrape"):
        us.check_politeness(FakeClient(rate_limit=0))


def test_a_too_fast_client_is_refused():
    with pytest.raises(RuntimeError, match="below the"):
        us.check_politeness(FakeClient(rate_limit=1.0))


def test_a_properly_throttled_client_is_accepted():
    client = FakeClient(rate_limit=us.REQUEST_INTERVAL_SECONDS)
    client._session = SpySession()
    us.throttle(client, rate_limit=us.REQUEST_INTERVAL_SECONDS, jitter=0.0)
    us.check_politeness(client)  # must not raise


def test_a_client_without_a_throttle_is_refused():
    """Setting rate_limit alone is not enough, and must not look like enough.

    Understat's reader fetches through its own method that never consults
    rate_limit, so a client with the attribute set but no throttle installed
    would still hammer the site.
    """
    client = FakeClient(rate_limit=us.REQUEST_INTERVAL_SECONDS)
    with pytest.raises(RuntimeError, match="no throttle"):
        us.check_politeness(client)


class SpySession:
    """Records when requests happen, so we can measure the gaps between them."""

    def __init__(self):
        self.calls: list[float] = []
        self.headers: dict[str, str] = {}

    def get(self, *args, **kwargs):
        import time

        self.calls.append(time.monotonic())
        return "response"


def test_the_throttle_actually_delays_real_requests():
    """Measured, not assumed - this is the check that caught the real bug."""
    import time

    client = FakeClient()
    client._session = SpySession()
    client.no_cache = False

    us.throttle(client, rate_limit=0.25, jitter=0.0)

    started = time.monotonic()
    for _ in range(3):
        client._session.get("https://understat.com/anything")
    elapsed = time.monotonic() - started

    gaps = [b - a for a, b in zip(client._session.calls, client._session.calls[1:])]
    assert all(gap >= 0.25 for gap in gaps), gaps
    assert elapsed >= 0.5


def test_the_throttle_covers_every_request_not_just_the_data_ones():
    """Understat primes cookies with a separate call that must also be throttled."""
    client = FakeClient()
    client._session = SpySession()
    us.throttle(client, rate_limit=0.2, jitter=0.0)

    # Whatever calls the session - cookie priming included - is delayed.
    client._session.get("https://understat.com/")
    client._session.get("https://understat.com/main/getPlayersStats/")
    gaps = [b - a for a, b in zip(client._session.calls, client._session.calls[1:])]
    assert all(gap >= 0.2 for gap in gaps), gaps


def test_make_client_throttles_and_requests_the_right_season(monkeypatch):
    """The whole point: soccerdata's defaults must be overridden, not trusted."""
    created = FakeClient()
    created._session = SpySession()
    created.no_cache = False

    class FakeUnderstatModule:
        UNDERSTAT_HEADERS: dict[str, str] = {}

    class FakeModule:
        understat = FakeUnderstatModule

        @staticmethod
        def Understat(**kwargs):
            created.kwargs = kwargs
            return created

    monkeypatch.setitem(__import__("sys").modules, "soccerdata", FakeModule)

    client = us.make_client(2024, cache_dir="/tmp/premforecaster-test-cache")
    assert client.rate_limit >= 6.0
    assert client.max_delay > 0
    assert client._premforecaster_throttled
    us.check_politeness(client)  # must not raise

    assert "premforecaster" in FakeUnderstatModule.UNDERSTAT_HEADERS["User-Agent"].lower()
    # And the season must still be requested unambiguously.
    assert created.kwargs["seasons"] == "2024-2025"


# ---------------------------------------------------------------------------
# Team match stats
# ---------------------------------------------------------------------------


def test_each_match_becomes_two_rows_one_per_team():
    tidy = us.parse_team_match_stats(raw_team_match([TEAM_MATCH_ROW]), 2024)
    assert len(tidy) == 2
    assert set(tidy["team"]) == {"Manchester United", "Fulham"}
    assert list(tidy["is_home"]) == [True, False]


def test_for_and_against_are_swapped_for_the_away_row():
    """The away side's "for" must be the home side's "against"."""
    tidy = us.parse_team_match_stats(raw_team_match([TEAM_MATCH_ROW]), 2024)
    home = tidy[tidy["is_home"]].iloc[0]
    away = tidy[~tidy["is_home"]].iloc[0]

    assert (home["goals_for"], home["goals_against"]) == (1, 0)
    assert (away["goals_for"], away["goals_against"]) == (0, 1)
    assert home["xg_for"] == pytest.approx(1.42)
    assert away["xg_against"] == pytest.approx(1.42)
    assert home["xg_against"] == away["xg_for"]


def test_opponent_is_the_other_team():
    tidy = us.parse_team_match_stats(raw_team_match([TEAM_MATCH_ROW]), 2024)
    for _, row in tidy.iterrows():
        assert row["team"] != row["opponent"]
    assert set(tidy["opponent"]) == {"Manchester United", "Fulham"}


def test_the_fixture_is_kept_on_both_rows_so_it_still_joins():
    """Reconciliation joins on season + home_team + away_team."""
    tidy = us.parse_team_match_stats(raw_team_match([TEAM_MATCH_ROW]), 2024)
    assert set(tidy["home_team"]) == {"Manchester United"}
    assert set(tidy["away_team"]) == {"Fulham"}


def test_understat_names_become_canonical_names():
    """Understat writes "Wolverhampton Wanderers" and "Leicester"."""
    row = TEAM_MATCH_ROW | {"home_team": "Leicester", "away_team": "Wolverhampton Wanderers"}
    tidy = us.parse_team_match_stats(raw_team_match([row]), 2024)
    assert set(tidy["team"]) == {"Leicester City", "Wolverhampton Wanderers"}


def test_an_unknown_understat_team_stops_the_build():
    row = TEAM_MATCH_ROW | {"home_team": "Real Madrid"}
    with pytest.raises(UnknownTeamError, match="Real Madrid"):
        us.parse_team_match_stats(raw_team_match([row]), 2024)


def test_dates_come_out_as_utc():
    """Understat writes UK local time; 20:00 in August is 19:00 UTC."""
    tidy = us.parse_team_match_stats(raw_team_match([TEAM_MATCH_ROW]), 2024)
    assert str(tidy["date"].dt.tz) == "UTC"
    assert tidy["date"].iloc[0] == pd.Timestamp("2024-08-16 19:00", tz="UTC")


def test_the_season_label_matches_the_other_tables():
    tidy = us.parse_team_match_stats(raw_team_match([TEAM_MATCH_ROW]), 2024)
    assert set(tidy["season"]) == {"2024-25"}


def test_a_missing_column_is_reported_clearly():
    broken = raw_team_match([TEAM_MATCH_ROW]).drop(columns=["home_xg"])
    with pytest.raises(us.UnderstatFormatError, match="home_xg"):
        us.parse_team_match_stats(broken, 2024)


def test_an_unparseable_date_is_reported_clearly():
    row = TEAM_MATCH_ROW | {"date": "the second Tuesday"}
    with pytest.raises(us.UnderstatFormatError, match="date"):
        us.parse_team_match_stats(raw_team_match([row]), 2024)


# ---------------------------------------------------------------------------
# Player and shot tables
# ---------------------------------------------------------------------------


def test_player_season_stats_are_tidied_and_mapped():
    raw = pd.DataFrame(
        [{
            "league_id": 1, "season_id": 1, "team_id": 1, "player_id": "647",
            "position": "F S", "matches": 38, "minutes": 3100, "goals": 27,
            "xg": 24.1, "np_goals": 22, "np_xg": 19.4, "assists": 5, "xa": 6.2,
            "shots": 110, "key_passes": 40, "yellow_cards": 2, "red_cards": 0,
            "xg_chain": 30.1, "xg_buildup": 8.0,
        }]
    )
    raw.index = pd.MultiIndex.from_arrays(
        [["ENG-Premier League"], ["2024"], ["Manchester City"], ["Erling Haaland"]],
        names=["league", "season", "team", "player"],
    )
    tidy = us.parse_player_season_stats(raw, 2024)
    assert tidy.loc[0, "team"] == "Manchester City"
    assert tidy.loc[0, "player"] == "Erling Haaland"
    assert tidy.loc[0, "npxg"] == pytest.approx(19.4)
    assert tidy.loc[0, "season"] == "2024-25"


def test_shot_events_flag_goals():
    raw = pd.DataFrame(
        [
            {"league_id": 1, "season_id": 1, "game_id": "26100", "date": "2024-08-16 20:00:00",
             "shot_id": "1", "team_id": 1, "player_id": "1", "assist_player_id": None,
             "assist_player": None, "xg": 0.76, "location_x": 0.9, "location_y": 0.5,
             "minute": 37, "body_part": "RightFoot", "situation": "OpenPlay", "result": "Goal"},
            {"league_id": 1, "season_id": 1, "game_id": "26100", "date": "2024-08-16 20:00:00",
             "shot_id": "2", "team_id": 1, "player_id": "2", "assist_player_id": None,
             "assist_player": None, "xg": 0.04, "location_x": 0.7, "location_y": 0.3,
             "minute": 55, "body_part": "Head", "situation": "FromCorner", "result": "MissedShots"},
        ]
    )
    raw.index = pd.MultiIndex.from_arrays(
        [["ENG-Premier League"] * 2, ["2024"] * 2, ["26100"] * 2,
         ["Manchester United"] * 2, ["Bruno Fernandes", "Harry Maguire"]],
        names=["league", "season", "game", "team", "player"],
    )
    tidy = us.parse_shot_events(raw, 2024)
    assert list(tidy["is_goal"]) == [True, False]
    assert set(tidy["team"]) == {"Manchester United"}
    assert tidy["xg"].between(0, 1).all()


# ---------------------------------------------------------------------------
# Penalties
# ---------------------------------------------------------------------------


def shots_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["is_goal"] = frame["result"].str.lower().eq("goal")
    return frame


def test_blank_situations_are_relabelled_as_penalties():
    """soccerdata has no mapping for Understat's Penalty, so it arrives blank.

    Losing that label would hide the most predictable shot in football from the
    goalscorer model.
    """
    shots = shots_frame(
        [
            {"situation": "Open Play", "xg": 0.09, "result": "MissedShots"},
            {"situation": None, "xg": 0.76, "result": "Goal"},
            {"situation": None, "xg": 0.78, "result": "Goal"},
        ]
    )
    restored = us.restore_penalty_situations(shots, 2024)
    assert list(restored["situation"]) == ["Open Play", "Penalty", "Penalty"]


def test_parse_shot_events_sets_an_is_penalty_flag():
    raw = pd.DataFrame(
        [{
            "league_id": 1, "season_id": 1, "game_id": "26100",
            "date": "2024-08-16 20:00:00", "shot_id": "1", "team_id": 1,
            "player_id": "1", "assist_player_id": None, "assist_player": None,
            "xg": 0.7602, "location_x": 0.88, "location_y": 0.5, "minute": 37,
            "body_part": "RightFoot", "situation": None, "result": "Goal",
        }]
    )
    raw.index = pd.MultiIndex.from_arrays(
        [["ENG-Premier League"], ["2425"], ["26100"], ["Manchester City"], ["Erling Haaland"]],
        names=["league", "season", "game", "team", "player"],
    )
    tidy = us.parse_shot_events(raw, 2024)
    assert bool(tidy.loc[0, "is_penalty"])
    assert tidy.loc[0, "situation"] == "Penalty"


def test_ordinary_shots_are_never_relabelled_as_penalties():
    """If soccerdata stops mapping some other situation, we must not invent one."""
    shots = shots_frame(
        [
            {"situation": None, "xg": 0.04, "result": "MissedShots"},
            {"situation": None, "xg": 0.07, "result": "MissedShots"},
        ]
    )
    with pytest.raises(us.UnderstatFormatError, match="outside the"):
        us.restore_penalty_situations(shots, 2024)


def test_nothing_happens_when_no_situations_are_missing():
    shots = shots_frame([{"situation": "Open Play", "xg": 0.09, "result": "Goal"}])
    assert list(us.restore_penalty_situations(shots, 2024)["situation"]) == ["Open Play"]


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------


def test_collect_reuses_staged_seasons_instead_of_refetching(tmp_path, monkeypatch):
    """A finished season must be read from disk, never fetched again."""
    staging = tmp_path / "staged"
    checkpoint = tmp_path / "cp.json"

    calls: list[int] = []

    def fake_fetch(table, start_year, *, cache_dir, staging_dir):
        calls.append(start_year)
        frame = us.parse_team_match_stats(raw_team_match([TEAM_MATCH_ROW]), start_year)
        path = us.staged_path(table, start_year, staging_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return frame

    monkeypatch.setattr(us, "fetch_season", fake_fetch)

    first = us.collect("team_match", [2023, 2024], staging_dir=staging, checkpoint_path=checkpoint)
    assert calls == [2023, 2024]
    assert len(first) == 4

    second = us.collect("team_match", [2023, 2024], staging_dir=staging, checkpoint_path=checkpoint)
    assert calls == [2023, 2024], "a second run must not fetch anything"
    assert len(second) == 4


def test_a_failed_season_is_retried_next_run(tmp_path, monkeypatch):
    staging = tmp_path / "staged"
    checkpoint = tmp_path / "cp.json"
    attempts: list[int] = []

    def flaky_fetch(table, start_year, *, cache_dir, staging_dir):
        attempts.append(start_year)
        if len(attempts) == 1:
            raise us.UnderstatFormatError("network blip")
        frame = us.parse_team_match_stats(raw_team_match([TEAM_MATCH_ROW]), start_year)
        path = us.staged_path(table, start_year, staging_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return frame

    monkeypatch.setattr(us, "fetch_season", flaky_fetch)

    with pytest.raises(us.UnderstatFormatError):
        us.collect("team_match", [2024], staging_dir=staging, checkpoint_path=checkpoint)

    tidy = us.collect("team_match", [2024], staging_dir=staging, checkpoint_path=checkpoint)
    assert attempts == [2024, 2024]
    assert len(tidy) == 2
