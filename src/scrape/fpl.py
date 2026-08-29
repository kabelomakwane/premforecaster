"""Fantasy Premier League API: squads, availability and fixtures.

The FPL API is free, needs no key, and is the best public source for the thing
that most often breaks a forecast: **who is actually going to play**. A model
that thinks Haaland is starting when he is injured will be confidently wrong.

Two endpoints:

``/api/bootstrap-static/``
    Every player in every squad, with the fields we care about: ``status`` (a
    one-letter availability flag), ``chance_of_playing_next_round``, the
    ``news`` string a human wrote about them, minutes played so far, and form.
``/api/fixtures/``
    The full fixture list, with kick-off times and which gameweek each match
    belongs to. Used to know what is coming up and when.

Snapshots, not history
----------------------
The API only ever tells you about **now**: ask it in March and there is no way
to learn who was injured in October. So the player table is written
**append-only**, one date-stamped snapshot per run. Run it on a schedule and a
history of availability builds up over time, which is what lets the goalscorer
model eventually learn how a "75% chance of playing" flag actually cashes out.
That history cannot be backfilled, so the sooner it starts the better.

Team names come back as FPL spells them ("Spurs", "Man Utd") and are mapped to
canonical names through the lookup, keyed on FPL's numeric team id.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.lookups import to_canonical

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "fpl"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

PLAYERS_PARQUET = PROCESSED_DIR / "fpl_players.parquet"
FIXTURES_PARQUET = PROCESSED_DIR / "fpl_fixtures.parquet"

SOURCE = "fpl"
BASE_URL = "https://fantasy.premierleague.com/api"
BOOTSTRAP_URL = f"{BASE_URL}/bootstrap-static/"
FIXTURES_URL = f"{BASE_URL}/fixtures/"
USER_AGENT = (
    "premforecaster/0.1 (personal, non-commercial football forecasting project)"
)

#: What FPL's one-letter availability flag means. This is the single most
#: valuable field in the whole API for our purposes.
STATUS_MEANINGS = {
    "a": "available",
    "d": "doubtful",
    "i": "injured",
    "s": "suspended",
    "u": "unavailable",
    "n": "on loan or not in squad",
}

#: FPL's numeric element types.
POSITIONS = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}

#: A Premier League season has 20 squads of ~25, so a healthy snapshot is
#: comfortably over 500 players. Far below that means a partial response.
MIN_EXPECTED_PLAYERS = 500

#: 20 clubs playing each other twice.
EXPECTED_FIXTURES = 380


class FPLUnavailableError(RuntimeError):
    """Raised when the FPL API cannot be reached."""


class FPLFormatError(ValueError):
    """Raised when the FPL API returns something we do not recognise."""


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------


def _get_json(url: str, *, session: requests.Session | None = None, timeout: int = 45) -> dict | list:
    owns_session = session is None
    session = session or requests.Session()
    try:
        response = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        raise FPLUnavailableError(
            f"Could not reach the FPL API at {url}: {type(error).__name__}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise FPLFormatError(
            f"The FPL API at {url} did not return JSON: {error}"
        ) from error
    finally:
        if owns_session:
            session.close()


def raw_path(name: str, taken_at: datetime, raw_dir: Path | str = RAW_DIR) -> Path:
    """Date-stamped raw file. Never overwrites an earlier snapshot."""
    return Path(raw_dir) / f"{name}_{taken_at:%Y-%m-%d}.json"


def fetch_bootstrap(
    *, session: requests.Session | None = None, save_to: Path | str | None = RAW_DIR,
    taken_at: datetime | None = None,
) -> dict:
    """Download bootstrap-static: every squad, player and availability flag."""
    taken_at = taken_at or datetime.now(timezone.utc)
    payload = _get_json(BOOTSTRAP_URL, session=session)

    if not isinstance(payload, dict) or "elements" not in payload or "teams" not in payload:
        raise FPLFormatError(
            "bootstrap-static did not contain the expected 'elements' and 'teams' "
            f"keys. Keys present: {list(payload)[:15] if isinstance(payload, dict) else type(payload)}"
        )

    if save_to is not None:
        destination = raw_path("bootstrap-static", taken_at, save_to)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_text(json.dumps(payload), encoding="utf-8")
            logger.info("Saved %s", destination.name)

    return payload


def fetch_fixtures(
    *, session: requests.Session | None = None, save_to: Path | str | None = RAW_DIR,
    taken_at: datetime | None = None,
) -> list:
    """Download the fixture list."""
    taken_at = taken_at or datetime.now(timezone.utc)
    payload = _get_json(FIXTURES_URL, session=session)

    if not isinstance(payload, list):
        raise FPLFormatError(f"The fixtures endpoint returned {type(payload)}, not a list.")

    if save_to is not None:
        destination = raw_path("fixtures", taken_at, save_to)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_text(json.dumps(payload), encoding="utf-8")
            logger.info("Saved %s", destination.name)

    return payload


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def team_id_map(bootstrap: dict) -> dict[int, str]:
    """FPL's numeric team id -> canonical name, via the lookup.

    Mapping on the id rather than the name is deliberate: ids are stable within
    a season and unambiguous, whereas FPL's display names are exactly the sort
    of thing that changes ("Spurs" vs "Tottenham") and silently breaks joins.
    """
    teams = pd.DataFrame(bootstrap["teams"])
    for column in ("id", "name"):
        if column not in teams.columns:
            raise FPLFormatError(
                f"FPL teams data has no {column!r} column. Columns: {list(teams.columns)}"
            )

    canonical = to_canonical(teams["name"], SOURCE)
    return dict(zip(teams["id"].astype(int), canonical))


def parse_players(bootstrap: dict, taken_at: datetime | None = None) -> pd.DataFrame:
    """One row per player, as of the moment the snapshot was taken."""
    taken_at = taken_at or datetime.now(timezone.utc)
    players = pd.DataFrame(bootstrap["elements"])

    required = [
        "id", "team", "element_type", "first_name", "second_name", "web_name",
        "status", "minutes", "news",
    ]
    missing = [column for column in required if column not in players.columns]
    if missing:
        raise FPLFormatError(
            f"FPL player data is missing the column(s) {missing}. The API may have "
            f"changed. Columns present: {sorted(players.columns)[:25]}"
        )

    teams = team_id_map(bootstrap)
    unknown_ids = sorted(set(players["team"].astype(int)) - set(teams))
    if unknown_ids:
        raise FPLFormatError(f"Players reference unknown FPL team id(s): {unknown_ids}")

    def numeric(column: str) -> pd.Series:
        if column not in players.columns:
            return pd.Series(pd.NA, index=players.index, dtype="Float64")
        return pd.to_numeric(players[column], errors="coerce").astype("Float64")

    status = players["status"].astype("string").str.strip().str.lower()
    unexpected = sorted(set(status.dropna()) - set(STATUS_MEANINGS))
    if unexpected:
        logger.warning(
            "FPL returned unfamiliar status flag(s) %s; they are kept as-is but "
            "status_meaning will be blank for them.", unexpected,
        )

    tidy = pd.DataFrame(
        {
            "snapshot_date": pd.Timestamp(taken_at).tz_convert("UTC").normalize()
            if pd.Timestamp(taken_at).tzinfo
            else pd.Timestamp(taken_at).tz_localize("UTC").normalize(),
            "player_id": players["id"].astype("int64"),
            "team": players["team"].astype(int).map(teams),
            "position": players["element_type"].astype(int).map(POSITIONS),
            "web_name": players["web_name"].astype("string").str.strip(),
            "full_name": (
                players["first_name"].astype("string").str.strip()
                + " "
                + players["second_name"].astype("string").str.strip()
            ),
            "status": status,
            "status_meaning": status.map(STATUS_MEANINGS).astype("string"),
            "chance_of_playing_next_round": numeric("chance_of_playing_next_round"),
            "chance_of_playing_this_round": numeric("chance_of_playing_this_round"),
            "news": players["news"].astype("string").str.strip().replace("", pd.NA),
            "minutes": numeric("minutes"),
            "starts": numeric("starts"),
            "form": numeric("form"),
            "points_per_game": numeric("points_per_game"),
            "total_points": numeric("total_points"),
            "goals_scored": numeric("goals_scored"),
            "assists": numeric("assists"),
            "expected_goals": numeric("expected_goals"),
            "expected_assists": numeric("expected_assists"),
            "selected_by_percent": numeric("selected_by_percent"),
            "now_cost": numeric("now_cost"),
        }
    )

    if tidy["team"].isna().any():
        raise FPLFormatError("Some players could not be mapped to a canonical team.")

    return tidy.sort_values(["team", "web_name"]).reset_index(drop=True)


def parse_fixtures(fixtures: list, bootstrap: dict) -> pd.DataFrame:
    """One row per fixture, with canonical team names and UTC kick-offs."""
    frame = pd.DataFrame(fixtures)
    if frame.empty:
        raise FPLFormatError("The FPL fixtures endpoint returned no fixtures.")

    required = ["id", "team_h", "team_a", "event", "finished", "kickoff_time"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise FPLFormatError(
            f"FPL fixture data is missing the column(s) {missing}. Columns present: "
            f"{list(frame.columns)}"
        )

    teams = team_id_map(bootstrap)
    unknown = sorted(
        (set(frame["team_h"].dropna().astype(int)) | set(frame["team_a"].dropna().astype(int)))
        - set(teams)
    )
    if unknown:
        raise FPLFormatError(f"Fixtures reference unknown FPL team id(s): {unknown}")

    # kickoff_time is ISO 8601 in UTC, but is null for fixtures not yet scheduled.
    kickoff = pd.to_datetime(frame["kickoff_time"], errors="coerce", utc=True)

    tidy = pd.DataFrame(
        {
            "fixture_id": frame["id"].astype("int64"),
            "gameweek": pd.to_numeric(frame["event"], errors="coerce").astype("Int64"),
            "kickoff_utc": kickoff,
            "home_team": frame["team_h"].astype(int).map(teams),
            "away_team": frame["team_a"].astype(int).map(teams),
            "finished": frame["finished"].astype(bool),
            "home_goals": pd.to_numeric(frame.get("team_h_score"), errors="coerce").astype("Int64"),
            "away_goals": pd.to_numeric(frame.get("team_a_score"), errors="coerce").astype("Int64"),
            "home_difficulty": pd.to_numeric(frame.get("team_h_difficulty"), errors="coerce").astype("Int64"),
            "away_difficulty": pd.to_numeric(frame.get("team_a_difficulty"), errors="coerce").astype("Int64"),
        }
    )

    same_team = tidy["home_team"] == tidy["away_team"]
    if same_team.any():
        raise FPLFormatError(f"{int(same_team.sum())} fixture(s) have a team playing itself.")

    return tidy.sort_values(["gameweek", "kickoff_utc", "home_team"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def append_snapshot(
    snapshot: pd.DataFrame, path: Path | str = PLAYERS_PARQUET
) -> pd.DataFrame:
    """Add today's player snapshot to the accumulated history.

    Append-only, because the API cannot tell us about the past: today's
    availability is only ever knowable today. Re-running on the same day
    replaces that day's rows rather than duplicating them, so the job is safe to
    run more than once.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_parquet(path)
        today = snapshot["snapshot_date"].iloc[0]
        replaced = int((existing["snapshot_date"] == today).sum())
        if replaced:
            logger.info("Replacing %d existing row(s) for %s.", replaced, today.date())
            existing = existing[existing["snapshot_date"] != today]
        combined = pd.concat([existing, snapshot], ignore_index=True)
    else:
        combined = snapshot

    combined = combined.sort_values(["snapshot_date", "team", "web_name"]).reset_index(drop=True)
    combined.to_parquet(path, index=False)
    logger.info(
        "Wrote %d player rows across %d snapshot(s) to %s",
        len(combined), combined["snapshot_date"].nunique(), path,
    )
    return combined


def write_fixtures(fixtures: pd.DataFrame, path: Path | str = FIXTURES_PARQUET) -> Path:
    """Fixtures are a full replacement each run - they change as they are rescheduled."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fixtures.to_parquet(path, index=False)
    logger.info("Wrote %d fixtures to %s", len(fixtures), path)
    return path


def build_all(
    *,
    raw_dir: Path | str = RAW_DIR,
    players_path: Path | str = PLAYERS_PARQUET,
    fixtures_path: Path | str = FIXTURES_PARQUET,
    taken_at: datetime | None = None,
) -> dict[str, Path]:
    """Take a snapshot of players and refresh the fixture list."""
    taken_at = taken_at or datetime.now(timezone.utc)

    with requests.Session() as session:
        bootstrap = fetch_bootstrap(session=session, save_to=raw_dir, taken_at=taken_at)
        fixtures = fetch_fixtures(session=session, save_to=raw_dir, taken_at=taken_at)

    players = parse_players(bootstrap, taken_at)
    if len(players) < MIN_EXPECTED_PLAYERS:
        raise FPLFormatError(
            f"Only {len(players)} players in the FPL snapshot, expected at least "
            f"{MIN_EXPECTED_PLAYERS}. The response was probably partial; refusing "
            "to record it as a snapshot."
        )

    append_snapshot(players, players_path)
    write_fixtures(parse_fixtures(fixtures, bootstrap), fixtures_path)

    return {"players": Path(players_path), "fixtures": Path(fixtures_path)}


def availability_summary(players: pd.DataFrame) -> pd.DataFrame:
    """How many players are in each availability state, per snapshot."""
    return (
        players.groupby(["snapshot_date", "status_meaning"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
