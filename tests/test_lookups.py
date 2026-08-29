"""Tests for the hand-maintained lookup tables in data/lookups/.

Every join in this project goes through team_names.csv, so a typo or a blank
cell there would quietly drop matches from the model. These tests fail loudly if
that happens: they check the columns are the ones we expect, that nothing is
blank, that no name is duplicated within a source (which would make the mapping
ambiguous), and that every club has a stadium with plausible UK coordinates.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

LOOKUPS_DIR = Path(__file__).resolve().parents[1] / "data" / "lookups"
TEAM_NAMES_CSV = LOOKUPS_DIR / "team_names.csv"
STADIUMS_CSV = LOOKUPS_DIR / "stadiums.csv"

SOURCE_COLUMNS = [
    "fbref_name",
    "understat_name",
    "footballdata_name",
    "clubelo_name",
    "fpl_name",
]

# The UK sits roughly between these bounds. Every ground in the lookup is in
# England or Wales, so anything outside this box is a typo (a flipped sign on
# the longitude, most likely).
UK_LAT_RANGE = (49.5, 61.0)
UK_LON_RANGE = (-8.5, 2.0)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def team_names() -> list[dict[str, str]]:
    return _read(TEAM_NAMES_CSV)


@pytest.fixture(scope="module")
def stadiums() -> list[dict[str, str]]:
    return _read(STADIUMS_CSV)


def test_team_names_has_expected_columns(team_names):
    assert list(team_names[0]) == ["canonical_name", *SOURCE_COLUMNS]


def test_team_names_covers_every_premier_league_club_since_2014_15(team_names):
    # 20 current clubs plus everyone relegated since the Understat horizon.
    assert len(team_names) == 36


def test_no_blank_cells_in_team_names(team_names):
    blanks = [
        (row["canonical_name"], column)
        for row in team_names
        for column, value in row.items()
        if value is None or not value.strip()
    ]
    assert blanks == [], f"blank cells in team_names.csv: {blanks}"


def test_canonical_names_are_unique(team_names):
    names = [row["canonical_name"] for row in team_names]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("column", SOURCE_COLUMNS)
def test_source_names_are_unique_within_a_source(team_names, column):
    """Two clubs sharing one source name would make the join ambiguous."""
    values = [row[column] for row in team_names]
    duplicates = {value for value in values if values.count(value) > 1}
    assert not duplicates, f"{column} maps more than one club to: {duplicates}"


def test_names_have_no_stray_whitespace(team_names):
    offenders = [
        (row["canonical_name"], column, repr(value))
        for row in team_names
        for column, value in row.items()
        if value != value.strip()
    ]
    assert offenders == [], f"leading/trailing whitespace: {offenders}"


def test_every_club_has_a_stadium(team_names, stadiums):
    clubs = {row["canonical_name"] for row in team_names}
    with_grounds = {row["canonical_name"] for row in stadiums}
    assert clubs == with_grounds, (
        f"missing a stadium: {sorted(clubs - with_grounds)}; "
        f"stadium for an unknown club: {sorted(with_grounds - clubs)}"
    )


def test_stadium_coordinates_are_in_the_uk(stadiums):
    for row in stadiums:
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])
        assert UK_LAT_RANGE[0] <= latitude <= UK_LAT_RANGE[1], row
        assert UK_LON_RANGE[0] <= longitude <= UK_LON_RANGE[1], row


def test_no_two_clubs_share_a_ground(stadiums):
    """True today. If it ever stops being true, we want to know about it."""
    coordinates = [(row["latitude"], row["longitude"]) for row in stadiums]
    assert len(coordinates) == len(set(coordinates))
