"""Loading the hand-maintained lookup tables in data/lookups/.

Every data source spells club names differently: football-data.co.uk writes
"Man United", FBref writes "Manchester Utd", Club Elo just writes "Forest". If we
joined on those raw names we would silently lose matches, and a model quietly
missing a third of Nottingham Forest's games is far worse than one that crashes.

So all joins go through ``data/lookups/team_names.csv``, and this module is the
single place that reads it. The important behaviour is that
:func:`to_canonical` **raises** when it meets a name it does not recognise,
rather than returning a blank. A loud failure tells us a source has renamed a
club or a newly promoted team needs adding; a quiet one would just corrupt the
model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LOOKUPS_DIR = REPO_ROOT / "data" / "lookups"
TEAM_NAMES_CSV = LOOKUPS_DIR / "team_names.csv"
STADIUMS_CSV = LOOKUPS_DIR / "stadiums.csv"

CANONICAL_COLUMN = "canonical_name"

#: The per-source name columns in team_names.csv. Add a column here when a new
#: data source is wired up.
SOURCE_COLUMNS = (
    "fbref_name",
    "understat_name",
    "footballdata_name",
    "clubelo_name",
    "fpl_name",
)


class UnknownTeamError(ValueError):
    """Raised when a source uses a club name that is not in the lookup table.

    This nearly always means one of two things: a newly promoted club needs a
    row adding to data/lookups/team_names.csv, or a source has changed how it
    spells an existing club.
    """


def load_team_names(path: Path | str | None = None) -> pd.DataFrame:
    """Read team_names.csv and check it is usable before anyone joins on it."""
    path = Path(path) if path is not None else TEAM_NAMES_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Team name lookup not found at {path}. This file is version "
            "controlled and must be present; see data/lookups/NOTES.md."
        )

    table = pd.read_csv(path, dtype=str)

    expected = [CANONICAL_COLUMN, *SOURCE_COLUMNS]
    missing = [column for column in expected if column not in table.columns]
    if missing:
        raise ValueError(
            f"{path} is missing the column(s) {missing}. Expected {expected}."
        )

    blank = table[expected].isna() | (table[expected].apply(lambda s: s.str.strip()) == "")
    if blank.to_numpy().any():
        offenders = sorted(
            {
                str(table.loc[row, CANONICAL_COLUMN])
                for row, _ in zip(*blank.to_numpy().nonzero())
            }
        )
        raise ValueError(
            f"{path} has blank cells for: {offenders}. Every club needs a name "
            "for every source, or joins will drop matches."
        )

    duplicated = table[CANONICAL_COLUMN][table[CANONICAL_COLUMN].duplicated()].tolist()
    if duplicated:
        raise ValueError(f"{path} has duplicate canonical names: {duplicated}.")

    return table


def team_name_map(source: str, path: Path | str | None = None) -> dict[str, str]:
    """Return a ``{source spelling: canonical name}`` dictionary for one source.

    ``source`` may be given with or without the ``_name`` suffix, so both
    ``"footballdata"`` and ``"footballdata_name"`` work.
    """
    column = source if source.endswith("_name") else f"{source}_name"
    if column not in SOURCE_COLUMNS:
        raise ValueError(
            f"Unknown source {source!r}. Known sources: "
            f"{[c.removesuffix('_name') for c in SOURCE_COLUMNS]}."
        )

    table = load_team_names(path)
    mapping = dict(zip(table[column], table[CANONICAL_COLUMN]))

    if len(mapping) != len(table):
        counts = table[column].value_counts()
        clashes = counts[counts > 1].index.tolist()
        raise ValueError(
            f"Column {column} maps more than one club to the same name: "
            f"{clashes}. The join would be ambiguous."
        )

    return mapping


def to_canonical(
    names: pd.Series,
    source: str,
    path: Path | str | None = None,
) -> pd.Series:
    """Translate a column of source team names into canonical names.

    Raises :class:`UnknownTeamError` listing every unrecognised name, so one run
    tells you everything that needs adding rather than one name at a time.
    """
    mapping = team_name_map(source, path)
    stripped = names.astype("string").str.strip()

    unknown = sorted(set(stripped.dropna()) - set(mapping))
    if unknown:
        raise UnknownTeamError(
            f"{source} used team name(s) not in the lookup: {unknown}. "
            f"Add a row for each to {TEAM_NAMES_CSV.relative_to(REPO_ROOT)} "
            "before this data can be joined."
        )

    return stripped.map(mapping)


def load_stadiums(path: Path | str | None = None) -> pd.DataFrame:
    """Read stadiums.csv (ground name and coordinates per canonical club).

    Note this holds one ground per club, which is right for upcoming fixtures
    but wrong for back-testing clubs that have moved. See data/lookups/NOTES.md.
    """
    path = Path(path) if path is not None else STADIUMS_CSV
    if not path.exists():
        raise FileNotFoundError(f"Stadium lookup not found at {path}.")

    table = pd.read_csv(path)
    expected = ["canonical_name", "stadium", "latitude", "longitude"]
    missing = [column for column in expected if column not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing the column(s) {missing}.")

    return table
