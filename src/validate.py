"""Data-quality guards for the processed tables.

Separate from ``src/reconcile.py``: that compares one source against another,
this checks a single table is internally sound before anything trusts it.

The guard here is about **time**. Almost every table in this project has a
season-shaped axis - a ``season`` column of labels like ``2014-15`` - and almost
everything downstream walks forward through it: the Dixon-Coles model decays
older matches, and the back-test replays history in order. A missing season in
the middle of such a table is uniquely nasty, because nothing looks broken. The
table still loads, every row is still valid, and the model just quietly trains
on a history with two years cut out of it.

That is not hypothetical. The player match table ended up holding 2014/15 to
2020/21 and 2023/24 to 2026/27, with 2021/22 and 2022/23 missing - a hole in the
middle that no existing test noticed, because every individual row was fine.

So: a table whose seasons run from A to B must contain every season between A
and B. Not covering the full history is fine and often deliberate - the shot
table is a documented recent window - but that has to be *declared*, in
:data:`TABLE_RULES`, rather than being indistinguishable from data we lost.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: A season label looks like "2014-15": four-digit start year, two-digit end.
SEASON_LABEL_PATTERN = re.compile(r"^(\d{4})-(\d{2})$")

DEFAULT_SEASON_COLUMN = "season"


class SeasonAxisError(ValueError):
    """Raised when a table's season axis has a hole in it.

    Means the table is missing a season between its own first and last, which
    would silently distort anything that walks forward through time.
    """


@dataclass(frozen=True)
class SeasonAxisRule:
    """What we accept as missing from one table, and why.

    ``documented_from``
        The table is only ever expected to cover this season onwards. Anything
        earlier may be absent without complaint, so a partial backfill of older
        seasons does not trip the guard.
    ``allowed_gaps``
        Specific seasons known to be missing and accepted. Use sparingly, and
        say why in ``reason`` - a whitelist entry is a promise that someone
        looked at the hole and decided it was fine.
    """

    documented_from: str | None = None
    allowed_gaps: frozenset[str] = field(default_factory=frozenset)
    reason: str = ""

    def permits(self, season: str) -> bool:
        if season in self.allowed_gaps:
            return True
        if self.documented_from is not None:
            return start_year(season) < start_year(self.documented_from)
        return False


#: Per-table exceptions. A table not listed here must have no holes at all.
TABLE_RULES: dict[str, SeasonAxisRule] = {
    # The shot table is deliberately a recent window: shot data costs one
    # request per match, so the full history would be most of a day of
    # continuous scraping for seasons the goalscorer model would not use. See
    # DEFAULT_SHOT_SEASONS in src/scrape/understat.py.
    #
    # Its current span (2023/24 onwards) is contiguous, so this rule changes
    # nothing today. It is here so that a *partial* backfill of older seasons -
    # which would leave a real hole between the backfilled year and the recent
    # window - is accepted as intended rather than failing the build.
    "shots": SeasonAxisRule(
        documented_from="2023-24",
        reason=(
            "documented four-season window (last three completed plus the "
            "current one); earlier seasons are intentionally not held"
        ),
    ),
}


def start_year(season: str) -> int:
    """``"2014-15"`` -> 2014. Raises on anything not season-shaped."""
    match = SEASON_LABEL_PATTERN.match(str(season).strip())
    if not match:
        raise ValueError(
            f"{season!r} is not a season label. Expected 'YYYY-YY', e.g. '2014-15'."
        )
    return int(match.group(1))


def season_label(start: int) -> str:
    """2014 -> ``"2014-15"``."""
    return f"{start}-{(start + 1) % 100:02d}"


def has_season_axis(
    frame: pd.DataFrame, column: str = DEFAULT_SEASON_COLUMN
) -> bool:
    """Does this table have a season-shaped time axis we should be guarding?"""
    if column not in frame.columns:
        return False
    values = frame[column].dropna().unique()
    if len(values) == 0:
        return False
    return all(SEASON_LABEL_PATTERN.match(str(value).strip()) for value in values)


def find_season_gaps(seasons) -> list[str]:
    """Seasons missing between the earliest and latest present.

    Only interior holes count. A table that simply starts late or stops early
    has no gap by this definition - that is coverage, not a hole.
    """
    present = {str(season).strip() for season in pd.Series(list(seasons)).dropna()}
    if not present:
        return []

    years = sorted(start_year(season) for season in present)
    expected = {season_label(year) for year in range(years[0], years[-1] + 1)}
    return sorted(expected - present, key=start_year)


def check_season_contiguity(
    frame: pd.DataFrame,
    table: str,
    *,
    column: str = DEFAULT_SEASON_COLUMN,
    rules: dict[str, SeasonAxisRule] | None = None,
) -> list[str]:
    """Fail if ``table`` is missing a season between its first and last.

    Returns the gaps that were allowed by the table's rule, so a caller can log
    them. Raises :class:`SeasonAxisError` on any gap that is not whitelisted.
    """
    if column not in frame.columns:
        raise SeasonAxisError(
            f"{table} has no {column!r} column, so its season axis cannot be "
            f"checked. Columns present: {list(frame.columns)}"
        )

    gaps = find_season_gaps(frame[column])
    if not gaps:
        return []

    rule = (rules if rules is not None else TABLE_RULES).get(table, SeasonAxisRule())
    permitted = [season for season in gaps if rule.permits(season)]
    unexplained = [season for season in gaps if not rule.permits(season)]

    if unexplained:
        held = sorted({str(s).strip() for s in frame[column].dropna()}, key=start_year)
        raise SeasonAxisError(
            f"{table} is missing {len(unexplained)} season(s) from the middle of "
            f"its own range: {unexplained}. It holds {held[0]} to {held[-1]}, so "
            "anything that walks forward through time would silently skip those "
            "years. Either fetch them, or add them to TABLE_RULES in "
            "src/validate.py with a reason if the hole is intended."
        )

    if permitted:
        logger.info(
            "%s: %d gap(s) allowed by rule (%s): %s",
            table, len(permitted), rule.reason or "no reason recorded",
            ", ".join(permitted),
        )
    return permitted


def check_tables(
    tables: dict[str, pd.DataFrame],
    *,
    rules: dict[str, SeasonAxisRule] | None = None,
) -> dict[str, list[str]]:
    """Run the season-axis guard over several tables at once.

    Tables without a season axis are skipped. Every failure is collected before
    raising, so one run tells you about all of them rather than one at a time.
    """
    allowed: dict[str, list[str]] = {}
    failures: list[str] = []

    for table, frame in tables.items():
        if frame is None or frame.empty:
            logger.debug("Skipping %s in season-axis check: no data.", table)
            continue
        if not has_season_axis(frame):
            logger.debug("Skipping %s in season-axis check: no season axis.", table)
            continue
        try:
            allowed[table] = check_season_contiguity(frame, table, rules=rules)
        except SeasonAxisError as error:
            failures.append(str(error))

    if failures:
        raise SeasonAxisError("\n\n".join(failures))

    return allowed


def season_span(frame: pd.DataFrame, column: str = DEFAULT_SEASON_COLUMN) -> str:
    """A short "2014-15 to 2026-27 (13 seasons)" description, for logging."""
    seasons = sorted({str(s).strip() for s in frame[column].dropna()}, key=start_year)
    if not seasons:
        return "no seasons"
    return f"{seasons[0]} to {seasons[-1]} ({len(seasons)} seasons)"


def validate_processed_tables(
    processed_dir: Path | str,
    *,
    rules: dict[str, SeasonAxisRule] | None = None,
) -> dict[str, list[str]]:
    """Load every processed parquet and run the season-axis guard over it.

    The table name used for the rules lookup is the filename without its
    suffix, so ``shots.parquet`` is checked against ``TABLE_RULES["shots"]``.
    """
    processed_dir = Path(processed_dir)
    tables = {
        path.stem: pd.read_parquet(path)
        for path in sorted(processed_dir.glob("*.parquet"))
    }
    return check_tables(tables, rules=rules)
