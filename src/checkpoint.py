"""Checkpoint files, so a long scrape can be interrupted and resumed.

Scraping Understat and FBref politely takes hours: FBref allows one request
every seven seconds, and a single season of shot data is one request per match.
A job that lost all its progress when the laptop slept, the network dropped or
GitHub Actions timed out would be unusable.

So the long jobs record each finished unit of work - normally one season of one
table - in a small JSON file under ``data/raw/<source>/checkpoints/``. On the
next run, anything already recorded is skipped.

This is belt and braces: ``soccerdata`` already caches every page it downloads,
so a rerun would not re-request anything anyway. The checkpoint saves the
parsing work on top of that, and more importantly gives an honest progress
report - it is the thing that can tell you "9 of 13 seasons done" after a crash.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)


class Checkpoint:
    """A record of which units of a long job are already finished.

    Keys are short strings naming a unit of work, e.g. ``"shots/2024"``. The
    file is rewritten after every change, so killing the process mid-run at
    worst loses the unit in flight.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._entries: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Checkpoint file {self.path} is corrupt: {error}. Delete it to "
                "start the job from scratch (cached downloads are kept, so this "
                "costs no extra requests)."
            ) from error

        entries = payload.get("completed", {})
        if not isinstance(entries, dict):
            raise ValueError(f"Checkpoint file {self.path} has an unexpected shape.")
        return entries

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "completed": self._entries,
        }
        # Write to a temporary file and move it into place, so an interrupted
        # write cannot leave a half-written checkpoint behind.
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)

    def is_done(self, key: str) -> bool:
        return key in self._entries

    def mark_done(self, key: str, **details: Any) -> None:
        """Record a unit of work as finished, with any details worth keeping."""
        self._entries[key] = {
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **details,
        }
        self._save()

    def forget(self, key: str) -> None:
        """Drop one unit so it will be redone next run."""
        if self._entries.pop(key, None) is not None:
            self._save()

    def clear(self) -> None:
        self._entries = {}
        self._save()

    def pending(self, keys: Iterable[str]) -> list[str]:
        """Which of these units still need doing, in the order given."""
        return [key for key in keys if not self.is_done(key)]

    @property
    def completed(self) -> dict[str, dict[str, Any]]:
        return dict(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Checkpoint({self.path.name}, {len(self._entries)} done)"

    @contextmanager
    def step(self, key: str, **details: Any) -> Iterator[None]:
        """Run one unit of work, marking it done only if it succeeds.

        Used as ``with checkpoint.step("shots/2024"): ...``. If the body raises,
        nothing is recorded, so the unit is retried on the next run.
        """
        yield
        self.mark_done(key, **details)


def report_progress(checkpoint: Checkpoint, keys: list[str], label: str) -> None:
    """Log a one-line "3 of 13 done" summary before a job starts."""
    outstanding = checkpoint.pending(keys)
    done = len(keys) - len(outstanding)
    if outstanding:
        logger.info(
            "%s: %d of %d already done, %d to fetch (%s)",
            label, done, len(keys), len(outstanding), ", ".join(outstanding[:8]),
        )
    else:
        logger.info("%s: all %d units already done, nothing to fetch.", label, len(keys))
