"""Tests for the resumable-job checkpoint file.

The point of a checkpoint is that a crash costs you one unit of work, not the
whole run, so these tests care mostly about what happens when things go wrong.
"""

from __future__ import annotations

import json

import pytest

from src.checkpoint import Checkpoint


def test_a_new_checkpoint_is_empty(tmp_path):
    checkpoint = Checkpoint(tmp_path / "job.json")
    assert len(checkpoint) == 0
    assert not checkpoint.is_done("shots/2024-25")


def test_marking_work_done_survives_a_restart(tmp_path):
    path = tmp_path / "job.json"
    Checkpoint(path).mark_done("shots/2024-25", rows=9878)

    reopened = Checkpoint(path)
    assert reopened.is_done("shots/2024-25")
    assert reopened.completed["shots/2024-25"]["rows"] == 9878


def test_pending_returns_only_unfinished_work_in_order(tmp_path):
    checkpoint = Checkpoint(tmp_path / "job.json")
    checkpoint.mark_done("a")
    checkpoint.mark_done("c")
    assert checkpoint.pending(["a", "b", "c", "d"]) == ["b", "d"]


def test_step_records_the_unit_only_when_it_succeeds(tmp_path):
    checkpoint = Checkpoint(tmp_path / "job.json")

    with checkpoint.step("good"):
        pass
    assert checkpoint.is_done("good")

    with pytest.raises(RuntimeError):
        with checkpoint.step("bad"):
            raise RuntimeError("network died")

    # The failed unit must be retried next run, so it must not be recorded.
    assert not checkpoint.is_done("bad")


def test_forget_puts_work_back_on_the_list(tmp_path):
    checkpoint = Checkpoint(tmp_path / "job.json")
    checkpoint.mark_done("shots/2024-25")
    checkpoint.forget("shots/2024-25")
    assert not checkpoint.is_done("shots/2024-25")


def test_clear_empties_the_checkpoint(tmp_path):
    checkpoint = Checkpoint(tmp_path / "job.json")
    checkpoint.mark_done("a")
    checkpoint.mark_done("b")
    checkpoint.clear()
    assert len(checkpoint) == 0
    assert len(Checkpoint(tmp_path / "job.json")) == 0


def test_a_corrupt_checkpoint_fails_loudly(tmp_path):
    """Better to stop and say so than to silently redo or skip everything."""
    path = tmp_path / "job.json"
    path.write_text("{not json at all")
    with pytest.raises(ValueError, match="corrupt"):
        Checkpoint(path)


def test_the_checkpoint_file_is_readable_json(tmp_path):
    """A human should be able to open it and see what happened."""
    path = tmp_path / "job.json"
    Checkpoint(path).mark_done("shots/2024-25", rows=10)

    payload = json.loads(path.read_text())
    assert "updated_at" in payload
    assert payload["completed"]["shots/2024-25"]["rows"] == 10


def test_no_temporary_file_is_left_behind(tmp_path):
    """Writes go via a .tmp file that must always be moved into place."""
    path = tmp_path / "job.json"
    Checkpoint(path).mark_done("a")
    assert not (tmp_path / "job.tmp").exists()
    assert path.exists()
