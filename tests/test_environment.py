"""A smoke test for the Python environment.

If this fails, nothing else in the project will work. It only checks that the
two libraries we cannot replace - penaltyblog (the Dixon-Coles model) and
soccerdata (the FBref and Understat scrapers) - import cleanly, and that the
Python version is new enough for the syntax used in this codebase.
"""

from __future__ import annotations

import sys


def test_python_is_at_least_3_11():
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version}"


def test_penaltyblog_imports():
    import penaltyblog  # noqa: F401


def test_soccerdata_imports():
    import soccerdata  # noqa: F401
