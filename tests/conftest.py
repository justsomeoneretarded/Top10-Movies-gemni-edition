

import pytest as _pytest


@_pytest.fixture(autouse=True)
def _no_genre_leak(monkeypatch):
    """Production sets VIDEO_GENRE for the run; tests start without it."""
    import os
    monkeypatch.delenv("VIDEO_GENRE", raising=False)
    yield
    os.environ.pop("VIDEO_GENRE", None)


@_pytest.fixture(autouse=True)
def _natural_narration_speed(monkeypatch):
    """Tests check the natural pacing; the workflow's 1.2x default is tested on its own."""
    monkeypatch.setenv("NARRATION_SPEED", "1")
