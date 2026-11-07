"""One speech-model selection for warm-up, final work paths and existing checkpoints."""
from __future__ import annotations

import json
import time

from movie_pipeline import cli


def test_run_checkpoints_are_canonical_and_separate_by_speech_model(monkeypatch, tmp_path):
    reference = tmp_path/"reference.ogg"
    reference.write_bytes(b"same-reference")
    films = [{"title": "A", "year": 2020, "kind": "movie", "genres": []},
             {"title": "B", "year": 2021, "kind": "movie", "genres": []}]

    class Research:
        def collect(self, items, path):
            return films

    monkeypatch.setattr("movie_pipeline.research.Research", Research)
    monkeypatch.setattr("movie_pipeline.availability.enrich_availability", lambda films, *args: films)
    monkeypatch.setattr(cli, "voice_reference", lambda *args: reference)
    monkeypatch.setattr(cli, "download_ceiling", lambda: 1080)
    monkeypatch.setenv("VIDEO_TYPE", "Movies")
    monkeypatch.setenv("NARRATOR", "Warguy")
    monkeypatch.setenv("VIDEO_MINUTES", "Auto")
    monkeypatch.setenv("SCRIPT_MODEL", "example")
    paths = {}
    for model in ("original", " Original ", "turbo", "nano"):
        monkeypatch.setenv("TTS_MODEL", model)
        ctx = cli.setup_run("Top 2 Movies", "Covert operations.", "A (2020); B (2021)",
                            tmp_path/"jobs", tmp_path/"output", time.time())
        paths[model] = ctx.work
        saved = json.loads((ctx.work/"input.json").read_text())
        assert saved["voice_model"] == model.strip().casefold()
    assert paths["original"] == paths[" Original "]
    assert len({paths["original"], paths["turbo"], paths["nano"]}) == 3
    # Removing the local override selects the same original checkpoint directory.
    monkeypatch.delenv("TTS_MODEL")
    ctx = cli.setup_run("Top 2 Movies", "Covert operations.", "A (2020); B (2021)",
                        tmp_path/"jobs", tmp_path/"output", time.time())
    assert ctx.work == paths["original"]
