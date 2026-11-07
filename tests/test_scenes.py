import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from movie_pipeline import scenes
from movie_pipeline.shots import ShotSelector, _zscore


@pytest.fixture(autouse=True)
def reset_scene_budget(monkeypatch):
    monkeypatch.setattr(scenes, "_DEADLINE", None)
    monkeypatch.setattr(scenes, "_READY", False)
    monkeypatch.setattr(scenes, "_UNAVAILABLE", False)

class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def fake_ollama(calls):
    def post(url, json=None, timeout=None):
        calls.append((url.rsplit("/", 1)[-1], json.get("model")))
        if url.endswith("/api/generate"):
            assert json["images"]
            return FakeResponse({"response": "close-up of an astronaut, tense, red cockpit light"})
        if url.endswith("/api/embed"):
            return FakeResponse({"embeddings": [[1.0, float(i)] for i, _ in enumerate(json["input"])]})
        return FakeResponse({})
    return post


def test_descriptions_are_generated_once_and_cached(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(scenes.requests, "post", fake_ollama(calls))
    monkeypatch.setattr("movie_pipeline.shots.get_frame", lambda path, seconds: Image.new("RGB", (1280, 720), "red"))
    trailer = tmp_path / "movie.mp4"
    shots = [{"start": 0, "end": 4}, {"start": 4, "end": 9}]
    data = scenes.describe_trailer(trailer, shots)
    assert data["descriptions"] == ["close-up of an astronaut, tense, red cockpit light"] * 2
    assert len(data["embeddings"]) == 2
    assert json.loads(scenes.cache_path(trailer).read_text())["version"] == scenes.VERSION
    calls.clear()
    assert scenes.describe_trailer(trailer, shots) == scenes.load(trailer, 2)
    assert calls == []  # cached
    assert scenes.load(trailer, 3) is None  # shot list changed


def test_descriptions_can_be_switched_off(monkeypatch, tmp_path):
    monkeypatch.setenv("SCENE_DESCRIPTIONS", "false")
    assert scenes.describe_trailer(tmp_path / "movie.mp4", [{"start": 0, "end": 4}]) is None


def test_unavailable_vision_model_falls_back_quietly(monkeypatch, tmp_path):
    import requests

    def down(*args, **kwargs):
        raise requests.ConnectionError("no server")
    monkeypatch.setattr(scenes.requests, "post", down)
    assert scenes.describe_trailer(tmp_path / "movie.mp4", [{"start": 0, "end": 4}]) is None


def test_relevance_blends_image_and_description_matches_on_clip_scale(monkeypatch):
    torch = pytest.importorskip("torch")
    selector = object.__new__(ShotSelector)
    selector.torch = torch
    path = "movie.mp4"
    selector.siglip = SimpleNamespace(text=lambda text: torch.tensor([[1.0, 0.0]]))
    selector.siglip_features = {path: torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])}
    selector.scene_embeddings = {path: np.array([[0, 1], [1, 0], [0.7, 0.7]], dtype=np.float32)}
    monkeypatch.setattr(scenes, "embed", lambda texts: np.array([[1.0, 0.0]], dtype=np.float32))
    scores = selector.relevance(path, "An astronaut panics.")
    assert scores.shape == (3,)
    assert 0.15 < scores.min() and scores.max() < 0.35
    assert scores[2] == pytest.approx(scores.mean(), abs=0.02) or scores.argmax() in (0, 1)


def test_zscore_handles_flat_scores():
    assert np.all(_zscore([0.3, 0.3, 0.3]) == 0)


def test_siglip_loader_reports_and_falls_back_to_clip(monkeypatch, capsys):
    import builtins

    from movie_pipeline.shots import load_siglip
    real = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name in ("open_clip", "transformers"):
            raise ImportError(name)
        return real(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    assert load_siglip(object()) is None
    assert "SigLIP 2 unavailable" in capsys.readouterr().out
