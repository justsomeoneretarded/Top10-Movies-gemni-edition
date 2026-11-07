"""Mock-only checks: Gemini advice cannot overrule local editing protections."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from movie_pipeline import gemini, gemini_editing as editing, moments, scenes


@pytest.fixture(autouse=True)
def isolated_optional_state(monkeypatch):
    monkeypatch.setattr(gemini, "enabled", lambda: False)
    monkeypatch.setattr(gemini, "identity", lambda: {"provider": "gemini", "model": "test-flash"})
    monkeypatch.setattr(gemini, "record_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(editing, "_ATTEMPTED", set())
    monkeypatch.setattr(editing, "_FINGERPRINTS", {})
    monkeypatch.setattr(editing, "_VIDEO_DEADLINE", None)
    monkeypatch.setattr(scenes, "_DEADLINE", None)
    monkeypatch.setattr(scenes, "_READY", False)
    monkeypatch.setattr(scenes, "_LOCAL_VISION_READY", False)
    monkeypatch.setattr(scenes, "_LOCAL_VISION_UNAVAILABLE", False)
    monkeypatch.setattr(scenes, "_UNAVAILABLE", False)
    monkeypatch.setattr(scenes, "_LAST_BACKEND", "ollama")
    monkeypatch.setenv("GEMINI_VIDEO_ANALYSIS", "true")
    monkeypatch.setenv("SCENE_DESCRIPTIONS", "true")


class Client:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def generate_json(self, prompt, schema, **kwargs):
        self.calls.append((prompt, schema, kwargs))
        if self.error:
            raise self.error
        return self.response


def use_client(monkeypatch, client):
    monkeypatch.setattr(gemini, "enabled", lambda: True)
    monkeypatch.setattr(gemini, "get_client", lambda: client)


def window(start=10, end=16, **kwargs):
    return {"start": start, "end": end, "description": "A tense exchange beside a bridge.",
            "interest": .9, "action": .4, "title_card": False, **kwargs}


def source_file(tmp_path, name="source", video_id="abcdefghijk"):
    source = tmp_path / f"{name}.mp4"
    source.write_bytes(b"actual accepted media bytes")
    source.with_suffix(".json").write_text(json.dumps({
        "source": f"https://www.youtube.com/watch?v={video_id}",
        "primary_source": "https://www.youtube.com/watch?v=zyxwvutsrqp"}))
    return source


def test_window_validation_discards_invalid_timing_and_bounds_content():
    rows = [
        window(), window(-1, 4), window(8, 7), window(30, 46),
        window(float("nan"), 12), window(True, 12), window(0, 30),
        window(20, 24, interest=float("inf")), window(20, 24, title_card="false"),
        window(20, 24, description=""), window(25, 29, interest=9, action=-2),
    ]
    valid = editing.validate_windows({"windows": rows}, 45)
    assert [(r["start"], r["end"]) for r in valid] == [(10, 16), (25, 29)]
    assert valid[1]["interest"] == 1 and valid[1]["action"] == 0
    assert len(editing.validate_windows({"windows": [window()] * 100}, 45)) == editing.MAX_WINDOWS
    assert editing.validate_windows({"windows": None}, 45) == []


@pytest.mark.parametrize("url", [
    "http://www.youtube.com/watch?v=abcdefghijk",
    "https://www.youtube.com.evil.test/watch?v=abcdefghijk",
    "https://user@www.youtube.com/watch?v=abcdefghijk",
    "https://www.youtube.com/watch?v=abcdefghijk&v=zyxwvutsrqp",
    "https://www.youtube.com/playlist?v=abcdefghijk",
    "https://youtu.be/abcdefghijk/extra",
    "file:///tmp/movie.mp4",
])
def test_video_analysis_rejects_untrusted_urls(url):
    assert editing.youtube_url(url) is None


def test_video_uses_actual_selected_source_duration_and_fingerprint_cache(monkeypatch, tmp_path):
    source = source_file(tmp_path)
    monkeypatch.setattr(editing, "duration", lambda path: 45.0)
    client = Client({"windows": [window()]})
    use_client(monkeypatch, client)
    first = editing.analyse_source(source)
    assert first["cache"] == "miss"
    options = client.calls[0][2]
    assert options["youtube_url"] == "https://www.youtube.com/watch?v=abcdefghijk"
    assert options["video_duration"] == 45.0
    assert options["max_output_tokens"] <= 4096
    assert editing.analyse_source(source)["cache"] == "hit"
    assert len(client.calls) == 1
    # Same-size replacement must not reuse advice for different bytes.
    source.write_bytes(b"changed accepted media data")
    editing._ATTEMPTED.clear()
    assert editing.analyse_source(source)["cache"] == "miss"
    assert len(client.calls) == 2
    editing._ATTEMPTED.clear()
    monkeypatch.setattr(gemini, "identity", lambda: {"provider": "gemini", "model": "another-flash"})
    assert editing.analyse_source(source)["cache"] == "miss"
    assert len(client.calls) == 3


def test_video_failure_or_unknown_duration_uses_no_advice(monkeypatch, tmp_path):
    source = source_file(tmp_path)
    client = Client(error=ValueError("not accessible"))
    use_client(monkeypatch, client)
    monkeypatch.setattr(editing, "duration", lambda path: float("nan"))
    assert editing.analyse_source(source) is None
    assert client.calls == []
    monkeypatch.setattr(editing, "duration", lambda path: 45)
    assert editing.analyse_source(source) is None
    assert editing.analyse_source(source) is None
    assert len(client.calls) == 1  # never repeatedly retry an inaccessible public URL


def test_video_requests_have_a_shared_twelve_source_budget(monkeypatch, tmp_path):
    client = Client({"windows": [window()]})
    use_client(monkeypatch, client)
    monkeypatch.setattr(editing, "duration", lambda path: 45)
    for number in range(13):
        source = source_file(tmp_path, str(number), f"{number:011d}")
        editing.analyse_source(source)
    assert len(client.calls) == 12


def test_disabled_video_analysis_never_probes_or_calls_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(editing, "duration", lambda path: pytest.fail("disabled analysis probed video"))
    monkeypatch.setattr(gemini, "get_client", lambda: pytest.fail("disabled Gemini requested client"))
    assert editing.analyse_source(tmp_path / "missing.mp4") is None


def test_gemini_scene_frames_avoid_local_vision_pull_and_keep_local_embeddings(monkeypatch, tmp_path):
    client = Client({"description": "A frightened astronaut watches the red cockpit."})
    use_client(monkeypatch, client)
    image = Image.new("RGB", (1280, 720), (180, 30, 50))
    monkeypatch.setattr("movie_pipeline.shots.shot_frame", lambda *_: image)
    pulled = []
    monkeypatch.setattr(scenes, "_pull", pulled.append)
    monkeypatch.setattr(scenes, "embed", lambda texts: np.ones((len(texts), 3), dtype=np.float32))
    source = source_file(tmp_path)
    shots = [{"start": 0, "end": 4}, {"start": 4, "end": 9}]
    data = scenes.describe_trailer(source, shots)
    assert data["backends"] == ["gemini"]
    assert pulled == [scenes.EMBED_MODEL]
    assert data["embedding_provider"] == "ollama"
    assert data["embedding_dimensions"] == 3
    supplied = client.calls[0][2]["images"][0]
    assert supplied.size == (448, 252)
    assert supplied.getpixel((0, 0)) == image.getpixel((0, 0))
    assert scenes.load(source, 2, shots) == data
    assert scenes.load(source, 2, [{"start": 0, "end": 3}, {"start": 3, "end": 9}]) is None
    monkeypatch.setattr(gemini, "identity", lambda: {"provider": "gemini", "model": "changed"})
    assert scenes.load(source, 2) is None


def test_scene_cache_rejects_embedding_model_source_and_malformed_vectors(monkeypatch, tmp_path):
    source = source_file(tmp_path)
    shots = [{"start": 0, "end": 4}]
    monkeypatch.setattr("movie_pipeline.shots.shot_frame", lambda *_: Image.new("RGB", (64, 36)))
    monkeypatch.setattr(scenes, "_describe", lambda image: "A quiet room.")
    monkeypatch.setattr(scenes, "_pull", lambda *_: None)
    monkeypatch.setattr(scenes, "embed", lambda texts: np.ones((len(texts), 2), dtype=np.float32))
    data = scenes.describe_trailer(source, shots)
    assert scenes.load(source, 1)
    monkeypatch.setattr(scenes, "EMBED_MODEL", "different-embedding-space")
    assert scenes.load(source, 1) is None
    monkeypatch.setattr(scenes, "EMBED_MODEL", data["embed_model"])
    original_url = scenes._url
    monkeypatch.setattr(scenes, "_url", lambda: "http://different-local-model-server:11434")
    assert scenes.load(source, 1) is None
    monkeypatch.setattr(scenes, "_url", original_url)
    scenes.cache_path(source).write_text(json.dumps({**data, "backends": []}))
    assert scenes.load(source, 1) is None
    malformed = {**data, "embeddings": [[float("nan"), 1]]}
    scenes.cache_path(source).write_text(json.dumps(malformed))
    assert scenes.load(source, 1) is None
    scenes.cache_path(source).write_text(json.dumps(data))
    source.write_bytes(b"replacement video")
    assert scenes.load(source, 1) is None


def test_scene_provider_failure_preserves_local_description_and_sanitized_reason(monkeypatch):
    use_client(monkeypatch, Client(error=RuntimeError("secret-value-that-must-not-be-logged")))
    events = []
    monkeypatch.setattr(gemini, "record_event", lambda *args, **kwargs: events.append((args, kwargs)))
    monkeypatch.setattr(scenes, "_pull", lambda *_: None)
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"response": "local tense close-up"})
    monkeypatch.setattr(scenes.requests, "post", lambda *args, **kwargs: response)
    assert scenes._describe(Image.new("RGB", (80, 45))) == "local tense close-up"
    assert "secret-value" not in repr(events)
    assert scenes._LAST_BACKEND == "ollama"


def test_thumbnail_gemini_returns_existing_shape_and_same_frame(monkeypatch):
    response = {"large_face": True, "emotion_intensity": 9, "action_or_spectacle": 8,
                "genre_fit": 10, "logo_or_text": False, "thumbnail_score": 9}
    client = Client(response)
    use_client(monkeypatch, client)
    monkeypatch.setattr(scenes.requests, "post", lambda *a, **kw: pytest.fail("local vision used"))
    image = Image.new("RGB", (1024, 576), (120, 40, 80))
    assert scenes.rate_thumbnail(image, "horror", "fear") == {
        "large_face": True, "emotion": 9, "action": 8, "genre_fit": 10, "logo": False, "score": 9}
    supplied = client.calls[0][2]["images"][0]
    assert supplied.size == (512, 288)
    assert supplied.getpixel((0, 0)) == image.getpixel((0, 0))


def test_invalid_gemini_thumbnail_falls_back_to_local_schema(monkeypatch):
    use_client(monkeypatch, Client({"large_face": "true"}))
    payload = {"large_face": False, "emotion_intensity": 4, "action_or_spectacle": 5,
               "genre_fit": 6, "logo_or_text": True, "thumbnail_score": 3}
    calls = []
    def post(url, json=None, **kwargs):
        calls.append(json)
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"response": __import__("json").dumps(payload)})
    monkeypatch.setattr(scenes.requests, "post", post)
    result = scenes.rate_thumbnail(Image.new("RGB", (100, 60)))
    assert result["score"] == 3 and result["logo"] is True
    assert calls[0]["images"] and calls[0]["format"]["required"]


def test_visual_hints_only_rerank_existing_fresh_noncard_candidates(monkeypatch):
    source = Path("accepted.mp4")
    shots = [{"start": 10, "end": 16}, {"start": 20, "end": 26}, {"start": 30, "end": 36}]
    class Selector:
        def identity_relevance(self, *_): return np.full(3, .25)
        def card_kind(self, path, index): return "promo" if index == 1 else None
        def freshness(self, path, start, length): return 0 if start == 30 else 1
        def visual_freshness(self, *_): return 1
    lookup = {str(source): shots}
    monkeypatch.setattr(editing, "analyse_source", lambda *_: None)
    original = moments.candidates(Selector(), [source], lookup, {"title": "Example"})
    analysis = {"provider": {"provider": "gemini"}, "cache": "miss",
                "windows": [window(10.25, 15.75), window(20, 26), window(30, 36)]}
    monkeypatch.setattr(editing, "analyse_source", lambda *_: analysis)
    enriched = moments.candidates(Selector(), [source], lookup, {"title": "Example"})
    assert len(original) == len(enriched) == 1
    for field in ("start", "duration", "action_strength", "scene_index", "cues", "muted"):
        assert enriched[0][field] == original[0][field]
    assert 0 < enriched[0]["score"]-original[0]["score"] <= .055
    assert enriched[0]["visual_analysis"]["notes"]


def test_visual_hint_cannot_authorize_unfinished_or_truncated_dialogue(monkeypatch, tmp_path):
    source = source_file(tmp_path)
    candidate = {"source": str(source), "start": 10.5, "duration": 4.5, "extent": 45,
                 "visual_analysis": {"score_adjustment": .055, "notes": ["An exciting exchange."]}}
    monkeypatch.setattr(moments, "probe", lambda path: {"streams": [{"codec_type": "audio"}]})
    incomplete = [{"start": 10, "end": 15, "text": "If you think that we"}]
    assert moments._enrich(candidate, {str(source): incomplete}, None, tmp_path) is None
    cues = [{"start": 9.3, "end": 11, "text": "We have to"},
            {"start": 11.1, "end": 14, "text": "get across the bridge"},
            {"start": 14.2, "end": 16, "text": "before it collapses."}]
    clip = moments._enrich(candidate, {str(source): cues}, None, tmp_path)
    assert clip["start"] < 9.3 and clip["start"]+clip["duration"] > 16
    assert len(clip["cues"]) == 3 and not clip["muted"]
    assert moments.complete_window(candidate, cues, (9.3, 16)) is None


def test_dead_local_vision_pull_is_attempted_once_while_gemini_stays_optional(monkeypatch):
    import requests
    client = Client(error=ValueError("optional Gemini unavailable"))
    use_client(monkeypatch, client)
    pulls = []
    def down(model):
        pulls.append(model)
        raise requests.ConnectionError("local unavailable")
    monkeypatch.setattr(scenes, "_pull", down)
    image = Image.new("RGB", (64, 36))
    for _ in range(2):
        with pytest.raises((ValueError, requests.RequestException)):
            scenes._describe(image)
    assert pulls == [scenes.VISION_MODEL]
    assert len(client.calls) == 2
