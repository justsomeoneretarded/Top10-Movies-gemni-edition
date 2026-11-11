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
    monkeypatch.setattr(editing, "_URL_REVIEWS", {})
    monkeypatch.setattr(editing, "REPORT", {})
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


def test_video_requests_have_a_shared_source_budget(monkeypatch, tmp_path):
    client = Client({"windows": [window()]})
    use_client(monkeypatch, client)
    monkeypatch.setattr(editing, "duration", lambda path: 45)
    for number in range(editing.MAX_VIDEOS + 1):
        source = source_file(tmp_path, str(number), f"{number:011d}")
        editing.analyse_source(source)
    assert len(client.calls) == editing.MAX_VIDEOS
    assert editing.REPORT[str(source)]["status"] == "not reviewed"  # the skipped source is reported as such


def test_disabled_video_analysis_never_probes_or_calls_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(editing, "duration", lambda path: pytest.fail("disabled analysis probed video"))
    monkeypatch.setattr(gemini, "get_client", lambda: pytest.fail("disabled Gemini requested client"))
    assert editing.analyse_source(tmp_path / "missing.mp4") is None


def test_gemini_scene_frames_avoid_local_vision_pull_and_keep_local_embeddings(monkeypatch, tmp_path):
    client = Client({"descriptions": [
        {"index": 1, "description": "A frightened astronaut watches the red cockpit."},
        {"index": 0, "description": "A frightened astronaut watches the red cockpit."}]})
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
    assert len(client.calls) == 1
    assert len(client.calls[0][2]["images"]) == 2
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
    # A scene verified against the local cuts may raise an existing candidate by up to about 0.2;
    # it still never creates a window, a cut or a cue.
    assert 0 < enriched[0]["score"]-original[0]["score"] <= .2
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


def prepare_batch_scenes(monkeypatch, tmp_path, client, count=3):
    use_client(monkeypatch, client)
    source = source_file(tmp_path)
    shots = [{"start": index*4, "end": index*4+4} for index in range(count)]
    monkeypatch.setattr("movie_pipeline.shots.shot_frame",
                        lambda source, shots, index: Image.new("RGB", (128, 72), (index*30, 0, 0)))
    monkeypatch.setattr(scenes, "_pull", lambda *_: None)
    monkeypatch.setattr(scenes, "embed", lambda texts: np.ones((len(texts), 2), dtype=np.float32))
    return source, shots


def test_batch_scene_captions_match_indexes_not_response_order(monkeypatch, tmp_path):
    client = Client({"descriptions": [{"index": 2, "description": "A blue-lit doorway."},
                                       {"index": 0, "description": "A quiet room."},
                                       {"index": 1, "description": "A running figure."}]})
    source, shots = prepare_batch_scenes(monkeypatch, tmp_path, client)
    monkeypatch.setattr(scenes, "_describe_local", lambda image: pytest.fail("valid caption fell back"))
    data = scenes.describe_trailer(source, shots)
    assert data["descriptions"] == ["A quiet room.", "A running figure.", "A blue-lit doorway."]
    assert len(client.calls) == 1 and len(client.calls[0][2]["images"]) == 3


def test_batch_scene_duplicate_and_invalid_indexes_fall_back_only_when_needed(monkeypatch, tmp_path):
    client = Client({"descriptions": [
        {"index": 0, "description": "Duplicate must be rejected."},
        {"index": 0, "description": None},
        {"index": True, "description": "Boolean must not address frame one."},
        {"index": 1.0, "description": "Float must not address frame one."},
        {"index": "1", "description": "String must not address frame one."},
        {"index": 7, "description": "Outside submitted frames."},
        {"index": 2, "description": "A real doorway."}]})
    source, shots = prepare_batch_scenes(monkeypatch, tmp_path, client)
    local = []
    def local_caption(image):
        local.append(image.getpixel((0, 0))[0])
        return "Local frame description."
    monkeypatch.setattr(scenes, "_describe_local", local_caption)
    data = scenes.describe_trailer(source, shots)
    assert local == [0, 30]
    assert data["descriptions"] == ["Local frame description.", "Local frame description.", "A real doorway."]
    assert data["backends"] == ["gemini", "ollama"]
    assert len(client.calls) == 1


def test_batch_scene_failure_never_retries_gemini_per_frame(monkeypatch, tmp_path):
    client = Client(error=gemini.GeminiUnavailable("quota exhausted"))
    source, shots = prepare_batch_scenes(monkeypatch, tmp_path, client, 6)
    calls = []
    monkeypatch.setattr(scenes, "_describe_local", lambda image: calls.append(image) or "Local frame.")
    data = scenes.describe_trailer(source, shots)
    assert len(client.calls) == 1 and len(calls) == 6
    assert data["descriptions"] == ["Local frame."]*6
    assert data["backends"] == ["ollama"]


def test_batch_scene_partial_caption_survives_failed_local_fallback(monkeypatch, tmp_path):
    client = Client({"descriptions": [{"index": 1, "description": "A visible figure."}]})
    source, shots = prepare_batch_scenes(monkeypatch, tmp_path, client)
    def unavailable(image):
        raise ValueError("local unavailable")
    monkeypatch.setattr(scenes, "_describe_local", unavailable)
    data = scenes.describe_trailer(source, shots)
    assert data["descriptions"] == ["", "A visible figure.", ""]
    assert data["described"] == [1] and data["backends"] == ["gemini"]
    assert len(client.calls) == 1


def test_batch_scene_indexes_compact_after_frame_extraction_failure(monkeypatch, tmp_path):
    from movie_pipeline.core import PipelineError
    client = Client({"descriptions": [{"index": 1, "description": "The last actual frame."},
                                       {"index": 0, "description": "The first actual frame."}]})
    source, shots = prepare_batch_scenes(monkeypatch, tmp_path, client)
    def frame(source, shots, index):
        if index == 1:
            raise PipelineError("missing frame")
        return Image.new("RGB", (128, 72), (index*30, 0, 0))
    monkeypatch.setattr("movie_pipeline.shots.shot_frame", frame)
    data = scenes.describe_trailer(source, shots)
    assert data["descriptions"] == ["The first actual frame.", "", "The last actual frame."]
    assert len(client.calls) == 1 and len(client.calls[0][2]["images"]) == 2


def test_scene_descriptions_disabled_prevents_all_cloud_analysis(monkeypatch, tmp_path):
    monkeypatch.setenv("SCENE_DESCRIPTIONS", "false")
    use_client(monkeypatch, Client())
    monkeypatch.setattr(gemini, "get_client", lambda: pytest.fail("disabled scene stage invoked Gemini"))
    monkeypatch.setattr("movie_pipeline.shots.shot_frame", lambda *_: pytest.fail("disabled stage decoded a frame"))
    assert scenes.describe_trailer(tmp_path / "source.mp4", [{"start": 0, "end": 4}]) is None


def thumbnail_rating(index, score=8, logo=False, **kwargs):
    return {"index": index, "large_face": True, "emotion_intensity": 7,
            "action_or_spectacle": 6, "genre_fit": 8,
            "logo_or_text": logo, "thumbnail_score": score, **kwargs}


def test_thumbnail_batch_keeps_per_image_results_and_logo_rejection(monkeypatch):
    from movie_pipeline import thumbnail_studio as studio
    client = Client({"ratings": [thumbnail_rating(1, 9), thumbnail_rating(0, 9, logo=True)]})
    use_client(monkeypatch, client)
    monkeypatch.setattr(scenes, "_rate_thumbnail_local", lambda *args: pytest.fail("valid rating fell back"))
    images = [Image.new("RGB", (1024, 576), (30, 40, 50)), Image.new("RGB", (1024, 576), (80, 90, 100))]
    candidates = [studio.Candidate(image, 0, str(index), score=5-index) for index, image in enumerate(images)]
    assert studio.rate_with_vision(candidates) == 2
    assert candidates[0].notes["logo"] is True and candidates[1].notes["logo"] is False
    assert candidates[0].score < candidates[1].score
    assert len(client.calls) == 1 and len(client.calls[0][2]["images"]) == 2
    assert client.calls[0][2]["images"][0].getpixel((0, 0)) == images[0].getpixel((0, 0))


def test_thumbnail_batch_invalid_rows_only_use_local_fallback(monkeypatch):
    client = Client({"ratings": [thumbnail_rating(0, 3), thumbnail_rating(0, 7),
                                  thumbnail_rating(1, logo="false"), thumbnail_rating(2, 9)]})
    use_client(monkeypatch, client)
    local = []
    def fallback(image, genre, genre_hint):
        local.append(image.getpixel((0, 0))[0])
        return {"large_face": False, "emotion": 1, "action": 2, "genre_fit": 3, "logo": False, "score": 4}
    monkeypatch.setattr(scenes, "_rate_thumbnail_local", fallback)
    images = [Image.new("RGB", (80, 45), (index*30, 0, 0)) for index in range(3)]
    result = scenes.rate_thumbnails(images)
    assert local == [0, 30]
    assert [row["score"] for row in result] == [4, 4, 9]
    assert len(client.calls) == 1


def test_thumbnail_batch_quota_failure_has_no_individual_cloud_retries(monkeypatch):
    client = Client(error=gemini.GeminiUnavailable("quota exhausted"))
    use_client(monkeypatch, client)
    local = []
    monkeypatch.setattr(scenes, "_rate_thumbnail_local", lambda *args: local.append(args[0]) or None)
    assert scenes.rate_thumbnails([Image.new("RGB", (80, 45)) for _ in range(6)]) == [None]*6
    assert len(client.calls) == 1 and len(local) == 6


def test_advisory_sources_cover_all_titles_before_optional_extras():
    pools = [[Path("a-primary"), Path("a-extra-2"), Path("a-extra-3")],
             [Path("b-primary")], [Path("c-primary"), Path("c-extra-2")]]
    assert editing.source_rounds(pools) == [
        Path("a-primary"), Path("b-primary"), Path("c-primary"),
        Path("a-extra-2"), Path("c-extra-2"), Path("a-extra-3")]
    assert editing.source_rounds([]) == []


# ---- Scene planning: review before download, verified scene maps, review openings, honest report ----
def scene(start, end, kind="fight", **kwargs):
    return window(start, end, kind=kind, energy=.8, sound="impact", complete=True, **kwargs)


def test_candidates_are_watched_before_download_and_unsuitable_ones_rejected(monkeypatch):
    good = Client({"windows": [scene(5, 12), scene(20, 30, kind="chase")],
                   "review": {"suitable": True, "variety": .8, "problem": "none"}})
    use_client(monkeypatch, good)
    verdict = editing.review_candidate("https://www.youtube.com/watch?v=aaaaaaaaaaa", 120, "Film - Car Chase Scene")
    assert verdict["suitable"] and verdict["worth"] > .5 and len(good.calls) == 1
    assert good.calls[0][2]["youtube_url"].endswith("aaaaaaaaaaa")
    assert editing.review_candidate("https://www.youtube.com/watch?v=aaaaaaaaaaa", 120)["worth"] == verdict["worth"]
    assert len(good.calls) == 1  # one request per video
    bad = Client({"windows": [scene(5, 12, kind="dialogue")],
                  "review": {"suitable": False, "variety": .1, "problem": "talking_heads"}})
    use_client(monkeypatch, bad)
    rejected = editing.review_candidate("https://www.youtube.com/watch?v=bbbbbbbbbbb", 200)
    assert rejected["suitable"] is False and rejected["worth"] == 0
    assert editing.REPORT["review:https://www.youtube.com/watch?v=bbbbbbbbbbb"]["problem"] == "talking_heads"


def test_a_failed_review_is_reported_as_not_reviewed(monkeypatch):
    use_client(monkeypatch, Client(error=gemini.GeminiUnavailable("quota_exhausted")))
    assert editing.review_candidate("https://www.youtube.com/watch?v=ccccccccccc", 90) is None
    entry = editing.REPORT["review:https://www.youtube.com/watch?v=ccccccccccc"]
    assert entry["status"] == "not reviewed" and "quota_exhausted" in entry["reason"]


def test_review_before_download_becomes_the_downloaded_files_scene_map(monkeypatch, tmp_path):
    client = Client({"windows": [scene(5, 12)], "review": {"suitable": True, "variety": .7, "problem": "none"}})
    use_client(monkeypatch, client)
    editing.review_candidate("https://www.youtube.com/watch?v=abcdefghijk", 45)
    monkeypatch.setattr(editing, "duration", lambda path: 45.4)
    analysis = editing.analyse_source(source_file(tmp_path))
    assert analysis["cache"] == "review" and len(client.calls) == 1  # no second request for the same video


def test_scene_times_are_verified_against_local_cuts_and_cards_dropped():
    shots = [{"start": 0, "end": 4}, {"start": 4, "end": 10.2}, {"start": 10.2, "end": 16.5}, {"start": 16.5, "end": 30},
             {"start": 30, "end": 36}]
    analysis = {"windows": [scene(10, 16), scene(19, 24, kind="dialogue"), scene(30.2, 35.8, kind="establishing"),
                            window(0, 4, title_card=True)]}
    mapped = editing.verify_scenes(analysis, shots, lambda index: index == 4)
    assert [(s["id"], s["verified"]) for s in mapped] == [("S1", True), ("S2", False)]  # card scene and title card dropped
    assert (mapped[0]["start"], mapped[0]["end"]) == (10.2, 16.5)  # snapped to the real cuts
    assert (mapped[1]["start"], mapped[1]["end"]) == (19, 24)      # unverified: kept only as a hint
    assert editing.scene_for(mapped, 10.2, 16.5)["id"] == "S1" and editing.scene_for(mapped, 0, 4) is None
    assert editing.scenes_for_writer(mapped) == ["fight: A tense exchange beside a bridge."]  # verified scenes only


def test_review_opening_prefers_the_genres_signature_scene():
    fight = {**scene(0, 8), "verified": True}
    talk = {**scene(0, 8, kind="dialogue"), "verified": True}
    joke = {**scene(0, 8, kind="comedy"), "verified": True}
    assert editing.opening_fit(fight, "action") > editing.opening_fit(talk, "action")
    assert editing.opening_fit(joke, "comedy") > editing.opening_fit(fight, "comedy")
    assert editing.opening_fit(talk, "drama") > editing.opening_fit(joke, "drama")
    assert editing.opening_fit({**fight, "verified": False}, "action") < editing.opening_fit(fight, "action")
    assert editing.opening_fit(None, "action") == 0


def test_edit_report_never_claims_an_unreviewed_source(monkeypatch, tmp_path):
    monkeypatch.setattr(gemini, "enabled", lambda: True)
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    editing.REPORT[str(a)] = {"status": "scene map", "scenes": 2}
    maps = {str(a): [{**scene(0, 8), "verified": True, "id": "S1"}, {**scene(9, 15), "verified": False, "id": "S2"}]}
    report = editing.edit_report([{"title": "One", "available_scenes": ["fight: x"]}, {"title": "Two"}], [[a], [b]], maps)
    assert report["sources_mapped"] == 1 and report["sources_total"] == 2
    one, two = report["titles"]
    assert one["sources"][0]["verified"] == 1 and one["script_saw_scenes"] is True
    assert two["sources"][0]["status"] == "not reviewed" and two["script_saw_scenes"] is False


def test_rejected_youtube_link_falls_back_to_an_uploaded_copy(monkeypatch, tmp_path):
    class TwoRoutes(Client):
        def generate_json(self, prompt, schema, **kwargs):
            self.calls.append((prompt, schema, kwargs))
            if "youtube_url" in kwargs:
                raise gemini.GeminiUnavailable("request_rejected (HTTP 400: video not supported)")
            return {"windows": [scene(5, 12)], "review": {"suitable": True, "variety": .7, "problem": "none"}}

        def upload_video(self, path, timeout=300):
            self.uploaded = path
            return {"uri": "https://generativelanguage.googleapis.com/v1beta/files/x", "mime_type": "video/mp4"}

    client = TwoRoutes()
    use_client(monkeypatch, client)
    monkeypatch.setattr(editing, "duration", lambda path: 45)
    monkeypatch.setattr(editing, "_proxy_copy", lambda source: tmp_path / "proxy.mp4")
    source = source_file(tmp_path)
    analysis = editing.analyse_source(source)
    assert analysis and len(analysis["windows"]) == 1
    assert client.calls[1][2]["video_file"]["mime_type"] == "video/mp4" and client.uploaded == tmp_path / "proxy.mp4"
    assert editing.REPORT[str(source)]["route"] == "uploaded_file"


def test_quota_errors_do_not_trigger_uploads(monkeypatch, tmp_path):
    client = Client(error=gemini.GeminiUnavailable("quota_exhausted"))
    client.upload_video = lambda *a, **k: pytest.fail("uploaded after a quota error")
    use_client(monkeypatch, client)
    monkeypatch.setattr(editing, "duration", lambda path: 45)
    source = source_file(tmp_path)
    assert editing.analyse_source(source) is None
    assert "quota_exhausted" in editing.REPORT[str(source)]["reason"]
