"""Free video analysis spends capacity on accepted footage before optional discovery.

All transport, file-duration and clock work is mocked; no model calls occur.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from movie_pipeline import gemini, gemini_editing as editing


SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}},
          "required": ["answer"], "additionalProperties": False}


def answer():
    return {"candidates": [{"finishReason": "STOP",
             "content": {"parts": [{"text": '{"answer":"ok"}'}]}}]}


class Response:
    def __init__(self, data=None, status=200):
        self.status_code, self.headers = status, {}
        self.data = answer() if data is None else data

    def json(self):
        return self.data


class Session:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            pytest.fail("Unexpected request: retry and fallback work must stay bounded")
        row = self.responses.pop(0)
        if isinstance(row, Exception):
            raise row
        return row


def busy():
    return Response({"error": {"status": "UNAVAILABLE"}}, 503)


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    settings = {"AI_BACKEND": "gemini", "GEMINI_API_KEY": "offline-key",
                "GEMINI_TEXT_MODEL": "gemini-3.8-flash",
                "GEMINI_VISION_MODEL": "gemini-3.8-flash",
                "GEMINI_MAX_REQUESTS": "100", "GEMINI_MAX_VIDEO_SECONDS": "600",
                "GEMINI_MAX_RETRIES": "0", "GEMINI_TEXT_RETRIES": "0",
                "GEMINI_VIDEO_RETRIES": "0", "GEMINI_MIN_INTERVAL_SECONDS": "0",
                "GEMINI_USAGE_REPORT": str(tmp_path / "usage.jsonl")}
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("GEMINI_API_KEY_2", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_3", raising=False)
    monkeypatch.delenv("GEMINI_API_2", raising=False)
    monkeypatch.delenv("GEMINI_API_3", raising=False)
    monkeypatch.setattr(gemini, "_BUDGET", gemini._Budget())
    monkeypatch.setattr(gemini, "_KEY_INDEX", [0])
    monkeypatch.setattr(gemini, "_ACTUAL_MODELS", {})
    monkeypatch.setattr(gemini, "_CLIENT", None)
    monkeypatch.setattr(gemini, "_CLIENT_SIGNATURE", None)
    clock = [1000.0]
    monkeypatch.setattr(gemini.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(gemini.time, "sleep", lambda *_: None)
    monkeypatch.setattr(requests.Session, "post",
                        lambda *_a, **_k: pytest.fail("Unexpected live model call"))
    return clock


def video(provider, *, seconds=120, model="gemini-3.5-flash-lite"):
    return provider.generate_json("Map an existing public clip", SCHEMA, model=model,
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk", video_duration=seconds)


def test_failed_discovery_keeps_attempt_charge_but_preserves_accepted_capacity(isolated):
    session = Session(requests.ReadTimeout("private transport details"), Response())
    provider = gemini.GeminiClient(session=session)
    with gemini.video_scope("discovery"), pytest.raises(gemini.GeminiUnavailable):
        video(provider)
    assert gemini.budget_snapshot()["attempted_video_seconds"] == 120
    assert gemini.budget_snapshot()["successful_video_seconds"] == 0
    # Another model can use the reserved accepted share, despite discovery failure.
    with gemini.video_scope("accepted"):
        assert video(provider, seconds=480, model="gemini-3.8-flash") == {"answer": "ok"}
    snapshot = gemini.budget_snapshot()
    assert snapshot["requests"] == 2 and snapshot["discovery_requests"] == 1
    assert snapshot["attempted_video_seconds"] == 600
    assert snapshot["accepted_successful_video_seconds"] == 480
    with pytest.raises(gemini.GeminiUnavailable, match="video_budget"):
        video(provider, seconds=1, model="gemini-3.8-flash")
    assert len(session.calls) == 2


def test_discovery_request_share_does_not_spend_writers_or_accepted_requests(isolated, monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_REQUESTS", "20")
    monkeypatch.setenv("GEMINI_MAX_VIDEO_SECONDS", "10000")
    session = Session(Response(), Response(), Response())
    provider = gemini.GeminiClient(session=session)
    with gemini.video_scope("discovery"):
        video(provider, seconds=1)
        video(provider, seconds=1)
        with pytest.raises(gemini.GeminiUnavailable, match="discovery_video_budget"):
            video(provider, seconds=1)
    video(provider, seconds=1)
    assert len(session.calls) == 3
    assert gemini.budget_snapshot()["discovery_requests"] == 2


def test_tiny_global_budget_is_reserved_for_an_accepted_request(isolated, monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_REQUESTS", "1")
    session = Session(Response())
    provider = gemini.GeminiClient(session=session)
    with gemini.video_scope("discovery"), pytest.raises(gemini.GeminiUnavailable, match="discovery"):
        video(provider, seconds=1)
    video(provider, seconds=1)
    assert len(session.calls) == 1 and gemini.budget_snapshot()["requests"] == 1


def test_video_scope_restores_after_an_exception(isolated):
    with pytest.raises(RuntimeError):
        with gemini.video_scope("discovery"):
            raise RuntimeError("test")
    session = Session(Response())
    provider = gemini.GeminiClient(session=session)
    video(provider)
    assert gemini.budget_snapshot()["discovery_requests"] == 0


def test_busy_service_backoff_grows_and_success_restores_normal_probes(isolated):
    session = Session(busy(), busy(), Response())
    provider = gemini.GeminiClient(session=session)
    def ask():
        return provider.generate_json("A bounded request", SCHEMA, model="gemini-3.5-flash-lite")
    with pytest.raises(gemini.GeminiUnavailable):
        ask()
    first = gemini._BUDGET.cooldown["gemini-3.5-flash-lite"]
    assert first - isolated[0] == 90
    isolated[0] = first + 1
    with pytest.raises(gemini.GeminiUnavailable):
        ask()
    second = gemini._BUDGET.cooldown["gemini-3.5-flash-lite"]
    assert second - isolated[0] == 180
    isolated[0] = second - 1
    with pytest.raises(gemini.GeminiUnavailable):
        ask()
    assert len(session.calls) == 2
    isolated[0] = second + 1
    assert ask() == {"answer": "ok"}
    assert "gemini-3.5-flash-lite" not in gemini._BUDGET.circuit_failures


def test_frame_service_failure_does_not_suppress_independent_writing(isolated):
    from PIL import Image
    session = Session(busy(), Response())
    provider = gemini.GeminiClient(session=session)
    with pytest.raises(gemini.GeminiUnavailable):
        provider.generate_json("Describe", SCHEMA, model="gemini-3.5-flash-lite",
                               images=[Image.new("RGB", (20, 20))])
    assert "gemini-3.5-flash-lite#vision" in gemini._BUDGET.blocked_models
    assert provider.generate_json("Write", SCHEMA, model="gemini-3.5-flash-lite") == {"answer": "ok"}


def test_preferred_model_is_not_retried_on_every_fallback_request(isolated):
    session = Session(busy(), Response(), Response())
    provider = gemini.GeminiClient(session=session)
    assert provider.generate_json("Write one", SCHEMA) == {"answer": "ok"}
    assert provider.generate_json("Write two", SCHEMA) == {"answer": "ok"}
    assert [url.rsplit("/", 1)[-1] for url, _ in session.calls] == [
        "gemini-3.8-flash:generateContent", "gemini-3.5-flash-lite:generateContent",
        "gemini-3.5-flash-lite:generateContent"]
    assert provider.text_model == "gemini-3.8-flash"


def test_default_video_retry_is_one_attempt_and_keeps_exposure(isolated):
    session = Session(requests.ReadTimeout("unavailable"))
    provider = gemini.GeminiClient(session=session)
    with pytest.raises(gemini.GeminiUnavailable):
        video(provider, seconds=60)
    assert len(session.calls) == 1
    assert gemini.budget_snapshot()["attempted_video_seconds"] == 60


def test_optional_video_retry_stays_bounded_and_counts_every_attempt(isolated, monkeypatch):
    monkeypatch.setenv("GEMINI_VIDEO_RETRIES", "99")
    session = Session(busy(), Response())
    provider = gemini.GeminiClient(session=session)
    assert video(provider, seconds=60) == {"answer": "ok"}
    snapshot = gemini.budget_snapshot()
    assert len(session.calls) == 2 and snapshot["video_requests"] == 2
    assert snapshot["attempted_video_seconds"] == 120
    assert snapshot["successful_video_seconds"] == 60


def scene():
    return {"start": 3, "end": 12, "description": "A tense exchange with a complete answer.",
            "interest": .9, "action": .4, "title_card": False, "kind": "confrontation",
            "energy": .5, "sound": "dialogue", "complete": True}


class SceneClient:
    vision_model = "gemini-3.5-flash-lite"

    def __init__(self, error=None, empty=False):
        self.calls, self.error, self.empty = [], error, empty

    def generate_json(self, prompt, schema, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {"windows": [] if self.empty else [scene()],
                "review": {"suitable": True, "variety": .8, "problem": "none"}}

    def upload_video(self, *args, **kwargs):
        pytest.fail("An unavailable service must not trigger a second upload route")


@pytest.fixture
def edit_state(isolated, monkeypatch):
    monkeypatch.setattr(editing, "_ATTEMPTED", set())
    monkeypatch.setattr(editing, "_VIDEO_DEADLINE", None)
    monkeypatch.setattr(editing, "_URL_REVIEWS", {})
    monkeypatch.setattr(editing, "_FINGERPRINTS", {})
    monkeypatch.setattr(editing, "_FAILURES", {})
    monkeypatch.setattr(editing, "_COVERAGE", None)
    monkeypatch.setattr(editing, "REPORT", {})
    monkeypatch.setattr(editing, "_video_pause", lambda: None)
    monkeypatch.setattr(editing, "duration", lambda _: 45.0)
    monkeypatch.setenv("GEMINI_VIDEO_ANALYSIS", "true")
    monkeypatch.setenv("GEMINI_VIDEO_BUDGET_SECONDS", "1800")
    return isolated


def source(tmp_path, name, number):
    path = tmp_path / (name + ".mp4")
    path.write_bytes(b"actual-" + name.encode())
    path.with_suffix(".json").write_text(json.dumps(
        {"source": "https://www.youtube.com/watch?v=" + f"{number:011d}"}))
    return path


def attach(monkeypatch, client):
    monkeypatch.setattr(gemini, "get_client", lambda: client)


def test_discovery_time_does_not_start_the_accepted_analysis_clock(edit_state, tmp_path, monkeypatch):
    client = SceneClient()
    attach(monkeypatch, client)
    assert editing.review_candidate("https://www.youtube.com/watch?v=00000000001", 45)
    assert editing._VIDEO_DEADLINE is None
    edit_state[0] += 1000  # Discovery and downloads have taken time.
    primary = source(tmp_path, "primary", 2)
    editing.begin_accepted_analysis([[primary]])
    assert editing._VIDEO_DEADLINE is None
    assert editing.analyse_source(primary)
    assert editing._VIDEO_DEADLINE == edit_state[0] + 1800
    assert len(client.calls) == 2


def test_every_primary_gets_a_turn_before_uncached_extras(edit_state, tmp_path, monkeypatch):
    client = SceneClient()
    attach(monkeypatch, client)
    a, extra, b = (source(tmp_path, "a", 1), source(tmp_path, "extra", 2),
                   source(tmp_path, "b", 3))
    editing.begin_accepted_analysis([[a, extra], [b]])
    assert editing.analyse_source(extra) is None and not client.calls
    assert editing._VIDEO_DEADLINE is None
    assert editing.analyse_source(a)
    assert editing.analyse_source(extra) is None
    assert editing.analyse_source(b)
    assert editing.analyse_source(extra)
    assert len(client.calls) == 3
    assert editing.coverage_snapshot()["pending_primary_sources"] == 0


def test_missing_primary_does_not_deadlock_remaining_titles_and_extras(edit_state, tmp_path, monkeypatch):
    client = SceneClient()
    attach(monkeypatch, client)
    a, extra, absent = (source(tmp_path, "a", 1), source(tmp_path, "extra", 2),
                        tmp_path / "missing.mp4")
    editing.begin_accepted_analysis([[a, extra], [absent]])
    assert editing.analyse_source(a)
    assert editing.analyse_source(absent) is None
    assert editing.analyse_source(extra)
    assert len(client.calls) == 2 and editing.coverage_snapshot()["pending_primary_sources"] == 0


def test_existing_exact_cache_is_reused_even_before_other_primaries_or_after_budget(edit_state, tmp_path, monkeypatch):
    client = SceneClient()
    attach(monkeypatch, client)
    cached = source(tmp_path, "cached", 1)
    assert editing.analyse_source(cached)
    other = source(tmp_path, "other", 2)
    editing.begin_accepted_analysis([[other, cached]])
    monkeypatch.setattr(editing, "_VIDEO_DEADLINE", 0)
    assert editing.analyse_source(cached)["cache"] == "hit"
    assert len(client.calls) == 1
    snapshot = editing.coverage_snapshot()
    assert snapshot["cache_reuses"] == 1 and snapshot["unique_sources_mapped"] == 1


def test_pre_download_review_is_reused_without_spending_accepted_capacity(edit_state, tmp_path, monkeypatch):
    client = SceneClient()
    attach(monkeypatch, client)
    primary = source(tmp_path, "primary", 1)
    assert editing.review_candidate("https://www.youtube.com/watch?v=00000000001", 45)
    editing.begin_accepted_analysis([[primary]])
    assert editing.analyse_source(primary)["cache"] == "review"
    assert len(client.calls) == 1 and editing._VIDEO_DEADLINE is None
    assert editing.coverage_snapshot()["candidate_review_reuses"] == 1


def test_outage_does_not_trigger_file_upload_route(edit_state, tmp_path, monkeypatch):
    client = SceneClient(error=gemini.GeminiUnavailable("service_unavailable"))
    attach(monkeypatch, client)
    assert editing.analyse_source(source(tmp_path, "primary", 1)) is None
    assert len(client.calls) == 1


def test_discovery_source_limit_preserves_accepted_work(edit_state, tmp_path, monkeypatch):
    client = SceneClient()
    attach(monkeypatch, client)
    for n in range(1, 10):
        editing.review_candidate("https://www.youtube.com/watch?v=" + f"{n:011d}", 45)
    assert len(client.calls) == 8
    assert editing._VIDEO_DEADLINE is None
    primary = source(tmp_path, "primary", 20)
    editing.begin_accepted_analysis([[primary]])
    assert editing.analyse_source(primary)
    assert len(client.calls) == 9
    assert editing.coverage_snapshot()["skipped"]["discovery_source_or_time_budget"] == 1


def test_accepted_time_limit_remains_finite(edit_state, tmp_path, monkeypatch):
    client = SceneClient()
    attach(monkeypatch, client)
    a, b = source(tmp_path, "a", 1), source(tmp_path, "b", 2)
    editing.begin_accepted_analysis([[a], [b]])
    assert editing.analyse_source(a)
    edit_state[0] += 1801
    assert editing.analyse_source(b) is None
    assert len(client.calls) == 1
    assert editing.coverage_snapshot()["skipped"]["accepted_source_or_time_budget"] == 1


def test_invalid_scene_result_never_claims_usable_coverage(edit_state, tmp_path, monkeypatch):
    client = SceneClient(empty=True)
    attach(monkeypatch, client)
    path = source(tmp_path, "a", 1)
    assert editing.analyse_source(path) is None
    assert editing.coverage_snapshot()["unique_sources_mapped"] == 0
    report = editing.edit_report([{"title": "A"}], [[path]], {})
    assert report["sources_mapped"] == 0
    assert report["coverage"]["accepted_source_attempts"] == 1


def test_late_discovery_does_not_spend_reserved_accepted_phase(edit_state, tmp_path, monkeypatch):
    client = SceneClient()
    attach(monkeypatch, client)
    editing.begin_accepted_analysis([[source(tmp_path, "a", 1)]])
    assert editing.review_candidate("https://www.youtube.com/watch?v=00000000002", 45) is None
    assert not client.calls
