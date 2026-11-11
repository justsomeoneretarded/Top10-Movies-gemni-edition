"""Provider tests are fully mocked: no live API calls or downloaded models."""
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json

from PIL import Image
import pytest
import requests

from movie_pipeline import gemini


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def answer(text='{"answer":"ok"}', *, finish="STOP", parts=None, grounding=None):
    candidate = {"finishReason": finish, "content": {"parts": parts or [{"text": text}]}}
    if grounding is not None:
        candidate["groundingMetadata"] = grounding
    return {"candidates": [candidate], "usageMetadata": {"totalTokenCount": 12}}


class Response:
    def __init__(self, data, status=200, headers=None):
        self.data, self.status_code, self.headers = data, status, headers or {}

    def json(self):
        return self.data


class Session:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        value = self.responses.pop(0) if self.responses else Response(answer())
        if isinstance(value, Exception):
            raise value
        return value


@pytest.fixture(autouse=True)
def isolated_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-secret")
    monkeypatch.delenv("GEMINI_API", raising=False)
    monkeypatch.setenv("AI_BACKEND", "gemini")
    for name in ("GEMINI_MODEL", "GEMINI_TEXT_MODEL", "GEMINI_RESEARCH_MODEL",
                 "GEMINI_VISION_MODEL", "GEMINI_TTS_MODEL", "GEMINI_MAX_VIDEO_SECONDS",
                 "GEMINI_USAGE_MAX_BYTES", "GEMINI_RESEARCH_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_MAX_REQUESTS", "120")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "0")
    monkeypatch.setenv("GEMINI_MIN_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("GEMINI_USAGE_REPORT", str(tmp_path / "usage.jsonl"))
    monkeypatch.setattr(gemini, "_BUDGET", gemini._Budget())
    monkeypatch.setattr(gemini, "_ACTUAL_MODELS", {})
    monkeypatch.setattr(gemini, "_CLIENT", None)
    monkeypatch.setattr(gemini, "_CLIENT_SIGNATURE", None)
    # A missed mock must fail immediately instead of contacting the internet.
    monkeypatch.setattr(requests.Session, "post",
                        lambda *a, **k: pytest.fail("Unexpected live Gemini request"))


def client(*responses):
    session = Session(*responses)
    return gemini.GeminiClient(session=session), session


def legacy_client(*responses):
    provider, session = client(*responses)
    provider.research_mode = "legacy_google_search"
    return provider, session


def test_enablement_alias_local_override_and_safe_identity(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    assert not gemini.enabled()
    with pytest.raises(gemini.GeminiUnavailable, match="key_missing"):
        gemini.get_client()
    monkeypatch.setenv("GEMINI_API", "alias-secret")
    assert gemini.enabled()
    identity = gemini.identity()
    assert identity["backend"] == "gemini"
    assert identity["text_model"] == "gemini-3.8-flash"
    assert identity["research_model"] == "gemini-2.5-flash"
    assert identity["research_mode"] == "public_sources"
    assert identity["tts_model"] == "gemini-3.8-flash-tts"
    assert identity["free_only"] is True
    assert "secret" not in json.dumps(identity)
    monkeypatch.setenv("AI_BACKEND", "local")
    assert not gemini.enabled() and gemini.identity()["backend"] == "local"
    with pytest.raises(gemini.GeminiUnavailable):
        gemini.get_client()


def test_get_client_is_lazy_and_reused():
    first = gemini.get_client()
    assert first is gemini.get_client()
    assert gemini._BUDGET.requests == 0


def test_json_payload_uses_current_thinking_and_schema_without_sampling():
    provider, session = client(Response(answer()))
    assert provider.generate_json("A private prompt", SCHEMA) == {"answer": "ok"}
    url, call = session.calls[0]
    assert url.endswith("/gemini-3.8-flash:generateContent")
    assert "?" not in url
    assert call["headers"]["x-goog-api-key"] == "test-gemini-secret"
    config = call["json"]["generationConfig"]
    assert config["responseJsonSchema"] == SCHEMA
    assert config["responseMimeType"] == "application/json"
    assert config["thinkingConfig"] == {"thinkingLevel": "low"}
    assert not {"temperature", "topP", "topK", "candidateCount"} & config.keys()
    assert call["timeout"][0] == 10 and call["timeout"][1] <= 180


@pytest.mark.parametrize("model", ["gemini-2.5-flash", "gemini-2.5-flash-lite"])
def test_legacy_models_are_not_used_for_normal_text_or_vision(model):
    provider, session = client()
    with pytest.raises(gemini.GeminiUnavailable, match="free_tier"):
        provider.generate_json("Example", SCHEMA, model=model)
    assert not session.calls


@pytest.mark.parametrize("model", ["gemini-3-pro-image", "gemini-nano-banana-2.1",
                                   "gemini-3.1-pro-preview", "untrusted/model"])
def test_paid_and_unknown_models_are_rejected_without_requests(model, monkeypatch):
    monkeypatch.setenv("GEMINI_FREE_ONLY", "false")
    provider, session = client()
    with pytest.raises(gemini.GeminiUnavailable, match="free_tier"):
        provider.generate_json("Example", SCHEMA, model=model)
    assert session.calls == []


def test_unavailable_preferred_model_reports_actual_model_without_changing_cache_identity():
    provider, session = client(Response({"error": {"code": 404}}, 404), Response(answer()))
    assert provider.generate_json("Example", SCHEMA) == {"answer": "ok"}
    assert [url.split("/")[-1] for url, _ in session.calls] == [
        "gemini-3.8-flash:generateContent", "gemini-3.5-flash-lite:generateContent"]
    assert session.calls[1][1]["json"]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    assert provider.text_model == provider.model == "gemini-3.5-flash-lite"
    assert gemini.identity()["text_model"] == "gemini-3.8-flash"


def test_response_thoughts_are_not_mistaken_for_final_json():
    provider, _ = client(Response(answer(parts=[
        {"thought": True, "text": '{"answer":"hidden"}'}, {"text": '{"answer":"visible"}'}])))
    assert provider.generate_json("Example", SCHEMA) == {"answer": "visible"}


@pytest.mark.parametrize("data", [
    {"promptFeedback": {"blockReason": "SAFETY"}},
    {"candidates": []},
    answer(finish="MAX_TOKENS"),
    answer(finish="SAFETY"),
    answer(parts=[{"thought": True, "text": '{"answer":"only thought"}'}]),
    answer('{"answer":'),
    answer('{"wrong":"missing required value"}'),
    answer('{"answer":3}'),
    answer('{"answer":"ok","extra":"not allowed"}'),
    answer('{"answer":NaN}'),
    answer('["not an object"]'),
])
def test_blocked_truncated_thought_only_or_invalid_json_falls_back(data):
    provider, _ = client(Response(data))
    with pytest.raises(gemini.GeminiUnavailable):
        provider.generate_json("Example", SCHEMA)


def test_images_support_paths_and_pil_without_changing_original(tmp_path):
    path = tmp_path / "large.png"
    original = Image.new("RGBA", (1600, 900), (120, 20, 10, 90))
    original.save(path)
    provider, session = client(Response(answer()))
    provider.generate_json("Describe", SCHEMA, images=(path, original))
    parts = session.calls[0][1]["json"]["contents"][0]["parts"]
    for part in parts[:2]:
        inline = part["inline_data"]
        assert inline["mime_type"] == "image/jpeg"
        raw = base64.b64decode(inline["data"])
        with Image.open(io.BytesIO(raw)) as small:
            assert small.mode == "RGB" and max(small.size) <= 768
    assert original.mode == "RGBA" and original.size == (1600, 900)


@pytest.mark.parametrize("url", [
    "http://youtu.be/abcdefghijk",
    "https://youtube.com.evil.example/watch?v=abcdefghijk",
    "https://user:pass@youtube.com/watch?v=abcdefghijk",
    "https://youtube.com/watch?v=short",
    "https://youtube.com/watch?v=abcdefghijk&v=12345678901",
    "file:///abcdefghijk",
])
def test_video_urls_are_validated_before_request(url):
    provider, session = client()
    with pytest.raises(gemini.GeminiUnavailable, match="invalid_youtube_url"):
        provider.generate_json("Example", SCHEMA, youtube_url=url, video_duration=30)
    assert not session.calls


def test_video_duration_and_shared_budget_are_required(monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_VIDEO_SECONDS", "35")
    provider, session = client(Response(answer()))
    with pytest.raises(gemini.GeminiUnavailable, match="video_duration"):
        provider.generate_json("Example", SCHEMA, youtube_url="https://youtu.be/abcdefghijk")
    provider.generate_json("Example", SCHEMA, youtube_url="https://youtu.be/abcdefghijk", video_duration=20)
    part = session.calls[0][1]["json"]["contents"][0]["parts"][0]
    assert part == {"file_data": {"file_uri": "https://www.youtube.com/watch?v=abcdefghijk"}}
    second, _ = client()
    with pytest.raises(gemini.GeminiUnavailable, match="video_budget"):
        second.generate_json("Example", SCHEMA, youtube_url="https://youtu.be/abcdefghijk", video_duration=20)
    assert len(session.calls) == 1 and gemini._BUDGET.video_seconds == 20


def test_grounding_preserves_actual_supports_and_source_indices():
    grounding = {
        "groundingChunks": [{"maps": {}}, {"web": {"uri": "https://example.com/film", "title": "Official film"}}],
        "groundingSupports": [{"segment": {"text": "A pilot returns.", "startIndex": 0, "endIndex": 16},
                               "groundingChunkIndices": [1]}],
        "webSearchQueries": ["Example film"], "searchEntryPoint": {"renderedContent": "<div>Search</div>"},
    }
    provider, session = legacy_client(Response(answer("A pilot returns.", grounding=grounding)))
    result = provider.grounded_text("Research the verified film")
    assert result["text"] == "A pilot returns."
    assert result["grounding"] == grounding
    assert result["sources"] == [{"url": "https://example.com/film", "uri": "https://example.com/film",
                                  "title": "Official film", "index": 1}]
    url, call = session.calls[0]
    assert "gemini-2.5-flash:" in url
    assert call["json"]["tools"] == [{"google_search": {}}]
    assert call["json"]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 512}
    assert "responseMimeType" not in call["json"]["generationConfig"]


def test_ungrounded_answer_does_not_invent_sources():
    provider, _ = legacy_client(Response(answer("Unverified answer")))
    result = provider.grounded_text("Research")
    assert result == {"text": "Unverified answer", "sources": [], "grounding": {}}


def test_audio_preserves_spoken_text_style_voice_and_wav_bytes():
    raw = b"RIFF" + b"\0" * 40
    response = answer(parts=[{"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(raw).decode()}}])
    provider, session = client(Response(response))
    text = "Hold the line. <short pause> We are going home."
    audio, mime = provider.generate_audio(text, "voice_warguy123", style="Controlled anger")
    assert audio == raw and mime == "audio/wav"
    payload = session.calls[0][1]["json"]
    assert payload["contents"][0]["parts"] == [{"text": text, "speechMetadata": {"style": "Controlled anger"}}]
    assert payload["generationConfig"]["speechConfig"]["voiceConfig"] == {"voice": "voice_warguy123"}
    assert payload["generationConfig"]["responseModalities"] == ["AUDIO"]


def test_audio_never_silently_substitutes_a_preset_voice():
    provider, session = client()
    with pytest.raises(gemini.GeminiUnavailable, match="registered_character"):
        provider.generate_audio("Hello", "Kore")
    assert not session.calls


@pytest.mark.parametrize("parts", [
    [{"text": "No audio"}],
    [{"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}}],
    [{"inlineData": {"mimeType": "audio/wav", "data": "%%%invalid%%%"}}],
])
def test_bad_audio_is_unavailable(parts):
    provider, _ = client(Response(answer(parts=parts)))
    with pytest.raises(gemini.GeminiUnavailable):
        provider.generate_audio("Hello", "voice_test123")


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failure_opens_circuit_without_leaking_key(status, tmp_path):
    data = {"error": {"message": "test-gemini-secret body must stay private"}}
    provider, session = client(Response(data, status))
    for _ in range(2):
        with pytest.raises(gemini.GeminiUnavailable) as caught:
            provider.generate_json("Private prompt", SCHEMA)
        assert "secret" not in str(caught.value)
    assert len(session.calls) == 1
    log = (tmp_path / "usage.jsonl").read_text()
    assert "test-gemini-secret" not in log and "Private prompt" not in log and "body must" not in log


def test_zero_quota_does_not_retry_and_circuits(monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "2")
    provider, session = client(Response({"error": {"details": [{"quotaValue": 0}]}}, 429))
    with pytest.raises(gemini.GeminiUnavailable, match="quota_exhausted"):
        provider.generate_json("Example", SCHEMA)
    with pytest.raises(gemini.GeminiUnavailable, match="quota_exhausted"):
        provider.generate_json("Example again", SCHEMA)
    assert len(session.calls) == 1


def test_transient_retry_is_bounded_honors_retry_info_and_counts_budget(monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "1")
    sleeps = []
    monkeypatch.setattr(gemini.time, "sleep", sleeps.append)
    provider, session = client(
        Response({"error": {"details": [{"retryDelay": "2s"}]}}, 429),
        Response(answer()))
    assert provider.generate_json("Example", SCHEMA) == {"answer": "ok"}
    assert sleeps == [2] and len(session.calls) == 2 and gemini._BUDGET.requests == 2


def test_transport_errors_are_safe_and_circuit_after_bounded_retry(monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "1")
    monkeypatch.setattr(gemini.time, "sleep", lambda seconds: None)
    provider, session = client(requests.ConnectionError("test-gemini-secret"),
                               requests.Timeout("private response body"))
    with pytest.raises(gemini.GeminiUnavailable, match="transport_unavailable"):
        provider.generate_json("Example", SCHEMA)
    with pytest.raises(gemini.GeminiUnavailable):
        provider.generate_json("Example", SCHEMA)
    assert len(session.calls) == 2


def test_budget_is_atomic_and_shared_between_threads(monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_REQUESTS", "3")
    provider, session = client()
    def run(_):
        try:
            provider.generate_json("Example", SCHEMA)
            return True
        except gemini.GeminiUnavailable:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(12)))
    assert sum(results) == 3 and len(session.calls) == 3 and gemini._BUDGET.requests == 3


def test_usage_log_redacts_secrets_and_drops_arbitrary_details(tmp_path):
    gemini.record_event("speech", "gemini", "fallback",
                        "test-gemini-secret\nhttps://example.com?key=test-gemini-secret voicekey_sensitive",
                        details={"prompt": "private prompt", "headers": {"key": "secret"},
                                 "tokens": 5, "not_finite": float("nan"), "model": {"bad": "value"}})
    text = (tmp_path / "usage.jsonl").read_text()
    row = json.loads(text)
    assert row["details"] == {"tokens": 5}
    assert "test-gemini-secret" not in text and "private prompt" not in text
    assert "voicekey_sensitive" not in text and "https://" not in text


def test_usage_log_is_bounded_and_cannot_break_fallback(monkeypatch, tmp_path):
    path = tmp_path / "usage.jsonl"
    path.write_text("x" * 4096)
    monkeypatch.setenv("GEMINI_USAGE_MAX_BYTES", "4096")
    gemini.record_event("test", "gemini", "fallback")
    assert path.stat().st_size == 4096
    monkeypatch.setenv("GEMINI_USAGE_REPORT", str(tmp_path))
    gemini.record_event("test", "gemini", "fallback")  # directory is harmless


@pytest.mark.parametrize("status,body,reason", [
    (429, {"error": {"details": [{"quotaValue": 0}]}}, "quota_exhausted"),
    (503, {"error": {"status": "UNAVAILABLE"}}, "service_unavailable"),
])
def test_research_model_circuit_does_not_disable_independent_text_model(status, body, reason):
    provider, session = legacy_client(Response(body, status), Response(answer()))
    with pytest.raises(gemini.GeminiUnavailable, match=reason):
        provider.grounded_text("Research the verified film")
    # Repeating this model is suppressed, preserving requests for other models.
    with pytest.raises(gemini.GeminiUnavailable, match=reason):
        provider.grounded_text("Research another film")
    assert provider.generate_json("Write from already verified facts", SCHEMA) == {"answer": "ok"}
    assert [url.split("/")[-1] for url, _ in session.calls] == [
        "gemini-2.5-flash:generateContent", "gemini-3.8-flash:generateContent"]
    assert gemini._BUDGET.blocked == ""
    assert gemini._BUDGET.blocked_models["gemini-2.5-flash"] == reason


def test_slow_primary_model_falls_back_without_disabling_grounded_research(tmp_path):
    provider, session = legacy_client(
        requests.ReadTimeout("private URL and test-gemini-secret must stay private"),
        Response(answer()), Response(answer("A pilot returns.")))
    configured_identity = gemini.identity()
    assert provider.generate_json("Write from verified facts", SCHEMA) == {"answer": "ok"}
    assert provider.text_model == provider.vision_model == "gemini-3.5-flash-lite"
    assert gemini.identity() == configured_identity
    assert provider.grounded_text("Research the next film")["text"] == "A pilot returns."
    assert [url.split("/")[-1] for url, _ in session.calls] == [
        "gemini-3.8-flash:generateContent", "gemini-3.5-flash-lite:generateContent",
        "gemini-2.5-flash:generateContent"]
    assert gemini._BUDGET.blocked == ""
    assert gemini._BUDGET.blocked_models["gemini-3.8-flash"] == "model_read_timeout"
    logs = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
    diagnostic = next(row for row in logs if row.get("details", {}).get("error_class"))
    assert diagnostic["details"]["error_class"] == "ReadTimeout"
    assert diagnostic["details"]["model"] == "gemini-3.8-flash"
    assert diagnostic["details"]["elapsed_ms"] >= 0
    assert "test-gemini-secret" not in json.dumps(logs)
    assert "private URL" not in json.dumps(logs)


def test_read_timeout_retry_and_fallback_respect_total_request_budget(monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "2")
    monkeypatch.setenv("GEMINI_MAX_REQUESTS", "1")
    monkeypatch.setattr(gemini.time, "sleep", lambda seconds: None)
    provider, session = client(requests.ReadTimeout("private response body"))
    with pytest.raises(gemini.GeminiUnavailable, match="request_budget_exhausted"):
        provider.generate_json("Example", SCHEMA)
    assert len(session.calls) == 1 and gemini._BUDGET.requests == 1


def test_http_failures_record_status_without_server_body(tmp_path):
    provider, _ = client(Response({"error": {"message": "private response test-gemini-secret"}}, 400))
    with pytest.raises(gemini.GeminiUnavailable, match="request_rejected"):
        provider.generate_json("Example", SCHEMA)
    rows = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
    diagnostic = next(row for row in rows if row["reason"] == "http_error")
    assert diagnostic["details"]["http_status"] == 400
    assert "private response" not in json.dumps(rows) and "test-gemini-secret" not in json.dumps(rows)


def test_public_research_mode_never_calls_google_search_by_default():
    provider, session = client()
    assert provider.research_mode == "public_sources"
    with pytest.raises(gemini.GeminiUnavailable, match="legacy_google_search_not_enabled"):
        provider.grounded_text("Research a film")
    assert session.calls == []


def test_legacy_search_requires_explicit_configured_mode(monkeypatch):
    monkeypatch.setenv("GEMINI_RESEARCH_MODE", "legacy_google_search")
    provider, session = client(Response(answer("Legacy sourced answer")))
    assert gemini.identity()["research_mode"] == provider.research_mode == "legacy_google_search"
    assert provider.grounded_text("Research")["text"] == "Legacy sourced answer"
    assert len(session.calls) == 1 and "gemini-2.5-flash:" in session.calls[0][0]


def test_modern_fallback_is_free_and_has_no_search_tool():
    provider, session = client(Response(answer()))
    assert provider.generate_json("Summarize fetched public sources", SCHEMA,
                                  model="gemini-3.5-flash-lite") == {"answer": "ok"}
    payload = session.calls[0][1]["json"]
    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    assert "tools" not in payload


@pytest.mark.parametrize("retries", [0, 1])
def test_exhausted_primary_service_failure_uses_modern_free_fallback(monkeypatch, retries, tmp_path):
    monkeypatch.setenv("GEMINI_MAX_RETRIES", str(retries))
    monkeypatch.setattr(gemini.time, "sleep", lambda seconds: None)
    failures = [Response({"error": {"status": "UNAVAILABLE", "message": "private body"}}, 503)
                for _ in range(retries + 1)]
    provider, session = client(*failures, Response(answer()))
    configured_identity = gemini.identity()
    assert provider.generate_json("Write from verified facts", SCHEMA) == {"answer": "ok"}
    assert [url.split("/")[-1] for url, _ in session.calls] == (
        ["gemini-3.8-flash:generateContent"] * (retries + 1)
        + ["gemini-3.5-flash-lite:generateContent"])
    assert provider.text_model == provider.vision_model == "gemini-3.5-flash-lite"
    assert gemini.identity() == configured_identity
    assert gemini._BUDGET.blocked == ""
    assert gemini._BUDGET.blocked_models["gemini-3.8-flash"] == "service_unavailable"
    rows = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert any(row["reason"] == "preferred_model_service_unavailable" for row in rows)
    assert "private body" not in json.dumps(rows)


def test_rejections_report_the_code_and_known_problem_words_but_never_the_text(tmp_path):
    provider, _ = client(Response({"error": {"status": "INVALID_ARGUMENT",
                                             "message": "YouTube video private response test-gemini-secret is not supported"}}, 400))
    with pytest.raises(gemini.GeminiUnavailable) as raised:
        provider.generate_json("Example", SCHEMA)
    assert "HTTP 400: INVALID_ARGUMENT" in str(raised.value) and "youtube" in str(raised.value)
    rows = json.dumps([json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()])
    assert "private response" not in rows and "test-gemini-secret" not in rows
