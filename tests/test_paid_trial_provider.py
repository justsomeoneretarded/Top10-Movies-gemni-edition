"""Offline transport contracts for the isolated paid trial."""
from __future__ import annotations

import json

import pytest
import requests

from movie_pipeline import gemini, paid_budget

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}},
          "required": ["answer"], "additionalProperties": False}
USAGE = {"promptTokenCount": 500, "candidatesTokenCount": 200,
         "thoughtsTokenCount": 100, "totalTokenCount": 800}


class Response:
    def __init__(self, body, status=200):
        self.body, self.status_code, self.headers = body, status, {}

    def json(self):
        return self.body


class Session:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            pytest.fail("Unexpected request: paid calls must not retry or switch keys")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def answer(usage=USAGE):
    result = {"candidates": [{"finishReason": "STOP",
               "content": {"parts": [{"text": '{"answer":"ok"}'}]}}]}
    if usage is not None:
        result["usageMetadata"] = dict(usage)
    return result


@pytest.fixture(autouse=True)
def isolated_paid_provider(monkeypatch, tmp_path):
    for name, value in {
        "AI_BACKEND": "gemini",
        "GEMINI_PAID_TRIAL": "true",
        "GEMINI_FREE_ONLY": "false",
        "GEMINI_API_KEY": "paid-private-test-key",
        "GEMINI_API_KEY_2": "unused-secondary-test-key",
        "GEMINI_API_KEY_3": "unused-tertiary-test-key",
        "GEMINI_TEXT_MODEL": "gemini-4-carbon",
        "GEMINI_VISION_MODEL": "gemini-3.8-flash",
        "GEMINI_VIDEO_MODEL": "gemini-3.8-flash",
        "GEMINI_MAX_REQUESTS": "100",
        "GEMINI_MAX_RETRIES": "2",
        "GEMINI_TEXT_RETRIES": "5",
        "GEMINI_MIN_INTERVAL_SECONDS": "0",
        "GEMINI_PAID_MAX_USD": "5",
        "GEMINI_PAID_MAX_REQUESTS": "100",
        "GEMINI_PAID_MAX_INPUT_TOKENS": "100000",
        "GEMINI_PAID_MAX_OUTPUT_TOKENS": "500000",
        "GEMINI_PAID_BUDGET_PATH": str(tmp_path / "paid-budget.json"),
        "GEMINI_USAGE_REPORT": str(tmp_path / "usage.jsonl"),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("GEMINI_API", raising=False)
    monkeypatch.setattr(gemini, "_BUDGET", gemini._Budget())
    monkeypatch.setattr(gemini, "_CLIENT", None)
    monkeypatch.setattr(gemini, "_CLIENT_SIGNATURE", None)
    monkeypatch.setattr(gemini, "_ACTUAL_MODELS", {})
    monkeypatch.setattr(gemini, "_KEY_INDEX", [0])
    monkeypatch.setattr(gemini.time, "sleep", lambda *_: None)
    monkeypatch.setattr(requests.Session, "post",
                        lambda *_a, **_k: pytest.fail("Unexpected live Gemini request"))


def test_paid_key_is_the_only_key_even_when_free_rotation_keys_exist(monkeypatch):
    assert gemini._keys() == ["paid-private-test-key"]
    monkeypatch.delenv("GEMINI_API_KEY")
    assert gemini._keys() == []


def test_paid_carbon_counts_tokens_then_generates_once():
    session = Session(Response({"totalTokens": 500}), Response(answer()))
    provider = gemini.GeminiClient(session=session)
    assert provider.generate_json("Private trial prompt", SCHEMA,
                                  max_output_tokens=32768, thinking="low") == {"answer": "ok"}
    assert len(session.calls) == 2
    count_url, count_call = session.calls[0]
    generation_url, generation_call = session.calls[1]
    assert count_url.endswith("/gemini-4-carbon:countTokens")
    assert generation_url.endswith("/gemini-4-carbon:generateContent")
    counted = count_call["json"]["generateContentRequest"]
    assert counted["model"] == "models/gemini-4-carbon"
    assert counted["contents"] == generation_call["json"]["contents"]
    config = generation_call["json"]["generationConfig"]
    assert config["responseJsonSchema"] == SCHEMA
    assert config["maxOutputTokens"] == 32768
    assert config["thinkingConfig"] == {"thinkingLevel": "low"}
    assert generation_call["timeout"][1] >= 900
    assert generation_call["headers"]["x-goog-api-key"] == "paid-private-test-key"
    report = paid_budget.snapshot()
    assert report["requests"] == 1
    assert report["output_tokens"] == 300
    assert all(r["status"] == "settled" for r in report["reservations"].values())


@pytest.mark.parametrize("failure", [
    requests.ReadTimeout("Do not retry an unknown charged request"),
    Response({"error": {"status": "RESOURCE_EXHAUSTED"}}, 429),
])
def test_paid_timeout_or_rate_limit_does_not_retry_or_switch_key(failure):
    session = Session(Response({"totalTokens": 500}), failure)
    provider = gemini.GeminiClient(session=session)
    with pytest.raises(gemini.GeminiUnavailable):
        provider.generate_json("Private trial prompt", SCHEMA,
                               max_output_tokens=32768, thinking="low")
    assert len(session.calls) == 2
    assert gemini._KEY_INDEX[0] == 0
    report = paid_budget.snapshot()
    assert report["requests"] == 1
    assert all(r["status"] == "reserved" for r in report["reservations"].values())


def test_paid_response_without_usage_keeps_full_reservation():
    session = Session(Response({"totalTokens": 500}), Response(answer(usage=None)))
    provider = gemini.GeminiClient(session=session)
    assert provider.generate_json("Private trial prompt", SCHEMA,
                                  max_output_tokens=32768, thinking="low") == {"answer": "ok"}
    report = paid_budget.snapshot()
    assert report["requests"] == 1
    assert report["estimated_usd"] == pytest.approx(0.657588)
    assert all(r["status"] == "reserved" for r in report["reservations"].values())


def test_paid_spend_limit_stops_generation_after_counting(monkeypatch):
    monkeypatch.setenv("GEMINI_PAID_MAX_USD", ".001")
    session = Session(Response({"totalTokens": 500}))
    provider = gemini.GeminiClient(session=session)
    with pytest.raises(gemini.GeminiUnavailable):
        provider.generate_json("Private trial prompt", SCHEMA,
                               max_output_tokens=32768, thinking="low")
    assert len(session.calls) == 1
    assert session.calls[0][0].endswith(":countTokens")
    assert paid_budget.snapshot()["requests"] == 0


def test_free_mode_still_rejects_carbon_before_network(monkeypatch):
    monkeypatch.setenv("GEMINI_PAID_TRIAL", "false")
    monkeypatch.setenv("GEMINI_FREE_ONLY", "true")
    session = Session()
    provider = gemini.GeminiClient(session=session)
    with pytest.raises(gemini.GeminiUnavailable):
        provider.generate_json("Private trial prompt", SCHEMA, model="gemini-4-carbon")
    assert session.calls == []
    assert gemini.identity()["free_only"] is True


def test_paid_identity_and_usage_do_not_expose_credentials(monkeypatch):
    session = Session(Response({"totalTokens": 500}), Response(answer()))
    gemini.GeminiClient(session=session).generate_json(
        "Private trial prompt", SCHEMA, max_output_tokens=32768, thinking="low")
    from pathlib import Path
    import os
    report = Path(os.environ["GEMINI_USAGE_REPORT"]).read_text(encoding="utf-8")
    rendered = json.dumps(gemini.identity()) + report + json.dumps(paid_budget.snapshot())
    assert gemini.identity()["free_only"] is False
    assert "paid-private-test-key" not in rendered
    assert "unused-secondary-test-key" not in rendered
    assert "Private trial prompt" not in rendered


@pytest.mark.parametrize("prompt,blocked", [(520, False), (558, True)])
def test_counted_input_headroom_is_reserved_not_unbounded(prompt, blocked):
    usage = {"promptTokenCount": prompt, "candidatesTokenCount": 200,
             "thoughtsTokenCount": 100, "totalTokenCount": prompt + 300}
    session = Session(Response({"totalTokens": 500}), Response(answer(usage)))
    provider = gemini.GeminiClient(session=session)
    if blocked:
        with pytest.raises(gemini.GeminiUnavailable):
            provider.generate_json("Headroom check", SCHEMA, max_output_tokens=32768, thinking="low")
    else:
        assert provider.generate_json("Headroom check", SCHEMA, max_output_tokens=32768, thinking="low") == {"answer": "ok"}
    assert bool(paid_budget.snapshot().get("blocked")) is blocked
