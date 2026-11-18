"""Offline budget and writer isolation contracts for one paid episode."""
from __future__ import annotations

import importlib

import pytest
import requests

from movie_pipeline import gemini, paid_budget
from movie_pipeline.core import PipelineError
from movie_pipeline.script import Writer


@pytest.fixture(autouse=True)
def isolated_trial(monkeypatch, tmp_path):
    settings = {
        "AI_BACKEND": "gemini",
        "GEMINI_PAID_TRIAL": "true",
        "GEMINI_FREE_ONLY": "false",
        "GEMINI_API_KEY": "private-contract-test-key",
        "GEMINI_TEXT_MODEL": "gemini-3.8-flash",
        "GEMINI_VISION_MODEL": "gemini-3.8-flash",
        "GEMINI_PAID_BUDGET_PATH": str(tmp_path / "paid-budget.json"),
        "GEMINI_USAGE_REPORT": str(tmp_path / "usage.jsonl"),
        "GEMINI_PAID_MAX_USD": "5",
        "GEMINI_PAID_MAX_REQUESTS": "100",
        "GEMINI_PAID_MAX_INPUT_TOKENS": "100000",
        "GEMINI_PAID_MAX_OUTPUT_TOKENS": "500000",
        "GITHUB_RUN_ID": "123456789",
    }
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("GEMINI_SCRIPT_MODEL", raising=False)
    monkeypatch.setattr(gemini, "_BUDGET", gemini._Budget())
    monkeypatch.setattr(gemini, "_CLIENT", None)
    monkeypatch.setattr(gemini, "_CLIENT_SIGNATURE", None)
    monkeypatch.setattr(gemini, "_ACTUAL_MODELS", {})
    monkeypatch.setattr(gemini, "_KEY_INDEX", [0])
    monkeypatch.setattr(requests, "post",
                        lambda *_a, **_k: pytest.fail("Paid writing must not call a local backend"))
    monkeypatch.setattr(requests.Session, "post",
                        lambda *_a, **_k: pytest.fail("Unexpected live API request"))


def test_three_generated_images_is_a_hard_episode_limit():
    for _ in range(3):
        paid_budget.reserve("gemini-3-pro-image", 1000, 4096, "image")
    before = paid_budget.snapshot()
    assert before["images"] == 3
    with pytest.raises(paid_budget.PaidBudgetExceeded):
        paid_budget.reserve("gemini-3-pro-image", 1000, 4096, "image")
    assert paid_budget.snapshot() == before


def test_image_flat_cost_and_text_allowance_are_never_refunded():
    reservation = paid_budget.reserve("gemini-3-pro-image", 1000, 4096, "image")
    before = paid_budget.snapshot()
    assert before["estimated_usd"] == pytest.approx(0.185152)
    paid_budget.settle(reservation, {
        "promptTokenCount": 1000,
        "candidatesTokenCount": 100,
        "thoughtsTokenCount": 0,
        "totalTokenCount": 1100,
    })
    report = paid_budget.snapshot()
    assert report["estimated_usd"] == before["estimated_usd"]
    assert report["reservations"][reservation["id"]]["status"] == "reserved"


@pytest.mark.parametrize("limit_name,input_tokens,output_tokens", [
    ("GEMINI_PAID_MAX_INPUT_TOKENS", 101, 50),
    ("GEMINI_PAID_MAX_OUTPUT_TOKENS", 50, 101),
])
def test_token_caps_reject_before_recording_generation(monkeypatch, limit_name,
                                                      input_tokens, output_tokens):
    monkeypatch.setenv(limit_name, "100")
    with pytest.raises(paid_budget.PaidBudgetExceeded):
        paid_budget.reserve("gemini-4-carbon", input_tokens, output_tokens, "text")
    assert paid_budget.snapshot()["requests"] == 0


@pytest.mark.parametrize("prompt,candidate,thought", [
    (1100, 100, 0),
    (1000, 1001, 50),
])
def test_actual_usage_above_reservation_blocks_future_spend(prompt, candidate, thought):
    reservation = paid_budget.reserve("gemini-4-carbon", 1000, 1000, "text")
    paid_budget.settle(reservation, {
        "promptTokenCount": prompt,
        "candidatesTokenCount": candidate,
        "thoughtsTokenCount": thought,
        "totalTokenCount": prompt + candidate + thought,
    })
    assert paid_budget.snapshot()["blocked"]
    importlib.reload(paid_budget)
    with pytest.raises(paid_budget.PaidBudgetExceeded):
        paid_budget.reserve("gemini-4-carbon", 50, 50, "text")
    assert paid_budget.snapshot()["requests"] == 1


def test_writer_cloud_failure_never_loads_or_uses_local_model(monkeypatch):
    class UnavailableClient:
        def generate_json(self, *_args, **_kwargs):
            raise gemini.GeminiUnavailable("model_read_timeout")

    monkeypatch.setattr(gemini, "get_client", lambda: UnavailableClient())
    monkeypatch.setattr(Writer, "_prepare_local_model",
                        lambda *_: pytest.fail("Paid writing must fail without local substitution"))
    writer = Writer()
    with pytest.raises(PipelineError, match="Paid trial writing failed"):
        writer.ask("Write the review.", {"type": "object"})
    assert writer._using_local is False


@pytest.mark.parametrize("explicit_override", [False, True])
def test_carbon_override_is_confined_to_writer(monkeypatch, explicit_override):
    if explicit_override:
        monkeypatch.setenv("GEMINI_SCRIPT_MODEL", "gemini-4-carbon")
    client = gemini.GeminiClient()
    assert client.text_model == "gemini-3.8-flash"
    calls = []

    def generate_json(prompt, schema, **kwargs):
        calls.append(kwargs)
        return {"answer": "ok"}

    monkeypatch.setattr(client, "generate_json", generate_json)
    monkeypatch.setattr(gemini, "get_client", lambda: client)
    assert Writer().ask("Write the review.", {"type": "object"}) == {"answer": "ok"}
    assert len(calls) == 1
    assert calls[0]["model"] == "gemini-4-carbon"
    assert calls[0]["max_output_tokens"] == 32768
    assert calls[0]["thinking"] == "low"
    assert client.text_model == "gemini-3.8-flash"
    assert gemini.identity()["text_model"] == "gemini-3.8-flash"
