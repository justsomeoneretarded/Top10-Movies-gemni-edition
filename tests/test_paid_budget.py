"""Offline checks for the one-episode paid trial spend ledger."""
from __future__ import annotations

import importlib
import json

import pytest

from movie_pipeline import paid_budget


@pytest.fixture(autouse=True)
def paid_trial(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_PAID_TRIAL", "true")
    monkeypatch.setenv("GEMINI_FREE_ONLY", "false")
    monkeypatch.setenv("GEMINI_PAID_BUDGET_PATH", str(tmp_path / "paid-budget.json"))
    monkeypatch.setenv("GEMINI_PAID_MAX_USD", "5")
    monkeypatch.setenv("GEMINI_PAID_MAX_REQUESTS", "100")
    monkeypatch.setenv("GEMINI_PAID_MAX_INPUT_TOKENS", "100000")
    monkeypatch.setenv("GEMINI_PAID_MAX_OUTPUT_TOKENS", "500000")


@pytest.mark.parametrize("paid,free_only,expected", [
    ("true", "false", True),
    ("false", "false", False),
    ("true", "true", False),
    ("", "false", False),
    ("true", "", False),
])
def test_billing_requires_both_explicit_gates(monkeypatch, paid, free_only, expected):
    monkeypatch.setenv("GEMINI_PAID_TRIAL", paid)
    monkeypatch.setenv("GEMINI_FREE_ONLY", free_only)
    assert paid_budget.enabled() is expected


def test_reservation_precedes_spend_and_survives_module_reload():
    reservation = paid_budget.reserve("gemini-4-carbon", 1000, 2000, "text")
    before = paid_budget.snapshot()
    assert before["estimated_usd"] == pytest.approx(0.044)
    assert before["requests"] == 1
    assert before["reservations"][reservation["id"]]["status"] == "reserved"
    importlib.reload(paid_budget)
    assert paid_budget.snapshot() == before


def test_actual_usage_includes_thinking_and_releases_unused_reservation():
    reservation = paid_budget.reserve("gemini-4-carbon", 1000, 2000, "text")
    paid_budget.settle(reservation, {
        "promptTokenCount": 1000,
        "candidatesTokenCount": 100,
        "thoughtsTokenCount": 50,
        "totalTokenCount": 1150,
    })
    report = paid_budget.snapshot()
    assert report["estimated_usd"] == pytest.approx(0.007)
    assert report["output_tokens"] == 150
    assert report["requests"] == 1
    assert report["reservations"][reservation["id"]]["status"] == "settled"


@pytest.mark.parametrize("usage", [
    {},
    {"promptTokenCount": 1000, "candidatesTokenCount": 100},
    {"promptTokenCount": -1, "candidatesTokenCount": 100, "totalTokenCount": 99},
    {"promptTokenCount": 1000, "candidatesTokenCount": 100, "thoughtsTokenCount": 50,
     "totalTokenCount": 1100},
])
def test_missing_or_invalid_usage_cannot_refund_unknown_spend(usage):
    reservation = paid_budget.reserve("gemini-4-carbon", 1000, 2000, "text")
    before = paid_budget.snapshot()
    paid_budget.settle(reservation, usage)
    assert paid_budget.snapshot()["estimated_usd"] == before["estimated_usd"]
    assert paid_budget.snapshot()["reservations"][reservation["id"]]["status"] == "reserved"


def test_repeated_settlement_does_not_double_count():
    reservation = paid_budget.reserve("gemini-4-carbon", 1000, 2000, "text")
    usage = {"promptTokenCount": 1000, "candidatesTokenCount": 100,
             "thoughtsTokenCount": 50, "totalTokenCount": 1150}
    paid_budget.settle(reservation, usage)
    before = paid_budget.snapshot()
    paid_budget.settle(reservation, usage)
    assert paid_budget.snapshot() == before


def test_budget_rejection_does_not_record_an_attempt(monkeypatch):
    monkeypatch.setenv("GEMINI_PAID_MAX_USD", ".001")
    with pytest.raises(paid_budget.PaidBudgetExceeded):
        paid_budget.reserve("gemini-4-carbon", 500, 32768, "text")
    assert paid_budget.snapshot()["requests"] == 0


def test_request_cap_persists_across_module_reload(monkeypatch):
    monkeypatch.setenv("GEMINI_PAID_MAX_REQUESTS", "1")
    paid_budget.reserve("gemini-4-carbon", 500, 1000, "text")
    importlib.reload(paid_budget)
    with pytest.raises(paid_budget.PaidBudgetExceeded):
        paid_budget.reserve("gemini-4-carbon", 500, 1000, "text")
    assert paid_budget.snapshot()["requests"] == 1


def test_corrupt_ledger_fails_closed(monkeypatch):
    from pathlib import Path
    import os
    path = Path(os.environ["GEMINI_PAID_BUDGET_PATH"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"reservations":', encoding="utf-8")
    with pytest.raises(paid_budget.PaidBudgetExceeded):
        paid_budget.reserve("gemini-4-carbon", 500, 1000, "text")


def test_ledger_contains_no_prompt_or_credentials(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "paid-private-test-key")
    reservation = paid_budget.reserve("gemini-4-carbon", 500, 1000, "text")
    rendered = json.dumps(paid_budget.snapshot())
    assert reservation["id"]
    assert "paid-private-test-key" not in rendered
    assert "prompt" not in rendered
