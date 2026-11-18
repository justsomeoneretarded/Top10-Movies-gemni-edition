"""Durable conservative estimates for the explicitly enabled, single paid trial.

This is an application estimate, not a Google billing limit. Unknown outcomes
retain their full reservation. No prompts, content, URLs or credentials persist.
Rates were checked 2026-11-18 against https://aistudio.google.com/docs/pricing.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import uuid

# Conservative standard rates include announced price increases. Cached inputs
# receive no discount here. No tools, grounding, TTS or other models are allowed.
RATES = {
    "gemini-4-carbon": (4.0, 20.0),
    "gemini-3.8-flash": (1.5, 7.5),
    "gemini-3.5-flash-lite": (0.5, 3.0),
    "gemini-3-pro-image": (2.0, 12.0),
}
IMAGE_MODEL = "gemini-3-pro-image"
IMAGE_2K_USD = 0.134
VALID_THROUGH = "2026-12-31"


class PaidBudgetExceeded(ValueError):
    """A bounded, nonsecret reason to stop before another paid generation."""


def enabled() -> bool:
    return (os.getenv("GEMINI_PAID_TRIAL", "").strip().lower() == "true"
            and os.getenv("GEMINI_FREE_ONLY", "true").strip().lower() == "false")


def _limit(name, default, maximum):
    try:
        value = float(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        raise PaidBudgetExceeded("paid_budget_invalid_configuration") from None
    if not math.isfinite(value) or value <= 0 or value > maximum:
        raise PaidBudgetExceeded("paid_budget_invalid_configuration")
    return value


def _path():
    return Path(os.getenv("GEMINI_PAID_BUDGET_PATH",
                          "work/runner-diagnostics/paid-budget.json"))


def _empty():
    return {"schema_version": 1, "estimate_only": True,
            "rates_verified": "2026-11-18", "rates_valid_through": VALID_THROUGH,
            "run_id": os.getenv("GITHUB_RUN_ID", "local"),
            "reservations": {}}


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _totals(ledger):
    rows = list(ledger["reservations"].values())
    ledger.update(requests=len(rows),
                  input_tokens=sum(row["input_tokens"] for row in rows),
                  output_tokens=sum(row["output_tokens"] for row in rows),
                  images=sum(row["model"] == IMAGE_MODEL for row in rows),
                  estimated_usd=round(sum(row["estimated_usd"] for row in rows), 12))
    return ledger


def _read(path):
    if not path.exists():
        return _totals(_empty())
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(ledger, dict) and ledger.get("schema_version") == 1
        assert ledger.get("run_id") == os.getenv("GITHUB_RUN_ID", "local")
        assert isinstance(ledger.get("reservations"), dict)
        for token, row in ledger["reservations"].items():
            assert isinstance(token, str) and isinstance(row, dict)
            assert row.get("id") == token and row.get("model") in RATES
            assert row.get("status") in {"reserved", "settled"}
            assert all(_integer(row.get(key)) for key in (
                "input_tokens", "max_output_tokens", "output_tokens"))
            cost = row.get("estimated_usd")
            assert isinstance(cost, (int, float)) and not isinstance(cost, bool)
            assert math.isfinite(cost) and cost >= 0
        return _totals(ledger)
    except (OSError, ValueError, TypeError, AssertionError, KeyError):
        raise PaidBudgetExceeded("paid_budget_ledger_invalid") from None


def _write(path, ledger):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(_totals(ledger), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _locked():
    path = _path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(path.suffix + ".lock").open("a", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            yield path
    except OSError:
        raise PaidBudgetExceeded("paid_budget_ledger_unavailable") from None


def snapshot() -> dict:
    with _locked() as path:
        return _read(path)


def reserve(model, input_tokens, max_output_tokens, stage, *, request_hash=None) -> dict:
    if not enabled():
        raise PaidBudgetExceeded("paid_budget_not_enabled")
    if datetime.now(timezone.utc).date().isoformat() > VALID_THROUGH:
        raise PaidBudgetExceeded("paid_budget_rates_expired")
    if model not in RATES or not _integer(input_tokens) or not _integer(max_output_tokens):
        raise PaidBudgetExceeded("paid_budget_invalid_request")
    if input_tokens <= 0 or max_output_tokens <= 0:
        raise PaidBudgetExceeded("paid_budget_invalid_request")
    cap = 8192 if model == IMAGE_MODEL else 32768 if model == "gemini-4-carbon" else 16384
    if max_output_tokens > cap:
        raise PaidBudgetExceeded("paid_budget_output_limit")
    if stage not in {"text", "vision", "image", "image_review", "research"}:
        raise PaidBudgetExceeded("paid_budget_unsupported_stage")
    image = model == IMAGE_MODEL
    if image != (stage == "image"):
        raise PaidBudgetExceeded("paid_budget_invalid_image_request")
    input_rate, output_rate = RATES[model]
    cost = input_tokens * input_rate / 1_000_000 + max_output_tokens * output_rate / 1_000_000
    if image:
        cost += IMAGE_2K_USD
    row = {"id": uuid.uuid4().hex, "model": model, "stage": stage,
           "input_tokens": input_tokens, "max_output_tokens": max_output_tokens,
           "output_tokens": max_output_tokens, "estimated_usd": round(cost, 12),
           "status": "reserved"}
    if request_hash is not None:
        if not isinstance(request_hash, str) or len(request_hash) != 64 or any(
                char not in "0123456789abcdef" for char in request_hash):
            raise PaidBudgetExceeded("paid_budget_invalid_request_hash")
        row["request_hash"] = request_hash
    with _locked() as path:
        ledger = _read(path)
        if ledger.get("blocked"):
            raise PaidBudgetExceeded("paid_budget_usage_exceeded_reservation")
        if request_hash and any(old.get("request_hash") == request_hash
                                and old["status"] == "reserved"
                                for old in ledger["reservations"].values()):
            raise PaidBudgetExceeded("paid_budget_duplicate_unknown_request")
        if ledger["requests"] + 1 > int(_limit("GEMINI_PAID_MAX_REQUESTS", 100, 100)):
            raise PaidBudgetExceeded("paid_budget_request_limit")
        if ledger["images"] + int(image) > 3:
            raise PaidBudgetExceeded("paid_budget_image_limit")
        if ledger["input_tokens"] + input_tokens > int(_limit(
                "GEMINI_PAID_MAX_INPUT_TOKENS", 2_000_000, 2_000_000)):
            raise PaidBudgetExceeded("paid_budget_input_limit")
        if ledger["output_tokens"] + max_output_tokens > int(_limit(
                "GEMINI_PAID_MAX_OUTPUT_TOKENS", 500_000, 2_000_000)):
            raise PaidBudgetExceeded("paid_budget_output_limit")
        if ledger["estimated_usd"] + cost > _limit("GEMINI_PAID_MAX_USD", 5.0, 5.0) + 1e-12:
            raise PaidBudgetExceeded("paid_budget_estimate_limit")
        ledger["reservations"][row["id"]] = row
        _write(path, ledger)
    return dict(row)


def settle(reservation, usage) -> dict:
    """Only complete, internally consistent provider usage releases a reservation."""
    token = reservation.get("id") if isinstance(reservation, dict) else None
    if not isinstance(usage, dict):
        return snapshot()
    fields = ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")
    if not all(_integer(usage.get(key)) for key in fields):
        return snapshot()
    thought = usage.get("thoughtsTokenCount", 0)
    cache = usage.get("cachedContentTokenCount", 0)
    if not _integer(thought) or not _integer(cache):
        return snapshot()
    prompt, output = usage["promptTokenCount"], usage["candidatesTokenCount"] + thought
    if prompt + output != usage["totalTokenCount"] or cache > prompt:
        return snapshot()
    with _locked() as path:
        ledger = _read(path)
        row = ledger["reservations"].get(token)
        if row is None:
            raise PaidBudgetExceeded("paid_budget_unknown_reservation")
        if row["status"] == "settled":
            return ledger
        input_rate, output_rate = RATES[row["model"]]
        # Image token accounting differs from text. Retain the complete image
        # reservation rather than guessing how many output tokens were image data.
        if row["model"] == IMAGE_MODEL:
            row["usage_verified"] = True
            if prompt > row["input_tokens"] or output > row["max_output_tokens"]:
                ledger["blocked"] = "usage_exceeded_reservation"
                cost = (prompt * input_rate / 1_000_000
                        + output * output_rate / 1_000_000 + IMAGE_2K_USD)
                row["estimated_usd"] = max(row["estimated_usd"], round(cost, 12))
                row["input_tokens"] = max(row["input_tokens"], prompt)
                row["output_tokens"] = max(row["output_tokens"], output)
            _write(path, ledger)
            return _totals(ledger)
        cost = prompt * input_rate / 1_000_000 + output * output_rate / 1_000_000
        if prompt > row["input_tokens"] or output > row["max_output_tokens"]:
            ledger["blocked"] = "usage_exceeded_reservation"
            row["estimated_usd"] = max(row["estimated_usd"], round(cost, 12))
            row["input_tokens"] = max(row["input_tokens"], prompt)
            row["output_tokens"] = max(row["output_tokens"], output)
        else:
            row.update(status="settled", input_tokens=prompt, output_tokens=output,
                       thought_tokens=thought, cached_tokens=cache,
                       estimated_usd=round(cost, 12))
        _write(path, ledger)
        return _totals(ledger)
