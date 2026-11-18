"""Fail closed on paid-trial access; share one spend ledger with production."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request

from movie_pipeline import gemini, paid_budget

REPORT = Path("work/gemini-preflight-paid-trial.json")
MODELS = ("gemini-4-carbon", "gemini-3.8-flash", "gemini-3-pro-image")


def verify_config():
    if not gemini.paid_trial() or os.getenv("AI_BACKEND") != "gemini":
        raise ValueError("paid_trial_requires_explicit_gates")
    if os.getenv("GITHUB_RUN_ATTEMPT", "1") != "1":
        raise ValueError("paid_trial_rerun_disabled")
    if not gemini._keys():
        raise ValueError("paid_secret_missing")
    if gemini.script_model() != MODELS[0]:
        raise ValueError("paid_script_model_mismatch")
    if any(os.getenv(name) != MODELS[1] for name in ("GEMINI_TEXT_MODEL", "GEMINI_VISION_MODEL", "GEMINI_VIDEO_MODEL")):
        raise ValueError("paid_routine_model_mismatch")
    if (os.getenv("NARRATOR") != "Warguy" or os.getenv("TTS_MODEL") != "original"
            or any(os.getenv(name, "").strip() for name in
                   ("GEMINI_VOICE_WARGUY", "GEMINI_VOICE_KIT_LASER", "GEMINI_VOICE_FEMALE"))):
        raise ValueError("trial_requires_original_chatterbox_warguy")
    if os.getenv("GEMINI_RESEARCH_MODE") != "public_sources":
        raise ValueError("trial_uses_verified_public_sources")


def listed_models():
    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
        headers={"x-goog-api-key": gemini._keys()[0], "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            payload = json.loads(response.read(2_000_001))
        methods = {item.get("name", "").removeprefix("models/"): item.get("supportedGenerationMethods", [])
                   for item in payload.get("models", []) if isinstance(item, dict)}
        return {name: "generateContent" in methods.get(name, []) for name in MODELS}
    except Exception:
        raise ValueError("paid_key_metadata_unavailable") from None


def save(report):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    report = {"schema_version": 1, "inference_verified": False,
              "research": "verified_public_sources", "voice": "Warguy / Original Chatterbox",
              "estimated_budget_usd": 5,
              "billing_note": "Application estimate and reservation guard, not a Google billing hard cap."}
    try:
        verify_config()
        report["listed_models"] = listed_models()
        if not all(report["listed_models"][m] for m in MODELS[:2]):
            raise ValueError("required_trial_model_not_listed")
        result = gemini.get_client().generate_json(
            'Return {"status":"ready"}. This is a paid-only structured-output contract check.',
            {"type": "object", "properties": {"status": {"type": "string", "enum": ["ready"]}},
             "required": ["status"]},
            model=MODELS[0], max_output_tokens=32768, thinking="low")
        if result != {"status": "ready"}:
            raise ValueError("paid_contract_unverified")
        report.update(inference_verified=True, paid_model=MODELS[0], budget=paid_budget.snapshot())
        save(report)
        print("Paid-only Carbon inference verified; one bounded production attempt enabled.", flush=True)
        print("Voice: Warguy / Original Chatterbox. Research uses verified public sources, not native Google Search.", flush=True)
        return 0
    except Exception as error:
        # Exception bodies may contain HTTP internals; report fixed reason codes only.
        known = {"paid_trial_requires_explicit_gates", "paid_trial_rerun_disabled", "paid_secret_missing",
                 "paid_script_model_mismatch", "paid_routine_model_mismatch",
                 "trial_requires_original_chatterbox_warguy", "trial_uses_verified_public_sources",
                 "paid_key_metadata_unavailable", "required_trial_model_not_listed", "paid_contract_unverified"}
        reason = str(error) if type(error) is ValueError and str(error) in known else "paid_generation_unavailable"
        report["failure"] = reason
        save(report)
        print("Paid trial cannot start: " + reason + ". No free-key or local substitute started.", flush=True)
        return 1


def receipts():
    destination = Path("output")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        (destination / "paid-trial-budget.json").write_text(
            json.dumps(paid_budget.snapshot(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if REPORT.is_file():
            shutil.copyfile(REPORT, destination / "paid-trial-preflight.json")
        print("Paid model and estimated-spend receipts attached.", flush=True)
        return 0
    except Exception:
        print("Receipt unavailable; original budget ledger remains in runner diagnostics.", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(receipts() if sys.argv[1:] == ["--receipts"] else main())
