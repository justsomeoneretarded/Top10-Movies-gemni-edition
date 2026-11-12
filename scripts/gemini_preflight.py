"""Check Gemini authentication and model metadata without generating content.

The probe records only bounded, credential-free status data. Model metadata
cannot establish whether a project has billing enabled; the provider separately
enforces the application's free-only model/tool allowlist and request budgets.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

_DEFAULTS = {
    "text": ("GEMINI_TEXT_MODEL", "gemini-3.8-flash"),
    "vision": ("GEMINI_VISION_MODEL", "gemini-3.8-flash"),
    "research": ("GEMINI_RESEARCH_MODEL", "gemini-2.5-flash"),
    "tts": ("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts"),
}


def _selected_models() -> dict[str, str]:
    result = {}
    for capability, (variable, default) in _DEFAULTS.items():
        value = os.getenv(variable, "").strip()
        if capability == "text" and not value:
            value = os.getenv("GEMINI_MODEL", "").strip()
        value = value or default
        if capability == "research" and os.getenv("GEMINI_RESEARCH_MODE", "public_sources").strip().lower() != "legacy_google_search":
            value = (os.getenv("GEMINI_TEXT_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-3.8-flash").strip()
        # Never echo arbitrary configuration or use it in an API URL.
        result[capability] = value if re.fullmatch(r"gemini-[a-z0-9._-]{1,100}", value) else "invalid-model-id"
    return result


def _save(report: dict) -> None:
    path = Path(os.getenv("GEMINI_PREFLIGHT_REPORT", "work/gemini-preflight.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _local_fallback(report: dict) -> int:
    if os.getenv("TMDB_READ_TOKEN", "").strip():
        # Job environment changes are inherited by bootstrap and every later step.
        env_file = os.getenv("GITHUB_ENV")
        if env_file:
            with open(env_file, "a", encoding="utf-8") as stream:
                stream.write("AI_BACKEND=local\n")
        report["fallback"] = "local-with-tmdb"
        _save(report)
        print("Gemini authentication check unavailable; using TMDB and the local backend.", flush=True)
        return 0
    report["fallback"] = "unavailable-without-tmdb"
    _save(report)
    print("Gemini authentication could not be verified. Configure a valid GEMINI_API secret or TMDB_READ_TOKEN before model setup.", flush=True)
    return 1


def main() -> int:
    report = {
        "schema_version": 1,
        "probe": "authenticated-model-metadata-only",
        "billing_verified": False,
        "inference_verified": False,
        "research_mode": os.getenv("GEMINI_RESEARCH_MODE", "public_sources").strip().lower(),
        "free_only_enforced_by_provider": True,
        "models": {},
    }
    if os.getenv("AI_BACKEND", "gemini").strip().lower() == "local":
        report["authentication"] = "skipped-local-selection"
        report["fallback"] = "selected-local"
        _save(report)
        print("Gemini preflight: Local backend selected.", flush=True)
        return 0
    from movie_pipeline.gemini import _keys
    keys = _keys()
    if not keys:
        report["authentication"] = "missing-key"
        return _local_fallback(report)
    payload, failure = None, "invalid-response"
    for number, api_key in enumerate(keys, 1):
        # Any working key is enough: the run switches keys when one's quota is used up.
        request = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
            headers={"x-goog-api-key": api_key, "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read(2_000_001))
            if isinstance(payload, dict) and isinstance(payload.get("models"), list):
                report["keys_configured"], report["verified_key_number"] = len(keys), number
                break
            payload, failure = None, "invalid-response"
        except urllib.error.HTTPError as exc:
            failure = ("invalid-or-denied" if exc.code in (401, 403) else
                       "rate-limited" if exc.code == 429 else "service-unavailable")
        except Exception:
            # Transport errors may contain the request headers or URL: never print
            # exceptions, response bodies, request objects, or credential values.
            failure = "network-or-response-unavailable"
    if payload is None:
        report["authentication"] = failure
        return _local_fallback(report)
    available = {}
    for item in payload["models"]:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            available[item["name"].removeprefix("models/")] = item.get("supportedGenerationMethods", [])
    report["authentication"] = "verified"
    for capability, model in _selected_models().items():
        report["models"][capability] = {
            "model": model,
            "status": "available" if "generateContent" in available.get(model, []) else "unavailable",
        }
    report["fallback"] = "provider-local-fallback-for-unavailable-capabilities"
    _save(report)
    count = sum(item["status"] == "available" for item in report["models"].values())
    print(f"Gemini authentication verified; {count}/4 configured model capabilities are listed. Unavailable capabilities use their existing fallbacks. Project billing status is not exposed by this metadata check.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
