"""Bounded Gemini Developer API transport; local backends remain the fallback.

Uses generateContent over requests, without SDK startup or network at import.
Only models documented with a free API tier are permitted. The API key must
belong to an unbilled Free Tier project: Google has no per-request free-only flag.
See https://ai.google.dev/gemini-api/docs/pricing (verified 2026-11-07).
"""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import parse_qs, urlsplit

import requests

SCHEMA_VERSION = 1
FREE_TEXT_MODELS = frozenset({
    "gemini-3.8-flash", "gemini-3.5-flash-lite",
})
FREE_TTS_MODELS = frozenset({
    "gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts",
})
# These Search models are available only to grandfathered prior users.
GROUNDING_MODELS = frozenset({"gemini-2.5-flash", "gemini-2.5-flash-lite"})
FREE_MODELS = FREE_TEXT_MODELS | FREE_TTS_MODELS | GROUNDING_MODELS
_ACTUAL_MODELS = {}
_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
_LOG_LOCK = threading.Lock()
_CLIENT_LOCK = threading.Lock()
_CLIENT = None
_CLIENT_SIGNATURE = None


class GeminiUnavailable(ValueError):
    """A safe, nonsecret reason to continue with an existing local backend."""


@dataclass
class _Budget:
    lock: object = field(default_factory=threading.RLock)
    requests: int = 0
    video_seconds: float = 0.0
    next_request: float = 0.0
    blocked: str = ""
    blocked_models: dict = field(default_factory=dict)


_BUDGET = _Budget()


def _key():
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API") or "").strip()


def enabled():
    """True when configured, without probing the network or exposing the key."""
    return os.getenv("AI_BACKEND", "gemini").strip().lower() != "local" and bool(_key())


def _models():
    text = os.getenv("GEMINI_TEXT_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"
    return {
        "text_model": text.strip(),
        "research_model": os.getenv("GEMINI_RESEARCH_MODEL", "gemini-2.5-flash").strip(),
        "vision_model": os.getenv("GEMINI_VISION_MODEL", text).strip(),
        "tts_model": os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts").strip(),
    }


def _research_mode():
    return os.getenv("GEMINI_RESEARCH_MODE", "public_sources").strip().lower()


def identity():
    """JSON-serializable cache identity; includes neither key nor voice tokens."""
    return {
        "backend": "gemini" if enabled() else "local",
        "research_mode": _research_mode(),
        **_models(),
        "schema_version": SCHEMA_VERSION,
        "free_only": True,
    }


def _number(name, default, minimum, maximum):
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return max(minimum, min(maximum, value))


def _redact(value):
    value = str(value)
    for secret in (_key(), os.getenv("GEMINI_API_KEY", ""), os.getenv("GEMINI_API", "")):
        if secret:
            value = value.replace(secret, "[redacted]")
    value = re.sub(r"AIza[A-Za-z0-9_-]{20,}", "[redacted]", value)
    value = re.sub(r"voicekey_[A-Za-z0-9_-]+", "[redacted]", value)
    value = re.sub(r"https?://\S+", "[url]", value)
    return re.sub(r"[\x00-\x1f\x7f]", " ", value)


def _label(value, maximum=80):
    return re.sub(r"[^A-Za-z0-9_.: /()\[\]-]", "_", _redact(value))[:maximum]


def record_event(stage, backend, status, reason="", details=None):
    """Append bounded, sanitized provenance; never write prompts or API bodies."""
    row = {
        "time": datetime.now(timezone.utc).isoformat(),
        "stage": _label(stage), "backend": _label(backend), "status": _label(status),
        "reason": _label(reason, 160),
    }
    safe = {}
    if isinstance(details, dict):
        for key, value in details.items():
            if key == "model" and isinstance(value, str) and value in FREE_MODELS:
                safe["model"] = value
            elif (key == "error_class" and isinstance(value, str)
                  and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", value)
                  and value == _redact(value)):
                safe["error_class"] = value
            elif (isinstance(key, str) and re.fullmatch(r"[a-z_]{1,40}", key)
                  and isinstance(value, (int, float, bool))
                  and abs(value) <= 1e12 and math.isfinite(value)):
                safe[key] = value
    if safe:
        row["details"] = safe
    try:
        path = Path(os.getenv("GEMINI_USAGE_REPORT", "work/gemini-usage.jsonl"))
        limit = int(_number("GEMINI_USAGE_MAX_BYTES", 2_000_000, 4096, 10_000_000))
        with _LOG_LOCK:
            if path.exists() and path.stat().st_size >= limit:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
    except (OSError, ValueError, TypeError):
        pass  # Reporting must never break a working local fallback.


def _unavailable(reason):
    record_event("provider", "gemini", "unavailable", reason)
    return GeminiUnavailable(reason)


def _check_model(model, allowed):
    if model not in allowed:
        raise _unavailable("model_not_available_on_free_tier")


def get_client():
    """Reuse transport in this process; budgets are shared even across clients."""
    global _CLIENT, _CLIENT_SIGNATURE
    if not enabled():
        raise _unavailable("gemini_disabled_or_key_missing")
    signature = (_key(), tuple(_models().items()), _research_mode())
    with _CLIENT_LOCK:
        if _CLIENT is None or signature != _CLIENT_SIGNATURE:
            _CLIENT = GeminiClient()
            _CLIENT_SIGNATURE = signature
        return _CLIENT


def _youtube_url(value):
    """Canonical public YouTube URL, rejecting hosts, credentials and bad IDs."""
    if not isinstance(value, str):
        raise _unavailable("invalid_youtube_url")
    try:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port:
            raise ValueError
        host = (parsed.hostname or "").lower()
        if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
            if parsed.path == "/watch":
                ids = parse_qs(parsed.query).get("v", [])
                video_id = ids[0] if len(ids) == 1 else ""
            else:
                match = re.fullmatch(r"/(?:shorts|embed)/([A-Za-z0-9_-]{11})/?", parsed.path)
                video_id = match.group(1) if match else ""
        elif host == "youtu.be":
            video_id = parsed.path.strip("/")
        else:
            video_id = ""
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError
        return "https://www.youtube.com/watch?v=" + video_id
    except (TypeError, ValueError):
        raise _unavailable("invalid_youtube_url") from None


def _image_part(path):
    try:
        from PIL import Image
        if isinstance(path, Image.Image):
            image = path.convert("RGB")
        else:
            with Image.open(Path(path)) as original:
                image = original.convert("RGB")
        with image:
            image.thumbnail((768, 768))
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=82, optimize=True)
            raw = output.getvalue()
        if not raw or len(raw) > 1_000_000:
            raise ValueError
    except (ImportError, OSError, TypeError, ValueError):
        raise _unavailable("image_unavailable") from None
    return {"inline_data": {
        "mime_type": "image/jpeg", "data": base64.b64encode(raw).decode("ascii"),
    }}


def _candidate(data):
    feedback = data.get("promptFeedback", {})
    if isinstance(feedback, dict) and feedback.get("blockReason") not in (None, "", "BLOCK_REASON_UNSPECIFIED"):
        raise _unavailable("response_blocked")
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        raise _unavailable("response_missing")
    candidate = candidates[0]
    if candidate.get("finishReason") != "STOP":
        raise _unavailable("response_incomplete")
    content = candidate.get("content", {})
    if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
        raise _unavailable("response_missing")
    return candidate


def _text(candidate):
    value = "".join(
        part["text"] for part in candidate["content"]["parts"]
        if isinstance(part, dict) and isinstance(part.get("text"), str) and not part.get("thought")
    ).strip()
    if not value:
        raise _unavailable("response_has_no_text")
    return value


def _reject_constant(_):
    raise ValueError("invalid_json_constant")


def _matches(value, schema, depth=0):
    """Validate the JSON Schema subset used by pipeline prompts."""
    if not isinstance(schema, dict) or depth > 40:
        return False
    if "anyOf" in schema and not any(_matches(value, item, depth + 1) for item in schema["anyOf"]):
        return False
    types = schema.get("type")
    if isinstance(types, str):
        types = [types]
    checks = {
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str), "null": value is None,
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
    }
    if types and not any(checks.get(kind, False) for kind in types):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if any(key not in value for key in schema.get("required", [])):
            return False
        for key, item in value.items():
            if key in properties and not _matches(item, properties[key], depth + 1):
                return False
            if key not in properties and schema.get("additionalProperties") is False:
                return False
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", 1e12):
            return False
        if "items" in schema and not all(_matches(item, schema["items"], depth + 1) for item in value):
            return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value) or value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            return False
    return True


def _permanent_quota(data):
    # Inspect structured errors internally, but never copy server messages to logs.
    error = json.dumps(data, ensure_ascii=True).lower()[:60000]
    return bool(re.search(r'"quota_?value"\s*:\s*"?0(?:\D|$)', error)) or any(token in error for token in (
        '"quotavalue": "0"', '"quota_value": "0"', "limit: 0", "limit is 0",
        "per_day", "perday", "requestsperday", "daily quota", "daily limit",
    ))


def _retry_delay(response, data, attempt):
    delay = None
    try:
        delay = float(response.headers.get("Retry-After", ""))
    except (AttributeError, TypeError, ValueError):
        pass
    error = data.get("error", {})
    details = error.get("details", []) if isinstance(error, dict) else []
    for detail in details if isinstance(details, list) else []:
        if isinstance(detail, dict) and "retryDelay" in detail:
            try:
                delay = float(str(detail["retryDelay"]).removesuffix("s"))
            except (TypeError, ValueError):
                pass
    if delay is None or not math.isfinite(delay):
        delay = float(2 ** attempt)
    return max(0.0, delay) if delay <= 30 else None


class GeminiClient:
    def __init__(self, api_key=None, session=None):
        if os.getenv("AI_BACKEND", "gemini").strip().lower() == "local":
            raise _unavailable("gemini_disabled_or_key_missing")
        self._api_key = (api_key or _key()).strip()
        if not self._api_key:
            raise _unavailable("gemini_disabled_or_key_missing")
        for name, model in _models().items():
            setattr(self, name, _ACTUAL_MODELS.get(model, model))
        self.model = self.text_model
        self.research_mode = _research_mode()
        self.session = session if session is not None else requests.Session()

    def _request(self, model, payload, stage, video_seconds=0.0):
        _check_model(model, FREE_MODELS)
        retries = int(_number("GEMINI_MAX_RETRIES", 1, 0, 2))
        timeout = (10, _number("GEMINI_TIMEOUT_SECONDS", 120, 10, 180))
        with _BUDGET.lock:
            if _BUDGET.blocked or model in _BUDGET.blocked_models:
                raise _unavailable(_BUDGET.blocked or _BUDGET.blocked_models[model])
            for attempt in range(retries + 1):
                if _BUDGET.requests >= int(_number("GEMINI_MAX_REQUESTS", 120, 1, 1000)):
                    _BUDGET.blocked = "request_budget_exhausted"
                    raise _unavailable(_BUDGET.blocked)
                if _BUDGET.video_seconds + video_seconds > _number("GEMINI_MAX_VIDEO_SECONDS", 3600, 1, 28800):
                    raise _unavailable("video_budget_exhausted")
                delay = max(0.0, _BUDGET.next_request - time.monotonic())
                if delay:
                    time.sleep(min(delay, 30))
                _BUDGET.requests += 1
                _BUDGET.video_seconds += video_seconds
                _BUDGET.next_request = time.monotonic() + _number("GEMINI_MIN_INTERVAL_SECONDS", 12, 0, 30)
                started = time.monotonic()
                try:
                    response = self.session.post(
                        _ENDPOINT + "/" + model + ":generateContent",
                        headers={"x-goog-api-key": self._api_key, "Content-Type": "application/json"},
                        json=payload, timeout=timeout,
                    )
                except requests.RequestException as error:
                    # Exception messages can contain credentials and request URLs.
                    # Log only the Python class, model and numeric diagnostics.
                    record_event(stage, "gemini", "request_failed", "transport_error", details={
                        "model": model, "error_class": type(error).__name__,
                        "attempt": attempt + 1, "request_count": _BUDGET.requests,
                        "elapsed_ms": round((time.monotonic() - started) * 1000),
                    })
                    if attempt < retries:
                        time.sleep(min(2 ** attempt, 30))
                        continue
                    if isinstance(error, requests.ReadTimeout):
                        # A slow model is not evidence that other models are offline.
                        _BUDGET.blocked_models[model] = "model_read_timeout"
                        raise _unavailable("model_read_timeout") from None
                    _BUDGET.blocked = "transport_unavailable"
                    raise _unavailable(_BUDGET.blocked) from None
                try:
                    data = response.json()
                except (TypeError, ValueError):
                    data = {}
                if not isinstance(data, dict):
                    data = {}
                status = response.status_code
                if status == 200 and data and "error" not in data:
                    usage = data.get("usageMetadata", {})
                    details = {"model": model, "http_status": status, "attempt": attempt + 1,
                               "request_count": _BUDGET.requests, "video_seconds": _BUDGET.video_seconds}
                    if isinstance(usage, dict):
                        for source, target in (("promptTokenCount", "input_tokens"),
                                               ("candidatesTokenCount", "output_tokens"),
                                               ("totalTokenCount", "total_tokens")):
                            if isinstance(usage.get(source), (int, float)):
                                details[target] = usage[source]
                    record_event(stage, "gemini", "received", details=details)
                    return data
                record_event(stage, "gemini", "request_failed", "http_error", details={
                    "model": model, "http_status": status, "attempt": attempt + 1,
                    "request_count": _BUDGET.requests,
                    "elapsed_ms": round((time.monotonic() - started) * 1000),
                })
                if status in (401, 403):
                    _BUDGET.blocked = "authentication_or_access_unavailable"
                    raise _unavailable(_BUDGET.blocked)
                if status == 404:
                    _BUDGET.blocked_models[model] = "model_unavailable"
                    raise _unavailable("model_unavailable")
                if status == 429 and _permanent_quota(data):
                    _BUDGET.blocked_models[model] = "quota_exhausted"
                    raise _unavailable("quota_exhausted")
                transient = status in (429, 500, 502, 503, 504)
                error_text = json.dumps(data).lower()[:60000]
                if any(word in error_text for word in ("consent", "voice_mismatch", "voice not found")):
                    transient = False
                if transient and attempt < retries:
                    delay = _retry_delay(response, data, attempt)
                    if delay is not None:
                        time.sleep(delay)
                        continue
                if transient:
                    reason = "quota_exhausted" if status == 429 else "service_unavailable"
                    _BUDGET.blocked_models[model] = reason
                    raise _unavailable(reason)
                # Google's text is never stored; only its error code and which known problem words it mentions.
                error = data.get("error") if isinstance(data.get("error"), dict) else {}
                code = str(error.get("status", ""))[:40] if re.fullmatch(r"[A-Z_]{3,40}", str(error.get("status", ""))) else ""
                lowered = str(error.get("message", "")).lower()
                hints = [word for word in ("youtube", "video", "file_uri", "fileuri", "mime", "schema", "json", "unsupported",
                                           "not supported", "public", "private", "too long", "too large", "duration",
                                           "token", "thinking", "model", "safety", "blocked", "unavailable", "permission")
                         if word in lowered][:6]
                record_event(stage, "gemini", "rejected_detail", f"http_{status}", details={"code": code, "hints": hints, "model": model})
                detail = ", ".join(part for part in (code, "mentions: " + "/".join(hints) if hints else "") if part)
                raise _unavailable(f"request_rejected (HTTP {status}{': ' + detail if detail else ''})")

    def generate_json(self, prompt, schema, *, model=None, max_output_tokens=8192,
                      images=(), youtube_url=None, video_duration=None, thinking="low", video_file=None):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 160000:
            raise _unavailable("invalid_prompt")
        if not isinstance(schema, dict):
            raise _unavailable("invalid_schema")
        images = tuple(images)
        if len(images) > 8:
            raise _unavailable("image_budget_exhausted")
        model = model or (self.vision_model if images or youtube_url or video_file else self.text_model)
        _check_model(model, FREE_TEXT_MODELS)
        parts = [_image_part(path) for path in images]
        video_seconds = 0.0
        if youtube_url is not None:
            url = _youtube_url(youtube_url)
            try:
                video_seconds = float(video_duration)
                if not math.isfinite(video_seconds) or not 0 < video_seconds <= 10800:
                    raise ValueError
            except (TypeError, ValueError):
                raise _unavailable("video_duration_required_or_too_long") from None
            parts.append({"file_data": {"file_uri": url}})
        elif video_file is not None:
            try:
                video_seconds = float(video_duration)
                if not math.isfinite(video_seconds) or not 0 < video_seconds <= 10800:
                    raise ValueError
            except (TypeError, ValueError):
                raise _unavailable("video_duration_required_or_too_long") from None
            parts.append({"file_data": {"mime_type": video_file["mime_type"], "file_uri": video_file["uri"]}})
        parts.append({"text": prompt})
        try:
            max_output_tokens = max(256, min(16384, int(max_output_tokens)))
            json.dumps(schema, allow_nan=False)
        except (TypeError, ValueError):
            raise _unavailable("invalid_schema_or_output_budget") from None
        config = {"maxOutputTokens": max_output_tokens,
                  "responseMimeType": "application/json", "responseJsonSchema": schema}
        if thinking not in {"low", "medium", "high"}:
            raise _unavailable("invalid_thinking_level")
        config["thinkingConfig"] = {"thinkingLevel": thinking}
        stage = "vision" if images or youtube_url or video_file else "text"
        payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": config}
        try:
            data = self._request(model, payload, stage, video_seconds)
        except GeminiUnavailable as error:
            fallback_reason = str(error)
            if fallback_reason not in {"model_unavailable", "model_read_timeout", "service_unavailable"} or model != "gemini-3.8-flash":
                raise
            # Only unavailable, persistently slow or exhausted server failures may
            # use this free alternative. Auth, quota, blocked content and connection
            # errors never switch models.
            config["thinkingConfig"] = {"thinkingLevel": "low"}
            model = "gemini-3.5-flash-lite"
            data = self._request(model, payload, stage, video_seconds)
            _ACTUAL_MODELS["gemini-3.8-flash"] = model
            if self.text_model == "gemini-3.8-flash":
                self.text_model = self.model = model
            if self.vision_model == "gemini-3.8-flash":
                self.vision_model = model
            fallback_reasons = {
                "model_unavailable": "preferred_model_unavailable",
                "model_read_timeout": "preferred_model_read_timeout",
                "service_unavailable": "preferred_model_service_unavailable",
            }
            record_event(stage, "gemini", "model_fallback", fallback_reasons[fallback_reason],
                         details={"model": model})
        text = _text(_candidate(data))
        try:
            value = json.loads(text, parse_constant=_reject_constant)
            valid = isinstance(value, dict) and _matches(value, schema)
        except (TypeError, ValueError, RecursionError):
            valid = False
        if not valid:
            raise _unavailable("invalid_or_incomplete_json")
        record_event(stage, "gemini", "success", details={"model": model})
        return value

    def grounded_text(self, prompt):
        # Search has no free entitlement for new Gemini API projects. The 2.5
        # allowance applies only to prior users and must be explicitly selected.
        if self.research_mode != "legacy_google_search":
            raise _unavailable("legacy_google_search_not_enabled")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 160000:
            raise _unavailable("invalid_prompt")
        _check_model(self.research_model, GROUNDING_MODELS)
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"maxOutputTokens": 8192, "thinkingConfig": {"thinkingBudget": 512}},
        }
        candidate = _candidate(self._request(self.research_model, payload, "research"))
        text = _text(candidate)
        grounding = candidate.get("groundingMetadata", {})
        if not isinstance(grounding, dict):
            grounding = {}
        sources = []
        chunks = grounding.get("groundingChunks", [])
        for index, chunk in enumerate(chunks if isinstance(chunks, list) else []):
            web = chunk.get("web", {}) if isinstance(chunk, dict) else {}
            if not isinstance(web, dict):
                continue
            uri = web.get("uri", "")
            if not isinstance(uri, str) or len(uri) > 4096:
                continue
            try:
                parsed = urlsplit(uri)
                valid = parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password
            except ValueError:
                valid = False
            if valid:
                sources.append({"url": uri, "uri": uri, "title": str(web.get("title", ""))[:400], "index": index})
        record_event("research", "gemini", "success" if sources else "ungrounded",
                     details={"model": self.research_model, "source_count": len(sources)})
        return {"text": text, "sources": sources, "grounding": grounding}

    def upload_video(self, path, timeout=300):
        """Upload a local video to the Gemini Files API (free, kept 48 hours); returns {uri, mime_type}."""
        size = os.path.getsize(path)
        if not 0 < size <= 400 * 1024 * 1024:
            raise _unavailable("upload_too_large")
        base = "https://generativelanguage.googleapis.com"
        headers = {"x-goog-api-key": self._api_key}
        try:
            start = self.session.post(base + "/upload/v1beta/files", headers={
                **headers, "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(size), "X-Goog-Upload-Header-Content-Type": "video/mp4",
                "Content-Type": "application/json"}, json={"file": {"display_name": os.path.basename(str(path))[:120]}}, timeout=(10, 60))
            upload_url = start.headers.get("x-goog-upload-url") or start.headers.get("X-Goog-Upload-URL")
            if start.status_code != 200 or not upload_url:
                raise _unavailable(f"upload_start_failed (HTTP {start.status_code})")
            with open(path, "rb") as stream:
                done = self.session.post(upload_url, headers={
                    "Content-Length": str(size), "X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"},
                    data=stream, timeout=(10, 600))
            info = done.json().get("file", {}) if done.status_code == 200 else {}
            name, uri = info.get("name"), info.get("uri")
            if not name or not uri:
                raise _unavailable(f"upload_failed (HTTP {done.status_code})")
            deadline = time.monotonic() + timeout
            while info.get("state") == "PROCESSING" and time.monotonic() < deadline:
                time.sleep(5)
                info = self.session.get(f"{base}/v1beta/{name}", headers=headers, timeout=(10, 60)).json()
            if info.get("state") not in (None, "ACTIVE"):
                raise _unavailable(f"upload_not_active ({info.get('state')})")
            record_event("vision", "gemini", "uploaded", details={"bytes": size})
            return {"uri": uri, "mime_type": info.get("mimeType") or "video/mp4"}
        except requests.RequestException as error:
            raise _unavailable(f"upload_transport_error ({type(error).__name__})") from None

    def generate_audio(self, text, voice_id, style=""):
        _check_model(self.tts_model, FREE_TTS_MODELS)
        if not isinstance(voice_id, str) or not re.fullmatch(r"(?:voice_|voicekey_)[A-Za-z0-9_-]{1,500}", voice_id):
            raise _unavailable("registered_character_voice_required")
        if not isinstance(text, str) or not text.strip() or len(text) > 24000:
            raise _unavailable("invalid_speech_text")
        if not isinstance(style, str) or len(style) > 2000:
            raise _unavailable("invalid_speech_style")
        part = {"text": text}
        if style:
            part["speechMetadata"] = {"style": style}
        payload = {
            "contents": [{"role": "user", "parts": [part]}],
            "generationConfig": {
                "maxOutputTokens": 16384, "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"voice": voice_id}},
            },
        }
        candidate = _candidate(self._request(self.tts_model, payload, "speech"))
        for part in candidate["content"]["parts"]:
            if not isinstance(part, dict) or part.get("thought"):
                continue
            inline = part.get("inlineData", part.get("inline_data", {}))
            if not isinstance(inline, dict):
                continue
            mime = inline.get("mimeType", inline.get("mime_type", ""))
            if not isinstance(mime, str) or not mime.lower().startswith("audio/"):
                continue
            try:
                raw = base64.b64decode(inline.get("data", ""), validate=True)
            except (TypeError, ValueError, binascii.Error):
                raise _unavailable("invalid_audio_response") from None
            if not raw or len(raw) > 24_000_000:
                raise _unavailable("invalid_audio_response")
            record_event("speech", "gemini", "success", details={"model": self.tts_model})
            return raw, mime
        raise _unavailable("response_has_no_audio")
