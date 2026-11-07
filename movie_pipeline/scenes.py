"""Short AI descriptions of each trailer shot, used to match footage to narration.

Gemini, with a small local vision-model fallback through Ollama, describes the middle
frame of every analysed shot in a few plain words ("close-up of a woman in a
spacesuit, tense, red light"). A small local text-embedding model turns the
descriptions and each narration sentence into vectors, so a line about a
character's fear can find a frightened close-up. Everything is optional:
without descriptions the edit uses SigLIP image matching alone.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import requests

from . import gemini, gemini_editing
from .core import PipelineError, save_json

VISION_MODEL = os.getenv("SCENE_MODEL", "qwen3-vl:2b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-minilm")
VERSION = 3
_DEADLINE = None
_READY = False
_UNAVAILABLE = False
_LOCAL_VISION_READY = False
_LOCAL_VISION_UNAVAILABLE = False
_LAST_BACKEND = "ollama"
PROMPT = ("Describe this movie frame in at most 14 words for a video editor: who or what is shown, "
          "the action, the mood and the setting. Plain words, no names, no quotes.")


def enabled() -> bool:
    return os.getenv("SCENE_DESCRIPTIONS", "true").strip().lower() != "false"


def _url() -> str:
    return os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


def _pull(model: str):
    requests.post(_url() + "/api/pull", json={"model": model, "stream": False}, timeout=180).raise_for_status()


def cache_path(trailer: Path) -> Path:
    return trailer.with_suffix(".scenes.json")


def _shot_fingerprint(shots: list[dict]) -> str:
    import hashlib
    values = [[float(shot["start"]), float(shot["end"])] for shot in shots]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def _embedding_identity() -> dict:
    import hashlib
    return {"provider": "ollama", "model": EMBED_MODEL,
            "endpoint_sha256": hashlib.sha256(_url().encode()).hexdigest(),
            "normalization": "unit-l2-f32-v1"}


def load(trailer: Path, count: int, shots: list[dict] | None = None) -> dict | None:
    """Load only same-source, same-provider descriptions in the local embedding space."""
    try:
        data = json.loads(cache_path(trailer).read_text())
        if (not isinstance(data, dict) or data.get("version") != VERSION
                or data.get("provider_policy") != gemini_editing.frame_provider_identity(VISION_MODEL)
                or data.get("source") != gemini_editing.source_fingerprint(trailer)
                or data.get("embed_model") != EMBED_MODEL
                or data.get("embedding_provider") != "ollama"
                or data.get("embedding_identity") != _embedding_identity()):
            return None
        backends = data.get("backends")
        if not isinstance(backends, list) or not backends or any(b not in ("gemini", "ollama") for b in backends):
            return None
        texts, rows = data.get("descriptions"), data.get("embeddings")
        if (not isinstance(texts, list) or not isinstance(rows, list)
                or len(texts) != count or len(rows) != count
                or any(not isinstance(text, str) for text in texts)):
            return None
        vectors = np.asarray(rows, dtype=np.float32)
        if (vectors.ndim != 2 or vectors.shape[1] <= 0 or not np.isfinite(vectors).all()
                or data.get("embedding_dimensions") != vectors.shape[1]):
            return None
        if shots is not None and data.get("shots") != _shot_fingerprint(shots):
            return None
        return data
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def embed(texts: list[str]) -> np.ndarray:
    response = requests.post(_url() + "/api/embed", json={"model": EMBED_MODEL, "input": texts}, timeout=300)
    response.raise_for_status()
    vectors = np.asarray(response.json()["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-9)


def _describe(image) -> str:
    global _LOCAL_VISION_READY, _LOCAL_VISION_UNAVAILABLE, _LAST_BACKEND
    small = image.convert("RGB").copy()
    small.thumbnail((448, 252))
    schema = {"type": "object", "properties": {"description": {"type": "string", "maxLength": 160}},
              "required": ["description"]}
    data = gemini_editing.frame_json(small, PROMPT, schema, "scene_description", 1024)
    if data is not None:
        text = data.get("description")
        text = " ".join(text.split()[:14]).strip(" \"'")[:160] if isinstance(text, str) else ""
        if text:
            _LAST_BACKEND = "gemini"
            gemini.record_event("scene_description", "gemini", "success", details={"frames": 1})
            return text
        gemini.record_event("scene_description", "gemini", "fallback", "invalid description")
    _LAST_BACKEND = "ollama"
    if _LOCAL_VISION_UNAVAILABLE:
        raise ValueError("Local scene vision unavailable")
    if not _LOCAL_VISION_READY:
        try:
            _pull(VISION_MODEL)
        except requests.RequestException:
            _LOCAL_VISION_UNAVAILABLE = True
            raise
        _LOCAL_VISION_READY = True
    buffer = io.BytesIO()
    small.save(buffer, format="JPEG", quality=88)
    response = requests.post(_url() + "/api/generate", json={
        "model": VISION_MODEL, "prompt": PROMPT, "images": [base64.b64encode(buffer.getvalue()).decode()],
        "stream": False, "think": False, "keep_alive": "10m",
        "options": {"temperature": 0.2, "num_predict": 40, "num_ctx": 2048}}, timeout=25)
    response.raise_for_status()
    text = " ".join(str(response.json().get("response", "")).replace("\n", " ").split())
    return text.strip(" \"'")[:160]


def describe_trailer(trailer: Path, shots: list[dict]) -> dict | None:
    """Describe representative frames within a shared optional-analysis budget."""
    global _DEADLINE, _READY, _UNAVAILABLE
    if not enabled() or not shots:
        return None
    cached = load(trailer, len(shots), shots)
    if cached:
        gemini.record_event("scene_description", cached.get("backends", ["ollama"])[0], "cache_hit",
                            details={"described": len(cached.get("described", []))})
        return cached
    if _UNAVAILABLE:
        return None
    if _DEADLINE is None:
        try:
            seconds = float(os.getenv("SCENE_BUDGET_SECONDS", "420"))
        except ValueError:
            seconds = 420
        _DEADLINE = time.monotonic() + max(30, min(900, seconds if math.isfinite(seconds) else 420))
    if time.monotonic() >= _DEADLINE:
        return None
    from .shots import shot_frame
    try:
        limit = max(1, min(16, int(os.getenv("SCENE_SAMPLE_FRAMES", "6"))))
        # Mix representative positions and impact moments; never caption the same frame twice.
        spread = np.linspace(0, len(shots)-1, min(limit, len(shots)), dtype=int).tolist()
        ranked = sorted(range(len(shots)), key=lambda i: -float(shots[i].get("peak", 0)))[:limit//2]
        indices = list(dict.fromkeys([*ranked, *spread]))[:limit]
        descriptions = [""] * len(shots)
        backends = set()
        for index in indices:
            if time.monotonic() >= _DEADLINE:
                break
            try:
                descriptions[index] = _describe(shot_frame(trailer, shots, index))
                if descriptions[index]:
                    backends.add(_LAST_BACKEND)
            except (PipelineError, requests.RequestException, ValueError):
                continue
        known = [i for i, text in enumerate(descriptions) if text]
        if not known:
            return None
        # Vision can succeed without downloading a local vision model. Only local
        # text embeddings are shared with narration queries; never mix providers.
        if not _READY:
            _pull(EMBED_MODEL)
            _READY = True
        embedded = embed([descriptions[i] for i in known])
        if embedded.ndim != 2 or len(embedded) != len(known) or not np.isfinite(embedded).all():
            raise ValueError("Invalid local embedding matrix")
        vectors = np.zeros((len(shots), embedded.shape[1]), dtype=np.float32)
        vectors[known] = embedded
    except (requests.RequestException, ValueError, KeyError) as exc:
        _UNAVAILABLE = True
        print(f"Optional scene description unavailable ({type(exc).__name__}); retaining image matching.", flush=True)
        return None
    data = {"version": VERSION, "model": VISION_MODEL, "embed_model": EMBED_MODEL,
            "provider_policy": gemini_editing.frame_provider_identity(VISION_MODEL),
            "source": gemini_editing.source_fingerprint(trailer), "shots": _shot_fingerprint(shots),
            "embedding_provider": "ollama", "embedding_identity": _embedding_identity(),
            "embedding_dimensions": int(vectors.shape[1]),
            "backends": sorted(backends), "descriptions": descriptions,
            "embeddings": vectors.round(5).tolist(), "described": known}
    save_json(cache_path(trailer), data)
    return data


def unload():
    for model in (VISION_MODEL,):
        try:
            requests.post(_url() + "/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
        except requests.RequestException:
            pass




def rate_thumbnail(image, genre: str = "", genre_hint: str = "") -> dict | None:
    """Vision-model judgement of a frame as thumbnail material; None when the model is unavailable.

    ``genre_hint`` describes what the genre looks like ("fear, screaming, creatures, masks,
    blood, darkness"), so the model rates how instantly the frame reads as that genre.
    """
    small = image.convert("RGB").copy()
    small.thumbnail((512, 288))
    buffer = io.BytesIO()
    small.save(buffer, format="JPEG", quality=88)
    schema = {"type": "object", "properties": {
        "large_face": {"type": "boolean"}, "emotion_intensity": {"type": "integer", "minimum": 0, "maximum": 10},
        "action_or_spectacle": {"type": "integer", "minimum": 0, "maximum": 10},
        "genre_fit": {"type": "integer", "minimum": 0, "maximum": 10},
        "logo_or_text": {"type": "boolean"},
        "thumbnail_score": {"type": "integer", "minimum": 0, "maximum": 10}},
        "required": ["large_face", "emotion_intensity", "action_or_spectacle", "genre_fit", "logo_or_text", "thumbnail_score"]}
    prompt = THUMBNAIL_PROMPT
    if genre:
        prompt += (f" The video is a {genre} countdown. genre_fit: how instantly a viewer scrolling past would know this "
                   f"is a {genre} film from this frame alone ({genre_hint}); calm, smiling or neutral everyday frames score low.")
    data = gemini_editing.frame_json(small, prompt, schema, "thumbnail", 1024)
    if data is not None:
        try:
            result = _thumbnail_result(data, strict=True)
            gemini.record_event("thumbnail", "gemini", "success", details={"frames": 1})
            return result
        except (ValueError, TypeError, KeyError):
            gemini.record_event("thumbnail", "gemini", "fallback", "invalid thumbnail judgement")
    try:
        response = requests.post(_url() + "/api/generate", json={
            "model": VISION_MODEL, "prompt": prompt, "images": [base64.b64encode(buffer.getvalue()).decode()],
            "format": schema, "stream": False, "think": False, "keep_alive": "10m",
            "options": {"temperature": 0.1, "num_predict": 100, "num_ctx": 2048}}, timeout=25)
        response.raise_for_status()
        data = json.loads(response.json().get("response", "{}"))
        return _thumbnail_result(data)
    except (requests.RequestException, ValueError, TypeError, KeyError):
        return None


def _thumbnail_result(data: dict, strict: bool = False) -> dict:
    if strict:
        for field in ("large_face", "logo_or_text"):
            if not isinstance(data.get(field), bool):
                raise ValueError("Expected a boolean vision judgement")
        for field in ("emotion_intensity", "action_or_spectacle", "genre_fit", "thumbnail_score"):
            value = data.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10:
                raise ValueError("Expected a bounded integer vision judgement")
    return {"large_face": bool(data.get("large_face")),
            "emotion": max(0, min(10, int(data.get("emotion_intensity", 0)))),
            "action": max(0, min(10, int(data.get("action_or_spectacle", 0)))),
            "genre_fit": max(0, min(10, int(data.get("genre_fit", 5)))),
            "logo": bool(data.get("logo_or_text")),
            "score": max(0, min(10, int(data.get("thumbnail_score", 0))))}


THUMBNAIL_PROMPT = ("You are choosing a YouTube thumbnail for a movie countdown. Judge this frame: "
                    "does it show a large, clear human face? how intense is the emotion (fear, rage, shock, focus)? "
                    "is there visible action, a weapon, a creature, an explosion or spectacle? "
                    "is any logo, watermark, channel bug or written text visible anywhere, even small in a corner? "
                    "Return JSON only.")
