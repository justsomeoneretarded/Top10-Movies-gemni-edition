"""Rare, context-matched public meme insert; every failure simply skips the insert."""
from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image

from .core import key, save_json

CATALOG_URL = "https://api.imgflip.com/get_memes"
TRIGGER = re.compile(r"\babsolute\s+(?:cinema|masterpieces?)\b", re.IGNORECASE)
MAX_IMAGE_BYTES = 5_000_000


def _rare(seed: str) -> bool:
    # Stable per video: at most one of eight seed buckets is eligible, never a forced insert.
    return int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16) % 8 == 0


def _image_url(value) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return (parsed.scheme == "https" and parsed.netloc == "i.imgflip.com"
            and not parsed.username and not parsed.password and not parsed.fragment)


def _download(url: str, limit: int, content_type: str) -> bytes:
    with requests.get(url, stream=True, allow_redirects=False, timeout=(5, 12)) as response:
        if response.status_code != 200:
            raise ValueError("Meme source unavailable")
        if not response.headers.get("Content-Type", "").lower().startswith(content_type):
            raise ValueError("Unexpected meme response")
        data = bytearray()
        for chunk in response.iter_content(65536):
            data.extend(chunk)
            if len(data) > limit:
                raise ValueError("Meme response too large")
        return bytes(data)


def _validate_image(data: bytes):
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Invalid meme size")
    with Image.open(io.BytesIO(data)) as image:
        if min(image.size) < 256 or image.width * image.height > 16_000_000:
            raise ValueError("Invalid meme dimensions")
        image.load()


def prepare_intro_meme(text: str, seed: str, work: Path) -> dict | None:
    """Return a validated still and source credit only for a rare, exact thematic match.

    The caller limits this to the intro. Source credits are provenance, not a license
    or an assertion of endorsement. No generic reaction image is substituted.
    """
    if not TRIGGER.search(text) or not _rare(seed):
        return None
    try:
        work.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    image_path = work / f"intro-meme-{key(seed)}.image"
    metadata_path = image_path.with_suffix(".json")
    if image_path.exists() or metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            if not isinstance(metadata, dict) or not _image_url(metadata.get("source")):
                raise ValueError("Invalid cached meme source")
            if image_path.stat().st_size > MAX_IMAGE_BYTES:
                raise ValueError("Invalid cached meme size")
            _validate_image(image_path.read_bytes())
            return {**metadata, "path": str(image_path.resolve())}
        except (OSError, ValueError, Image.DecompressionBombError):
            image_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            return None
    try:
        catalog = json.loads(_download(CATALOG_URL, 2_000_000, "application/json"))
        if not isinstance(catalog, dict) or catalog.get("success") is not True:
            return None
        entries = catalog.get("data", {}).get("memes", [])
        if not isinstance(entries, list):
            return None
        template = next((entry for entry in entries if isinstance(entry, dict)
                         and str(entry.get("name", "")).strip().casefold() == "absolute cinema"
                         and _image_url(entry.get("url"))), None)
        if template is None:
            return None
        image_url = template["url"]
        data = _download(image_url, MAX_IMAGE_BYTES, "image/")
        _validate_image(data)
        metadata = {"source": image_url, "template": "Absolute Cinema",
                    "attribution": f"Meme template: Absolute Cinema — Imgflip ({image_url})."}
        image_path.write_bytes(data)
        save_json(metadata_path, metadata)
        return {**metadata, "path": str(image_path.resolve())}
    except (requests.RequestException, OSError, ValueError, TypeError, AttributeError,
            Image.DecompressionBombError):
        image_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        print("Optional intro meme unavailable; retaining the cinematic intro.", flush=True)
        return None
