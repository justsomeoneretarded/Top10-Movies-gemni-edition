"""Export resolution and conservative scratch-space checks, independent of thumbnails."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from .core import PipelineError, probe

MAJORITY = 0.6
MAX_UPSCALE = 1.5
SIZES = {144: (256, 144), 240: (426, 240), 360: (640, 360), 480: (854, 480), 720: (1280, 720), 1080: (1920, 1080), 1440: (2560, 1440), 2160: (3840, 2160)}


def download_ceiling() -> int:
    """The workflow checkbox controls the maximum; brief text cannot override it."""
    value = os.getenv("TRY_4K", "true").strip().lower()
    if value not in {"true", "false"}:
        raise PipelineError("TRY_4K must be true or false.")
    return 2160 if value == "true" else 1080


MIN_EXPORT_HEIGHT = 1080  # never below Full HD: one low-quality trailer must not blur the whole video


def choose_output_size(sources: list[dict], ceiling: int) -> tuple[int, int]:
    """The highest standard canvas most sources fill natively, and never below 1080p.

    A weak trailer (a 480x260 upload, say) is enlarged for its own title only; it no
    longer lowers the whole video (a comedy list once came out at 640x360 this way).
    """
    width, height = _shared_size(sources, ceiling)
    floor = min(MIN_EXPORT_HEIGHT, ceiling)
    return (width, height) if height >= floor else SIZES[floor]


def _shared_size(sources: list[dict], ceiling: int) -> tuple[int, int]:
    """Choose a standard canvas that no source needs enlarging to fill."""
    if not sources:
        raise PipelineError("No trailer sources were provided for quality selection.")
    if any(not isinstance(source.get(axis), int) or source[axis] <= 0 for source in sources for axis in ("width", "height")):
        raise PipelineError("A downloaded trailer has no readable video dimensions.")
    def detail(source):
        return min(source["height"], source["width"] * 9 / 16)

    for height, (width, _height) in sorted(SIZES.items(), reverse=True):
        if height > ceiling:
            continue
        native = sum(source["width"] >= width and source["height"] >= height for source in sources)
        # Most sources must support the size natively; a minority may be enlarged by at most 1.5x,
        # so one weaker trailer no longer lowers the whole video.
        if native == len(sources) or (native / len(sources) >= MAJORITY and
                                       all(height / detail(source) <= MAX_UPSCALE for source in sources)):
            return width, height
    # Extremely small footage remains usable; fit an even 16:9 canvas inside it.
    scale = min(min(source["width"]/16, source["height"]/9) for source in sources)
    width = max(2, int(16*scale)//2*2)
    height = max(2, int(9*scale)//2*2)
    return width, height


def inspect_sources(trailers: list[Path]) -> list[dict]:
    sources = []
    for path in trailers:
        video = next((stream for stream in probe(path).get("streams", []) if stream.get("codec_type") == "video"), {})
        sources.append({"file": path.name, "width": video.get("width"), "height": video.get("height")})
    return sources


def resolution_from_brief(brief: str) -> int:
    """Legacy local helper; the production workflow uses its checkbox instead."""
    requested = set()
    for pattern, height in ((r"\b(?:4k|2160p)\b", 2160), (r"\b(?:1440p|2k|qhd)\b", 1440),
                            (r"\b(?:1080p|full[ -]?hd)\b", 1080), (r"\b720p\b", 720)):
        if re.search(pattern, brief, re.IGNORECASE):
            requested.add(height)
    return max(requested) if requested else export_size()[1]


def export_size() -> tuple[int, int]:
    value = os.getenv("VIDEO_RESOLUTION", "1080").strip().lower().removesuffix("p")
    value = {"4k": "2160", "2k": "1440"}.get(value, value)
    if "x" in value:
        parts = value.split("x")
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            width, height = map(int, parts)
            if 2 <= width <= 3840 and 2 <= height <= 2160 and width % 2 == height % 2 == 0:
                return width, height
    if not value.isdigit() or int(value) not in SIZES:
        raise PipelineError("VIDEO_RESOLUTION must be a supported height or an even WIDTHxHEIGHT canvas.")
    return SIZES[int(value)]


def require_disk(path: Path, seconds: float = 0):
    # Parts, concatenated input, and final encode coexist. This is a reserve, not a size guarantee.
    height = export_size()[1]
    rate = {1080: 10, 1440: 16, 2160: 28}.get(height, 6) * 1_000_000 / 8
    needed = max(1_500_000_000, seconds * (rate * 3 + 576_000) + 1_000_000_000)
    free = shutil.disk_usage(path).free
    if free < needed:
        raise PipelineError(f"Not enough working disk for {height}p: {free/1e9:.1f} GB free, about {needed/1e9:.1f} GB reserved. Uncheck Try 4K or use fewer titles; artifact expiry does not free running-job disk.")
