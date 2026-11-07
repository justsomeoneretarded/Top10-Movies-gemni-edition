"""Provisional resource/word planning; Auto never dictates finished duration."""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

from .core import PipelineError, selected_content_type


@dataclass(frozen=True)
class DurationSpec:
    target: float = 540
    minimum: float = 480
    maximum: float = 600
    automatic: bool = False


def _auto_duration(title: str, count: int, films: list[dict] | None = None) -> DurationSpec:
    # A conservative fallback budget for downloads and model resource planning.
    # Auto writing receives no word target and is never expanded to meet this estimate.
    count = max(2, min(15, count))
    kind = selected_content_type()
    if kind:
        shows = kind == "tv"
    elif films:
        shows = sum(film.get("kind") == "tv" for film in films) > len(films)/2
    else:
        shows = bool(re.search(r"\b(shows?|series|tv)\b", title, re.IGNORECASE))
    target = min(22*60, 30 + count*(85 if shows else 80))
    return DurationSpec(target, target*.75, target*1.25, automatic=True)


def workflow_duration(title: str, count: int, films: list[dict] | None = None) -> DurationSpec:
    """Read the workflow selector; Auto uses content size and no hard deadline.

    Invalid stale/API selections fall back to Auto rather than losing production.
    Brief text is deliberately not parsed here: the workflow selector is the
    authority for specified lengths, and Auto stays free of brief-based limits.
    """
    value = os.getenv("VIDEO_MINUTES", "Auto").strip()
    if not value or value.casefold() == "auto":
        return _auto_duration(title, count, films)
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)?", value, re.IGNORECASE)
    minutes = float(match[1]) if match else float("nan")
    if not math.isfinite(minutes) or not 1 <= minutes <= 30:
        print("Duration selection is unavailable or outside 1–30 minutes; using Auto planning.", flush=True)
        return _auto_duration(title, count, films)
    seconds = minutes*60
    return DurationSpec(seconds, seconds*.75, seconds*1.25)


def requested_duration(brief: str) -> DurationSpec:
    """Legacy local brief parser; production uses workflow_duration instead.

    Single durations allow ±25%. Invalid or conflicting instructions no longer
    cause a video failure; they fall back to the legacy default planning target.
    """
    text = brief.casefold().replace("–", "-").replace("—", "-")
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15,
             "twenty": 20, "thirty": 30, "forty": 40, "sixty": 60}
    for word, number in words.items():
        text = re.sub(rf"\b{word}(?=[ -]+(?:minute|min|hour|hr|second|sec))", str(number), text)
    number = r"\d+(?:\.\d+)?"
    unit = r"(?:minutes?|mins?|hours?|hrs?)\b"
    matches = list(re.finditer(rf"\b({number})(?:\s*(?:-|to)\s*({number}))?[ -]*({unit})", text))
    # Seconds commonly describe editing pauses, so require an explicit runtime/video context.
    matches += list(re.finditer(rf"(?:duration|runtime|length)\s*[:=]?\s*({number})(?:\s*(?:-|to)\s*({number}))?[ -]*(seconds?|secs?)\b", text))
    matches += list(re.finditer(rf"\b({number})(?:\s*(?:-|to)\s*({number}))?[ -]*(seconds?|secs?)\s+video\b", text))
    specs = []
    for match in matches:
        factor = 3600 if match[3].startswith(("hour", "hr")) else (1 if match[3].startswith(("second", "sec")) else 60)
        low, high = float(match[1])*factor, float(match[2] or match[1])*factor
        if not math.isfinite(low) or not math.isfinite(high) or low < 30 or high < low:
            print("Invalid duration in brief; keeping the default planning length.", flush=True)
            return DurationSpec()
        target = (low+high)/2
        specs.append(DurationSpec(target, low if match[2] else target*.75, high if match[2] else target*1.25))
    if len(set(specs)) > 1:
        print("Conflicting durations in brief; keeping the default planning length.", flush=True)
        return DurationSpec()
    return specs[0] if specs else DurationSpec()


def final_tempo(original: float, spec: DurationSpec, ending_hold: float = 3) -> float:
    """Keep the finished footage at natural speed; runtime remains guidance."""
    if not math.isfinite(original) or original <= 0 or not math.isfinite(ending_hold) or ending_hold < 0:
        raise PipelineError("Cannot time an empty or invalid media stream.")
    return 1.0


def trailer_budget(spec: DurationSpec, count: int) -> float:
    per_film = min(4.5, spec.target * 0.08 / count)
    return per_film if per_film >= 1.5 else 0.0


def narration_words(seconds: float, count: int, trailer_seconds: float = 0, scale: float = 1) -> tuple[int, int, int]:
    total = max(20, round(max(1, seconds-trailer_seconds) * 145 / 60 * scale))
    # A brief useful opening stays near eight seconds even for a long list.
    # Timing revisions belong in the reviews, rather than expanding the hook.
    opening = min(18, max(8, round(total*.055)))
    closing = min(35, max(4, round(total*0.03)))
    body = max(3, round((total-opening-closing)/count))
    return opening, body, closing
