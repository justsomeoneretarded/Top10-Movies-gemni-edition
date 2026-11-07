"""Conservative character holds sourced only from the reviewed film's trailers."""
from __future__ import annotations

import math
import re
from pathlib import Path


def _mentions(text: str, person: dict) -> bool:
    aliases = [person.get("name") or "", *re.split(r"\s*/\s*", person.get("character") or "")]
    return any(len(alias.strip()) >= 4 and re.search(r"(?<!\w)"+re.escape(alias.strip())+r"(?!\w)", text, re.IGNORECASE)
               for alias in aliases)


def trailer_still_for(text: str, film: dict, selector, sources: list[Path], shot_lookup: dict) -> dict | None:
    """Soft CLIP cast matching; skip ambiguity instead of fetching a headshot.

    A face detector alone does not identify an actor. Require a visible face,
    a positive target match and separation from other cast descriptions.
    This is an editorial preference, not a face-identity verification claim.
    """
    cast = [person for person in film.get("cast_profiles", []) if isinstance(person, dict)]
    target = next((person for person in cast if _mentions(text, person)), None)
    if target is None:
        return None
    candidates = []
    for source in sources:
        shots = shot_lookup.get(str(source), [])
        faces = getattr(selector, "faces", {}).get(str(source), [])
        try:
            query = f"A close-up of {target['name']} as {target.get('character', '')} in {film['title']}."
            # Character holds use raw CLIP similarity, which their thresholds are tuned for.
            score_of = getattr(selector, "identity_relevance", None) or selector.relevance
            scores = score_of(source, query)
            alternatives = [score_of(source, f"A close-up of {p['name']} as {p.get('character', '')} in {film['title']}.")
                            for p in cast if p is not target]
            for i, shot in enumerate(shots):
                if selector.card_kind(source, i) is not None:
                    continue
                score = float(scores[i])
                other = max((float(values[i]) for values in alternatives), default=score-.03)
                face = i < len(faces) and faces[i]
                recognizable_character = target.get("character") and score >= .34 and score-other >= .07
                if not (face or recognizable_character) or not math.isfinite(score) or not math.isfinite(other) or score < .25 or score-other < .025:
                    continue
                when = (shot['start']+shot['end'])/2
                score -= selector.reuse_penalty(source, shot['start'], shot['end']-shot['start'])
                candidates.append((score, str(source), when))
        except (AttributeError, RuntimeError, ValueError, IndexError, KeyError, TypeError):
            continue
    if not candidates:
        return None
    _, source, when = max(candidates)
    return {"source":source, "start":when, "image":False, "still":True,
            "still_motion":"zoom_in", "person":target['name'],
            "visual_reason":"trailer-character-hold"}


def insert_trailer_hold(plan: list[dict], portrait: dict, seconds: float = 5.5, *, when: float = 0) -> list[dict]:
    """Replace time with a trailer freeze without adding or repeating footage."""
    total = sum(shot["duration"] for shot in plan)
    beginning = max(0, min(when, max(0, total-4)))
    end = min(total, beginning+seconds)
    def window(low, high):
        result, clock = [], 0.0
        for shot in plan:
            start = max(low, clock)
            stop = min(high, clock+shot["duration"])
            if stop > start:
                revised = {**shot, "duration": stop-start}
                if not shot.get("still"):
                    revised["start"] += start-clock
                revised["freeze"] = min(revised.get("freeze", 0), revised["duration"])
                result.append(revised)
            clock += shot["duration"]
        return result
    return [*window(0, beginning), {**portrait, "duration": end-beginning}, *window(end, total)]
