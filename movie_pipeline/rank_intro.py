"""Separate spoken rank cards from review bodies before generating any speech.

No waveform is cut: each announcement and each remaining review paragraph gets
its own verified take. A preview can then sit between two complete utterances.
"""
from __future__ import annotations

from copy import deepcopy
import re

RANK_INTRO_VERSION = 1


def _announcement(rank: int, title: str) -> str:
    return f"Number {rank}: {title}" + ("" if title.endswith((".", "!", "?")) else ".")


def _body_after_lead(text: str, title: str) -> str:
    """Remove a leading announcement, never a premise containing a title.

    Titles are matched literally (including colons, periods and parentheses).
    If the title is the subject of an unfinished sentence, retain that subject:
    'Number 2: Arrival follows a linguist' becomes 'Arrival follows a linguist'.
    """
    text = text.strip()
    lead = re.compile(
        r"^(?:(?:at\s+)?(?:number|no\.?|rank)\s+(?:\d{1,2}|[a-z]+)"
        r"(?:\s+on\s+(?:our|the)\s+list)?\s*[,.:—–-]*\s*"
        r"(?:(?:it[’']s|is|we\s+have)\s+)?|#\s*\d{1,2}\s*[:.,—–-]*\s*)",
        re.IGNORECASE,
    )
    match = lead.match(text)
    offset = match.end() if match else 0
    literal = re.match(re.escape(title), text[offset:], re.IGNORECASE)
    if literal is None:
        return text
    end = offset + literal.end()
    tail = text[end:]
    # Prevent partial matches such as 'It' in 'It's' or 'Up' in 'Upgrade'.
    if tail and tail[0].isalnum():
        return text
    year = re.match(r"^\s*\(\d{4}\)", tail)
    if year:
        end += year.end()
        tail = text[end:]
    boundary = re.match(r"^\s*[.!?:;—–-]+\s*", tail)
    if not tail.strip():
        return ""
    if boundary:
        return tail[boundary.end():].strip()
    if title.endswith((".", "!", "?")) and re.match(r"^\s+[A-Z0-9]", tail):
        # The title itself may supply the sentence-ending punctuation.
        return tail.strip()
    # A bare title can be part of the actual review; remove only a ranking prefix.
    return text[offset:].strip() if match else text


def prepare_rank_intros(script: dict, films: list[dict]) -> dict:
    """Return an idempotent, delivery-aligned script with one reveal per review."""
    from .script import aligned_delivery

    prepared = deepcopy(script)
    for section in prepared.get("sections", []):
        index = section.get("film")
        if section.get("kind") != "film" or not isinstance(index, int) or not 0 <= index < len(films):
            continue
        title = str(films[index]["title"]).strip()
        paragraphs = list(section.get("paragraphs", []))
        labels = aligned_delivery(section)
        body, delivery = [], []
        for p, paragraph in enumerate(paragraphs):
            cleaned = _body_after_lead(paragraph, title) if p == 0 else paragraph
            if cleaned.strip():
                body.append(cleaned)
                delivery.append(labels[p])
        section["paragraphs"] = [_announcement(index + 1, title), *body]
        section["delivery"] = ["hype", *delivery]
        section["rank_intro_version"] = RANK_INTRO_VERSION
    return prepared
