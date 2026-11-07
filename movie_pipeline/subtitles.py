"""Subtitle cues that use the exact script words with speech-recognition timing.

Narration text is known, so captions never need the recognizer's spelling of
names or titles. Recognized words only supply timing: script words are aligned
to recognized words, unmatched words (for example spelled-out numbers) are
interpolated, and cues are rebuilt to broadcast-style limits.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

MAX_CUE_CHARS = 84      # two lines of up to 42 characters
MAX_CUE_SECONDS = 7.0
MIN_MATCH_RATIO = 0.5


def _norm(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.casefold())


def _heard_words(cues: list[dict]) -> list[tuple[str, float, float]]:
    """Recognized words with times spread across each cue by character length."""
    words = []
    for cue in cues:
        tokens = str(cue.get("text", "")).split()
        start, end = float(cue["start"]), float(cue["end"])
        total = sum(len(token) + 1 for token in tokens) or 1
        position = start
        for token in tokens:
            span = (end - start) * (len(token) + 1) / total
            words.append((token, position, position + span))
            position += span
    return words


def script_cues(text: str, cues: list[dict]) -> list[dict]:
    """Rebuild cues from the script text; falls back to the recognized cues when timing is unreliable."""
    script = text.split()
    heard = _heard_words(cues)
    if not script or not heard:
        return cues
    matcher = SequenceMatcher(None, [_norm(w) for w in script], [_norm(w) for w, _, _ in heard], autojunk=False)
    times: list[tuple[float, float] | None] = [None] * len(script)
    matched = 0
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            _, start, end = heard[block.b + offset]
            times[block.a + offset] = (start, end)
            matched += 1
    if matched / len(script) < MIN_MATCH_RATIO:
        return cues
    first_start, last_end = heard[0][1], heard[-1][2]
    index = 0
    while index < len(script):
        if times[index] is not None:
            index += 1
            continue
        gap_end = index
        while gap_end < len(script) and times[gap_end] is None:
            gap_end += 1
        low = times[index - 1][1] if index > 0 else first_start
        high = times[gap_end][0] if gap_end < len(script) else last_end
        high = max(high, low)
        count = gap_end - index
        for k in range(count):
            times[index + k] = (low + (high - low) * k / count, low + (high - low) * (k + 1) / count)
        index = gap_end
    result = []
    current: list[int] = []

    def emit(indices: list[int]):
        if indices:
            result.append({"start": times[indices[0]][0], "end": times[indices[-1]][1],
                           "text": " ".join(script[i] for i in indices)})

    for index, word in enumerate(script):
        candidate = " ".join(script[i] for i in [*current, index])
        if current and (len(candidate) > MAX_CUE_CHARS or times[index][1] - times[current[0]][0] > MAX_CUE_SECONDS):
            # Prefer ending the cue at the last comma/clause/sentence mark in its second half.
            text_so_far = " ".join(script[i] for i in current)
            split = None
            for position in range(len(current) - 1, 0, -1):
                prefix = " ".join(script[i] for i in current[:position + 1])
                if re.search(r"[,;:.!?]['\")\]]*$", script[current[position]]) and len(prefix) >= len(text_so_far) * 0.4:
                    split = position + 1
                    break
            if split:
                emit(current[:split])
                current = current[split:]
            else:
                emit(current)
                current = []
        current.append(index)
        if re.search(r"[.!?]['\")\]]*$", word) and len(" ".join(script[i] for i in current)) >= MAX_CUE_CHARS * 0.45:
            emit(current)
            current = []
    emit(current)
    if result:
        result[-1]["end"] = max(result[-1]["end"], last_end)
    for previous, following in zip(result, result[1:]):
        previous["end"] = min(previous["end"], following["start"])
    return [cue for cue in result if cue["end"] > cue["start"]]


def two_lines(text: str, limit: int = 42) -> str:
    """Split a caption into two balanced lines when it is longer than one line."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    words = text.split()
    best, best_score = None, None
    for split in range(1, len(words)):
        first, second = " ".join(words[:split]), " ".join(words[split:])
        score = abs(len(first) - len(second)) + (0 if len(first) <= limit and len(second) <= limit else 100)
        if best_score is None or score < best_score:
            best, best_score = (first, second), score
    return "\n".join(best) if best else text
