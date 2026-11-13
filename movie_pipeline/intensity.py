"""How intense a stretch of trailer is: the hook after each rank announcement should be the
peak of the movie (a fight, a battle, gunfire, a chase, a scare, a furious argument), whatever
the genre. Combines what the picture shows, how loud and punchy the sound is, and how much moves.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np

STEP = 0.1  # seconds per loudness frame
_ENVELOPES: dict[str, np.ndarray | None] = {}

GENERIC = ("The most intense moment of a film: a brutal fight, a battle, gunfire, an explosion, a chase, "
           "a terrifying attack, someone screaming, or a furious face-to-face argument.")
GENRE_PEAKS = {
    "action": "People fighting hand to hand, shooting guns, an explosion or a high-speed chase.",
    "adventure": "A dangerous chase, a battle, a fall from a height or an explosion.",
    "war": "Soldiers in battle under fire, explosions and chaos.",
    "horror": "A terrifying creature or killer attacking, a person screaming in fear in the dark.",
    "thriller": "A violent struggle, a gun pointed at someone, a tense chase or a shocking attack.",
    "crime": "A violent confrontation, a shootout, a gun to someone's head or a furious interrogation.",
    "drama": "A furious shouting argument, someone crying in anger, a slap or a violent outburst.",
    "romance": "A passionate, emotional confrontation between two people, tears and shouting.",
    "comedy": "A big chaotic physical comedy moment: people falling, crashing or panicking.",
    "science fiction": "A huge space battle, alien attack, laser fire or explosion.",
    "fantasy": "A massive battle with creatures, magic blasts or a dragon attacking.",
    "animation": "A huge chaotic action moment: a chase, a fall, a battle or an explosion.",
    "mystery": "A shocking discovery, a sudden attack or a tense confrontation.",
}


def intensity_query(genre: str = "") -> str:
    peak = GENRE_PEAKS.get((genre or "").strip().lower(), "")
    return (GENERIC + " " + peak).strip()


def envelope(source: Path | str) -> np.ndarray | None:
    """Loudness per STEP seconds in dB (cached per file); None when the file has no usable sound."""
    name = str(source)
    if name in _ENVELOPES:
        return _ENVELOPES[name]
    result = None
    try:
        raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", name, "-vn", "-ac", "1", "-ar", "8000",
                              "-f", "s16le", "-"], capture_output=True, timeout=180, check=True).stdout
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        frame = int(8000 * STEP)
        count = len(samples) // frame
        if count >= 10:
            rms = np.sqrt(np.mean(samples[:count * frame].reshape(count, frame) ** 2, axis=1) + 1e-10)
            result = 20 * np.log10(rms)
    except (OSError, subprocess.SubprocessError, ValueError):
        result = None
    _ENVELOPES[name] = result
    return result


def loudness(env: np.ndarray | None, start: float, end: float) -> float:
    """0..1: how loud and punchy this window is compared with the rest of the same trailer.
    A window that is loud throughout and hits peaks (impacts, shouting, gunfire) scores high."""
    if env is None or not end > start:
        return 0.5
    a, b = max(0, int(start / STEP)), min(len(env), int(math.ceil(end / STEP)))
    if b - a < 5:
        return 0.5
    window = env[a:b]
    audible = env[env > -60]
    if len(audible) < 10:
        return 0.5
    median = float(np.median(audible))
    body = float(np.percentile(window, 60)) - median
    peaks = float(np.mean(window >= median + 9.0))  # impacts, shouting, gunfire well above the trailer's norm
    return float(max(0.0, min(1.0, 0.5 + body / 16.0)) * 0.6 + min(1.0, peaks * 2.0) * 0.4)


def ranks(values) -> np.ndarray:
    """Percentile rank 0..1 within one trailer, robust to each feature's scale."""
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.full(len(values), 0.5)
    # Average rank for ties, so equal values always score the same.
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    first = np.concatenate(([0], np.cumsum(counts)[:-1]))
    average = first + (counts - 1) / 2
    return average[inverse] / (len(values) - 1)


def score(picture: float, sound: float, motion: float, gemini_energy: float | None = None) -> float:
    """Combined 0..1 intensity. Gemini's watched energy, when verified, counts most."""
    local = 0.42 * picture + 0.36 * sound + 0.22 * motion
    if gemini_energy is None:
        return round(local, 4)
    return round(0.55 * float(gemini_energy) + 0.45 * local, 4)
