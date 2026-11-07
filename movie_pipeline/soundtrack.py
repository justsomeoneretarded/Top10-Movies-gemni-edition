"""Bundled licensed recorded instrumentals; one quiet musical theme per episode."""
from __future__ import annotations

import math
import re
import wave
import os
from pathlib import Path

import numpy as np

RATE = 48000


def theme_for(title: str, brief: str = "") -> str:
    from .editorial_policy import explicit_genre
    return explicit_genre(title, brief)


def narration_rate(theme: str) -> float:
    # Gentle base pace per theme; per-paragraph delivery (hype/dramatic/...) shapes it further.
    return {"action": 1.08, "mystery": 1.04, "drama": 1.0, "romance": 1.0}.get(theme, 1.0)


MUSIC_ROOT = Path(__file__).resolve().parents[1] / "assets" / "music"


def selected_track(theme: str) -> dict:
    import json
    name = {"action": "titan", "adventure": "titan", "mystery": "celestial", "horror": "celestial", "thriller": "celestial", "scifi": "celestial", "romance": "aurora", "drama": "aurora", "neutral": "aurora"}.get(theme, "aurora")
    return json.loads((MUSIC_ROOT / "catalog.json").read_text())[name]


# A genre score generated in the parallel music job (ACE-Step 1.5, MIT licence; output is ours to use).
GENERATED_SCORE = Path(os.getenv("GENERATED_SCORE", "work/music-generated/score.wav"))

# What the score should sound like for each genre (instrumental only).
GENRE_MUSIC = {
    "neutral": "restrained neutral instrumental underscore, soft muted keys, gentle understated pulse, clear space for a narrator, consistent warm mood, no dramatic stings, no vocals",
    "horror": "dark horror film score, low drones, dissonant strings, sparse eerie piano notes, distant metallic textures, slow heartbeat pulse, creeping tension, no vocals",
    "thriller": "suspenseful thriller underscore, ticking pulse, low synth bass, tense staccato strings, building tension, no vocals",
    "action": "epic action trailer score, driving taiko percussion, brass hits, aggressive strings, powerful and fast, no vocals",
    "scifi": "atmospheric science fiction score, analog synth arpeggios, deep pads, wide reverb, sense of wonder and scale, no vocals",
    "comedy": "playful upbeat comedy underscore, pizzicato strings, light percussion, bouncy bass, cheeky and fun, no vocals",
    "family": "whimsical family film score, bright orchestra, glockenspiel, warm strings, joyful and adventurous, no vocals",
    "adventure": "sweeping adventure film score, soaring horns, heroic strings, timpani, epic and hopeful, no vocals",
    "romance": "romantic cinematic piano with warm strings, tender and hopeful, gentle build, no vocals",
    "fantasy": "magical fantasy film score, harp, celesta, wordless choir pads, epic strings, enchanted, instrumental",
    "documentary": "thoughtful documentary underscore, soft piano, ambient pads, steady pulse, no vocals",
    "drama": "emotional cinematic drama score, piano and strings, slow build, heartfelt, no vocals",
}


def generated_score() -> Path | None:
    try:
        import soundfile as sf
        if not GENERATED_SCORE.exists():
            return None
        samples, rate = sf.read(GENERATED_SCORE, dtype="float32", always_2d=True)
        if (rate < 8000 or samples.shape[1] not in (1,2) or len(samples)/rate < 20
                or not np.isfinite(samples).all() or float(np.sqrt(np.mean(samples**2))) < .0001
                or float(np.mean(np.abs(samples) >= 1)) > .02):
            return None
        return GENERATED_SCORE
    except (OSError, RuntimeError, ValueError):
        return None


def music_credit(theme: str) -> str:
    if generated_score():
        return "Music: original score generated for this video with ACE-Step 1.5 (open-source, MIT licence)."
    track = selected_track(theme)
    return (f"Music: '{track['title']}' by Scott Buckley — CC BY 4.0. "
            f"{track['source']} License: {track['license_url']} "
            "Excerpt, volume adjustment and crossfaded repetition.")


def music_bed(path: Path, seconds: float, action: bool = False, *, theme: str | None = None):
    """Quiet licensed recorded music, with smooth loops and no synthesized bed."""
    import hashlib

    import soundfile as sf

    from .core import PipelineError
    track = selected_track(theme or ("action" if action else "drama"))
    source = MUSIC_ROOT / track["file"]
    score = generated_score()
    if score is not None:
        try:
            samples, rate = sf.read(score, dtype="float32", always_2d=True)
            if samples.shape[1] == 1:
                samples = np.repeat(samples, 2, axis=1)
            if rate != RATE:
                positions = np.arange(0, len(samples) - 1, rate / RATE)
                samples = np.stack([np.interp(positions, np.arange(len(samples)), samples[:, c]) for c in range(2)], axis=1).astype(np.float32)
                rate = RATE
            source = score
        except Exception:  # noqa: BLE001 -- the bundled track remains
            score = None
    if score is None:
        if hashlib.sha256(source.read_bytes()).hexdigest() != track["sha256"]:
            raise PipelineError("Bundled music failed its integrity check.")
        samples, rate = sf.read(source, dtype="float32", always_2d=True)
    if rate != RATE or len(samples) < RATE*8 or samples.shape[1] != 2:
        raise PipelineError("Bundled music must be stereo 48 kHz with a readable loop.")
    from .core import save_json
    save_json(path.with_suffix(".source.json"), {"genre": theme or "neutral", "source": str(source),
              "generated": score is not None, "fallback": None if score is not None else "bundled licensed instrumental"})
    overlap = RATE*3
    weight = np.linspace(0, 1, overlap, dtype=np.float32)[:, None]
    loop = np.concatenate((samples[overlap:-overlap],
                           samples[-overlap:]*(1-weight)+samples[:overlap]*weight))
    rms = float(np.sqrt(np.mean(loop.astype(np.float64)**2)))
    peak = float(np.max(np.abs(loop)))
    gain = min(.0126/max(rms, 1e-9), .08/max(peak, 1e-9))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        for block in range(math.ceil(seconds)):
            count = min(RATE, round(seconds*RATE)-block*RATE)
            indices = np.arange(count)+block*RATE
            time = indices/RATE
            envelope = np.minimum(1, time/2)*np.minimum(1, np.maximum(0, seconds-time)/2)
            audio = loop[indices % len(loop)]*gain*envelope[:, None]
            stream.writeframes((audio*32767).astype("<i2").tobytes())
