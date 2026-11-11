"""Segment-aware music levels and original synthesized transition effects.

The music bed is shaped by what is on screen: louder over the cold open, rank
reveals, title cards and the end screen; low under narration (where sidechain
ducking still applies); almost silent under trailer dialogue. Whooshes and
impacts mark rank reveals. Effects are generated here from noise and sine
sweeps, so there is no third-party audio or license involved.
"""
from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

RATE = 48000
CONTROL_RATE = 100  # gain automation points per second
RAMP_SECONDS = 1.2  # music level changes glide over about a second, never jump

# Multipliers on the quiet base bed (about 20 dB under narration).
MUSIC_LEVELS = {
    "brand-ident": 0.0,
    "cold-open": 1.8,
    "review-preview": 3.5,          # muted preview: music carries the reveal
    "review-preview-dialogue": 0.12,  # preview keeps a complete trailer line: music steps back for it
    "title-card": 3.0,
    "teaser": 2.5,                   # the number one flash in the intro
    "subscribe": 1.8,
    "narration": 0.75,               # about 2.5 dB quieter under the narrator
    "narration-top": 0.95,           # number one gets a slightly bigger bed
    "trailer-audio": 0.10,           # trailer dialogue moments: the line must be heard
    "trailer-audio-separated": 0.18,  # trailer score removed: our bed carries the moment
    "end-screen": 3.5,
    "ending": 3.5,
}


def segment_level(entry: dict, top_title: str | None = None) -> float:
    kind = entry.get("type", "narration")
    if kind == "cold-open" and entry.get("shots") and not entry["shots"][0].get("muted", True):
        return MUSIC_LEVELS["review-preview-dialogue"]
    if kind == "review-preview":
        shots = entry.get("shots") or [{}]
        return MUSIC_LEVELS["review-preview" if shots[0].get("muted", True) else "review-preview-dialogue"]
    if kind == "trailer-audio" and entry.get("separated"):
        return MUSIC_LEVELS["trailer-audio-separated"]
    if kind == "narration" and top_title and entry.get("section_kind") == "film" and entry.get("film") == top_title:
        return MUSIC_LEVELS["narration-top"]
    return MUSIC_LEVELS.get(kind, 1.0)


def gain_curve(timeline: list[dict], speed: float, total: float, ending_start: float,
               top_title: str | None = None) -> np.ndarray:
    """Smooth gain automation at CONTROL_RATE for the final (tempo-adjusted) timeline."""
    points = max(1, math.ceil(total * CONTROL_RATE) + 1)
    target = np.ones(points, dtype=np.float32)
    for entry in timeline:
        start = float(entry.get("start", 0)) / speed
        end = start + float(entry.get("duration", 0)) / speed
        a, b = max(0, round(start * CONTROL_RATE)), min(points, round(end * CONTROL_RATE))
        if b > a:
            target[a:b] = segment_level(entry, top_title)
    target[max(0, round(ending_start * CONTROL_RATE)):] = MUSIC_LEVELS["ending"]
    window = max(1, round(RAMP_SECONDS * CONTROL_RATE))
    padded = np.pad(target, (window, window), mode="edge")
    smooth = np.convolve(padded, np.ones(window, dtype=np.float32) / window, mode="same")[window:-window]
    return smooth.astype(np.float32)


def shape_music(path: Path, curve: np.ndarray):
    """Apply a gain curve to a 16-bit stereo WAV in place, one second at a time."""
    shaped = path.with_suffix(".shaped.wav")
    with wave.open(str(path), "rb") as source, wave.open(str(shaped), "wb") as target:
        target.setparams(source.getparams())
        rate = source.getframerate()
        position = 0
        control_times = np.arange(len(curve)) / CONTROL_RATE
        while True:
            raw = source.readframes(rate)
            if not raw:
                break
            block = np.frombuffer(raw, dtype="<i2").reshape(-1, source.getnchannels()).astype(np.float32)
            times = (position + np.arange(len(block))) / rate
            gain = np.interp(times, control_times, curve).astype(np.float32)
            block *= gain[:, None]
            target.writeframes(np.clip(block, -32767, 32767).astype("<i2").tobytes())
            position += len(block)
    shaped.replace(path)


def _one_pole_lowpass(signal: np.ndarray, cutoff: np.ndarray) -> np.ndarray:
    """Time-varying one-pole low-pass (cutoff in Hz per sample)."""
    alpha = 1 - np.exp(-2 * np.pi * np.clip(cutoff, 20, RATE / 2.2) / RATE)
    out = np.empty_like(signal)
    state = 0.0
    for index, (value, a) in enumerate(zip(signal, alpha)):
        state += a * (value - state)
        out[index] = state
    return out


def whoosh(seconds: float = 0.75, seed: int = 1) -> np.ndarray:
    """Stereo noise sweep that rises to a peak and falls away, panning left to right."""
    count = round(seconds * RATE)
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, count, dtype=np.float32)
    noise = rng.standard_normal(count).astype(np.float32)
    peak = 0.62
    shape = np.where(t < peak, (t / peak) ** 2.2, np.exp(-(t - peak) * 9)).astype(np.float32)
    cutoff = 300 + 5200 * shape
    band = _one_pole_lowpass(noise, cutoff) - _one_pole_lowpass(noise, cutoff * 0.18)
    band *= shape
    band /= max(float(np.max(np.abs(band))), 1e-9)
    pan = 0.15 + 0.7 * t
    return np.column_stack((band * np.sqrt(1 - pan), band * np.sqrt(pan))).astype(np.float32)


def impact(seconds: float = 1.1, seed: int = 2) -> np.ndarray:
    """Low cinematic hit: a falling sine 'boom' plus a short filtered transient."""
    count = round(seconds * RATE)
    t = np.arange(count, dtype=np.float32) / RATE
    frequency = 38 + 62 * np.exp(-t * 7)
    phase = 2 * np.pi * np.cumsum(frequency) / RATE
    boom = np.sin(phase) * np.exp(-t * 3.2)
    rng = np.random.default_rng(seed)
    transient = _one_pole_lowpass(rng.standard_normal(count).astype(np.float32), np.full(count, 1800.0)) * np.exp(-t * 40)
    hit = boom + 0.6 * transient
    attack = np.minimum(1, t / 0.004)
    hit = (hit * attack).astype(np.float32)
    hit /= max(float(np.max(np.abs(hit))), 1e-9)
    return np.column_stack((hit, hit)).astype(np.float32)


def riser(seconds: float = 1.6, seed: int = 3) -> np.ndarray:
    """Tension build into a reveal: filtered noise and a pitch climbing to a sudden stop."""
    count = round(seconds * RATE)
    t = np.linspace(0, 1, count, dtype=np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(count).astype(np.float32)
    band = _one_pole_lowpass(noise, 400 + 7000 * t ** 2) - _one_pole_lowpass(noise, 120 + 900 * t ** 2)
    frequency = 110 * 2 ** (t * 2.5)
    tone = np.sin(2 * np.pi * np.cumsum(frequency) / RATE) * 0.35
    signal = (band / max(float(np.max(np.abs(band))), 1e-9) + tone) * t ** 2.6
    signal /= max(float(np.max(np.abs(signal))), 1e-9)
    return np.column_stack((signal, signal)).astype(np.float32)


def stinger(seconds: float = 1.4, seed: int = 4) -> np.ndarray:
    """Horror stab: a dissonant cluster of detuned saw tones with tremolo, sharp attack, long decay."""
    count = round(seconds * RATE)
    t = np.arange(count, dtype=np.float32) / RATE
    cluster = sum(2 * ((t * f) % 1) - 1 for f in (233.0, 246.9, 261.6, 277.2, 466.2))
    tremolo = 0.75 + 0.25 * np.sin(2 * np.pi * 11 * t)
    envelope = np.minimum(1, t / 0.006) * np.exp(-t * 2.4)
    signal = _one_pole_lowpass((cluster * tremolo * envelope).astype(np.float32), np.full(count, 3200.0))
    signal /= max(float(np.max(np.abs(signal))), 1e-9)
    return np.column_stack((signal, np.roll(signal, 240))).astype(np.float32)


def subdrop(seconds: float = 1.5) -> np.ndarray:
    """Deep falling sub-bass that you feel more than hear."""
    count = round(seconds * RATE)
    t = np.arange(count, dtype=np.float32) / RATE
    frequency = 28 + 50 * np.exp(-t * 3)
    signal = np.sin(2 * np.pi * np.cumsum(frequency) / RATE) * np.minimum(1, t / 0.01) * np.exp(-t * 1.8)
    return np.column_stack((signal, signal)).astype(np.float32)


def shimmer(seconds: float = 1.6) -> np.ndarray:
    """Bright magical/sci-fi sparkle: high bell partials with a soft decay."""
    count = round(seconds * RATE)
    t = np.arange(count, dtype=np.float32) / RATE
    signal = sum(np.sin(2 * np.pi * f * t + i) * np.exp(-t * (2 + i)) for i, f in enumerate((1568.0, 2093.0, 2637.0, 3136.0)))
    signal = signal * np.minimum(1, t / 0.02)
    signal /= max(float(np.max(np.abs(signal))), 1e-9)
    return np.column_stack((signal, np.roll(signal, 480))).astype(np.float32)


def pop(seconds: float = 0.35) -> np.ndarray:
    """Light comedy/family accent: a quick rising pluck."""
    count = round(seconds * RATE)
    t = np.arange(count, dtype=np.float32) / RATE
    frequency = 520 + 900 * np.minimum(1, t / 0.05)
    signal = np.sin(2 * np.pi * np.cumsum(frequency) / RATE) * np.exp(-t * 16) * np.minimum(1, t / 0.003)
    return np.column_stack((signal, signal)).astype(np.float32)


def swell(seconds: float = 1.8) -> np.ndarray:
    """Soft warm swell for romance/drama: a gentle chord that breathes in and out."""
    count = round(seconds * RATE)
    t = np.linspace(0, 1, count, dtype=np.float32)
    seconds_axis = np.arange(count, dtype=np.float32) / RATE
    chord = sum(np.sin(2 * np.pi * f * seconds_axis) for f in (220.0, 277.2, 329.6, 440.0))
    signal = chord * np.sin(np.pi * t) ** 2
    signal /= max(float(np.max(np.abs(signal))), 1e-9)
    return np.column_stack((signal, signal)).astype(np.float32)


# Per genre: what leads into a reveal, what hits on it (with gains), and an optional low layer.
GENRE_SOUNDS = {
    "horror": {"pre": ("riser", 0.55), "hits": (("stinger", 0.55), ("subdrop", 0.7)), "open": ("stinger", 0.45)},
    "thriller": {"pre": ("riser", 0.45), "hits": (("impact", 0.6), ("subdrop", 0.5)), "open": ("impact", 0.5)},
    "action": {"pre": ("riser", 0.45), "hits": (("impact", 0.8), ("subdrop", 0.45)), "open": ("impact", 0.6)},
    "scifi": {"pre": ("riser", 0.4), "hits": (("impact", 0.55), ("shimmer", 0.3)), "open": ("shimmer", 0.35)},
    "fantasy": {"pre": ("whoosh", 0.45), "hits": (("impact", 0.5), ("shimmer", 0.4)), "open": ("shimmer", 0.4)},
    "adventure": {"pre": ("whoosh", 0.5), "hits": (("impact", 0.65),), "open": ("impact", 0.5)},
    "comedy": {"pre": ("whoosh", 0.35), "hits": (("pop", 0.45),), "open": ("pop", 0.4)},
    "family": {"pre": ("whoosh", 0.35), "hits": (("pop", 0.4), ("shimmer", 0.25)), "open": ("shimmer", 0.35)},
    "romance": {"pre": ("whoosh", 0.25), "hits": (("swell", 0.35),), "open": ("swell", 0.3)},
    "drama": {"pre": ("whoosh", 0.4), "hits": (("impact", 0.55),), "open": ("impact", 0.45)},
    "documentary": {"pre": ("whoosh", 0.3), "hits": (("swell", 0.3),), "open": ("swell", 0.25)},
}
LEADS = {"riser": 1.6, "whoosh": 0.32}


def effect_events(timeline: list[dict], speed: float, genre: str | None = None) -> list[tuple[float, str, float]]:
    """(time, effect, gain) in final-timeline seconds, shaped by the video's genre."""
    import os
    if os.getenv("TRANSITION_FX", "false").strip().lower() != "true":
        return []
    pack = GENRE_SOUNDS.get((genre or os.getenv("VIDEO_GENRE", "drama")).strip().lower(), GENRE_SOUNDS["drama"])
    events = []
    for index, entry in enumerate(timeline):
        start = float(entry.get("start", 0)) / speed
        kind = entry.get("type")
        if kind == "cold-open" and index == 0:
            events.append((0.02, *pack["open"]))
        elif entry.get("rank_announcement") or (kind == "review-preview" and
                not (index > 0 and timeline[index-1].get("rank_announcement"))):
            # New edits reveal the rank with its spoken announcement. Legacy
            # timelines still place the card on the preview; never add both.
            hit = start + 0.5
            name, gain = pack["pre"]
            style = entry.get("reveal_style", "punch")
            if style == "whip":  # the sound follows the picture: a whip-pan gets a whoosh
                name, gain = "whoosh", max(gain, 0.45)
            elif style == "pull" and name != "riser":
                name, gain = "riser", 0.4
            events.append((max(0.0, hit - LEADS.get(name, 0.8) - (0.0 if name == "riser" else 0.5)), name, gain))
            events += [(hit, name, gain) for name, gain in pack["hits"]]
        elif kind == "teaser":
            events.append((max(0.0, start - 0.3), "whoosh", 0.35))
            events.append((start + 0.05, *pack["open"]))
        elif kind == "subscribe":
            events.append((max(0.0, start - 0.25), "whoosh", 0.3))
        elif kind == "narration" and entry.get("section_kind") == "outro" and \
                (index == 0 or timeline[index - 1].get("section_kind") != "outro"):
            events.append((max(0.0, start - 0.3), "whoosh", 0.25))
    return events


def effects_track(path: Path, timeline: list[dict], speed: float, total: float) -> int:
    """Write a stereo 16-bit WAV with every transition effect; returns the event count."""
    samples = np.zeros((max(1, round(total * RATE)), 2), dtype=np.float32)
    sounds = {"whoosh": whoosh(), "impact": impact(), "riser": riser(), "stinger": stinger(), "subdrop": subdrop(),
              "shimmer": shimmer(), "pop": pop(), "swell": swell()}
    events = effect_events(timeline, speed)
    for when, name, gain in events:
        sound = sounds[name]
        start = round(when * RATE)
        if start >= len(samples):
            continue
        end = min(len(samples), start + len(sound))
        samples[start:end] += sound[:end - start] * gain * 0.25  # about -12 dBFS peaks before the final limiter
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return len(events)


DELIVERY_PACE = {"hype": 1.05, "normal": 1.0, "dramatic": 0.94, "calm": 0.97}


def narration_speed() -> float:
    """The run's narration speed (workflow dropdown: 1, 1.1, 1.2 or 1.3; default 1.2)."""
    import os
    try:
        value = float(os.getenv("NARRATION_SPEED", "1.2"))
    except ValueError:
        value = 1.2
    return min(1.3, max(1.0, value))


def delivery_speed(theme_rate: float, delivery: str, *, words: int | None = None,
                   seconds: float | None = None) -> float:
    """Use theme as direction, then respond to the actual complete recorded take.

    Natural readings within a broad comfort band remain untouched. Short title
    announcements retain their articulation. Pacing never rejects a recording.
    """
    preference = min(1.18, max(.9, theme_rate * DELIVERY_PACE.get(delivery, 1.0)))
    if words is None or seconds is None:
        return round(preference, 3)  # compatibility for local callers without measurements
    if words < 12 or not math.isfinite(seconds) or seconds <= 0:
        return 1.0
    heard_rate = words * 60 / seconds
    center = 155 * preference
    lower, upper = center*.90, center*1.10
    if lower <= heard_rate <= upper:
        return 1.0
    destination = lower if heard_rate < lower else upper
    return round(min(1.18, max(.90, destination/max(1, heard_rate))), 3)
