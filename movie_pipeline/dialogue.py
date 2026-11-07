"""Keep trailer dialogue, drop most of the trailer's own score (Demucs, local CPU).

Only the short windows the edit actually plays with trailer sound are separated:
dialogue inserts and previews that keep a complete spoken line. The voices are
kept at full level and the trailer's music/effects are mixed in quietly, so our
own music bed continues smoothly under the moment instead of two scores
clashing. Any failure returns None and the original trailer audio is used.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .core import ffmpeg, key

RATE = 44100
MUSIC_UNDER_DIALOGUE = 0.22  # about -13 dB of the trailer's own music and effects
_MODEL = {"loaded": False, "model": None}


def enabled() -> bool:
    return os.getenv("DIALOGUE_SEPARATION", "true").strip().lower() != "false"


def _model():
    if not _MODEL["loaded"]:
        _MODEL["loaded"] = True
        try:
            from demucs.pretrained import get_model
            model = get_model("htdemucs")
            model.eval()
            _MODEL["model"] = model
        except Exception as exc:  # noqa: BLE001 -- optional enhancement
            print(f"Dialogue separation unavailable ({type(exc).__name__}); trailer audio plays as released.", flush=True)
            _MODEL["model"] = None
    return _MODEL["model"]


def isolate_dialogue(source: Path, start: float, seconds: float, work: Path) -> Path | None:
    """A 48 kHz stereo WAV of [start, start+seconds) with voices kept and the trailer score lowered."""
    if not enabled() or seconds <= 0:
        return None
    work.mkdir(parents=True, exist_ok=True)
    target = work / f"dialogue-{key('dialogue-v1', str(source), round(start, 3), round(seconds, 3))}.wav"
    if target.exists():
        return target
    model = _model()
    if model is None:
        return None
    try:
        import soundfile as sf
        import torch
        from demucs.apply import apply_model
        raw = work / f"{target.stem}-raw.wav"
        # Half a second of context on each side helps the model; it is trimmed afterwards.
        pad = min(0.5, start)
        ffmpeg("-ss", max(0, start - pad), "-i", source, "-t", seconds + pad + 0.5, "-vn",
               "-ac", "2", "-ar", RATE, "-c:a", "pcm_f32le", raw)
        audio, rate = sf.read(raw, dtype="float32", always_2d=True)
        raw.unlink(missing_ok=True)
        if rate != RATE or len(audio) < RATE * 0.3:
            return None
        wav = torch.from_numpy(audio.T.copy())
        reference = wav.mean(0)
        mean, std = float(reference.mean()), float(reference.std()) or 1.0
        with torch.inference_mode():
            stems = apply_model(model, ((wav - mean) / std)[None], device="cpu", split=True, overlap=0.25, progress=False)[0]
        stems = stems * std + mean
        names = list(model.sources)
        vocals = stems[names.index("vocals")].numpy()
        rest = sum(stems[i].numpy() for i, name in enumerate(names) if name != "vocals")
        mixed = (vocals + MUSIC_UNDER_DIALOGUE * rest).T
        begin = round(pad * RATE)
        mixed = mixed[begin:begin + round(seconds * RATE)]
        peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
        if peak > 0.99:
            mixed = mixed * (0.99 / peak)
        interim = work / f"{target.stem}-44k.wav"
        sf.write(interim, mixed, RATE, subtype="FLOAT")
        ffmpeg("-i", interim, "-ar", 48000, "-ac", 2, "-c:a", "pcm_s24le", target)
        interim.unlink(missing_ok=True)
        return target
    except Exception as exc:  # noqa: BLE001 -- the released trailer audio remains a good fallback
        print(f"Dialogue separation skipped for {source.name} ({type(exc).__name__}).", flush=True)
        target.unlink(missing_ok=True)
        return None
