"""Bounded checks that verified paragraph recordings survive the final mix.

This is a secondary export check, not another script-length or ASR rejection
gate. Compare the actual verified recording first; recognition is only a lazy,
unprompted second opinion when the acoustic comparison is uncertain.
"""
from __future__ import annotations

import math
import wave
from itertools import pairwise
from pathlib import Path

import numpy as np

from .core import ffmpeg, save_json

RATE = 16000
HOP = 160
PADDING = .12


def _read(path):
    with wave.open(str(path), "rb") as stream:
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float64) / 32768


def _decode(source, target, *, start=0.0, seconds=None, speed=1.0):
    args = ["-ss", max(0, start), "-i", source, "-vn"]
    if seconds is not None:
        args += ["-t", seconds]
    args += ["-af", f"aresample=48000,atempo={speed}", "-ar", RATE, "-ac", "1", "-c:a", "pcm_s16le", target]
    ffmpeg(*args)
    return _read(target)


def _features(samples):
    """Short speech-band envelopes, insensitive to AAC and waveform phase."""
    size = 512
    if len(samples) < size:
        samples = np.pad(samples, (0, size-len(samples)))
    frames = np.lib.stride_tricks.sliding_window_view(samples, size)[::HOP]
    power = np.abs(np.fft.rfft(frames*np.hanning(size), axis=1))**2
    frequencies = np.fft.rfftfreq(size, 1/RATE)
    edges = (300, 500, 800, 1200, 1800, 2600, 3800, 6000)
    bands = np.column_stack([np.sqrt(power[:, (frequencies >= lo) & (frequencies < hi)].mean(axis=1))
                             for lo, hi in pairwise(edges)])
    # Time stretching the combined timeline has a different phase/context than
    # stretching a single take. Fifty-millisecond smoothing tolerates that.
    return np.column_stack([np.convolve(bands[:, i], np.ones(5)/5, mode="same") for i in range(bands.shape[1])])


def _cosine(first, second):
    denominator = np.linalg.norm(first)*np.linalg.norm(second)
    return float(np.sum(first*second)/denominator) if denominator > 1e-12 else 0.0


def _identity(reference, exported, offset):
    # The intersection can sound identical even when the export contains only
    # the beginning of a paragraph. Preserve independent duration evidence;
    # lag search and a high prefix correlation must never override it.
    available = max(0, len(exported)-round(offset*RATE))
    missing = max(0, len(reference)-available)/RATE
    coverage = {"missing_reference_seconds": round(missing, 4),
                "reference_coverage": round(min(1.0, available/max(1, len(reference))), 4)}
    original, mixed = _features(reference), _features(exported)
    weights = 1/np.maximum(np.sqrt(np.mean(original**2, axis=0)), np.max(original)*.02+1e-8)
    original *= weights
    mixed *= weights
    base = round(offset*RATE/HOP)
    best = None
    for shift in range(-12, 13):
        begin = base+shift
        left = max(0, -begin)
        right = min(len(original), len(mixed)-begin)
        if right-left < 8:
            continue
        first, second = original[left:right], mixed[begin+left:begin+right]
        shape = _cosine(first, second)
        variation = float(np.std(first)/max(np.mean(first), 1e-8))
        temporal = _cosine(first-first.mean(axis=0), second-second.mean(axis=0))
        score = min(shape, max(0.0, temporal)) if variation > .18 else shape
        if best is None or score > best[0]:
            best = score, shape, temporal, begin, left, right
    if best is None:
        return {"identity": 0.0, "shape": 0.0, "edges": [0.0, 0.0], "speech_level_ratio": 0.0, **coverage}
    score, shape, temporal, begin, left, right = best
    first, second = original[left:right], mixed[begin+left:begin+right]
    energy = np.linalg.norm(first, axis=1)
    active = np.flatnonzero(energy > max(float(np.quantile(energy, .8))*.12, 1e-7))
    edges = []
    for edge in ([active[0], active[-1]] if len(active) else [0, len(first)-1]):
        a, b = max(0, edge-30), min(len(first), edge+31)
        edges.append(_cosine(first[a:b], second[a:b]))
    level = float(np.linalg.norm(second)/max(np.linalg.norm(first), 1e-8))
    ratios = np.linalg.norm(second, axis=1)/np.maximum(energy, 1e-8)
    usual_level = float(np.median(ratios[active])) if len(active) else 0.0
    boundary_levels = []
    for edge in ([active[0], active[-1]] if len(active) else [0, len(first)-1]):
        positions = active[(active >= edge-30) & (active <= edge+30)]
        boundary_levels.append(float(np.median(ratios[positions]))/max(usual_level, 1e-8) if len(positions) else 0.0)
    return {"identity": round(score, 4), "shape": round(shape, 4), "temporal": round(temporal, 4),
            "edges": [round(value, 4) for value in edges], "speech_level_ratio": round(level, 4),
            "boundary_level_ratios": [round(value, 4) for value in boundary_levels],
            "alignment_seconds": round((begin-base)*HOP/RATE, 3), **coverage}


def _samples(events):
    voiced = [(index, event) for index, event in enumerate(events)
              if event.get("type") in ("narration", "subscribe") and str(event.get("text", "")).strip()]
    if len(voiced) <= 3:
        return voiced
    midpoint = (float(voiced[0][1]["start"])+float(voiced[-1][1]["start"]))/2
    middle = min(voiced[1:-1], key=lambda item: abs(float(item[1]["start"])-midpoint))
    return [voiced[0], middle, voiced[-1]]


def verify_export_audio(video: Path, events: list[dict], speed: float, recovery, work: Path) -> dict:
    """Check first, middle and last complete voiced paragraphs after final tempo.

    ``events`` use original timeline seconds and ``source_audio`` is the complete
    verified recording, never a subtitle slice. ``recovery`` may be a checker,
    a lazy zero-argument checker factory, or None. Inconclusive evidence is
    reported explicitly and allowed through; clear voice loss is rejected.
    """
    work = Path(work)/"export-audio-review"
    work.mkdir(parents=True, exist_ok=True)
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Export playback speed must be positive and finite.")
    checks = []
    checker = recovery
    for index, event in _samples(events):
        check = {"event_index": index, "type": event["type"], "status": "inconclusive"}
        target = work/f"paragraph-{index:03}.wav"
        try:
            start = float(event["start"])/speed
            seconds = float(event["duration"])/speed
            padded_start = max(0.0, start-PADDING)
            try:
                target.unlink(missing_ok=True)
                exported = _decode(video, target, start=padded_start, seconds=seconds+start-padded_start+PADDING)
            except Exception:
                # A prior successful check or a decoder's partial output must
                # never turn this attempt's unreadable export into uncertainty.
                check.update(status="failed", reason="Final paragraph audio could not be decoded.")
                raise
            check.update({"start": start, "duration": seconds, "audio": str(target)})
            source = event.get("source_audio")
            metrics = None
            if source and Path(source).is_file():
                reference = _decode(Path(source), work/f"reference-{index:03}.wav", speed=speed)
                metrics = _identity(reference, exported, start-padded_start)
                reference_rms = float(np.sqrt(np.mean(reference**2))) if len(reference) else 0.0
                estimated_voice_rms = reference_rms*metrics.get("speech_level_ratio", 1.0)
                metrics["estimated_voice_rms"] = round(estimated_voice_rms, 6)
                check["acoustic"] = metrics
                if metrics.get("missing_reference_seconds", 0) > PADDING:
                    check.update(status="failed", method="recording-duration", reason="Final audio ends before the complete verified paragraph recording.")
                elif reference_rms > .005 and metrics.get("speech_level_ratio", 1) < .04 and estimated_voice_rms < .003:
                    check.update(status="failed", method="recording-audibility", reason="Verified narration survived but became effectively inaudible in the final mix.")
                elif metrics["identity"] >= .78 and min(metrics["edges"]) >= .65:
                    check.update(status="passed", method="verified-recording", reason="Complete recorded speech retained through the final mix.")
                elif (np.max(np.abs(exported), initial=0) < .0001 and np.max(np.abs(reference), initial=0) > .002
                      or metrics["speech_level_ratio"] < .06 and metrics["identity"] < .5
                      or metrics["identity"] < .1 and metrics["shape"] < .8
                      or min(metrics.get("boundary_level_ratios", [1])) < .1 and min(metrics["edges"]) < .4):
                    check.update(status="failed", method="verified-recording", reason="Verified narration is absent from this final paragraph window.")
            if check["status"] == "inconclusive":
                if checker is not None:
                    if not hasattr(checker, "check"):
                        checker = checker()
                    # SpeechRecovery.check never passes expected text to its ASR
                    # decoder. It compares the independent hypothesis afterwards.
                    cues = checker.check(event["text"], target, label=f"final export paragraph {index+1}")
                    check["recognition_accepted"] = cues is not None
                    if cues is not None:
                        check.update(status="passed", method="unprompted-recognition", reason="Complete script independently recognized in the export.")
                    elif metrics and metrics["identity"] < .25 and max(metrics["edges"]) < .5:
                        check.update(status="failed", method="acoustic-and-recognition", reason="Final audio differs strongly from verified narration and the script could not be recognized.")
                if check["status"] == "inconclusive":
                    check["reason"] = "Acoustic identity is uncertain; recognition alone cannot establish lost words."
        except Exception as exc:  # noqa: BLE001 -- optional ASR failure is evidence of uncertainty, not of missing words.
            check["error"] = type(exc).__name__
            check.setdefault("reason", "Could not complete the secondary export audio check.")
            if not target.exists():
                check.update(status="failed", reason="Final paragraph audio could not be decoded.")
        checks.append(check)
    failed = [check["event_index"] for check in checks if check["status"] == "failed"]
    status = "failed" if failed else "inconclusive" if not checks or any(check["status"] == "inconclusive" for check in checks) else "passed"
    report = {"accepted": not failed, "status": status, "sampled_paragraphs": len(checks), "failed_events": failed, "checks": checks}
    if status == "inconclusive":
        report["warning"] = "Secondary export audio identity was inconclusive; no definite loss of verified narration was found."
        print(report["warning"], flush=True)
    save_json(work/"report.json", report)
    return report
