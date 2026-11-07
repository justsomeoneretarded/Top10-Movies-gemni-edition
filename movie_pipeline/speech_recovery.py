"""Verify recorded words before changing speech, with bounded same-voice recovery."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

from . import voice
from .core import PipelineError, duration, ffmpeg, key, save_json

POLICY_VERSION = "speech-recovery-v1"


class SpeechRecoveryError(PipelineError):
    """Every recognition/generation route was exhausted without verified speech."""


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _usable_cues(cues, seconds):
    result = []
    for cue in cues or ():
        if not isinstance(cue, dict):
            continue
        try:
            start, end = float(cue["start"]), float(cue["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        start, end = max(0, start), min(seconds, end)
        if end > start and isinstance(cue.get("text"), str) and cue["text"].strip():
            result.append({"start": start, "end": end, "text": cue["text"].strip()})
    result.sort(key=lambda cue: (cue["start"], cue["end"]))
    last_end = 0.0
    for cue in result:
        # The subtitle writer enforces chronological cues. Accepting nested
        # timestamps here could make it discard words later, despite a matching
        # transcript. Retry recognition instead of accepting that timing.
        if cue["start"] < last_end-.001:
            return []
        cue["start"] = max(cue["start"], last_end)
        if cue["end"] <= cue["start"]:
            return []
        last_end = cue["end"]
    return result


def _speech_chunks(text: str) -> list[str]:
    """Smaller complete thoughts, retaining every word and its original order.

    Avoid tiny isolated phrases and splitting initials, abbreviations or decimal
    numbers. A single long sentence is retained rather than chopped mid-thought.
    """
    protected = re.sub(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|vs|etc)\.",
                       lambda match: match[0].replace(".", "\u2024"), text)
    protected = re.sub(r"\b[A-Z]\.(?=\s+[A-Z])",
                       lambda match: match[0].replace(".", "\u2024"), protected)
    sentences = re.split(r"(?<=[.!?])\s+(?=[\"'“‘A-Z0-9])", protected.strip())
    chunks, current = [], ""
    for sentence in sentences:
        sentence = sentence.replace("\u2024", ".")
        if current and len(current.split()) >= 8 and len((current+" "+sentence).split()) > 26:
            chunks.append(current)
            current = ""
        current = (current+" "+sentence).strip()
    if current:
        if chunks and len(current.split()) < 5:
            chunks[-1] += " "+current
        else:
            chunks.append(current)
    return chunks


class SpeechRecovery:
    """Shared recovery for hooks, reviews, availability lines, outros and CTAs.

    No expected script is supplied to recognition. Each complete hypothesis must
    pass the same word and boundary checks on its own. Larger ASR loads only for
    uncertain speech; an unavailable second model still permits fresh TTS tries.
    """

    def __init__(self, recognizer, speaker_factory, work: Path, *, fallback_factory=None):
        self.recognizer = recognizer
        self.speaker_factory = speaker_factory
        self.fallback_factory = fallback_factory
        self.work = work
        self.work.mkdir(parents=True, exist_ok=True)
        self.speaker = None
        self.fallback = None
        self.fallback_attempted = False
        self.events = []
        self.report = self.work / "speech-verification.json"

    def _record(self, event):
        self.events.append(event)
        save_json(self.report, {"policy": POLICY_VERSION, "checks": self.events})

    def _fallback_model(self):
        if not self.fallback_attempted:
            self.fallback_attempted = True
            if self.fallback_factory is not None:
                try:
                    self.fallback = self.fallback_factory()
                except Exception as exc:  # noqa: BLE001 -- second opinion is optional, verification is not.
                    self._record({"stage": "second-opinion-load", "accepted": False, "error": type(exc).__name__})
                    print("Additional speech recognizer unavailable; continuing bounded same-voice recovery.", flush=True)
        return self.fallback

    def check(self, text: str, audio: Path, *, label: str = "speech") -> list[dict] | None:
        try:
            seconds = duration(audio)
            if not math.isfinite(seconds) or seconds <= 0:
                raise PipelineError("Recorded speech has no usable duration.")
            fingerprint = _fingerprint(audio)
        except Exception as exc:  # noqa: BLE001 -- unreadable takes require regeneration, never acceptance.
            self._record({"label": label, "audio": str(audio), "stage": "audio-read",
                          "accepted": False, "error": type(exc).__name__})
            return None
        cache = self.work / f"{key(POLICY_VERSION, text, fingerprint)}.verified.json"
        if cache.exists():
            try:
                saved = json.loads(cache.read_text())
                cues = _usable_cues(saved["cues"], seconds)
                if saved["audio_sha256"] == fingerprint and voice.speech_verified(text, cues):
                    return cues
            except (OSError, ValueError, TypeError, KeyError):
                pass
        for stage in ("base-vad", "base-no-vad", "second-opinion-no-vad"):
            model = self._fallback_model() if stage == "second-opinion-no-vad" else self.recognizer
            if model is None:
                continue
            try:
                cues = voice.captions(audio, model, vad_filter=stage == "base-vad",
                                      beam_size=3 if stage == "base-vad" else 5,
                                      condition_on_previous_text=False)
                cues = _usable_cues(cues, seconds)
                verdict = voice.speech_check(text, cues)
            except Exception as exc:  # noqa: BLE001 -- includes lazy decoder failures; never accept them.
                self._record({"label": label, "audio": str(audio), "stage": stage,
                              "accepted": False, "error": type(exc).__name__})
                print(f"Speech recognition retry for {label}: {stage} raised {type(exc).__name__}.", flush=True)
                continue
            self._record({"label": label, "audio": str(audio), "audio_sha256": fingerprint,
                          "seconds": seconds, "stage": stage, "expected": text, **verdict})
            if verdict["accepted"]:
                save_json(cache, {"audio_sha256": fingerprint, "cues": cues, "stage": stage})
                if stage != "base-vad":
                    print(f"Speech verified for {label} with {stage}; original audio retained.", flush=True)
                return cues
            print(f"Speech check for {label} ({stage}): {verdict['reason']}; "
                  f"coverage {verdict['coverage']:.0%}. Expected: {text[:220]!r}; "
                  f"heard: {verdict['heard'][:220]!r}", flush=True)
        # Retain one small rejected recording for diagnosing failed jobs. It is
        # never considered verified or substituted into the finished video.
        diagnostic = self.work / "last-unverified.wav"
        try:
            if audio.stat().st_size <= 4*1024*1024:
                if audio != diagnostic:
                    shutil.copyfile(audio, diagnostic)
                self._record({"label": label, "audio": str(audio), "stage": "unverified-audio",
                              "accepted": False, "diagnostic": str(diagnostic)})
        except OSError as exc:
            self._record({"label": label, "audio": str(audio), "stage": "unverified-audio",
                          "accepted": False, "error": type(exc).__name__})
        return None

    def _generate(self, text, attempt):
        if self.speaker is None:
            self.speaker = self.speaker_factory()
        return self.speaker.speak(text, attempt_offset=attempt*3, force=True)

    def recover(self, text: str, initial: Path | None = None, *, label: str = "speech",
                max_seconds: float | None = None, allow_split: bool = True,
                attempts: int = 3) -> tuple[Path, list[dict]]:
        attempts = max(1, min(3, attempts))
        for attempt in range(attempts):
            try:
                audio = initial if attempt == 0 and initial is not None else self._generate(text, attempt+1)
                cues = self.check(text, audio, label=label)
                seconds = duration(audio) if cues is not None else None
            except Exception as exc:  # noqa: BLE001 -- a failed voice take does not exhaust the remaining bounded tries.
                self._record({"label": label, "stage": "voice-take", "attempt": attempt+1,
                              "accepted": False, "error": type(exc).__name__})
                cues = None
            if cues is not None:
                if max_seconds is None or seconds <= max_seconds:
                    return audio, cues
                # A long but correct take may be used elsewhere. Keep it intact;
                # retry the delivery rather than excessively speeding it up.
                self._record({"label": label, "audio": str(audio), "stage": "duration",
                              "accepted": False, "seconds": seconds, "maximum": max_seconds})
                print(f"{label} is complete but {seconds:.1f}s; trying another natural take.", flush=True)
            if attempt+1 < attempts:
                print(f"Regenerating {label} with the selected voice (take {attempt+2}/{attempts}).", flush=True)
        chunks = _speech_chunks(text) if allow_split else []
        # Normal script paragraphs produce two or three thoughts. Even unusually
        # large external inputs receive a bounded repair without losing words.
        if len(chunks) > 6:
            chunks = chunks[:5] + [" ".join(chunks[5:])]
        if len(chunks) > 1:
            print(f"Recovering {label} as {len(chunks)} complete thoughts with the same voice.", flush=True)
            paths, cues, offset = [], [], 0.0
            for index, chunk in enumerate(chunks):
                path, aligned = self.recover(chunk, label=f"{label}, thought {index+1}",
                                             allow_split=False, attempts=2)
                # Retain the exact verified bytes before another generation can
                # reuse its output path, including repeated identical thoughts.
                retained = self.work / f"{key('verified-thought', _fingerprint(path))}{path.suffix}"
                if retained != path:
                    shutil.copyfile(path, retained)
                paths.append(retained)
                cues.extend({**cue, "start": offset+cue["start"], "end": offset+cue["end"]} for cue in aligned)
                offset += duration(retained)
            output = self.work / f"{key('reconstructed', text, [_fingerprint(p) for p in paths])}.wav"
            listing = self.work / f"{output.stem}.concat.txt"
            listing.write_text("\n".join("file '"+str(path.resolve()).replace("'", "'\\''")+"'" for path in paths)+"\n")
            try:
                ffmpeg("-f", "concat", "-safe", "0", "-i", listing, "-vn", "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", output)
                output_seconds = duration(output)
                cues = _usable_cues(cues, output_seconds)
            except Exception as exc:
                self._record({"label": label, "stage": "thought-assembly", "accepted": False,
                              "error": type(exc).__name__})
                raise SpeechRecoveryError(f"Could not assemble verified speech for {label}. Details: {self.report}") from exc
            finally:
                listing.unlink(missing_ok=True)
            if voice.speech_verified(text, cues) and (max_seconds is None or output_seconds <= max_seconds):
                self._record({"label": label, "stage": "verified-thoughts", "audio": str(output),
                              "accepted": True, "parts": [str(path) for path in paths]})
                return output, cues
        raise SpeechRecoveryError(f"Could not verify complete speech for {label} after recognition retries and fresh voice takes. Details: {self.report}")

    def release_voice(self):
        """Release TTS memory while keeping the optional recognizer available."""
        if self.speaker is not None:
            try:
                self.speaker.close()
            finally:
                self.speaker = None
        gc.collect()

    def close(self):
        self.release_voice()
        self.fallback = None
        self.recognizer = None
        gc.collect()
