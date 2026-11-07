"""Registered Gemini character voices with an episode-wide local fallback.

Register voices separately with Google's Voices API using real same-adult source
and consent recordings. Set GEMINI_VOICE_WARGUY, GEMINI_VOICE_KIT_LASER or
GEMINI_VOICE_FEMALE to the returned voice_ ID / voicekey_ key. Preset names and
synthetic consent are never used here. Original references remain untouched.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import threading
import wave
from pathlib import Path

from . import gemini
from .core import PipelineError, ffmpeg, key, save_json
from .voice import REVEAL_LEAD, VOICE_CHAIN, phrases, selected_tts_model, sentences, spoken_form

CACHE_VERSION = "gemini-character-voice-v1"
MAX_AUDIO_BYTES = 64 * 1024 * 1024
DELIVERY_STYLES = {
    "normal": "Natural conversational film narration, clear and measured.",
    "calm": "Calm, reflective and warm; unhurried, clear articulation.",
    "hype": "Energetic and engaged, with natural emphasis; clear articulation.",
    "dramatic": "Controlled dramatic tension, deliberate pacing and natural pauses.",
}
_ROUTES = {}
_ROUTES_LOCK = threading.RLock()


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def registered_voice(reference: Path, narrator: str | None = None) -> str | None:
    """Only explicitly registered custom voices may represent these references."""
    name = narrator or os.getenv("NARRATOR", "").strip() or reference.stem
    name = re.sub(r"[\s_-]+", "", name).casefold()
    variable = {"warguy": "GEMINI_VOICE_WARGUY", "kitlaser": "GEMINI_VOICE_KIT_LASER",
                "female": "GEMINI_VOICE_FEMALE"}.get(name)
    value = os.getenv(variable, "").strip() if variable else ""
    return value if re.fullmatch(r"(?:voice_|voicekey_)[^\s]+", value) else None


def has_registered_voice(narrator: str | None = None) -> bool:
    name = narrator or os.getenv("NARRATOR", "").strip() or "warguy"
    return registered_voice(Path(name + ".ogg"), name) is not None


def route_summary(reference: Path, cache: Path) -> dict:
    """Safe reporting for CLI/checkpoints; never expose an ID, key or audio."""
    reference_hash = _fingerprint(Path(reference))
    identifier = (str(Path(cache).resolve()), reference_hash)
    route = _ROUTES.get(identifier)
    if route is not None:
        state = route.state
    else:
        path = Path(cache) / f"voice-backend-{reference_hash[:20]}.json"
        if path.exists():
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(state, dict):
                    raise ValueError("Invalid route state.")
            except (OSError, ValueError):
                return {"backend": "unknown", "status": "invalid_checkpoint"}
        else:
            voice_id = registered_voice(Path(reference))
            state = {"backend": "gemini" if gemini.enabled() and voice_id else "local",
                     "model": gemini.identity().get("tts_model", "gemini-3.8-flash-tts"),
                     "voice_hash": hashlib.sha256((voice_id or "").encode()).hexdigest(),
                     "started": False}
    backend = state.get("backend")
    return {"backend": backend,
            "model": state.get("model") if backend == "gemini" else selected_tts_model(),
            "voice_fingerprint": state.get("voice_hash") if backend == "gemini" else reference_hash,
            "status": "started" if state.get("started") else "selected",
            "reason": state.get("reason", "")}


def _wav_info(data: bytes) -> tuple[int, int, int, int]:
    """Decode all PCM frames; format validity does not establish spoken words."""
    if not isinstance(data, bytes) or not 44 <= len(data) <= MAX_AUDIO_BYTES:
        raise ValueError("Empty, oversized or truncated WAV audio.")
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Expected a RIFF/WAVE container.")
    if int.from_bytes(data[4:8], "little") + 8 != len(data):
        raise ValueError("WAV container length does not match its payload.")
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            rate, channels, width, frames = (
                audio.getframerate(), audio.getnchannels(),
                audio.getsampwidth(), audio.getnframes(),
            )
            if (audio.getcomptype() != "NONE" or channels != 1
                    or width not in (2, 3) or not 8000 <= rate <= 48000
                    or frames <= 0):
                raise ValueError("Unsupported WAV channel, sample or encoding metadata.")
            if len(audio.readframes(frames + 1)) != frames * channels * width:
                raise ValueError("Truncated PCM frames.")
    except (wave.Error, EOFError) as exc:
        raise ValueError("Invalid PCM WAV audio.") from exc
    return rate, channels, width, frames


def audio_wav(payload: bytes, mime: str) -> bytes:
    """Keep WAV intact; wrap only declared Gemini little-endian PCM once."""
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_AUDIO_BYTES:
        raise ValueError("Empty or oversized Gemini audio.")
    if not isinstance(mime, str):
        raise ValueError("Missing Gemini audio MIME type.")
    fields = [part.strip().lower() for part in mime.split(";")]
    media, parameters = fields[0], {}
    if media not in ("audio/wav", "audio/x-wav", "audio/l16", "audio/pcm"):
        raise ValueError("Unsupported Gemini audio MIME type.")
    for field in fields[1:]:
        if not field or "=" not in field:
            raise ValueError("Invalid Gemini audio metadata.")
        name, value = (part.strip().strip('"') for part in field.split("=", 1))
        if name in parameters:
            raise ValueError("Duplicate Gemini audio metadata.")
        parameters[name] = value
    if set(parameters) - {"rate", "channels", "codec", "bits"}:
        raise ValueError("Unsupported audio metadata.")
    # Sniff RIFF even when an upstream response incorrectly labels WAV as L16.
    if payload[:4] == b"RIFF":
        rate, channels, width, _ = _wav_info(payload)
        try:
            if any(int(parameters[name]) != expected for name, expected in
                   (("rate", rate), ("channels", channels), ("bits", width * 8))
                   if name in parameters):
                raise ValueError("WAV metadata contradicts the decoded frames.")
        except ValueError as exc:
            raise ValueError("Invalid or contradictory WAV numeric metadata.") from exc
        if parameters.get("codec", "pcm") not in ("pcm", f"pcm_s{width * 8}le"):
            raise ValueError("Unsupported WAV codec.")
        return payload
    if media in ("audio/wav", "audio/x-wav"):
        raise ValueError("Declared WAV response has no WAV header.")
    if parameters.get("codec", "pcm") not in ("pcm", "pcm_s16le"):
        raise ValueError("Unsupported PCM codec.")
    try:
        rate = int(parameters.get("rate", "24000"))
        channels = int(parameters.get("channels", "1"))
        bits = int(parameters.get("bits", "16"))
    except ValueError as exc:
        raise ValueError("Invalid PCM numeric metadata.") from exc
    if rate not in (8000, 16000, 22050, 24000, 32000, 44100, 48000) or channels != 1 or bits != 16:
        raise ValueError("Unsupported PCM rate, channel count or bit depth.")
    if len(payload) % 2:
        raise ValueError("Incomplete PCM16 frame.")
    result = io.BytesIO()
    with wave.open(result, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(payload)
    data = result.getvalue()
    _wav_info(data)
    return data


class EpisodeVoiceRoute:
    """A shared, checkpointed backend choice; never change speakers mid-episode."""

    def __init__(self, cache: Path, reference_hash: str):
        self.path = cache / f"voice-backend-{reference_hash[:20]}.json"
        self.lock = threading.RLock()
        self.state = {"version": 1, "reference_sha256": reference_hash,
                      "backend": None, "started": False}
        if self.path.exists():
            try:
                state = json.loads(self.path.read_text(encoding="utf-8"))
                if (state.get("version") != 1 or state.get("reference_sha256") != reference_hash
                        or state.get("backend") not in ("gemini", "local")
                        or not isinstance(state.get("started"), bool)):
                    raise ValueError("Invalid route identity.")
                self.state = state
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                raise PipelineError("Cannot read the episode voice backend checkpoint; "
                                    "restore it or start a fresh run directory.") from exc

    def _save(self):
        save_json(self.path, self.state)

    def choose(self, available: bool, model: str, voice_hash: str):
        with self.lock:
            backend = self.state["backend"]
            if backend == "local":
                return
            if backend == "gemini":
                if not available or self.state.get("model") != model or self.state.get("voice_hash") != voice_hash:
                    if self.state["started"]:
                        raise PipelineError("This episode already uses a registered Gemini voice. "
                                            "Restore the same API key access, TTS model and voice ID; "
                                            "a different speaker will not be substituted.")
                    self.fallback("configuration_unavailable")
                return
            self.state.update(backend="gemini" if available else "local",
                              model=model, voice_hash=voice_hash)
            self._save()
            gemini.record_event("voice", self.state["backend"], "selected",
                                reason="registered_voice" if available else "local_or_unregistered_voice")

    def started(self):
        with self.lock:
            self.state["started"] = True
            self._save()

    def fallback(self, reason: str):
        with self.lock:
            if self.state["started"]:
                gemini.record_event("voice", "gemini", "failed", reason=reason)
                raise PipelineError("Gemini narration failed after this episode began using its "
                                    "registered voice. Retry with the same model and voice ID, "
                                    "or start a fresh episode with AI_BACKEND=local. "
                                    "The pipeline will not change speakers mid-episode.")
            self.state.update(backend="local", reason=reason)
            self._save()
            gemini.record_event("voice", "local", "fallback", reason=reason)


def _episode_route(cache: Path, reference_hash: str) -> EpisodeVoiceRoute:
    identifier = (str(cache.resolve()), reference_hash)
    with _ROUTES_LOCK:
        if identifier not in _ROUTES:
            _ROUTES[identifier] = EpisodeVoiceRoute(cache, reference_hash)
        return _ROUTES[identifier]


class GeminiNarrator:
    """Narrator-compatible adapter; local construction remains lazy and unchanged."""

    def __init__(self, reference: Path, cache: Path, local_factory, *,
                 narrator: str | None = None, route: EpisodeVoiceRoute | None = None):
        self.reference, self.cache = Path(reference), Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.reference_hash = _fingerprint(self.reference)
        self.voice_id = registered_voice(self.reference, narrator)
        self.voice_hash = hashlib.sha256((self.voice_id or "").encode()).hexdigest()
        self.model_name = str(gemini.identity().get("tts_model", "gemini-3.8-flash-tts"))
        self.local_factory, self.local = local_factory, None
        self.deliveries, self.emotion_enabled, self.takes = {}, True, 1
        self.route = route or _episode_route(self.cache, self.reference_hash)
        self.route.choose(bool(gemini.enabled() and self.voice_id), self.model_name, self.voice_hash)

    @property
    def backend(self) -> str:
        return self.route.state["backend"]

    def _local(self):
        if self.local is None:
            self.local = self.local_factory(self.reference, self.cache)
        return self.local

    def _cache_path(self, text: str, delivery: str, style: str,
                    attempt_offset: int, force: bool) -> Path:
        baseline = key(CACHE_VERSION, spoken_form(text), self.reference_hash,
                       self.model_name, self.voice_hash, delivery, style)
        token = key("gemini-voice-retry-v1", baseline, attempt_offset) if force else baseline
        output = self.cache / f"{token}.wav"
        if force:
            suffix = 0
            while output.exists() or output.with_suffix(".captions.json").exists():
                suffix += 1
                output = self.cache / f"{token}-{suffix}.wav"
        return output

    def _generate(self, text: str, output: Path, style: str):
        client = gemini.get_client()
        if str(client.tts_model) != self.model_name:
            raise ValueError("Gemini TTS model changed within the episode.")
        chunks = phrases(sentences(spoken_form(text)), limit=120, maximum=120)
        frames, metadata = [], None
        for chunk in chunks:
            data, mime = client.generate_audio(chunk, self.voice_id, style=style)
            data = audio_wav(data, mime)
            rate, channels, width, count = _wav_info(data)
            seconds = count / rate
            if not 0.15 <= seconds <= max(30, len(chunk.split()) * 2 + 10):
                raise ValueError("Gemini audio duration is empty or runaway.")
            # A valid provider take locks the identity, even if later conversion
            # fails. Frame validity is NOT speech acceptance; ASR still follows.
            self.route.started()
            current = (rate, channels, width)
            if metadata is not None and metadata != current:
                raise ValueError("Gemini changed PCM format between phrases.")
            metadata = current
            with wave.open(io.BytesIO(data), "rb") as audio:
                frames.append(audio.readframes(count))
        if metadata is None:
            raise ValueError("Gemini returned no audio.")
        raw_file = tempfile.NamedTemporaryFile(dir=self.cache, suffix=".source.wav", delete=False)
        raw = Path(raw_file.name)
        raw_file.close()
        rendered_file = tempfile.NamedTemporaryFile(dir=self.cache, suffix=".render.wav", delete=False)
        rendered = Path(rendered_file.name)
        rendered_file.close()
        try:
            with wave.open(str(raw), "wb") as audio:
                audio.setframerate(metadata[0])
                audio.setnchannels(metadata[1])
                audio.setsampwidth(metadata[2])
                audio.writeframes(b"".join(frames))
            reveal = bool(re.match(r"\s*Number\s+\w+\s*:", text))
            lead = int((REVEAL_LEAD if reveal else 0.16) * 1000)
            ffmpeg("-i", raw, "-af", VOICE_CHAIN.format(lead=lead),
                   "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", rendered)
            rate, channels, _, count = _wav_info(rendered.read_bytes())
            if rate != 48000 or channels != 1 or not 0.15 <= count / rate <= max(30, len(text.split()) * 2 + 10):
                raise ValueError("Invalid converted Gemini narration.")
            rendered.replace(output)
            try:
                shutil.copyfile(raw, output.with_suffix(".dry.wav"))
            except OSError:
                pass  # Matched diagnostics are optional; the rendered take is retained.
        finally:
            raw.unlink(missing_ok=True)
            rendered.unlink(missing_ok=True)

    def speak(self, text: str, attempt_offset: int = 0, *, force: bool = False,
              delivery: str | None = None) -> Path:
        if delivery:
            self.deliveries[text] = delivery
        delivery = delivery or self.deliveries.get(text, "normal")
        # Serializing takes also makes the first-failure fallback decision atomic.
        with self.route.lock:
            if self.backend == "local":
                return self._local().speak(text, attempt_offset, force=force, delivery=delivery)
            style = DELIVERY_STYLES.get(delivery, delivery)
            output = self._cache_path(text, delivery, style, attempt_offset, force)
            if not force and output.exists():
                try:
                    rate, channels, _, count = _wav_info(output.read_bytes())
                    if rate != 48000 or channels != 1 or count / rate <= 0.5:
                        raise ValueError("Invalid cached Gemini narration.")
                    self.route.started()
                    gemini.record_event("voice", "gemini", "cache_hit",
                                        details={"model": self.model_name, "voice_hash": self.voice_hash})
                    return output
                except (OSError, ValueError):
                    pass
            try:
                self._generate(text, output, style)
            except (ValueError, OSError, RuntimeError) as exc:
                self.route.fallback(type(exc).__name__)
                return self._local().speak(text, attempt_offset, force=force, delivery=delivery)
            gemini.record_event("voice", "gemini", "generated",
                                details={"model": self.model_name, "voice_hash": self.voice_hash,
                                         "speech_verification": "pending"})
            return output

    def close(self):
        if self.local is not None:
            self.local.close()
            self.local = None


def create_narrator(reference: Path, cache: Path, local_factory, *,
                    narrator: str | None = None) -> GeminiNarrator:
    return GeminiNarrator(reference, cache, local_factory, narrator=narrator)


factory = create_narrator
