import io
import json
import shutil
import wave
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from movie_pipeline import gemini_voice as voice
from movie_pipeline.core import PipelineError


def wav_bytes(rate=24000, seconds=1, width=2, frames=None):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(width)
        audio.setframerate(rate)
        audio.writeframes(frames if frames is not None else b"\x01" * width * int(rate * seconds))
    return stream.getvalue()


@pytest.fixture
def setup_voice(monkeypatch, tmp_path):
    voice._ROUTES.clear()
    monkeypatch.setenv("NARRATOR", "warguy")
    monkeypatch.setenv("GEMINI_VOICE_WARGUY", "voice_registered_warguy")
    monkeypatch.setattr(voice.gemini, "enabled", lambda: True)
    monkeypatch.setattr(voice.gemini, "identity", lambda: {"tts_model": "gemini-3.8-flash-tts"})
    client = SimpleNamespace(tts_model="gemini-3.8-flash-tts",
                             generate_audio=Mock(return_value=(wav_bytes(), "audio/wav")))
    get_client = Mock(return_value=client)
    events = Mock()
    monkeypatch.setattr(voice.gemini, "get_client", get_client)
    monkeypatch.setattr(voice.gemini, "record_event", events)
    local_path = tmp_path / "local.wav"
    local = SimpleNamespace(speak=Mock(return_value=local_path), close=Mock())
    local_factory = Mock(return_value=local)
    reference = tmp_path / "warguy.ogg"
    reference.write_bytes(b"unchanged-user-reference")
    cache = tmp_path / "speech"
    conversions = []

    def convert(*args):
        raw = args[args.index("-i") + 1]
        source = raw.read_bytes()
        rate, channels, width, count = voice._wav_info(source)
        conversions.append((args, source))
        args[-1].write_bytes(wav_bytes(rate=48000, seconds=count / rate, width=2))

    monkeypatch.setattr(voice, "ffmpeg", convert)
    return SimpleNamespace(reference=reference, cache=cache, client=client,
                           get_client=get_client, events=events, local=local,
                           factory=local_factory, conversions=conversions,
                           build=lambda: voice.create_narrator(reference, cache, local_factory))


def test_wav_is_preserved_instead_of_receiving_a_second_header():
    data = wav_bytes()
    assert voice.audio_wav(data, "audio/wav") == data
    assert voice.audio_wav(data, "audio/L16;codec=pcm;rate=24000") == data


def test_declared_pcm_has_exactly_one_header_and_preserves_frame_bytes():
    pcm = b"\x01\x00\xff\x7f" * 6000
    result = voice.audio_wav(pcm, "audio/L16;codec=pcm;rate=24000;channels=1")
    with wave.open(io.BytesIO(result), "rb") as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (24000, 1, 2)
        assert audio.readframes(audio.getnframes()) == pcm
    assert result.count(b"RIFF") == 1


@pytest.mark.parametrize("payload,mime", [
    (b"\x00", "audio/l16;rate=24000"),
    (b"\x00\x00", "audio/l16;rate=0"),
    (b"\x00\x00", "audio/l16;rate=abc"),
    (b"\x00\x00", "audio/l16;channels=2"),
    (b"\x00\x00", "audio/l16;bits=24"),
    (b"\x00\x00", "audio/l16;codec=pcm_s16be"),
    (b"\x00\x00", "audio/l16;rate=24000;rate=16000"),
    (b"\x00\x00", "audio/l16;unexpected=1"),
    (b"\x00\x00", "audio/wav"),
    (b"\x00\x00", "audio/mp3"),
    (b"", "audio/l16"),
    (b"not-audio", None),
])
def test_rejects_invalid_audio_metadata_and_incomplete_frames(payload, mime):
    with pytest.raises(ValueError):
        voice.audio_wav(payload, mime)


def test_rejects_truncated_or_contradictory_wav():
    data = wav_bytes()
    for payload, mime in [(data[:-2], "audio/wav"), (data, "audio/wav;rate=16000"),
                          (data, "audio/wav;channels=two")]:
        with pytest.raises(ValueError):
            voice.audio_wav(payload, mime)


def test_audio_size_is_bounded(monkeypatch):
    monkeypatch.setattr(voice, "MAX_AUDIO_BYTES", 16)
    with pytest.raises(ValueError, match="oversized"):
        voice.audio_wav(b"\x00" * 18, "audio/l16")


def test_decodable_silence_is_not_claimed_as_verified_speech():
    # Decoding is a container check. Existing ASR completeness checks remain the
    # only acceptance evidence and reject empty transcripts, regardless of PCM.
    from movie_pipeline.voice import speech_check

    pcm = b"\x00\x00" * 24000
    assert voice.audio_wav(pcm, "audio/l16")
    assert not speech_check("Words must really be spoken.", [])["accepted"]


def test_registered_voice_maps_all_existing_characters(monkeypatch, tmp_path):
    monkeypatch.delenv("NARRATOR", raising=False)
    for name, variable in [("warguy", "WARGUY"), ("kit-laser", "KIT_LASER"), ("female", "FEMALE")]:
        monkeypatch.setenv("GEMINI_VOICE_" + variable, "voice_custom_" + variable)
        assert voice.registered_voice(tmp_path / (name + ".ogg")) == "voice_custom_" + variable
    monkeypatch.setenv("GEMINI_VOICE_WARGUY", "Kore")
    assert voice.registered_voice(tmp_path / "warguy.ogg") is None
    assert voice.registered_voice(tmp_path / "unknown.ogg") is None


def test_valid_generation_uses_registered_identity_and_converts_to_48k(setup_voice):
    env = setup_voice
    narrator = env.build()
    path = narrator.speak("Number 3: M3GAN. A complete thought.", delivery="dramatic")
    env.factory.assert_not_called()
    env.client.generate_audio.assert_called_once_with(
        "Number 3: Megan. A complete thought.", "voice_registered_warguy",
        style=voice.DELIVERY_STYLES["dramatic"])
    assert voice._wav_info(path.read_bytes())[:3] == (48000, 1, 2)
    args, raw = env.conversions[0]
    assert "adelay=180" in args[args.index("-af") + 1]
    assert args[args.index("-c:a") + 1] == "pcm_s16le"
    assert raw == wav_bytes()
    assert env.reference.read_bytes() == b"unchanged-user-reference"
    assert path.with_suffix(".dry.wav").read_bytes() == raw
    saved = narrator.route.path.read_text()
    assert "voice_registered_warguy" not in saved
    assert json.loads(saved)["started"] is True


@pytest.mark.parametrize("missing_id,disabled", [(True, False), (False, True), (True, True)])
def test_missing_registered_id_or_api_key_uses_unchanged_local_factory(
        setup_voice, monkeypatch, missing_id, disabled):
    env = setup_voice
    if missing_id:
        monkeypatch.delenv("GEMINI_VOICE_WARGUY")
    monkeypatch.setattr(voice.gemini, "enabled", lambda: not disabled)
    narrator = env.build()
    assert narrator.speak("A local line.", 12, force=True, delivery="calm") == env.local.speak.return_value
    env.factory.assert_called_once_with(env.reference, env.cache)
    env.local.speak.assert_called_once_with("A local line.", 12, force=True, delivery="calm")
    env.get_client.assert_not_called()
    narrator.close()
    env.local.close.assert_called_once()


def test_stock_preset_is_never_substituted_for_character(setup_voice, monkeypatch):
    monkeypatch.setenv("GEMINI_VOICE_WARGUY", "Algenib")
    narrator = setup_voice.build()
    narrator.speak("Still the original character.")
    assert narrator.backend == "local"
    setup_voice.client.generate_audio.assert_not_called()


def test_first_request_failure_locks_local_for_recovery_and_next_stage(setup_voice):
    env = setup_voice
    env.client.generate_audio.side_effect = ValueError("quota exhausted")
    first = env.build()
    first.speak("First line.", delivery="hype")
    assert first.backend == "local"
    first.close()
    # A different object and a later process both keep the original choice.
    env.client.generate_audio.side_effect = None
    env.build().speak("Second line.")
    voice._ROUTES.clear()
    restored = env.build()
    restored.speak("Third line.")
    assert restored.backend == "local"
    assert env.client.generate_audio.call_count == 1
    assert env.local.speak.call_count == 3


def test_backend_never_changes_after_any_gemini_take(setup_voice):
    env = setup_voice
    first = env.build()
    first.speak("The opening line.")
    env.client.generate_audio.side_effect = ValueError("quota exhausted")
    first.close()
    voice._ROUTES.clear()
    recovered = env.build()
    with pytest.raises(PipelineError, match="will not change speakers"):
        recovered.speak("The next line.")
    assert recovered.backend == "gemini"
    env.factory.assert_not_called()


def test_lost_key_or_changed_voice_cannot_switch_an_existing_episode(setup_voice, monkeypatch):
    env = setup_voice
    env.build().speak("Opening.")
    monkeypatch.setattr(voice.gemini, "enabled", lambda: False)
    with pytest.raises(PipelineError, match="already uses"):
        env.build()
    monkeypatch.setattr(voice.gemini, "enabled", lambda: True)
    monkeypatch.setenv("GEMINI_VOICE_WARGUY", "voice_different")
    with pytest.raises(PipelineError, match="already uses"):
        env.build()
    env.factory.assert_not_called()


def test_conversion_failure_after_valid_audio_never_changes_speaker(setup_voice, monkeypatch):
    def bad_conversion(*args):
        raise PipelineError("conversion failed")

    monkeypatch.setattr(voice, "ffmpeg", bad_conversion)
    narrator = setup_voice.build()
    with pytest.raises(PipelineError, match="will not change speakers"):
        narrator.speak("The complete opening.")
    setup_voice.factory.assert_not_called()


def test_cache_reuse_delivery_memory_and_forced_attempts_preserve_existing_takes(setup_voice):
    env = setup_voice
    narrator = env.build()
    text = "The same complete script."
    original = narrator.speak(text, delivery="calm")
    original_bytes = original.read_bytes()
    original.with_suffix(".captions.json").write_text("verified captions")
    assert narrator.speak(text) == original
    assert env.client.generate_audio.call_count == 1
    forced = narrator.speak(text, attempt_offset=3, force=True)
    next_forced = narrator.speak(text, attempt_offset=3, force=True)
    assert len({original, forced, next_forced}) == 3
    assert original.read_bytes() == original_bytes
    assert original.with_suffix(".captions.json").read_text() == "verified captions"


def test_cache_identity_includes_reference_model_voice_and_delivery(setup_voice):
    narrator = setup_voice.build()

    def cache_path():
        return narrator._cache_path("Script.", "normal", "style", 0, False)

    original = cache_path()
    narrator.reference_hash = "different-reference-sha256"
    assert cache_path() != original
    narrator.reference_hash = voice._fingerprint(setup_voice.reference)
    narrator.voice_hash = "different-registered-voice-hash"
    assert cache_path() != original
    narrator.voice_hash = __import__("hashlib").sha256(narrator.voice_id.encode()).hexdigest()
    narrator.model_name = "gemini-3.8-flash-lite-tts"
    assert cache_path() != original
    narrator.model_name = "gemini-3.8-flash-tts"
    assert narrator._cache_path("Script.", "dramatic", "style", 0, False) != original


def test_multi_phrase_generation_keeps_every_word_and_one_backend(setup_voice):
    env = setup_voice
    text = " ".join("word" + str(index) for index in range(121)) + "."
    narrator = env.build()
    path = narrator.speak(text)
    calls = env.client.generate_audio.call_args_list
    assert len(calls) == 2
    assert " ".join(call.args[0] for call in calls) == text
    assert all(call.args[1] == "voice_registered_warguy" for call in calls)
    assert voice._wav_info(path.read_bytes())[3] == 96000
    env.factory.assert_not_called()


def test_later_phrase_failure_keeps_gemini_lock(setup_voice):
    env = setup_voice
    env.client.generate_audio.side_effect = [(wav_bytes(), "audio/wav"), ValueError("unavailable")]
    narrator = env.build()
    with pytest.raises(PipelineError, match="will not change speakers"):
        narrator.speak(" ".join(["word"] * 121))
    env.factory.assert_not_called()


def test_corrupt_backend_checkpoint_requires_explicit_recovery(setup_voice):
    env = setup_voice
    narrator = env.build()
    narrator.route.path.write_text("{broken")
    voice._ROUTES.clear()
    with pytest.raises(PipelineError, match="checkpoint"):
        env.build()


def test_registered_helper_and_route_summary_never_expose_id(setup_voice, monkeypatch):
    env = setup_voice
    assert voice.has_registered_voice("Warguy")
    narrator = env.build()
    summary = voice.route_summary(env.reference, env.cache)
    assert summary["backend"] == "gemini"
    assert summary["model"] == "gemini-3.8-flash-tts"
    assert "voice_registered_warguy" not in json.dumps(summary)
    assert summary["status"] == "selected"
    narrator.speak("An opening line.")
    assert voice.route_summary(env.reference, env.cache)["status"] == "started"
    monkeypatch.delenv("GEMINI_VOICE_WARGUY")
    assert not voice.has_registered_voice("warguy")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_real_ffmpeg_output_is_decodable_on_supported_python_311(setup_voice, monkeypatch):
    # PCM24 FFmpeg WAV uses WAVE_FORMAT_EXTENSIBLE, which wave cannot read on
    # Python 3.11. The production adapter must emit classic PCM16 instead.
    from movie_pipeline.core import ffmpeg

    monkeypatch.setattr(voice, "ffmpeg", ffmpeg)
    output = setup_voice.build().speak("A complete narrated sentence.")
    assert voice._wav_info(output.read_bytes())[:3] == (48000, 1, 2)
