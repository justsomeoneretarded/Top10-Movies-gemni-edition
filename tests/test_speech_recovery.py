import json
import math
import wave
from array import array
from collections import deque
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from movie_pipeline import speech_recovery as recovery
from movie_pipeline.core import PipelineError, duration

TEXT = "Hit like and subscribe because your movie nights deserve better."


class Recognizer:
    """Exercise the real captions adapter without loading an ASR model."""

    def __init__(self, *answers, callback=None):
        self.answers = deque(answers)
        self.callback = callback
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((Path(audio), kwargs))
        answer = self.callback(Path(audio), kwargs) if self.callback else self.answers.popleft()

        def segments():
            if isinstance(answer, Exception):
                raise answer
            yield SimpleNamespace(words=[], text=answer, start=.1, end=4.2)

        return segments(), SimpleNamespace()


@pytest.fixture
def audio(tmp_path, monkeypatch):
    path = tmp_path / "original.wav"
    path.write_bytes(b"original recorded audio")
    monkeypatch.setattr(recovery, "duration", lambda path: 4.4)
    return path


def make_recovery(tmp_path, model, *, speaker=None, fallback_factory=None):
    speaker_factory = Mock(return_value=speaker)
    instance = recovery.SpeechRecovery(model, speaker_factory, tmp_path / "checks",
                                       fallback_factory=fallback_factory)
    return instance, speaker_factory


def test_short_clip_retries_same_bytes_without_vad_before_generating(audio, tmp_path):
    model = Recognizer("because your movie nights deserve better", TEXT)
    fallback = Mock()
    service, speaker = make_recovery(tmp_path, model, fallback_factory=fallback)
    before = audio.read_bytes()

    result, cues = service.recover(TEXT, audio, max_seconds=8, label="midpoint CTA")

    assert result == audio
    assert audio.read_bytes() == before
    assert " ".join(cue["text"] for cue in cues) == TEXT
    speaker.assert_not_called()
    fallback.assert_not_called()
    assert [path for path, _ in model.calls] == [audio, audio]
    assert [options["vad_filter"] for _, options in model.calls] == [True, False]
    assert [options["beam_size"] for _, options in model.calls] == [3, 5]
    for _, options in model.calls:
        assert options["condition_on_previous_text"] is False
        assert "initial_prompt" not in options
        assert "hotwords" not in options


def test_nested_cached_cues_are_recognized_again_instead_of_losing_subtitle_words(audio, tmp_path):
    from movie_pipeline.core import key

    model = Recognizer(TEXT)
    service, speaker = make_recovery(tmp_path, model)
    fingerprint = recovery._fingerprint(audio)
    cache = service.work / f"{key(recovery.POLICY_VERSION, TEXT, fingerprint)}.verified.json"
    cache.write_text(json.dumps({"audio_sha256": fingerprint, "cues": [
        {"start": 0, "end": 4, "text": "Hit like and subscribe"},
        {"start": 1, "end": 2, "text": "because your movie nights deserve better."},
    ]}))

    cues = service.check(TEXT, audio)

    assert len(model.calls) == 1
    assert " ".join(cue["text"] for cue in cues) == TEXT
    assert all(left["end"] <= right["start"] for left, right in pairwise(cues))
    speaker.assert_not_called()


def test_stronger_recognizer_is_lazy_reused_and_unprompted(audio, tmp_path):
    base = Recognizer(callback=lambda *_: "wrong movie")
    stronger = Recognizer(TEXT, "A second complete sentence belongs to another review.")
    fallback = Mock(return_value=stronger)
    service, speaker = make_recovery(tmp_path, base, fallback_factory=fallback)

    assert service.check(TEXT, audio)
    other = audio.with_name("another.wav")
    other.write_bytes(b"another recording")
    assert service.check("A second complete sentence belongs to another review.", other)

    fallback.assert_called_once_with()
    speaker.assert_not_called()
    assert len(base.calls) == 4
    assert len(stronger.calls) == 2
    for _, options in stronger.calls:
        assert options == {"language": "en", "beam_size": 5, "word_timestamps": True,
                           "vad_filter": False, "condition_on_previous_text": False}


def test_opposing_incomplete_hypotheses_are_never_combined(audio, tmp_path):
    text = "These incredible movies make every quiet weekend feel exciting with unforgettable characters and surprising stories worth your time."
    words = text.split()
    base = Recognizer(" ".join(words[4:]), " ".join(words[:-4]))
    stronger = Recognizer(" ".join(words[4:]))
    service, _ = make_recovery(tmp_path, base, fallback_factory=lambda: stronger)

    assert service.check(text, audio) is None
    assert not list(service.work.glob("*.verified.json"))
    assert all(not check["accepted"] for check in service.events)
    assert {check["reason"] for check in service.events if "reason" in check} == {"opening", "ending"}


def test_success_cache_is_bound_to_exact_audio_bytes_and_script(audio, tmp_path):
    model = Recognizer(TEXT, "unrelated missing speech", "unrelated missing speech", TEXT)
    service, _ = make_recovery(tmp_path, model)
    assert service.check(TEXT, audio)
    assert service.check(TEXT, audio)
    assert len(model.calls) == 1

    # Same filename and duration cannot validate a new take with old captions.
    audio.write_bytes(b"new recording with missing speech")
    assert service.check(TEXT, audio) is None
    assert len(model.calls) == 3
    assert service.check(TEXT, audio)
    assert len(model.calls) == 4


def test_cache_for_another_script_is_not_reused(audio, tmp_path):
    model = Recognizer(TEXT, "Different words make a completely separate closing sentence.")
    service, _ = make_recovery(tmp_path, model)
    assert service.check(TEXT, audio)
    assert service.check("Different words make a completely separate closing sentence.", audio)
    assert len(model.calls) == 2


def test_lazy_decoder_exception_uses_next_recognition_pass(audio, tmp_path):
    model = Recognizer(RuntimeError("error while iterating decoded segments"), TEXT)
    service, speaker = make_recovery(tmp_path, model)
    assert service.recover(TEXT, audio)[0] == audio
    speaker.assert_not_called()
    assert service.events[0]["error"] == "RuntimeError"
    assert service.events[1]["accepted"]


def test_unavailable_second_model_still_permits_fresh_same_voice(audio, tmp_path):
    fresh = audio.with_name("fresh.wav")
    fresh.write_bytes(b"fresh recording")
    model = Recognizer(callback=lambda path, _: TEXT if path == fresh else "bad speech")
    fallback = Mock(side_effect=OSError("model download unavailable"))
    narrator = SimpleNamespace(speak=Mock(return_value=fresh), close=Mock())
    service, factory = make_recovery(tmp_path, model, speaker=narrator, fallback_factory=fallback)

    result, _ = service.recover(TEXT, audio)

    assert result == fresh
    fallback.assert_called_once_with()
    factory.assert_called_once_with()
    narrator.speak.assert_called_once_with(TEXT, attempt_offset=6, force=True)
    assert any(event.get("stage") == "second-opinion-load" for event in service.events)


def test_fresh_generation_and_offsets_are_used_after_a_failed_take(audio, tmp_path):
    bad = audio.with_name("bad.wav")
    good = audio.with_name("good.wav")
    bad.write_bytes(b"bad take")
    good.write_bytes(b"good take")
    model = Recognizer(callback=lambda path, _: TEXT if path == good else "missing speech")
    narrator = SimpleNamespace(speak=Mock(side_effect=[bad, good]), close=Mock())
    service, factory = make_recovery(tmp_path, model, speaker=narrator)

    result, _ = service.recover(TEXT, audio)

    assert result == good
    factory.assert_called_once_with()
    assert [call.kwargs for call in narrator.speak.call_args_list] == [
        {"attempt_offset": 6, "force": True}, {"attempt_offset": 9, "force": True}]


def test_voice_generation_exception_does_not_abort_remaining_attempts(audio, tmp_path):
    model = Recognizer(TEXT)
    narrator = SimpleNamespace(speak=Mock(side_effect=[PipelineError("empty take"), audio]), close=Mock())
    service, _ = make_recovery(tmp_path, model, speaker=narrator)

    assert service.recover(TEXT)[0] == audio
    assert narrator.speak.call_count == 2
    assert service.events[0]["stage"] == "voice-take"
    assert service.events[0]["error"] == "PipelineError"


def test_broken_initial_audio_is_replaced_without_weakening_verification(audio, tmp_path, monkeypatch):
    missing = audio.with_name("missing.wav")
    monkeypatch.setattr(recovery, "duration", lambda path: 4.4 if path.exists() else (_ for _ in ()).throw(PipelineError("missing audio")))
    narrator = SimpleNamespace(speak=Mock(return_value=audio), close=Mock())
    service, _ = make_recovery(tmp_path, Recognizer(TEXT), speaker=narrator)

    assert service.recover(TEXT, missing)[0] == audio
    assert narrator.speak.call_count == 1
    assert service.events[0]["stage"] == "audio-read"


def test_duration_cap_tries_natural_take_and_preserves_long_source(audio, tmp_path, monkeypatch):
    fresh = audio.with_name("short-natural-take.wav")
    fresh.write_bytes(b"complete faster natural recording")
    original = audio.read_bytes()
    monkeypatch.setattr(recovery, "duration", lambda path: 9 if path == audio else 7.2)
    narrator = SimpleNamespace(speak=Mock(return_value=fresh), close=Mock())
    service, _ = make_recovery(tmp_path, Recognizer(TEXT, TEXT), speaker=narrator)

    result, _ = service.recover(TEXT, audio, max_seconds=8)

    assert result == fresh
    assert audio.read_bytes() == original
    assert any(event.get("stage") == "duration" for event in service.events)


def test_all_generation_failures_are_bounded_and_reported(tmp_path):
    narrator = SimpleNamespace(speak=Mock(side_effect=RuntimeError("TTS unavailable")), close=Mock())
    service, _ = make_recovery(tmp_path, Recognizer(), speaker=narrator)

    with pytest.raises(recovery.SpeechRecoveryError, match="after recognition retries"):
        service.recover(TEXT, allow_split=False, attempts=99)

    assert narrator.speak.call_count == 3
    assert len(json.loads(service.report.read_text())["checks"]) == 3


def test_all_incomplete_takes_remain_rejected(audio, tmp_path):
    model = Recognizer(callback=lambda *_: "subscribe because your movie nights deserve better")
    narrator = SimpleNamespace(speak=Mock(return_value=audio), close=Mock())
    service, _ = make_recovery(tmp_path, model, speaker=narrator)

    with pytest.raises(recovery.SpeechRecoveryError):
        service.recover(TEXT, allow_split=False)

    assert narrator.speak.call_count == 3
    assert len(model.calls) == 6
    assert not any(event["accepted"] for event in service.events)


def test_malformed_cues_cannot_hide_or_invent_valid_speech(audio, tmp_path, monkeypatch):
    malformed = [{}, None, {"start": None, "end": 1, "text": TEXT},
                 {"start": math.nan, "end": 2, "text": TEXT},
                 {"start": 2, "end": 1, "text": TEXT},
                 {"start": 8, "end": 9, "text": TEXT}]
    decoder = Mock(return_value=malformed)
    monkeypatch.setattr(recovery.voice, "captions", decoder)
    service, _ = make_recovery(tmp_path, object())
    assert service.check(TEXT, audio) is None
    decoder.return_value = malformed + [{"start": -.1, "end": 5, "text": TEXT}]
    assert service.check(TEXT, audio) == [{"start": 0, "end": 4.4, "text": TEXT}]


def test_corrupt_verification_cache_is_rechecked(audio, tmp_path):
    model = Recognizer(TEXT, TEXT)
    service, _ = make_recovery(tmp_path, model)
    assert service.check(TEXT, audio)
    cache = next(service.work.glob("*.verified.json"))
    cache.write_text('{"cues": null}')
    assert service.check(TEXT, audio)
    assert len(model.calls) == 2


def test_rejected_audio_diagnostic_is_single_bounded_copy(audio, tmp_path):
    model = Recognizer(callback=lambda *_: "wrong movie")
    service, _ = make_recovery(tmp_path, model)
    assert service.check(TEXT, audio, label="failed CTA") is None
    diagnostic = service.work / "last-unverified.wav"
    assert diagnostic.read_bytes() == audio.read_bytes()
    assert service.events[-1]["diagnostic"] == str(diagnostic)
    assert service.events[-1]["label"] == "failed CTA"

    audio.write_bytes(b"next rejected recording")
    assert service.check(TEXT, audio) is None
    assert diagnostic.read_bytes() == b"next rejected recording"
    assert len(list(service.work.glob("last-unverified*"))) == 1

    audio.write_bytes(b"x" * (4*1024*1024+1))
    assert service.check(TEXT, audio) is None
    assert diagnostic.stat().st_size <= 4*1024*1024


def test_release_voice_keeps_second_opinion_and_can_recreate_speaker(audio, tmp_path):
    stronger = Recognizer(TEXT, TEXT)
    narrator = SimpleNamespace(speak=Mock(return_value=audio), close=Mock())
    fallback = Mock(return_value=stronger)
    base = Recognizer(callback=lambda *_: "incomplete")
    service, factory = make_recovery(tmp_path, base, speaker=narrator, fallback_factory=fallback)
    assert service.recover(TEXT)[0] == audio

    service.release_voice()
    assert service.speaker is None
    assert service.fallback is stronger
    narrator.close.assert_called_once_with()
    assert service.recover(TEXT)[0] == audio
    assert factory.call_count == 2
    fallback.assert_called_once_with()

    service.close()
    assert service.speaker is None
    assert service.fallback is None
    assert narrator.close.call_count == 2


def write_wave(path, seconds, value):
    samples = array("h", [value] * round(48000*seconds))
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        stream.writeframes(samples.tobytes())


def test_thought_reconstruction_preserves_exact_words_order_markers_and_cue_offsets(tmp_path, monkeypatch):
    first = "Dr. J. K. Simmons gives this strange story a warm and surprisingly funny human heart despite its enormous stakes."
    second = "The adventure stays exciting because these characters feel real and every bold decision has a meaningful cost for them."
    text = first + " " + second
    assert recovery._speech_chunks(text) == [first, second]
    shared = tmp_path / "reused.wav"
    recognized = {}
    emitted = []

    def speak(words, **options):
        assert options["force"] is True
        emitted.append(words)
        value = 1200 if words == first else 2400
        seconds = .35 if words == first else .55
        write_wave(shared, seconds, value)
        recognized[shared] = words
        return shared

    narrator = SimpleNamespace(speak=Mock(side_effect=speak), close=Mock())
    service, factory = make_recovery(tmp_path, object(), speaker=narrator)
    monkeypatch.setattr(recovery, "duration", duration)

    def captions(path, model, **kwargs):
        words = recognized[path]
        # The full paragraph is genuinely unverifiable, while each separate
        # thought has complete independently checked recognition.
        return [{"start": .02, "end": duration(path)-.02,
                 "text": words if words != text else "wrong speech"}]

    monkeypatch.setattr(recovery.voice, "captions", captions)
    result, cues = service.recover(text, attempts=1)

    factory.assert_called_once_with()
    assert emitted == [text, first, second]
    assert " ".join(cue["text"] for cue in cues) == text
    assert cues[0]["start"] == pytest.approx(.02)
    assert cues[0]["end"] == pytest.approx(.33)
    assert cues[1]["start"] == pytest.approx(.37)
    assert cues[1]["end"] == pytest.approx(.88)
    assert duration(result) == pytest.approx(.9)
    with wave.open(str(result), "rb") as stream:
        samples = array("h", stream.readframes(stream.getnframes()))
    assert samples[0] == samples[round(.35*48000)-1] == 1200
    assert samples[round(.35*48000)] == samples[-1] == 2400
    assert not list(service.work.glob("*.concat.txt"))


@pytest.mark.parametrize("text", [
    "Mr. Smith gives Dr. J. K. Simmons 3.5 seconds to solve the mystery. Then our unlikely heroes rush into another dangerous situation with no useful plan whatsoever.",
    "A single long sentence keeps all the original words rather than chopping the thought into tiny isolated phrases that sound awkward in narration",
    "One complete thought has enough words to stand alone and keep its meaning. Another complete thought also has enough words to sound quite natural when spoken. Thanks!",
])
def test_thought_chunks_keep_original_words_and_protect_names_and_decimals(text):
    chunks = recovery._speech_chunks(text)
    assert " ".join(chunks).split() == text.split()
    assert all(len(chunk.split()) >= 5 for chunk in chunks)
    if "J. K." in text:
        assert any("Dr. J. K. Simmons 3.5 seconds" in chunk for chunk in chunks)


def test_reconstruction_failure_has_recovery_error_and_cleans_listing(tmp_path, monkeypatch):
    first = "This first sentence explains why the strange adventure stays funny and exciting for every viewer at home."
    second = "This second sentence adds another complete thought with enough words to make the repair useful and natural."
    text = first + " " + second
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")
    monkeypatch.setattr(recovery, "duration", lambda _: 1)
    narrator = SimpleNamespace(speak=Mock(return_value=path), close=Mock())
    service, _ = make_recovery(tmp_path, object(), speaker=narrator)
    monkeypatch.setattr(service, "check", lambda words, *_args, **_kwargs: None if words == text else [{"start": 0, "end": 1, "text": words}])
    monkeypatch.setattr(recovery, "ffmpeg", Mock(side_effect=PipelineError("broken assembly")))

    with pytest.raises(recovery.SpeechRecoveryError, match="assemble verified speech"):
        service.recover(text, attempts=1)

    assert not list(service.work.glob("*.concat.txt"))
    assert service.events[-1]["stage"] == "thought-assembly"
