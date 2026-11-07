"""Real codec/mix checks plus conservative handling of uncertain recognition."""
from __future__ import annotations

import math
import shutil
import wave
from unittest.mock import Mock

import numpy as np
import pytest

from movie_pipeline import audio_review
from movie_pipeline.core import ffmpeg

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")


def recording(path, seconds=3.6):
    rate = 48000
    time = np.arange(round(seconds*rate))/rate
    # Changing vowel/formant bands and word-like pauses exercise temporal
    # identity, rather than merely matching a constant test tone.
    frequency = 450+180*np.sin(time*3.7)+100*np.sin(time*7.1)
    phase = 2*np.pi*np.cumsum(frequency)/rate
    envelope = (.08+.12*(np.sin(time*9.3)**2))*(.2+.8*(np.sin(time*4.1) > -.6))
    values = envelope*(np.sin(phase)+.3*np.sin(2.7*phase)+.15*np.sin(5.1*phase))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((values*32767).astype("<i2").tobytes())
    return path


def event(source, *, start=1.37, seconds=3.6, kind="narration", text="The full original paragraph starts here and finishes with these last words."):
    return {"type": kind, "start": start, "duration": seconds, "text": text, "source_audio": str(source)}


@pytest.mark.parametrize("speed", [.85, 1.0, 1.18])
def test_aac_music_loudnorm_and_global_tempo_keep_verified_voice_without_asr(tmp_path, speed):
    source = recording(tmp_path/"source.wav")
    final = tmp_path/"final.m4a"
    graph = (f"[0:a]adelay=1370:all=1,atempo={speed},apad=pad_dur=0.5[voice];"
             "[1:a]volume=.04[bed];[voice][bed]amix=inputs=2:duration=first:normalize=0,"
             "loudnorm=I=-16:TP=-1.5:LRA=10[out]")
    ffmpeg("-i", source, "-f", "lavfi", "-i", "sine=frequency=165:sample_rate=48000:duration=8",
           "-filter_complex", graph, "-map", "[out]", "-c:a", "aac", "-b:a", "192k", final)
    factory = Mock(spec=lambda: None, side_effect=AssertionError("Healthy acoustic identity must not load ASR"))
    report = audio_review.verify_export_audio(final, [event(source)], speed, factory, tmp_path)
    assert report["accepted"] and report["status"] == "passed"
    assert report["checks"][0]["method"] == "verified-recording"
    factory.assert_not_called()


@pytest.mark.parametrize("speed", [.85, 1.0, 1.18])
def test_identical_exported_prefix_cannot_verify_a_physically_truncated_paragraph(tmp_path, speed):
    source = recording(tmp_path/"source.wav")
    final = tmp_path/"truncated.m4a"
    ffmpeg("-i", source, "-af", f"adelay=1370:all=1,atempo={speed}",
           "-t", (1.37+3.6-.8)/speed, "-c:a", "aac", "-b:a", "192k", final)
    checker = Mock(check=Mock(return_value=[{"text": "An unreliable recognition result"}]))
    report = audio_review.verify_export_audio(final, [event(source)], speed, checker, tmp_path)
    check = report["checks"][0]
    assert report["accepted"] is False
    assert check["method"] == "recording-duration"
    assert check["acoustic"]["missing_reference_seconds"] > .6
    checker.check.assert_not_called()


@pytest.mark.parametrize("gain,accepted", [(.02, False), (.2, True)])
def test_acoustic_identity_still_requires_audible_voice_without_rejecting_normal_gain_changes(tmp_path, gain, accepted):
    source = recording(tmp_path/"source.wav")
    final = tmp_path/"quiet.m4a"
    ffmpeg("-i", source, "-af", f"volume={gain}", "-c:a", "aac", "-b:a", "192k", final)
    report = audio_review.verify_export_audio(final, [event(source, start=0)], 1, None, tmp_path)
    assert report["accepted"] is accepted
    if not accepted:
        assert report["checks"][0]["method"] == "recording-audibility"


def test_only_first_middle_and_last_complete_paragraph_are_sampled_after_previews(monkeypatch, tmp_path):
    source = recording(tmp_path/"source.wav")
    seen = []

    def decode(path, target, **options):
        seen.append((str(path), options))
        return np.ones(16000)

    monkeypatch.setattr(audio_review, "_decode", decode)
    monkeypatch.setattr(audio_review, "_identity", lambda *args: {"identity": .99, "edges": [.99, .99]})
    events = [{"type": "cold-open", "start": 0, "duration": 5}]
    events += [event(source, start=5+index*20, kind="subscribe" if index == 2 else "narration") for index in range(7)]
    report = audio_review.verify_export_audio(tmp_path/"final.mp4", events, 1.18, None, tmp_path)
    assert [check["event_index"] for check in report["checks"]] == [1, 4, 7]
    exported = [options for path, options in seen if path.endswith("final.mp4")]
    assert [options["start"] for options in exported] == pytest.approx([start/1.18-.12 for start in (5, 65, 125)])
    assert all(options["seconds"] == pytest.approx(3.6/1.18+.24) for options in exported)


@pytest.mark.parametrize("replacement", ["anullsrc=r=48000:cl=mono", "sine=frequency=165:sample_rate=48000", "anoisesrc=color=white:sample_rate=48000:seed=42"])
def test_clear_silence_music_or_corrupt_noise_in_place_of_narration_is_rejected(tmp_path, replacement):
    source = recording(tmp_path/"source.wav")
    final = tmp_path/"final.m4a"
    ffmpeg("-f", "lavfi", "-i", replacement, "-t", 3.8, "-c:a", "aac", final)
    checker = Mock(check=Mock(return_value=None))
    report = audio_review.verify_export_audio(final, [event(source, start=0)], 1, checker, tmp_path)
    assert report["accepted"] is False
    assert report["failed_events"] == [0]


@pytest.mark.parametrize("edge", ["beginning", "ending"])
def test_missing_boundary_speech_is_detected_even_when_most_of_paragraph_survives(tmp_path, edge):
    source = recording(tmp_path/"source.wav")
    final = tmp_path/"final.m4a"
    expression = "lt(t,0.8)" if edge == "beginning" else "gt(t,2.8)"
    ffmpeg("-i", source, "-af", f"volume=0:enable='{expression}'", "-c:a", "aac", final)
    report = audio_review.verify_export_audio(final, [event(source, start=0)], 1, Mock(check=Mock(return_value=None)), tmp_path)
    assert report["accepted"] is False


def test_asr_disagreement_does_not_reject_uncertain_but_audible_recording(monkeypatch, tmp_path):
    source = recording(tmp_path/"source.wav")
    metrics = {"identity": .45, "shape": .7, "edges": [.6, .6], "speech_level_ratio": 1, "boundary_level_ratios": [1, 1]}
    monkeypatch.setattr(audio_review, "_identity", lambda *args: metrics)
    checker = Mock(check=Mock(return_value=None))
    factory = Mock(spec=lambda: None, return_value=checker)
    text = "Preserve the complete first sentence. And these final words too."
    report = audio_review.verify_export_audio(source, [event(source, start=0, text=text)], 1, factory, tmp_path)
    assert report["accepted"] and report["status"] == "inconclusive"
    assert report["warning"]
    factory.assert_called_once_with()
    assert checker.check.call_args.args[0] == text  # Compare afterwards; never an ASR initial_prompt.
    with wave.open(str(checker.check.call_args.args[1]), "rb") as stream:
        assert stream.getnframes()/stream.getframerate() == pytest.approx(3.6, abs=.01)


def test_missing_reference_can_be_verified_by_lazy_unprompted_checker(tmp_path):
    source = recording(tmp_path/"source.wav")
    item = event(tmp_path/"missing-reference.wav", start=0)
    checker = Mock(check=Mock(return_value=[{"start": 0, "end": 3.5, "text": item["text"]}]))
    report = audio_review.verify_export_audio(source, [item], 1, lambda: checker, tmp_path)
    assert report["status"] == "passed"
    assert report["checks"][0]["method"] == "unprompted-recognition"


def test_unreadable_final_audio_is_a_clear_failure(tmp_path):
    source = recording(tmp_path/"source.wav")
    report = audio_review.verify_export_audio(tmp_path/"missing.mp4", [event(source)], 1, None, tmp_path)
    assert report["accepted"] is False


def test_failed_rerun_cannot_reuse_a_previous_successful_export_window(tmp_path):
    source = recording(tmp_path/"source.wav")
    final = tmp_path/"final.m4a"
    ffmpeg("-i", source, "-c:a", "aac", final)
    item = event(source, start=0)
    first = audio_review.verify_export_audio(final, [item], 1, None, tmp_path)
    assert first["status"] == "passed"
    assert (tmp_path/"export-audio-review"/"paragraph-000.wav").exists()
    final.unlink()
    second = audio_review.verify_export_audio(final, [item], 1, None, tmp_path)
    assert second["accepted"] is False
    assert second["checks"][0]["reason"] == "Final paragraph audio could not be decoded."


@pytest.mark.parametrize("speed", [0, math.inf, math.nan])
def test_invalid_tempo_is_not_used_for_window_offsets(tmp_path, speed):
    with pytest.raises(ValueError):
        audio_review.verify_export_audio(tmp_path/"final.mp4", [], speed, None, tmp_path)
