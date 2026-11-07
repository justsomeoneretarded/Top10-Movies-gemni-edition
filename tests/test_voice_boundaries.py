import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from movie_pipeline.core import ffmpeg, key
from movie_pipeline.voice import (
    Narrator,
    captions,
    speech_check,
    speech_verified,
    spoken_coverage,
)

PARAGRAPH = (
    "Stop scrolling because your weekend deserves better movies. "
    "These five picks bring strange worlds dangerous missions and characters worth following. "
    "Stick around because the last recommendation might become your new favorite."
)


def cues(text):
    return [{"start": 0, "end": 10, "text": text}]


def test_complete_speech_is_verified_across_caption_chunks():
    words = PARAGRAPH.split()
    transcript = cues(" ".join(words[:12])) + cues(" ".join(words[12:]))
    assert speech_verified(PARAGRAPH, transcript)


def test_missing_opening_clause_is_rejected_despite_high_overall_coverage():
    transcript = cues(" ".join(PARAGRAPH.split()[6:]))
    assert spoken_coverage(PARAGRAPH, transcript) >= .72
    assert not speech_verified(PARAGRAPH, transcript)


def test_missing_closing_clause_is_rejected_despite_high_overall_coverage():
    transcript = cues(" ".join(PARAGRAPH.split()[:-6]))
    assert spoken_coverage(PARAGRAPH, transcript) >= .72
    assert not speech_verified(PARAGRAPH, transcript)


@pytest.mark.parametrize("missing_index", [0, 10, -1])
def test_one_unheard_word_is_tolerated_at_either_boundary_or_in_middle(missing_index):
    words = PARAGRAPH.split()
    del words[missing_index]
    assert speech_verified(PARAGRAPH, cues(" ".join(words)))


def test_minor_transcription_error_does_not_reject_complete_boundaries():
    words = PARAGRAPH.split()
    words[11] = "stranger"
    words[17] = "character"
    assert speech_verified(PARAGRAPH, cues(" ".join(words)))


@pytest.mark.parametrize("expected,heard", [
    ("We've found 5 movies you'll want to watch this weekend.",
     "We have found five movies you will want to watch this weekend."),
    ("Don't skip these 2026 movies. They're worth your time.",
     "Do not skip these twenty twenty six movies. They are worth your time."),
    ("2026 brings five great picks. Let's find your next watch.",
     "Two thousand twenty six brings 5 great picks. Let us find your next watch."),
])
def test_contractions_and_number_formatting_are_tolerated(expected, heard):
    assert speech_verified(expected, cues(heard))


@pytest.mark.parametrize("expected,heard", [("", ""), (PARAGRAPH, ""), (PARAGRAPH, "Thanks for watching.")])
def test_empty_or_unrelated_transcript_is_not_verified(expected, heard):
    assert not speech_verified(expected, cues(heard))


def test_legacy_overall_coverage_api_keeps_float_result():
    assert spoken_coverage("One two three four", cues("One two three")) == .75


@pytest.mark.parametrize("expected,heard", [
    ("Like and subscribe. I'll handle both sides of the high five.",
     "Like and subscribe. I will handle both sides of the highfive."),
    ("Like and subscribe. Virtual high five. Zero awkward eye contact.",
     "Like and subscribe. Virtual high-five. Zero awkward eyecontact."),
    ("Keep your notebook ready because these five recommendations deserve a closer look.",
     "Keep your note book ready because these 5 recommendations deserve a closer look."),
])
def test_bounded_word_join_and_split_equivalents_preserve_complete_speech(expected, heard):
    check = speech_check(expected, cues(heard))
    assert check['accepted']
    assert check['coverage'] == 1
    assert check['opening']['accepted'] and check['ending']['accepted']


@pytest.mark.parametrize("expected,heard", [
    ("One. Two. Three.", "Six."),
    ("One, two, three.", "Six."),
    ("One two three.", "Six."),
    ("Five. Zero distractions.", "Five distractions."),
])
def test_independent_number_phrases_are_never_added_together(expected, heard):
    assert not speech_verified(expected, cues(heard))


@pytest.mark.parametrize("expected,heard", [
    ("One, two, three.", "One two three."),
    ("There are 1,000 great films and twenty-five recommendations worth discussing.",
     "There are one thousand great films and 25 recommendations worth discussing."),
])
def test_real_number_phrases_and_number_lists_still_match(expected, heard):
    assert speech_verified(expected, cues(heard))


def test_full_sentence_repeated_three_times_is_not_accepted():
    expected = "Like and subscribe. Protect your next movie night from terrible recommendations."
    check = speech_check(expected, cues(" ".join([expected]*3)))
    assert check['coverage'] == 1
    assert not check['accepted']
    assert check['reason'] == 'extra_or_repeated_speech'


def test_one_short_asr_correction_is_tolerated():
    expected = "Like and subscribe. Your next movie night deserves better recommendations."
    heard = "Like and subscribe. Your next, your next movie night deserves better recommendations."
    assert speech_verified(expected, cues(heard))


@pytest.mark.parametrize("move_clause", [False, True])
def test_missing_or_reordered_middle_clause_cannot_hide_inside_good_overall_coverage(move_clause):
    beginning = "Keep this original opening because your next movie night deserves much better recommendations than another boring sequel."
    middle = "The middle clause must remain exactly here."
    ending = "Finish with these closing words because they complete the entire thought for our audience."
    expected = f"{beginning} {middle} {ending}"
    heard = f"{beginning} {ending}" + (f" {middle}" if move_clause else "")
    check = speech_check(expected, cues(heard))
    assert check['coverage'] >= .72
    assert not check['accepted']
    assert check['reason'] == 'missing_or_reordered_phrase'


def test_speech_check_reports_boundary_evidence_and_original_transcript():
    heard = " ".join(PARAGRAPH.split()[6:])
    check = speech_check(PARAGRAPH, cues(heard))
    assert not check['accepted']
    assert check['reason'] == 'opening'
    assert check['heard'] == heard
    assert check['coverage'] >= .72
    assert check['opening']['unmatched_tokens'] == 6
    assert check['ending']['accepted']


@pytest.mark.parametrize("options,expected", [
    ({}, {'vad_filter': True, 'beam_size': 3, 'condition_on_previous_text': False}),
    ({'vad_filter': False, 'beam_size': 5}, {'vad_filter': False, 'beam_size': 5, 'condition_on_previous_text': False}),
])
def test_captions_support_independent_decode_modes_without_expected_text_prompts(options, expected):
    word = SimpleNamespace(word=' Complete.', start=.16, end=1.2)
    segment = SimpleNamespace(words=[word], text='Complete.', start=.16, end=1.2)
    model = SimpleNamespace(transcribe=Mock(return_value=([segment], None)))
    assert captions(Path('speech.wav'), model, **options) == [{'start': .16, 'end': 1.2, 'text': 'Complete.'}]
    model.transcribe.assert_called_once_with('speech.wav', language='en', word_timestamps=True, **expected)


class Wave:
    def __init__(self, samples):
        self.samples = samples

    def squeeze(self):
        return self

    detach = squeeze
    cpu = squeeze

    def numpy(self):
        return self.samples


def fake_speaker(monkeypatch, tmp_path: Path, samples):
    monkeypatch.setitem(__import__("sys").modules, "torch", SimpleNamespace(
        manual_seed=Mock(), inference_mode=nullcontext,
    ))
    monkeypatch.setitem(__import__("sys").modules, "soundfile", SimpleNamespace(write=write_wav))
    speaker = object.__new__(Narrator)
    speaker.reference_hash = "reference"
    speaker.model_name = "turbo"
    speaker.cache = tmp_path
    speaker.model = SimpleNamespace(sr=24000, generate=Mock(return_value=Wave(samples)))
    speaker.deliveries, speaker.emotion_enabled, speaker.takes, speaker.emotional = {}, False, 1, None
    speaker.reference = tmp_path / "ref.wav"
    return speaker


def write_wav(path, samples, rate, **kwargs):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((np.clip(samples, -1, 1)*32767).astype("<i2").tobytes())


def read_wav(path):
    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(float)/32768, stream.getframerate()


def test_new_voice_cache_does_not_reuse_unpadded_v1_audio(monkeypatch, tmp_path):
    text = "Keep the complete opening and closing words."
    old = tmp_path / f"{key('voice-v1', text, 'reference', 'turbo')}.wav"
    write_wav(old, np.full(24000, .1), 24000)
    speaker = fake_speaker(monkeypatch, tmp_path, np.full(24000, .1))
    output = speaker.speak(text)
    assert output != old
    assert old.exists()
    assert speaker.model.generate.call_count == 1
    assert speaker.speak(text) == output
    assert speaker.model.generate.call_count == 1


def test_forced_retry_generates_distinct_audio_without_overwriting_verified_cache(monkeypatch, tmp_path):
    speaker = fake_speaker(monkeypatch, tmp_path, np.full(24000, .1))
    text = 'Keep this verified speech unchanged while another use retries.'
    baseline = speaker.speak(text)
    original = baseline.read_bytes()
    baseline.with_suffix('.captions.json').write_text('[{"text":"verified"}]')
    first = speaker.speak(text, attempt_offset=3, force=True)
    second = speaker.speak(text, attempt_offset=3, force=True)
    third = speaker.speak(text, attempt_offset=6, force=True)
    assert len({baseline, first, second, third}) == 4
    assert baseline.read_bytes() == original
    assert baseline.with_suffix('.captions.json').read_text() == '[{"text":"verified"}]'
    assert speaker.model.generate.call_count == 4
    seeds = [call.args[0] for call in __import__('sys').modules['torch'].manual_seed.call_args_list]
    assert len(set(seeds)) == 4
    assert speaker.speak(text) == baseline
    assert speaker.model.generate.call_count == 4


def test_padding_preserves_first_and_last_audio_markers_through_aac(monkeypatch, tmp_path):
    rate = 24000
    seconds = .8
    time = np.arange(round(rate*seconds))/rate
    samples = .03*np.sin(2*np.pi*300*time)
    opening = time < .08
    closing = time >= seconds-.08
    samples[opening] = .25*np.sin(2*np.pi*600*time[opening])
    samples[closing] = .25*np.sin(2*np.pi*1200*time[closing])
    speaker = fake_speaker(monkeypatch, tmp_path, samples)
    output = speaker.speak("Every opening and closing sound must survive.")
    padded, sample_rate = read_wav(output)
    assert len(padded)/sample_rate == pytest.approx(seconds+.16+.26, abs=.015)
    assert np.max(np.abs(padded[:round(.14*sample_rate)])) < .001
    assert np.max(np.abs(padded[-round(.24*sample_rate):])) < .001

    encoded = tmp_path / "encoded.m4a"
    decoded = tmp_path / "decoded.wav"
    ffmpeg("-i", output, "-c:a", "aac", "-b:a", "192k", encoded)
    ffmpeg("-i", encoded, "-ar", "48000", "-ac", "1", decoded)
    recovered, decoded_rate = read_wav(decoded)

    def marker_frequency(start, end):
        segment = recovered[round(start*decoded_rate):round(end*decoded_rate)]
        assert float(np.sqrt(np.mean(segment**2))) > .01
        spectrum = np.abs(np.fft.rfft(segment*np.hanning(len(segment))))
        return np.fft.rfftfreq(len(segment), 1/decoded_rate)[int(np.argmax(spectrum))]

    assert marker_frequency(.17, .23) == pytest.approx(600, abs=20)
    assert marker_frequency(.89, .95) == pytest.approx(1200, abs=20)
