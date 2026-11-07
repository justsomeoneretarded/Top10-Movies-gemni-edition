from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from movie_pipeline.core import PipelineError
from movie_pipeline.voice import Narrator


class Wave:
    def __init__(self, samples):
        self.samples = samples

    def squeeze(self):
        return self

    detach = squeeze
    cpu = squeeze

    def numpy(self):
        return self.samples


def setup_speaker(monkeypatch, tmp_path, seconds, samples):
    torch = SimpleNamespace(manual_seed=Mock(), inference_mode=nullcontext)
    soundfile = SimpleNamespace(write=lambda path, data, sr, **kwargs: path.write_bytes(b"raw"))
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    monkeypatch.setitem(__import__("sys").modules, "soundfile", soundfile)
    monkeypatch.setattr("movie_pipeline.voice.duration", lambda path: seconds)
    monkeypatch.setattr("movie_pipeline.voice.ffmpeg", lambda *args: args[-1].write_bytes(b"normalized"))
    speaker = object.__new__(Narrator)
    speaker.reference_hash = "test-reference"
    speaker.model_name = "turbo"
    speaker.cache = tmp_path
    speaker.model = SimpleNamespace(sr=24000, generate=Mock(return_value=Wave(samples)))
    speaker.deliveries, speaker.emotion_enabled, speaker.takes, speaker.emotional = {}, False, 1, None
    speaker.reference = tmp_path / "ref.wav"
    return speaker


@pytest.mark.parametrize("word_count, seconds", [(36, 6), (80, 75)])
def test_fast_or_slow_delivery_alone_does_not_abort(monkeypatch, tmp_path, word_count, seconds):
    speaker = setup_speaker(monkeypatch, tmp_path, seconds, np.full(24000, 0.1))
    path = speaker.speak(" ".join(["word"]*word_count))
    assert path.exists()
    assert speaker.model.generate.call_count == 1


@pytest.mark.parametrize("samples", [np.zeros(200), np.array([]), np.array([float("nan")])])
def test_empty_silent_or_invalid_audio_is_still_rejected(monkeypatch, tmp_path, samples):
    speaker = setup_speaker(monkeypatch, tmp_path, 6, samples)
    with pytest.raises(PipelineError, match="empty, silent"):
        speaker.speak("A closing sentence for the video.")
    assert speaker.model.generate.call_count == 3


def test_runaway_audio_is_rejected_with_diagnostics(monkeypatch, tmp_path):
    speaker = setup_speaker(monkeypatch, tmp_path, 600, np.full(24000, 0.1))
    with pytest.raises(PipelineError, match="runaway"):
        speaker.speak("Thanks for watching.")
    assert speaker.model.generate.call_count == 3


def speech_like(seconds=2.0, rate=24000, gap=0.0):
    t = np.arange(int(rate * seconds)) / rate
    wave = 0.3 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    if gap:
        middle = len(wave) // 2
        wave[middle: middle + int(rate * gap)] = 0
    return wave


def test_best_of_two_takes_drops_a_take_with_dead_air(monkeypatch, tmp_path):
    from movie_pipeline.voice import take_score
    clean, broken = speech_like(), speech_like(gap=1.6)
    assert take_score(clean, 24000, 6) > take_score(broken, 24000, 6)
    speaker = setup_speaker(monkeypatch, tmp_path, 3, clean)
    speaker.takes = 2
    speaker.model.generate = Mock(side_effect=[Wave(broken), Wave(clean)])
    written = []
    monkeypatch.setitem(__import__("sys").modules, "soundfile",
                        SimpleNamespace(write=lambda path, data, sr, **kw: (written.append(data), path.write_bytes(b"raw"))))
    speaker.speak("A short line with six words.")
    assert speaker.model.generate.call_count == 2
    assert not np.any(np.all(np.abs(np.lib.stride_tricks.sliding_window_view(written[0], 24000)[::2400]) < 1e-6, axis=1))


def test_original_uses_the_same_model_for_dramatic_lines_and_reveals_get_a_lead(monkeypatch, tmp_path):
    speaker = setup_speaker(monkeypatch, tmp_path, 3, speech_like())
    speaker.model_name, speaker.emotion_enabled = "original", True
    filters = []
    monkeypatch.setattr("movie_pipeline.voice.ffmpeg", lambda *args: (filters.append(args[3]), args[-1].write_bytes(b"n")))
    speaker.speak("Number 3: Sinners. Two brothers come home.", delivery="dramatic")
    assert speaker.model.generate.call_count == 1
    assert speaker.model.generate.call_args.kwargs["exaggeration"] > 0.5
    assert speaker.model.generate.call_args.kwargs["cfg_weight"] > 0
    assert "adelay=180" in filters[0] and "deesser" in filters[0] and "acompressor" in filters[0]


def test_spoken_form_fixes_trouble_titles_but_coverage_still_matches():
    from movie_pipeline.voice import spoken_coverage, spoken_form
    assert spoken_form("M3GAN vs. Se7en: Part II, 2025–2026") == "Megan versus Seven: Part Two, 2025 to 2026"
    cues = [{"text": "Megan versus Seven part two"}]
    assert spoken_coverage("M3GAN vs. Se7en Part II", cues) == 1.0



def test_original_delivery_settings_have_distinct_cache_without_replacing_accepted_audio(monkeypatch, tmp_path):
    speaker = setup_speaker(monkeypatch, tmp_path, 3, speech_like())
    speaker.model_name = "original"
    text = "A complete thought with enough room to breathe."
    regular = speaker.speak(text, delivery="dramatic")
    speaker.emotion_enabled = True
    emotional = speaker.speak(text, delivery="dramatic")
    assert regular != emotional and regular.exists() and emotional.exists()
    assert speaker.model.generate.call_count == 2
    speaker.emotion_enabled = False
    assert speaker.speak(text, delivery="dramatic") == regular
    assert speaker.model.generate.call_count == 2


@pytest.mark.parametrize("selection,expected", [(None, "original"), ("original", "original"),
                                               (" Original ", "original"), ("turbo", "turbo"), ("nano", "nano")])
def test_constructor_routes_exactly_one_selected_model(monkeypatch, tmp_path, selection, expected):
    import sys
    from types import ModuleType

    if selection is None:
        monkeypatch.delenv("TTS_MODEL", raising=False)
    else:
        monkeypatch.setenv("TTS_MODEL", selection)
    monkeypatch.delenv("VOICE_EMOTION", raising=False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=Mock(), inference_mode=nullcontext))
    original = Mock(return_value=SimpleNamespace(prepare_conditionals=Mock()))
    turbo = Mock(return_value=SimpleNamespace(prepare_conditionals=Mock()))
    monkeypatch.setitem(sys.modules, "chatterbox", ModuleType("chatterbox"))
    monkeypatch.setitem(sys.modules, "chatterbox.tts", SimpleNamespace(ChatterboxTTS=SimpleNamespace(from_pretrained=original)))
    monkeypatch.setitem(sys.modules, "chatterbox.tts_turbo", SimpleNamespace(ChatterboxTurboTTS=SimpleNamespace(from_pretrained=turbo)))
    reference = tmp_path/"warguy.ogg"
    reference.write_bytes(b"same-user-reference")
    narrator = Narrator(reference, tmp_path/"speech")
    assert narrator.model_name == expected
    if expected == "original":
        original.assert_called_once_with(device="cpu")
        turbo.assert_not_called()
        narrator.model.prepare_conditionals.assert_called_once_with(str(reference), exaggeration=.5)
        assert narrator.emotion_enabled is True
    else:
        turbo.assert_called_once_with(device="cpu", nano=expected == "nano")
        original.assert_not_called()
        narrator.model.prepare_conditionals.assert_called_once_with(str(reference))
    narrator.close()


def test_original_load_failure_does_not_fall_back_to_another_model(monkeypatch, tmp_path):
    import sys
    from types import ModuleType

    monkeypatch.setenv("TTS_MODEL", "original")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=Mock()))
    original = Mock(side_effect=RuntimeError("model unavailable"))
    turbo = Mock()
    monkeypatch.setitem(sys.modules, "chatterbox", ModuleType("chatterbox"))
    monkeypatch.setitem(sys.modules, "chatterbox.tts", SimpleNamespace(ChatterboxTTS=SimpleNamespace(from_pretrained=original)))
    monkeypatch.setitem(sys.modules, "chatterbox.tts_turbo", SimpleNamespace(ChatterboxTurboTTS=SimpleNamespace(from_pretrained=turbo)))
    with pytest.raises(RuntimeError, match="model unavailable"):
        Narrator(tmp_path/"warguy.ogg", tmp_path/"speech")
    turbo.assert_not_called()


def test_all_deliveries_and_expression_toggle_keep_one_original_model(monkeypatch, tmp_path):
    speaker = setup_speaker(monkeypatch, tmp_path, 3, speech_like())
    speaker.model_name, speaker.emotion_enabled = "original", True
    model = speaker.model
    for label in ("normal", "calm", "hype", "dramatic"):
        samples, rate = speaker._generate("One consistent voice.", label, 123)
        assert speaker.model is model and rate == model.sr
        assert samples.size
        assert model.generate.call_args.kwargs["cfg_weight"] > 0
    assert model.generate.call_count == 4
    speaker.emotion_enabled = False
    speaker._generate("Still the same speaker.", "hype", 124)
    assert model.generate.call_args.kwargs == {"exaggeration": .5, "cfg_weight": .5, "temperature": .8}


def test_original_long_sentences_are_bounded_and_keep_every_word(monkeypatch, tmp_path):
    from movie_pipeline.voice import ORIGINAL_PHRASE_WORDS

    speaker = setup_speaker(monkeypatch, tmp_path, 6, speech_like())
    speaker.model_name, speaker.emotion_enabled = "original", True
    text = " ".join(f"word{i}" for i in range(135))+"."
    speaker.speak(text)
    requests = [call.args[0] for call in speaker.model.generate.call_args_list]
    assert all(len(request.split()) <= ORIGINAL_PHRASE_WORDS for request in requests)
    assert " ".join(requests) == text


def test_original_and_turbo_cannot_reuse_each_others_cached_speech(monkeypatch, tmp_path):
    speaker = setup_speaker(monkeypatch, tmp_path, 3, speech_like())
    text = "The same script still needs the selected model."
    turbo = speaker.speak(text)
    speaker.model_name = "original"
    original = speaker.speak(text)
    assert original != turbo and original.exists() and turbo.exists()
    assert speaker.model.generate.call_count == 2
    assert speaker.speak(text) == original
    speaker.model_name = "turbo"
    assert speaker.speak(text) == turbo
    assert speaker.model.generate.call_count == 2


def test_unknown_model_is_rejected_before_generation(monkeypatch):
    from movie_pipeline.voice import selected_tts_model
    monkeypatch.setenv("TTS_MODEL", "unsupported")
    with pytest.raises(PipelineError, match="original, turbo or nano"):
        selected_tts_model()
