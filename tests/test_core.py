
import runpy
from unittest.mock import Mock

import pytest

from movie_pipeline.cli import choose_beat
from movie_pipeline.core import (
    PipelineError,
    narrator_from_brief,
    parse_movies,
    selected_content_type,
    selected_narrator,
    write_srt,
)
from movie_pipeline.footage import candidate_ok
from movie_pipeline.voice import spoken_coverage


def test_parse_titles_preserves_order_and_punctuation():
    movies = parse_movies("Spider-Man: No Way Home (2021); Arrival 2016\nCoherence (2013)")
    assert [(m.title, m.year) for m in movies] == [("Spider-Man: No Way Home", 2021), ("Arrival", 2016), ("Coherence", 2013)]


@pytest.mark.parametrize("text", ["Arrival", "Arrival (2016)", "Arrival (2016); Arrival (2016)", "Arrival (9999); Inception (2010)"])
def test_invalid_inputs_stop_early(text):
    with pytest.raises(PipelineError):
        parse_movies(text)


@pytest.mark.parametrize("brief, expected", [
    ("An energetic action countdown.", "warguy"),
    ("Use male audio.", "warguy"),
    ("Use Kit Laser for narration.", "kit-laser"),
    ("Use the female audio.", "female"),
    ("Female voice, slow and cinematic.", "female"),
    ("Don't use female audio.", "warguy"),
    ("Use male, not female.", "warguy"),
    ("Movies with female protagonists.", "warguy"),
    ("Use Warguy. Avoid the female voice.", "warguy"),
])
def test_narrator_selection(brief, expected):
    assert narrator_from_brief(brief) == expected


def test_ambiguous_narrator_rejected():
    with pytest.raises(PipelineError):
        narrator_from_brief("Use Kit Laser and the female voice")


@pytest.mark.parametrize("choice, expected", [("Warguy", "warguy"), ("Kit Laser", "kit-laser"), ("Female", "female")])
def test_explicit_narrator_overrides_conflicting_brief(monkeypatch, choice, expected):
    monkeypatch.setenv("NARRATOR", choice)
    assert selected_narrator("Use Kit Laser and the female voice with Warguy.") == expected


def test_unspecified_narrator_preserves_local_brief_selection(monkeypatch):
    monkeypatch.delenv("NARRATOR", raising=False)
    assert selected_narrator("An energetic countdown.") == "warguy"
    assert selected_narrator("Use the female voice.") == "female"
    monkeypatch.setenv("NARRATOR", "")
    assert selected_narrator("Use Kit Laser.") == "kit-laser"


def test_selected_warguy_does_not_allow_brief_to_choose_female(monkeypatch):
    monkeypatch.setenv("NARRATOR", "Warguy")
    assert selected_narrator("Use the female voice.") == "warguy"


def test_invalid_narrator_selector_reports_early(monkeypatch):
    monkeypatch.setenv("NARRATOR", "unknown")
    with pytest.raises(PipelineError, match="narrator selector"):
        selected_narrator("Use Warguy.")


@pytest.mark.parametrize("choice, expected", [("Movies", "movie"), ("Shows", "tv"), ("movies", "movie"), ("Shows ", "tv")])
def test_explicit_content_type(monkeypatch, choice, expected):
    monkeypatch.setenv("VIDEO_TYPE", choice)
    assert selected_content_type() == expected


def test_unspecified_content_type_preserves_automatic_lookup(monkeypatch):
    monkeypatch.delenv("VIDEO_TYPE", raising=False)
    assert selected_content_type() is None
    monkeypatch.setenv("VIDEO_TYPE", " ")
    assert selected_content_type() is None


def test_invalid_content_type_selector_reports_early(monkeypatch):
    monkeypatch.setenv("VIDEO_TYPE", "unknown")
    with pytest.raises(PipelineError, match="content-type selector"):
        selected_content_type()


def test_preflight_uses_same_explicit_media_and_voice_before_models(monkeypatch, capsys):
    for name, value in {
        "VIDEO_TITLE": "Top 2 Movies", "VIDEO_BRIEF": "Use Warguy.",
        "MOVIE_LIST": "Example A (2020); Example B (2022)", "TMDB_READ_TOKEN": "placeholder",
        "VIDEO_TYPE": "Shows", "NARRATOR": "Female", "VIDEO_MINUTES": "Auto", "TRY_4K": "false",
    }.items():
        monkeypatch.setenv(name, value)
    reference = Mock()
    monkeypatch.setattr("movie_pipeline.core.voice_reference", reference)
    runpy.run_module("scripts.preflight", run_name="__main__")
    assert reference.call_args.args[0] == "female"
    assert "content type tv; narrator female" in capsys.readouterr().out


def test_srt_is_uploadable_and_nonoverlapping(tmp_path):
    path = tmp_path / "subtitles.srt"
    write_srt(path, [{"start": 1.125, "end": 2.5, "text": "Hello world"}, {"start": 2.3, "end": 4, "text": "Next line"}])
    text = path.read_text()
    assert "00:00:01,125 --> 00:00:02,500" in text
    assert "00:00:02,500 --> 00:00:04,000" in text
    assert text.startswith("1\n")


def test_speech_verification_detects_missing_words():
    expected = "This movie takes us deep into space and asks a difficult question about humanity"
    assert spoken_coverage(expected, [{"text": expected}]) == 1
    assert spoken_coverage(expected, [{"text": "This movie takes us"}]) < 0.72


def test_trailer_rejects_fake_and_wrong_titles_without_uploader_gate():
    film = {"title": "Arrival"}
    base = {"title": "Arrival Official Trailer", "duration": 150, "channel_is_verified": True}
    assert candidate_ok(base, film)
    assert not candidate_ok({**base, "title": "Arrival Fan-Made Concept Trailer"}, film, curated=True)
    assert candidate_ok({**base, "channel_is_verified": False}, film)
    assert not candidate_ok({**base, "title": "Interstellar Official Trailer"}, film)


def test_beat_avoids_promos_and_respects_bounds():
    cues = [{"start": 1, "end": 3, "text": "We have to go"}, {"start": 20, "end": 23, "text": "In theaters this summer"}, {"start": 30, "end": 33, "text": "We are running out of time"}]
    chosen = choose_beat(cues, [{"start": 29, "end": 34, "motion": 30}], 100, True)
    assert chosen["text"] == "We are running out of time"


def test_trailer_dialogue_skips_title_cards_even_when_the_cue_starts_in_a_scene():
    cues = [{"start": 30, "end": 33, "text": "We are running out of time"},
            {"start": 40, "end": 42, "text": "We have to leave now"}]
    cards = [{"start": 32.5, "end": 35, "motion": 0}]
    shots = [{"start": 29, "end": 32.5, "motion": 30}, *cards,
             {"start": 39, "end": 43, "motion": 10}]
    chosen = choose_beat(cues, shots, 100, True, excluded=cards)
    assert chosen["text"] == "We have to leave now"
    assert choose_beat(cues[:1], shots, 100, True, excluded=cards) is None
