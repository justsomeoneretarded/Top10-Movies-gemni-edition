import pytest

from movie_pipeline.core import PipelineError
from movie_pipeline.timing import (
    DurationSpec,
    final_tempo,
    narration_words,
    requested_duration,
    trailer_budget,
    workflow_duration,
)


@pytest.mark.parametrize("brief, seconds", [
    ("Make a 5-minute video. Use Warguy.", 300),
    ("Duration: 5 min", 300), ("Make a five minute video", 300),
    ("Aim for 8–10 minutes", 540), ("Make a 15-minute video", 900),
    ("Make a 1.5 minute video", 90), ("Make a 1-hour video", 3600),
    ("Duration: 45 seconds", 45), ("A 30-second video", 30),
    ("Pause for 3 seconds during each trailer beat", 540),
])
def test_duration_from_brief(brief, seconds):
    assert requested_duration(brief).target == seconds


def test_default_preserves_original_eight_to_ten_minute_range():
    assert requested_duration("Cinematic and energetic") == DurationSpec()


def test_conflicting_or_malformed_brief_duration_falls_back_without_aborting():
    assert requested_duration("Make a 5 minute video. Aim for 8-10 minutes.") == DurationSpec()
    assert requested_duration("Make a 10-5 minute video") == DurationSpec()


def test_shorter_duration_gets_shorter_total_script():
    five = narration_words(300, 5, 15)
    fifteen = narration_words(900, 5, 15)
    assert five[1] < fifteen[1]
    assert sum((five[0], five[1]*5, five[2])) < sum((fifteen[0], fifteen[1]*5, fifteen[2]))


@pytest.mark.parametrize("seconds, count, scale", [(300, 5, 1), (900, 10, 1), (1800, 15, 2)])
def test_intro_word_goal_stays_brief_when_video_duration_or_revision_scale_grows(seconds, count, scale):
    opening, _, _ = narration_words(seconds, count, scale=scale)
    assert 16 <= opening <= 20


def test_tiny_video_does_not_spend_its_duration_on_trailer_breaks():
    assert trailer_budget(requested_duration("30-second video"), 5) == 0


@pytest.mark.parametrize("minutes", [1, 3, 5, 8, 10, 15, 20, 30, 7.5])
def test_workflow_minutes_always_allow_twenty_five_percent(monkeypatch, minutes):
    monkeypatch.setenv("VIDEO_MINUTES", str(minutes))
    spec = workflow_duration("Top 5 shows", 5)
    assert spec.target == minutes*60
    assert spec.minimum == spec.target*.75
    assert spec.maximum == spec.target*1.25
    assert not spec.automatic


def test_eight_minutes_is_six_to_ten(monkeypatch):
    monkeypatch.setenv("VIDEO_MINUTES", "8")
    assert workflow_duration("Top 5 movies", 5) == DurationSpec(480, 360, 600)


def test_auto_scales_to_list_and_uses_resolved_title_types(monkeypatch):
    monkeypatch.delenv("VIDEO_MINUTES", raising=False)
    five_shows = workflow_duration("Top 5 Shows", 5)
    ten_movies = workflow_duration("Top 10 Movies", 10)
    assert five_shows.automatic and 7*60 <= five_shows.target <= 9*60
    assert ten_movies.automatic and 13*60 <= ten_movies.target <= 15*60
    assert five_shows.target < ten_movies.target
    assert workflow_duration("Our favorites", 5, [{"kind":"tv"}]*5).target == five_shows.target
    assert workflow_duration("Top 15 Shows", 15).target <= 22*60
    assert workflow_duration("Top 100 Movies", 100).target <= 22*60


@pytest.mark.parametrize("choice, title, film_kind, per_title", [
    ("Shows", "Top 5 Movies", "movie", 85),
    ("Movies", "Top 5 Shows", "tv", 80),
])
def test_auto_respects_content_dropdown_over_title_and_metadata(monkeypatch, choice, title, film_kind, per_title):
    monkeypatch.setenv("VIDEO_MINUTES", "Auto")
    monkeypatch.setenv("VIDEO_TYPE", choice)
    assert workflow_duration(title, 5).target == 30 + 5*per_title
    assert workflow_duration(title, 5, [{"kind": film_kind}]*5).target == 30 + 5*per_title


@pytest.mark.parametrize("value", ["bad", "-5", "0", "NaN", "inf", "100", "99999999999999999999999999999999999999999999999999999999999999999999"])
def test_bad_workflow_minutes_fall_back_to_auto(monkeypatch, value):
    monkeypatch.setenv("VIDEO_MINUTES", value)
    assert workflow_duration("Top 5 movies", 5).automatic


def test_auto_does_not_adjust_speech_even_outside_its_planning_range(monkeypatch):
    monkeypatch.setenv("VIDEO_MINUTES", "Auto")
    spec = workflow_duration("Top 5 shows", 5)
    assert final_tempo(120, spec) == 1
    assert final_tempo(1000, spec) == 1


def test_runtime_guidance_never_retimes_finished_footage():
    spec = DurationSpec(480,360,600)
    assert final_tempo(480, spec) == 1
    assert all(final_tempo(seconds, spec) == 1 for seconds in (30, 350, 610, 2000))


@pytest.mark.parametrize("seconds", [0,-1,float("nan"),float("inf")])
def test_actual_empty_or_broken_media_still_reports_a_media_error(seconds):
    with pytest.raises(PipelineError, match="invalid media"):
        final_tempo(seconds, DurationSpec())


def test_legacy_single_duration_tolerance_is_also_twenty_five_percent():
    assert requested_duration("Make a 5-minute video") == DurationSpec(300,225,375)
