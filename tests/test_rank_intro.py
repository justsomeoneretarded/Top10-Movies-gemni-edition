"""A rank reveal is a complete utterance, followed by a full native scene."""
from copy import deepcopy
from pathlib import Path

import pytest

from movie_pipeline.rank_intro import prepare_rank_intros


@pytest.mark.parametrize("title,lead", [
    ("Arrival", "Number 1: Arrival."),
    ("Spider-Man: Brand New Day", "At number 1, it's Spider-Man: Brand New Day (2026)."),
    ("Mission: Impossible - The Final Reckoning", "No. 1: Mission: Impossible - The Final Reckoning."),
    ("(500) Days of Summer", "#1: (500) Days of Summer."),
    ("Dr. Strangelove", "Number one: Dr. Strangelove."),
])
def test_split_announcement_retains_complete_body_and_delivery(title, lead):
    premise = "Two strangers make a dangerous bargain."
    original = {"sections": [{"kind": "film", "film": 0,
        "paragraphs": [lead+" "+premise, "Neither can afford to lose."],
        "delivery": ["dramatic", "calm"]}]}
    before = deepcopy(original)
    prepared = prepare_rank_intros(original, [{"title": title}])
    section = prepared["sections"][0]
    assert section["paragraphs"] == [f"Number 1: {title}.", premise, "Neither can afford to lose."]
    assert section["delivery"] == ["hype", "dramatic", "calm"]
    assert original == before
    assert prepare_rank_intros(prepared, [{"title": title}]) == prepared


@pytest.mark.parametrize("body", [
    "A translator has to understand a language before time runs out.",
    "Arrival follows a translator.",
    "The director makes Arrival feel like a memory.",
    "It's about listening before fighting.",
])
def test_original_premise_without_rank_is_never_discarded(body):
    original = {"sections": [{"kind": "film", "film": 0, "paragraphs": [body]}]}
    actual = prepare_rank_intros(original, [{"title": "Arrival"}])
    assert actual["sections"][0]["paragraphs"] == ["Number 1: Arrival.", body]


def test_integrated_title_keeps_its_subject():
    original = {"sections": [{"kind": "film", "film": 0,
                             "paragraphs": ["Number 1: Arrival follows a translator."]}]}
    actual = prepare_rank_intros(original, [{"title": "Arrival"}])
    assert actual["sections"][0]["paragraphs"] == ["Number 1: Arrival.", "Arrival follows a translator."]


def test_separate_old_title_paragraph_does_not_create_empty_speech():
    original = {"sections": [{"kind": "film", "film": 1,
        "paragraphs": ["Number 2: It.", "A town has a secret."], "delivery": ["hype", "dramatic"]}]}
    section = prepare_rank_intros(original, [{"title": "Other"}, {"title": "It"}])["sections"][0]
    assert section["paragraphs"] == ["Number 2: It.", "A town has a secret."]
    assert section["delivery"] == ["hype", "dramatic"]


def test_partial_title_match_never_eats_a_premise():
    original = {"sections": [{"kind": "film", "film": 0, "paragraphs": ["It's about a town."]}]}
    actual = prepare_rank_intros(original, [{"title": "It"}])
    assert actual["sections"][0]["paragraphs"][-1] == "It's about a town."


def test_intro_and_outro_and_unknown_sections_are_unchanged():
    original = {"sections": [{"kind": "intro", "film": 0, "paragraphs": ["Start here."]},
                            {"kind": "outro", "film": 0, "paragraphs": ["See you next time."]},
                            {"kind": "film", "film": 10, "paragraphs": ["Unchanged."]}]}
    assert prepare_rank_intros(original, [{"title": "A"}]) == original


def test_native_insert_planner_never_inserts_between_announcement_and_preview(monkeypatch, tmp_path):
    import movie_pipeline.moments as moments

    source = Path("movie.mp4")
    candidates = [{"source": str(source), "start": 10+i*20, "duration": 8,
                   "score": 1-i*.1, "action_strength": 0, "motion": 0} for i in range(5)]
    monkeypatch.setattr(moments, "candidates", lambda *args: candidates)
    monkeypatch.setattr(moments, "_enrich", lambda clip, *args: {**clip, "muted": False, "cues": []})
    monkeypatch.setattr(moments, "_usable", lambda *args: True)
    monkeypatch.setattr(moments, "_reserve", lambda *args: None)
    monkeypatch.setattr("movie_pipeline.core.duration", lambda path: 2 if str(path) == "name.wav" else 45)
    class Selector:
        def identity_relevance(self, *args):
            return [1]
    section = {"kind": "film", "film": 0, "rank_intro_version": 1,
               "paragraphs": ["Number 1: A.", "First review thought.", "Second review thought."]}
    result = moments.plan_episode(Selector(), [[source]], {}, [{"title": "A"}], [section],
        {(0, 0): Path("name.wav"), (0, 1): Path("one.wav"), (0, 2): Path("two.wav")}, {}, None, tmp_path)
    assert result["previews"][0]["duration"] == 8
    assert set(result["inserts"][0]) == {1, 2}


@pytest.mark.parametrize("title", ["Air!", "Who Framed Roger Rabbit?"])
def test_title_punctuation_ends_the_announcement_without_repeating_it(title):
    script = {"sections": [{"kind": "film", "film": 0,
                           "paragraphs": [f"Number 1: {title} A detective needs answers."]}]}
    result = prepare_rank_intros(script, [{"title": title}])["sections"][0]
    assert result["paragraphs"] == [f"Number 1: {title}", "A detective needs answers."]
    assert prepare_rank_intros({"sections": [result]}, [{"title": title}])["sections"][0] == result


def test_punctuated_title_as_subject_keeps_the_review_verb():
    title = "Air!"
    script = {"sections": [{"kind": "film", "film": 0,
                           "paragraphs": [f"Number 1: {title} follows two friends."]}]}
    result = prepare_rank_intros(script, [{"title": title}])["sections"][0]
    assert result["paragraphs"] == [f"Number 1: {title}", "Air! follows two friends."]
