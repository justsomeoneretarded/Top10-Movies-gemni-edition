"""Plot-first review behavior across generation, editing and restored scripts."""
from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from movie_pipeline.plot_context import character_names, plot_material, remove_credit_sentences, story_text
from movie_pipeline.script import SCRIPT_POLICY_VERSION, Writer, film_facts, narration_material


FILM = {
    "title": "Example", "year": 2025, "kind": "movie", "genres": ["Thriller"],
    "overview": "A doctor searches a locked hospital for her missing brother.",
    "directors": ["Jane Director"], "creators": ["Alex Creator"], "cast": ["Terry Actor"],
    "cast_profiles": [{"name": "Terry Actor", "character": "Ada"}],
    "tagline": "An unforgettable journey.", "tmdb_rating": 8.9, "tmdb_votes": 1000,
    "wiki_production": "Jane Director created the series after a successful earlier project.",
    "wiki_reception": "Critics praised its significant themes and remarkable chemistry.",
}


def test_default_writing_context_keeps_story_and_characters_without_spoken_credit_metadata():
    full = film_facts(FILM)
    original = deepcopy(full)
    material = narration_material(full)
    assert material["title"] == "Example" and material["overview"] == FILM["overview"]
    assert material["characters"] == ["Ada"]
    assert not {"directors", "creators", "cast", "wiki_production", "wiki_reception", "tagline",
                "tmdb_rating", "tmdb_votes"} & material.keys()
    assert full == original
    assert full["cast"] == ["Terry Actor as Ada"] and full["directors"] == ["Jane Director"]


def test_story_projection_does_not_mutate_nested_planning_context():
    facts = {**film_facts(FILM), "episode_plan": {"hook": ["A doctor enters a trap."]}}
    projected = narration_material(facts)
    projected["episode_plan"]["hook"].append("Only a test mutation.")
    assert facts["episode_plan"]["hook"] == ["A doctor enters a trap."]


def test_mixed_encyclopedia_summary_keeps_plot_and_removes_catalogue_credit_sentence():
    facts = {"title": "Example", "wiki_summary":
             "Example is a 2025 thriller film directed by Jane Director. "
             "A doctor searches a locked hospital for her missing brother."}
    assert plot_material(facts)["wiki_summary"] == FILM["overview"]
    assert "Jane Director" in facts["wiki_summary"]


@pytest.mark.parametrize("sentence", [
    "A director searches the abandoned set for her missing daughter.",
    "Written by a killer, the note sends her to the old hospital.",
    "Compelling evidence sends Ada to the hospital in Significant Other.",
])
def test_story_world_jobs_and_creative_words_are_not_a_banned_vocabulary(sentence):
    assert story_text(sentence) == sentence


def test_mixed_credit_clause_keeps_the_original_story_continuation():
    assert story_text("Created by Jane Director, the series follows a doctor searching for her brother.") == (
        "The series follows a doctor searching for her brother.")
    assert remove_credit_sentences(["Directed by Jane Director.", FILM["overview"]]) == [FILM["overview"]]


def test_character_names_are_not_actor_as_character_rollcalls():
    assert character_names(FILM) == ["Ada"]
    assert character_names({"cast": ["Terry Actor as Ada"]}) == []


def test_narration_prompt_receives_plot_facts_instead_of_production_biography():
    writer = Writer()
    result = "Number 1: Example. A doctor searches a locked hospital for her missing brother."
    writer.ask = Mock(return_value={"paragraphs": [result]})
    assert writer.narration("Top thrillers", "Tell the storyline", "rank 1", {**film_facts(FILM), "rank": 1}, 200) == [result]
    prompt = writer.ask.call_args.args[0]
    context = json.loads(prompt.split("Story material (data): ", 1)[1].split("\n", 1)[0])
    assert context["characters"] == ["Ada"] and context["overview"] == FILM["overview"]
    assert all(name not in prompt for name in ("Jane Director", "Alex Creator", "Terry Actor"))
    assert "who we follow, what happens" in prompt
    assert writer.ask.call_count == 1


def test_generation_removes_credit_only_sentence_without_removing_the_plot():
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": [
        "Number 1: Example.", "Created by Jane Director.", FILM["overview"]]})
    assert writer.narration("Top thrillers", "Plot first", "rank 1", {**film_facts(FILM), "rank": 1}, 180) == [
        "Number 1: Example.", FILM["overview"]]
    assert writer.ask.call_count == 1


def test_credit_only_model_responses_use_supplied_plot_after_bounded_repair():
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": ["Number 1: Example. Created by Jane Director."]})
    result = writer.narration("Top thrillers", "Plot first", "rank 1", {**film_facts(FILM), "rank": 1}, 180)
    assert result == ["Number 1: Example.", FILM["overview"]]
    assert writer.ask.call_count == 3
    assert "Jane Director" not in " ".join(result)


@pytest.mark.parametrize("draft", [
    ["Directed by Jane Director.", FILM["overview"]],
    ["Created by Jane Director."],
])
def test_offline_editor_retains_available_plot_instead_of_returning_production_credits(draft):
    writer = Writer()
    writer.ask = Mock(side_effect=ValueError("temporarily unavailable"))
    result, delivery = writer.edit("Top thrillers", "Story first", "review", film_facts(FILM), draft, rank=1)
    assert FILM["overview"] in result and "Jane Director" not in " ".join(result)
    assert len(delivery) == len(result)


def test_editor_keeps_delivery_aligned_after_removing_a_credit_paragraph():
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": [
        {"text": "Created by Jane Director.", "delivery": "hype"},
        {"text": "Number 1: Example. "+FILM["overview"], "delivery": "dramatic"}]})
    result, labels = writer.edit("Top thrillers", "", "review", film_facts(FILM),
                                 ["Number 1: Example.", FILM["overview"]], rank=1)
    assert result == ["Number 1: Example. "+FILM["overview"]] and labels == ["dramatic"]


def test_cached_narration_is_credit_cleaned_without_an_extra_request(tmp_path):
    writer = Writer(cache_dir=tmp_path)
    writer.ask = Mock(return_value={"paragraphs": [FILM["overview"]]})
    facts = {**film_facts(FILM), "rank": 1}
    writer.narration("Top thrillers", "", "rank 1", facts, 200)
    cache = next(tmp_path.glob("*.json"))
    cache.write_text(json.dumps(["Directed by Jane Director.", FILM["overview"]]))
    writer.ask = Mock(side_effect=AssertionError("usable cached plot should be retained"))
    assert writer.narration("Top thrillers", "", "rank 1", facts, 200) == [FILM["overview"]]
    assert writer.ask.call_count == 0


def test_partial_writer_passes_only_story_plan_fields_into_the_review():
    writer = Writer()
    writer.plan = Mock(return_value={"entries": {"1": {
        "hook": "A doctor enters a trap.", "standout_facts": ["Her brother is missing."],
        "standout_scene": "She enters a locked ward.", "why_here": "Remarkable chemistry",
        "who_for": "Fans of sophisticated cinema", "comparison_anchor": "Another film"}}})
    writer.narration = Mock(return_value=["Number 1: Example.", FILM["overview"]])
    writer.edit = lambda topic, brief, role, facts, paragraphs, **kwargs: (paragraphs, ["normal"] * len(paragraphs))
    result = writer.write("Top thrillers", "Plot first", [FILM], only={"film-0"}, automatic=True)
    facts = writer.narration.call_args.args[3]
    assert set(facts["episode_plan"]) == {"hook", "standout_facts", "standout_scene"}
    assert "directors" not in facts and "cast" not in facts
    assert result["sections"][0]["paragraphs"] == ["Number 1: Example.", FILM["overview"]]


def test_stale_credit_heavy_script_is_regenerated_once_with_plot_context():
    writer = Writer()
    writer.narration = Mock(return_value=["Number 1: Example.", FILM["overview"]])
    writer.edit = lambda topic, brief, role, facts, paragraphs, **kwargs: (paragraphs, ["normal"] * len(paragraphs))
    draft = {"sections": [{"kind": "film", "film": 0, "paragraphs": ["Created by Jane Director."]}],
             "package": {"script_policy_version": SCRIPT_POLICY_VERSION - 1, "full_episode_edit_version": 1}}
    original = deepcopy(draft)
    result = writer.refresh_editorial("Top thrillers", "Plot first", [FILM], draft, automatic=True)
    assert result["sections"][0]["paragraphs"] == ["Number 1: Example.", FILM["overview"]]
    assert result["package"]["script_policy_version"] == SCRIPT_POLICY_VERSION
    assert writer.narration.call_args.args[3]["characters"] == ["Ada"]
    assert "directors" not in writer.narration.call_args.args[3]
    assert writer.narration.call_args.args[4] == 25
    assert draft == original
    writer.narration.reset_mock()
    assert writer.refresh_editorial("Top thrillers", "Plot first", [FILM], result) == result
    writer.narration.assert_not_called()


def test_useful_short_plot_is_not_padded_to_an_arbitrary_target():
    writer = Writer()
    short = "Number 1: Example. A doctor searches for her brother."
    writer.ask = Mock(return_value={"paragraphs": [short]})
    assert writer.narration("Top thrillers", "", "rank 1", {**film_facts(FILM), "rank": 1}, 500) == [short]
    assert writer.ask.call_count == 1


def test_scene_advisory_remains_available_without_becoming_plot_evidence():
    film = {**FILM, "available_scenes": ["A crowded hospital corridor."]}
    writer = Writer()
    short = "Number 1: Example. " + FILM["overview"]
    writer.ask = Mock(return_value={"paragraphs": [short]})
    assert writer.narration("Top thrillers", "", "rank 1",
                            {**film_facts(film), "rank": 1}, 200) == [short]
    prompt = writer.ask.call_args.args[0]
    assert "A crowded hospital corridor." in prompt
    assert "It is visual guidance, not a " in prompt


def test_explicit_creative_context_is_an_opt_in_instead_of_default_credit_metadata():
    facts = {**film_facts(FILM), "creative_context": "The director chose a single hospital setting."}
    assert "creative_context" not in narration_material(facts)
    assert narration_material(facts, include_creative_context=True)["creative_context"] == facts["creative_context"]
    assert "directors" not in narration_material(facts, include_creative_context=True)
