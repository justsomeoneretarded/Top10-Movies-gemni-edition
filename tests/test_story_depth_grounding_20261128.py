"""Story-depth repairs keep explicit source facts without production or live models."""
from copy import deepcopy
from unittest.mock import Mock

from movie_pipeline.script import Writer
from movie_pipeline.story_depth import _quote_supported, _source_material


def test_editorial_notes_and_visual_labels_cannot_certify_new_plot_facts():
    invented = "Their supervisor sends a helicopter to capture them"
    material = {
        "overview": "Seven engineers cross a closed border.",
        "episode_plan": {"standout_facts": [invented]},
        "previous_narration_context": [invented],
        "available_scenes": [invented],
    }
    assert _quote_supported("Seven engineers cross a closed border", material)
    assert not _quote_supported(invented, material)


def test_partial_character_list_does_not_replace_explicit_crew_and_relationship_evidence():
    film = {
        "title": "Border Shift", "year": 2025, "kind": "movie",
        "overview": "Seven former engineers cross a closed border to rescue a missing colleague. "
                    "Mara and Ivo are siblings.",
        "characters": ["Mara", "Ivo"],
        "available_scenes": ["Two people wait beside a helicopter."],
    }
    original = {
        "sections": [{
            "kind": "film", "film": 0,
            "paragraphs": [
                "Number 1: Border Shift.",
                "Two former engineers, Mara and Ivo, cross the border to rescue a colleague.",
            ],
            "delivery": ["hype", "normal"],
        }],
        "package": {"keep": True},
    }
    before = deepcopy(original)
    detail = "Seven former engineers cross a closed border to rescue a missing colleague"
    replacement = [
        "Number 1: Border Shift.",
        "Seven former engineers cross a closed border to rescue a missing colleague. "
        "Among them are Mara and Ivo, two siblings who must work together.",
    ]
    writer = Writer(cache_dir=None)
    writer.ask = Mock(side_effect=[
        {"reviews": [{
            "rank": 1, "status": "thin",
            "reason": "The draft mistakes two named people for the full crew.",
            "supported_details": [detail, "Mara and Ivo are siblings"],
        }]},
        {"paragraphs": replacement, "used_details": [detail, "Mara and Ivo are siblings"],
         "adds_substance": True, "preserves_existing_detail": True},
    ])

    result = writer.repair_depth(
        "One border thriller", "Keep the ending and hidden motives unrevealed",
        [film], original, target_seconds=180, max_repairs=1,
    )

    assert original == before
    assert result["sections"][0]["paragraphs"] == replacement
    assert result["package"]["keep"] is True
    assert writer.ask.call_count == 2
    assert result["package"]["depth_review"]["repair_attempts"] == 1
    assert result["package"]["depth_review"]["changed_film_indices"] == [0]
    material = _source_material(film)
    assert "characters" not in material
    assert "available_scenes" not in material
    assert material["overview"] == film["overview"]
    assessment_prompt = writer.ask.call_args_list[0].args[0]
    repair_prompt = writer.ask.call_args_list[1].args[0]
    for prompt in (assessment_prompt, repair_prompt):
        assert "partial" in prompt
        assert "relationships" in prompt
        assert "spoiler boundary" in prompt
    assert "correct a contradiction" in assessment_prompt


def test_unsupported_visual_expansion_retains_draft_without_another_model_call():
    film = {
        "title": "Harbour Watch", "year": 2025,
        "overview": "A mechanic shelters a stranded family during a storm.",
        "available_scenes": ["The family is secretly hunted by cartel mercenaries."],
    }
    script = {
        "sections": [{
            "kind": "film", "film": 0,
            "paragraphs": ["Number 1: Harbour Watch.",
                           "A mechanic shelters a stranded family during a storm."],
        }],
        "package": {},
    }
    writer = Writer(cache_dir=None)
    writer.ask = Mock(return_value={"reviews": [{
        "rank": 1, "status": "thin", "reason": "Add a pursuer.",
        "supported_details": ["The family is secretly hunted by cartel mercenaries"],
    }]})

    result = writer.repair_depth(
        "One storm thriller", "No spoilers", [film], script, target_seconds=180,
    )

    assert result["sections"] == script["sections"]
    assert writer.ask.call_count == 1
    assert result["package"]["depth_review"]["reviews"][0]["action"] == "retained_no_supported_expansion"
