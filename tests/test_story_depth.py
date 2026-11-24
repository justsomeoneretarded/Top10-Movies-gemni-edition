"""Optional semantic depth repair, tested without live models or timing padding."""
from copy import deepcopy
from unittest.mock import Mock

import pytest
import requests

from movie_pipeline.script import Writer
from movie_pipeline.story_depth import REVIEW_SCHEMA, _quote_supported, _source_material

FILMS = [
    {"title": "Night Crossing", "year": 2025, "kind": "movie",
     "overview": "A courier crosses a closed border to rescue her brother. "
                 "Her only guide is a detective who once arrested her.",
     "wiki_summary": "She must choose between keeping the evidence and saving her brother.",
     "available_scenes": [{"description": "A villain launches a nuclear missile."}]},
    {"title": "Quiet Harbour", "year": 2025, "kind": "movie",
     "overview": "A mechanic hides a stranded family while a storm closes the harbour."},
]
DETAIL = "Her only guide is a detective who once arrested her"
BETTER = [
    "Number 1: Night Crossing.",
    "A courier crosses a closed border to rescue her brother. Her only guide is the detective "
    "who once arrested her, so she has to depend on someone she has every reason to distrust.",
]


def draft():
    return {
        "sections": [
            {"kind": "intro", "film": 0, "paragraphs": ["Two stories about dangerous choices."],
             "delivery": ["hype"]},
            {"kind": "film", "film": 1, "paragraphs": [
                "Number 2: Quiet Harbour. A mechanic hides a family as a storm traps them in port."],
             "delivery": ["normal"], "planning_words": 140},
            {"kind": "film", "film": 0, "paragraphs": [
                "Number 1: Night Crossing.", "A courier crosses a border to rescue her brother."],
             "delivery": ["hype", "normal"], "planning_words": 160},
            {"kind": "outro", "film": 0, "paragraphs": [
                "Like and subscribe, and watch our other recommendations."], "delivery": ["calm"]},
        ],
        "package": {"title": "Two movies worth watching", "existing_metadata": {"keep": True}},
    }


def assessment(status="thin", details=None, *, other="adequate"):
    return {"reviews": [
        {"rank": 1, "status": status, "reason": "The guide's history makes the choice specific.",
         "supported_details": [DETAIL] if details is None else details},
        {"rank": 2, "status": other, "reason": "Clear situation and stakes.",
         "supported_details": []},
    ]}


def improvement(paragraphs=None, *, used=None, adds=True, preserves=True):
    return {"paragraphs": BETTER if paragraphs is None else paragraphs,
            "used_details": [DETAIL] if used is None else used,
            "adds_substance": adds, "preserves_existing_detail": preserves}


def writer_with(*responses, cache_dir=None):
    writer = Writer(cache_dir=cache_dir)
    writer.ask = Mock(side_effect=list(responses))
    return writer


def repair(writer, original=None, **kwargs):
    return writer.repair_depth(
        "Two movies worth watching", "Plain, useful storytelling", FILMS,
        draft() if original is None else original, target_seconds=480, **kwargs)


def test_repairs_supported_missing_thought_without_rewriting_other_sections():
    original = draft()
    writer = writer_with(assessment(), improvement())
    result = repair(writer, original)
    assert original == draft()  # repair is isolated from the saved original
    assert result["sections"][2]["paragraphs"] == BETTER
    assert result["sections"][2]["delivery"] == ["hype", "normal"]
    for index in (0, 1, 3):
        assert result["sections"][index] == original["sections"][index]
    assert result["package"]["existing_metadata"] == {"keep": True}
    assert result["package"]["depth_review"]["changed_film_indices"] == [0]
    assert writer.ask.call_count == 2
    assert "About 160 words" in writer.ask.call_args_list[1].args[0]
    assert "never a minimum" in writer.ask.call_args_list[1].args[0]


@pytest.mark.parametrize("automatic", [False, True])
def test_short_substantive_review_stays_short_even_far_below_soft_runtime(automatic):
    original = draft()
    writer = writer_with(assessment("adequate", details=[]))
    result = repair(writer, original, automatic=automatic, estimated_seconds=55)
    assert result["sections"] == original["sections"]
    assert writer.ask.call_count == 1
    receipt = result["package"]["depth_review"]
    assert receipt["changed_film_indices"] == []
    assert receipt["estimated_seconds"] == 55
    assert receipt["phase"] == "measured"


@pytest.mark.parametrize("status,details,action", [
    ("limited_material", [], "retained"),
    ("thin", [], "retained_no_supported_expansion"),
    ("thin", ["The villain launches a nuclear missile"], "retained_no_supported_expansion"),
    ("generic", ["A completely invented plot detail"], "retained_no_supported_expansion"),
    ("generic", ["Her only"], "retained_no_supported_expansion"),
])
def test_limited_or_unsupported_material_never_becomes_padding(status, details, action):
    writer = writer_with(assessment(status, details))
    result = repair(writer)
    assert result["sections"] == draft()["sections"]
    assert writer.ask.call_count == 1
    record = next(r for r in result["package"]["depth_review"]["reviews"] if r["film"] == 0)
    assert record["action"] == action


@pytest.mark.parametrize("candidate,action", [
    (improvement(adds=False), "retained_no_semantic_improvement"),
    (improvement(preserves=False), "retained_no_semantic_improvement"),
    (improvement(used=["She steals a spaceship from a secret base"]), "retained_unsupported_repair"),
    (improvement(used=[]), "retained_unsupported_repair"),
    (improvement(paragraphs=[]), "retained_invalid_repair"),
    (improvement(paragraphs=["Number 1: Night Crossing. Number 2: Quiet Harbour."]),
     "retained_invalid_repair"),
])
def test_unusable_candidate_retains_complete_original(candidate, action):
    writer = writer_with(assessment(), candidate)
    result = repair(writer)
    assert result["sections"] == draft()["sections"]
    record = next(r for r in result["package"]["depth_review"]["reviews"] if r["film"] == 0)
    assert record["action"] == action


@pytest.mark.parametrize("failure", [
    requests.Timeout("SECRET_SENTINEL"), ValueError("SECRET_SENTINEL"),
    TypeError("SECRET_SENTINEL"), RuntimeError("SECRET_SENTINEL"),
])
def test_optional_model_failure_is_nonfatal_and_does_not_record_exception_body(failure):
    writer = writer_with(failure)
    result = repair(writer)
    assert result["sections"] == draft()["sections"]
    receipt = result["package"]["depth_review"]
    assert receipt["status"] == "unavailable"
    assert receipt["error_class"] == type(failure).__name__
    assert "SECRET_SENTINEL" not in repr(result)
    assert writer.ask.call_count == 1


def test_failed_targeted_repair_does_not_retry_or_discard_existing_speech():
    writer = writer_with(assessment(), requests.Timeout("private-error-body"))
    result = repair(writer)
    assert result["sections"] == draft()["sections"]
    receipt = result["package"]["depth_review"]
    record = next(r for r in receipt["reviews"] if r["film"] == 0)
    assert record["action"] == "retained_repair_unavailable"
    assert record["error_class"] == "Timeout"
    assert receipt["repair_attempts"] == 1
    assert writer.ask.call_count == 2
    assert "private-error-body" not in repr(result)


@pytest.mark.parametrize("row", [
    {"rank": 1, "status": "unknown", "supported_details": [DETAIL]},
    {"rank": 999, "status": "thin", "supported_details": [DETAIL]},
    {"rank": True, "status": "thin", "supported_details": [DETAIL]},
])
def test_bad_identity_or_status_does_not_repair_another_film(row):
    writer = writer_with({"reviews": [row]})
    result = repair(writer)
    assert result["sections"] == draft()["sections"]
    assert writer.ask.call_count == 1
    assert result["package"]["depth_review"]["changed_film_indices"] == []


def test_duplicate_rank_assessments_are_ambiguous_not_last_row_wins():
    row = assessment()["reviews"][0]
    writer = writer_with({"reviews": [row, deepcopy(row)]})
    result = repair(writer)
    assert result["sections"] == draft()["sections"]
    assert writer.ask.call_count == 1


@pytest.mark.parametrize("limit,expected", [(0, 0), (1, 1), (2, 2), (3, 3), (50, 3)])
def test_repair_calls_are_bounded_across_the_whole_episode(limit, expected):
    films = [{**FILMS[0], "title": f"Story {i}"} for i in range(5)]
    original = {"sections": [
        {"kind": "film", "film": i, "paragraphs": [f"Number {i+1}: Story {i}. A courier must cross a border."]}
        for i in range(5)], "package": {}}
    rows = {"reviews": [{"rank": i+1, "status": "thin", "reason": "Omitted conflict.",
                         "supported_details": [DETAIL]} for i in range(5)]}
    # Even rejected candidate repairs consume the bounded optional budget.
    writer = writer_with(rows, *[improvement(adds=False) for _ in range(expected)])
    result = writer.repair_depth("Five stories", "", films, original,
                                 target_seconds=600, max_repairs=limit)
    assert result["sections"] == original["sections"]
    assert writer.ask.call_count == (1+expected if limit else 0)
    receipt = result["package"]["depth_review"]
    assert receipt.get("repair_attempts", 0) == expected


def test_verified_standalone_announcement_stays_byte_exact():
    original = draft()
    announcement = "Number 1: Night Crossing."
    original["sections"][2]["paragraphs"][0] = announcement
    combined = ["Number 1: Night Crossing. A courier crosses a border with the detective who once arrested her."]
    writer = writer_with(assessment(), improvement(combined))
    result = repair(writer, original, estimated_seconds=180, max_repairs=2)
    assert result["sections"][2]["paragraphs"] == [
        announcement, "A courier crosses a border with the detective who once arrested her."]
    assert result["sections"][0] == original["sections"][0]
    assert result["sections"][1] == original["sections"][1]


def test_cache_reuse_avoids_model_calls_and_keeps_current_packaging(tmp_path):
    first = writer_with(assessment(), improvement(), cache_dir=tmp_path)
    expected = repair(first)
    second = writer_with(cache_dir=tmp_path)
    original = draft()
    original["package"]["new_report"] = {"keep": 42}
    result = repair(second, original, estimated_seconds=195)
    assert result["sections"] == expected["sections"]
    assert result["package"]["new_report"] == {"keep": 42}
    receipt = result["package"]["depth_review"]
    assert receipt["cache"] == "hit"
    assert receipt["phase"] == "measured"
    assert receipt["estimated_seconds"] == 195
    second.ask.assert_not_called()


def test_truncated_review_is_not_rewritten_based_on_missing_tail():
    original = draft()
    original["sections"][2]["paragraphs"] = [
        "Number 1: Night Crossing.", "A courier must cross a border. " * 200]
    writer = writer_with(assessment())
    result = repair(writer, original)
    assert result["sections"] == original["sections"]
    assert writer.ask.call_count == 1
    record = next(r for r in result["package"]["depth_review"]["reviews"] if r["film"] == 0)
    assert record["action"] == "retained_full_review_not_truncated"


@pytest.mark.parametrize("target", [None, 0, float("nan"), float("inf"), -4, True])
def test_bad_soft_target_cannot_fail_a_usable_script(target):
    writer = writer_with()
    result = writer.repair_depth("Two movies", "", FILMS, draft(), target_seconds=target)
    assert result["sections"] == draft()["sections"]
    assert result["package"]["depth_review"]["status"] == "unavailable"
    writer.ask.assert_not_called()


def configured_writer(*, depth_status="completed"):
    """Exercise actual write ordering while avoiding unrelated model packaging."""
    writer = Writer()
    writer.plan = Mock(return_value={})
    writer.bridges = Mock(return_value={})
    writer.thumbnail_concepts = Mock(return_value=[])
    writer.narration = Mock(side_effect=lambda title, brief, role, facts, words: [
        ("A useful opening." if "opening hook" in role else
         "Like and subscribe for more recommendations." if "closing thought" in role else
         f"Number {facts['rank']}: {facts['title']}. A courier must rescue her brother. "
         "Her guide once arrested her, so trust is dangerous.")])
    writer.edit = Mock(side_effect=lambda *args, **kwargs: (args[4], ["normal"]*len(args[4])))
    writer.ask = Mock(return_value={
        "titles": ["Two movies worth watching"], "description": "Two movie recommendations.",
        "thumbnail_texts": ["Dangerous Choices", "Trust Nobody", "Two Picks"], "tags": []})

    def episode_edit(title, brief, sections):
        result = deepcopy(sections)
        section = next(s for s in result if s["kind"] == "film" and s["film"] == 0)
        section["paragraphs"] = ["Number 1: Night Crossing. A courier must rescue her brother."]
        section["delivery"] = ["normal"]
        return result
    writer.episode_edit = Mock(side_effect=episode_edit)

    def depth(title, brief, films, script, **kwargs):
        # Depth review must see the result of episode editing, before packaging.
        current = next(s for s in script["sections"] if s["kind"] == "film" and s["film"] == 0)
        assert "guide" not in " ".join(current["paragraphs"])
        result = deepcopy(script)
        result["package"]["depth_review"] = {
            "status": depth_status, "changed_film_indices": [], "reviews": []}
        return result
    writer.repair_depth = Mock(side_effect=depth)
    return writer


def test_auto_write_gives_every_review_soft_guidance_and_runs_depth_before_packaging():
    writer = configured_writer()
    result = writer.write("Two movies worth watching", "", FILMS,
                          target_seconds=480, automatic=True)
    review_calls = [call for call in writer.narration.call_args_list
                    if call.args[3].get("rank")]
    assert {call.args[3]["rank"] for call in review_calls} == {1, 2}
    assert all(isinstance(call.args[4], int) and call.args[4] > 0 for call in review_calls)
    assert all("planning_words" in call.args[3] for call in review_calls)
    writer.repair_depth.assert_called_once()
    assert writer.repair_depth.call_args.kwargs["automatic"] is True
    assert result["package"]["depth_review"]["status"] == "completed"
    assert all(s["planning_words"] > 0 for s in result["sections"] if s["kind"] == "film")


def test_unavailable_depth_check_restores_substance_removed_by_optional_episode_edit():
    writer = configured_writer(depth_status="unavailable")
    result = writer.write("Two movies worth watching", "", FILMS,
                          target_seconds=480, automatic=True)
    section = next(s for s in result["sections"] if s["kind"] == "film" and s["film"] == 0)
    assert "Her guide once arrested her" in " ".join(section["paragraphs"])
    assert result["package"]["depth_review"]["retained_pre_episode_edit"] == [0]


def test_partial_warm_write_does_not_start_an_episode_repair():
    writer = configured_writer()
    result = writer.write("Two movies worth watching", "", FILMS,
                          target_seconds=480, automatic=True, only={"film-0"})
    assert result["partial"] == ["film-0"]
    writer.repair_depth.assert_not_called()
    writer.episode_edit.assert_not_called()


def test_repair_without_standalone_announcement_cannot_lose_film_identity():
    original = draft()
    original["sections"][2]["paragraphs"] = [
        "Number 1: Night Crossing. A courier must rescue her brother."]
    writer = writer_with(assessment(), improvement(["A courier must trust the detective who arrested her."]))
    result = repair(writer, original)
    assert result["sections"] == original["sections"]
    record = next(r for r in result["package"]["depth_review"]["reviews"] if r["film"] == 0)
    assert record["action"] == "retained_invalid_repair"


def test_optional_cache_write_failure_does_not_undo_a_supported_repair(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("cache unavailable")
    monkeypatch.setattr("movie_pipeline.story_depth.save_json", unavailable)
    writer = writer_with(assessment(), improvement(), cache_dir=tmp_path)
    result = repair(writer)
    assert result["sections"][2]["paragraphs"] == BETTER
    receipt = result["package"]["depth_review"]
    assert receipt["status"] == "completed"
    assert receipt["changed_film_indices"] == [0]
    assert receipt["cache"] == "write_unavailable"


def test_depth_source_material_keeps_story_notes_without_credit_metadata():
    film = {**FILMS[0], "directors": ["Jane Director"], "creators": ["Alex Creator"],
            "cast_profiles": [{"name": "Terry Actor", "character": "Ada"}],
            "wiki_production": "The director built the border set.",
            "research_notes": [
                "Created by Jane Director.",
                "The border guards are searching for the evidence she carries."],
            "grounded_notes": {"plot": "Her brother is trapped on the other side.",
                               "cast": ["Terry Actor as Ada"]}}
    material = _source_material(film)
    assert material["research_notes"] == ["The border guards are searching for the evidence she carries."]
    assert material["grounded_notes"] == {"plot": "Her brother is trapped on the other side."}
    assert not {"directors", "creators", "cast", "characters", "wiki_production", "available_scenes"} & material.keys()
    assert "Jane Director" not in repr(material) and "Terry Actor" not in repr(material)


@pytest.mark.parametrize("quote,material", [
    ("Directed by Jane Director", {"overview": "Directed by Jane Director."}),
    ("created by david fincher", {"research_notes": "created by david fincher."}),
    ("Terry Actor as Ada", {"cast": ["Terry Actor as Ada"]}),
    ("Jane Director built the hospital set", {"wiki_production": "Jane Director built the hospital set."}),
    ("Night Crossing the movie", {"title": "Night Crossing the movie"}),
    ("The series stars Terry Actor as a doctor fighting to save her brother",
     {"wiki_summary": "The series stars Terry Actor as a doctor fighting to save her brother."}),
])
def test_depth_expansion_quotes_cannot_be_credits_or_identity(quote, material):
    assert not _quote_supported(quote, material)


def test_depth_expansion_quotes_retain_plot_inside_mixed_sources_and_fictional_jobs():
    material = {"wiki_summary": "Example is a 2025 film directed by Jane Director. "
                               "A director searches the abandoned set for her daughter.",
                "research_notes": "Created by Umbrella, the virus spreads through the city.",
                "grounded_notes": "The series stars Terry Actor as a doctor fighting to save her brother."}
    assert _quote_supported("A director searches the abandoned set for her daughter", material)
    assert _quote_supported("Created by Umbrella, the virus spreads through the city", material)
    assert _quote_supported("a doctor fighting to save her brother", material)
    assert not _quote_supported("directed by Jane Director", material)


def test_depth_editor_assesses_plot_instead_of_awarding_substance_to_credits():
    original = draft()
    original["sections"][2]["paragraphs"] = [
        "Number 1: Night Crossing.", "Created by Jane Director.",
        "A courier crosses a border to rescue her brother."]
    writer = writer_with(assessment("adequate", details=[]))
    result = repair(writer, original)
    prompt = writer.ask.call_args.args[0]
    assert "Jane Director" not in prompt
    assert "does not count as story substance" in prompt
    assert "a broad genre or vague problem is still thin" in prompt
    assert "quote up to two exact unused details" in prompt
    assert result["sections"] == original["sections"]


def test_credit_only_unused_quote_does_not_trigger_an_expansion_request():
    films = deepcopy(FILMS)
    films[0]["research_notes"] = "Created by Jane Director."
    writer = writer_with(assessment("thin", ["Created by Jane Director"]))
    result = writer.repair_depth("Two movies", "", films, draft(), target_seconds=480)
    assert writer.ask.call_count == 1
    assert result["sections"] == draft()["sections"]
    record = next(row for row in result["package"]["depth_review"]["reviews"] if row["film"] == 0)
    assert record["action"] == "retained_no_supported_expansion"


def test_credit_only_depth_repair_cannot_erase_existing_story():
    writer = writer_with(assessment(), improvement([
        "Number 1: Night Crossing.", "Directed by Jane Director."]))
    result = repair(writer)
    assert result["sections"] == draft()["sections"]
    record = next(row for row in result["package"]["depth_review"]["reviews"] if row["film"] == 0)
    assert record["action"] == "retained_invalid_repair"
