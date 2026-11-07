import math
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from movie_pipeline.previews import choose_preview, cold_open_plan, outro_recap_plan


def shots(*values):
    return [{"start": start, "end": end, "motion": motion} for start, end, motion in values]


def selector(values=None):
    return Mock(relevance=Mock(return_value=np.array(values if values is not None else [0, 0])))


def test_action_preview_prefers_the_stronger_action_in_one_continuous_window():
    source = Path("action.mp4")
    lookup = {str(source): shots((10, 18, 5), (40, 48, 45))}
    result = choose_preview(selector([.2, .2]), [source], lookup, "Action movies", 3.5, True, {str(source): []})
    assert result["start"] == pytest.approx(40.08)
    assert result["duration"] == 3.5
    assert result["muted"] is False
    assert result["start"] + result["duration"] <= 48


def test_non_action_preview_prefers_context_and_stability_over_motion():
    source = Path("mystery.mp4")
    lookup = {str(source): shots((10, 18, 4), (40, 48, 50))}
    result = choose_preview(selector([.6, .6]), [source], lookup, "Quiet mystery", 3.5, False)
    assert result["start"] == pytest.approx(10.08)


def test_full_readable_source_interval_wins_over_a_short_high_motion_shot():
    source = Path("action.mp4")
    lookup = {str(source): shots((10, 11.5, 60), (40, 48, 10))}
    result = choose_preview(selector([.9, .4]), [source], lookup, "Action", 3.5, True)
    assert result["start"] >= 40


def test_preview_moves_after_dialogue_when_a_clean_window_fits():
    source = Path("dialogue.mp4")
    lookup = {str(source): shots((10, 18, 20))}
    dialogue = {str(source): [{"start": 9.8, "end": 12.0, "text": "Not half a sentence"}]}
    result = choose_preview(selector([.5]), [source], lookup, "Film", 3.5, True, dialogue)
    assert result["start"] == pytest.approx(12.14)
    assert result["muted"] is False
    assert result["start"] + result["duration"] <= 18


def test_trailer_dialogue_is_muted_instead_of_cutting_into_its_sentence():
    source = Path("dialogue.mp4")
    lookup = {str(source): shots((10, 14, 20))}
    dialogue = {str(source): [{"start": 9, "end": 17, "text": "A much longer spoken sentence"}]}
    result = choose_preview(selector([.5]), [source], lookup, "Film", 3.5, True, dialogue)
    assert result["muted"] is True


@pytest.mark.parametrize("dialogue", [None, {}, {"extra.mp4": None}, {"extra.mp4": [{"start": "bad", "end": 7}]}])
def test_unknown_or_invalid_dialogue_on_extra_sources_is_muted(dialogue):
    source = Path("extra.mp4")
    result = choose_preview(selector([.5]), [source], {str(source): shots((10, 18, 20))}, "Film", 3.5, True, dialogue)
    assert result["muted"] is True


@pytest.mark.parametrize("scores", [[float("nan"), float("inf")], [-.9, -.8]])
def test_weak_or_nonfinite_relevance_never_rejects_a_preview(scores):
    source = Path("film.mp4")
    result = choose_preview(selector(scores), [source], {str(source): shots((10, 18, 20), (40, 48, 30))}, "Film", 3.5, True)
    assert result["duration"] == 3.5
    assert result["source"] == str(source)


def test_unavailable_relevance_and_analysis_keep_a_quiet_continuous_preview():
    source = Path("film.mp4")
    unavailable = Mock(relevance=Mock(side_effect=RuntimeError("Unavailable")))
    result = choose_preview(unavailable, [source], {}, "Film", 3.5, True)
    assert result == {"source": str(source), "start": 0, "duration": 3.5, "muted": True}


def test_all_short_shots_use_one_continuous_interval_without_freeze_or_extra_cuts():
    source = Path("film.mp4")
    result = choose_preview(selector([.5, .6]), [source], {str(source): shots((10, 12, 30), (40, 42, 40))}, "Film", 3.5, True)
    assert result["duration"] == 3.5
    assert result["start"] + result["duration"] <= 42
    assert "freeze" not in result
    assert "still" not in result


def test_cold_open_is_exactly_five_seconds_from_two_different_best_movies():
    pools = [[Path(f"{i}.mp4")] for i in range(5)]
    lookup = {str(pool[0]): shots((20, 28, i * 10)) for i, pool in enumerate(pools)}
    plan = cold_open_plan(selector([.3]), pools, lookup, "Action movies", True)
    assert [entry["film_index"] for entry in plan] == [4, 3]
    assert [entry["duration"] for entry in plan] == [2.5, 2.5]
    assert sum(round(entry["duration"] * 30) for entry in plan) == 150


@pytest.mark.parametrize("seconds,expected_count", [(5, 2), (5.0333, 2), (2, 1)])
def test_cold_open_never_introduces_rapid_flashes(seconds, expected_count):
    pools = [[Path("first.mp4")], [Path("second.mp4")]]
    lookup = {str(pool[0]): shots((10, 18, 20)) for pool in pools}
    plan = cold_open_plan(selector([0]), pools, lookup, "Films", False, seconds=seconds)
    assert len(plan) == expected_count
    assert all(entry["duration"] >= 1.5 for entry in plan)
    assert sum(round(entry["duration"] * 30) for entry in plan) == round(seconds * 30)


def test_one_available_film_still_gets_one_uninterrupted_five_second_opening():
    source = Path("one.mp4")
    plan = cold_open_plan(selector([0]), [[], [source]], {str(source): shots((10, 18, 20))}, "Film", True)
    assert len(plan) == 1
    assert plan[0]["duration"] == 5
    assert plan[0]["film_index"] == 1


def outro_sources(length=8):
    pools = [[Path("first.mp4")], [Path("second.mp4")], [Path("third.mp4")]]
    lookup = {str(pool[0]): shots((10, 10 + length, motion)) for pool, motion in zip(pools, (5, 15, 50))}
    return pools, lookup


def test_outro_revisits_two_movies_without_long_motion_or_repeated_freezes():
    pools, lookup = outro_sources()
    plans = outro_recap_plan(selector([.5]), pools, lookup, [9, 9], "Watch our other movie recommendations.", True)
    assert [[cut["film_index"] for cut in plan] for plan in plans] == [[0], [1]]
    assert [sum(cut["duration"] for cut in plan) for plan in plans] == [9, 9]
    assert all(cut["duration"] <= 12 and cut["freeze"] == 0 for plan in plans for cut in plan)
    assert all(cut["visual_reason"] == "calm-outro-recap" for plan in plans for cut in plan)


def test_short_outro_continues_the_same_image_across_paragraphs_without_restarting():
    pools, lookup = outro_sources()
    plans = outro_recap_plan(selector([.5]), pools, lookup, [3, 4, 5], "Save your next movie night.")
    assert {cut["film_index"] for plan in plans for cut in plan} == {0}
    assert plans[1][0]["start"] == pytest.approx(plans[0][0]["start"]+3)
    assert all(cut["duration"] <= 12 and cut["freeze"] == 0 for plan in plans for cut in plan)
    assert sum(cut["duration"] for plan in plans for cut in plan) == 12


def test_outro_paragraph_boundary_continues_the_current_excerpt_without_restarting():
    pools, lookup = outro_sources()
    plans = outro_recap_plan(selector([.5]), pools, lookup, [8, 10], "See our other picks.")
    assert [cut["film_index"] for cut in plans[0]] == [0]
    assert [cut["film_index"] for cut in plans[1]] == [0, 1]
    assert plans[1][0]["start"] == pytest.approx(plans[0][-1]["start"]+8)
    assert plans[1][0]["duration"] == 1
    assert plans[1][0]["still"] is False
    assert plans[1][1]["duration"] == 9


def test_outro_weak_matching_keeps_exact_frame_based_paragraph_lengths():
    pools, lookup = outro_sources()
    weak = Mock(relevance=Mock(side_effect=RuntimeError("Weak model match")))
    lengths = [3.13, 7.17, 8.23]
    plans = outro_recap_plan(weak, pools, lookup, lengths, "Like and subscribe.")
    assert len({cut["film_index"] for plan in plans for cut in plan}) == 2
    for length, plan in zip(lengths, plans):
        assert sum(round(cut["duration"] * 30) for cut in plan) == math.ceil(length * 30)


def test_long_closing_uses_readable_excerpts_from_at_most_two_titles():
    pools, lookup = outro_sources()
    plans = outro_recap_plan(selector([.5]), pools, lookup, [26], "Find your next movie.")
    assert len(plans[0]) == 3
    assert {cut["film_index"] for cut in plans[0]} == {0, 1}
    assert [cut["duration"] for cut in plans[0]] == pytest.approx([26/3]*3)
    assert all(cut["freeze"] == 0 for cut in plans[0])
