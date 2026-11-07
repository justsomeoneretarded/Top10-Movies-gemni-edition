import math
from itertools import pairwise
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from movie_pipeline.shots import ShotSelector, intro_montage


class FakeSelector:
    def __init__(self):
        self.prompts = []

    def hook_score(self, path, shots, prompt, action):
        self.prompts.append(prompt)
        return int(path.stem)

    def choose(self, path, shots, prompt, seconds, action, maximum_shot=None):
        return [{"start": 50.0, "duration": seconds, "freeze": 0}]


@pytest.mark.parametrize("count, seconds", [(2, [10]), (5, [12]), (8, [1, 2]), (15, [0.7, 0.6])])
def test_intro_covers_at_least_half_of_movies_across_paragraphs(count, seconds):
    selector = FakeSelector()
    paths = [Path(f"{i}.mp4") for i in range(count)]
    plans = intro_montage(selector, paths, [[{"start": 50, "end": 55, "motion": 20}]]*count, seconds, "Top action movies", True)
    assert len({cut["film_index"] for plan in plans for cut in plan}) >= min(math.ceil(count/2), max(1, int(sum(seconds)/2.5)))
    flattened = [cut for plan in plans for cut in plan]
    assert all(a["source"] != b["source"] for a, b in pairwise(flattened))
    for plan, duration in zip(plans, seconds):
        assert sum(round(c["duration"]*30) for c in plan) == math.ceil(duration*30)
    assert "fight" in selector.prompts[0]
    assert flattened[0]["start"] == 50  # A trailer's middle can open the video.


def bare_selector(scores):
    selector = object.__new__(ShotSelector)
    selector.relevance = Mock(return_value=np.array(scores))
    selector.uses = {}
    selector.last_shots = {}
    selector.cut_count = 0
    return selector


def test_shots_can_start_in_middle_and_are_not_played_as_original_sequence():
    selector = bare_selector([0.9, 0.8, 1.0])
    shots = [{"start": 0, "end": 4, "motion": 5}, {"start": 4, "end": 8, "motion": 5}, {"start": 40, "end": 44, "motion": 5}]
    cuts = selector.choose(Path("film.mp4"), shots, "cinematic action", 6, True)
    assert cuts[0]["start"] == 40
    assert len(cuts) == 1  # Hold the final image rather than flash a short tail.
    assert sum(c["duration"] for c in cuts) == pytest.approx(6)
    assert cuts[0]["freeze"] == pytest.approx(2)


def test_next_paragraph_avoids_replaying_the_same_source_shot():
    selector = bare_selector([1.0, 0.99, 0.9])
    shots = [{"start": 0, "end": 4, "motion": 5}, {"start": 4, "end": 8, "motion": 5}, {"start": 40, "end": 44, "motion": 5}]
    selector.choose(Path("film.mp4"), shots, "action", 2, True)
    second = selector.choose(Path("film.mp4"), shots, "action", 2, True)
    assert second[0]["start"] != 0


def test_readable_long_beats_and_face_still_without_similarity_threshold():
    selector = bare_selector([-0.4, -0.5])
    path = Path('film.mp4')
    selector.faces = {str(path): [True, False]}
    selector.people = {str(path): ['Example Actor']}
    shots = [{'start': 12, 'end': 20, 'motion': 8}, {'start': 40, 'end': 48, 'motion': 40}]
    cuts = selector.choose(path, shots, 'Example Actor gives a great performance.', 20, True)
    assert cuts[0]['still']
    assert cuts[0]['duration'] == pytest.approx(5.5)
    assert cuts[0]['start'] == 16
    assert sum(c['duration'] for c in cuts) == pytest.approx(20)
    assert sum(bool(c['still']) for c in cuts) == 1
    assert len(cuts) <= 4
    assert all(c['duration'] >= 3 for c in cuts)


def test_midroll_boundary_uses_measured_time_and_never_splits_a_review():
    from movie_pipeline.editorial import subscription_boundary
    sections = [{'kind':'intro', 'paragraphs':['Hook']}] + [{'kind':'film', 'paragraphs':['Review']} for _ in range(5)] + [{'kind':'outro','paragraphs':['End']}]
    lengths = {(i,0): 15 if i in (0,6) else 90 for i in range(7)}
    boundary, fraction = subscription_boundary(sections, lengths, {i:3 for i in range(1,6)}, 6)
    assert boundary == 3
    assert .3 <= fraction <= .5
    assert sections[boundary]['kind'] == 'film'


def test_meme_replaces_one_existing_beat_near_its_spoken_joke():
    from movie_pipeline.editorial import place_intro_meme
    plan = [{'start':10,'duration':3},{'start':20,'duration':3},{'start':40,'duration':3}]
    result, used = place_intro_meme(plan,'Here is our list. Absolute cinema.',
                                  [{'start':6,'end':8,'text':'Absolute cinema.'}], {'path':'meme.jpg'})
    assert used
    assert len(result) == len(plan)
    assert result[2]['image']
    assert result[0] == plan[0]
    assert sum(s['duration'] for s in result) == 9


def test_next_paragraph_continues_the_trailer_in_story_order():
    # After playing 0-12 s, the next paragraph picks up just after it (the trailer's own next
    # scenes) instead of jumping to a slightly better-matching but unrelated moment.
    selector = bare_selector([0.6, 0.5, 0.4])
    shots = [{"start": 0, "end": 12, "motion": 5}, {"start": 12, "end": 24, "motion": 5}, {"start": 60, "end": 72, "motion": 5}]
    first = selector.choose(Path("film.mp4"), shots, "action", 12, False)
    selector.relevance = Mock(return_value=np.array([0.5, 0.5, 0.6]))
    second = selector.choose(Path("film.mp4"), shots, "action", 12, False)
    assert first[0]["start"] == 0 and 12 <= second[0]["start"] <= 20
