from pathlib import Path

import pytest

from movie_pipeline.source_pool import best_intro_sources, choose_from_sources


class Selector:
    def __init__(self, primary: Path, secondary: Path, scores=(.4, .4)):
        self.scores = {str(primary): scores[0], str(secondary): scores[1]}
        self.features = dict.fromkeys(self.scores, object())
        self.calls = []
        self.fail_score = set()
        self.fail_choose = set()

    def hook_score(self, path, shots, text, action):
        self.calls.append(('score', str(path), text, action))
        if str(path) in self.fail_score:
            raise RuntimeError('Optional image matching unavailable')
        return self.scores[str(path)]

    def choose(self, path, shots, text, seconds, action):
        self.calls.append(('choose', str(path), text, action))
        if str(path) in self.fail_choose:
            raise OSError('Optional footage unreadable')
        return [{'start': 10, 'duration': seconds/2, 'freeze': .5},
                {'start': 40, 'duration': seconds/2, 'freeze': 0}]


def setup(tmp_path, scores=(.4, .4)):
    primary, secondary = tmp_path/'primary.mp4', tmp_path/'extra.mp4'
    for path in (primary, secondary):
        path.write_bytes(b'media')
    shots = {str(path): [{'start':10,'end':20,'motion':8}] for path in (primary,secondary)}
    return primary, secondary, shots, Selector(primary, secondary, scores)


def test_varies_source_between_paragraphs_without_adding_cuts(tmp_path):
    primary, secondary, shots, selector = setup(tmp_path)
    first = choose_from_sources(selector,[primary,secondary],shots,'A chase.',16,True)
    second = choose_from_sources(selector,[primary,secondary],shots,'The escape.',16,True)
    third = choose_from_sources(selector,[primary,secondary],shots,'The stakes.',16,True)
    assert {s['source'] for s in first} == {str(primary)}
    assert {s['source'] for s in second} == {str(secondary)}
    assert {s['source'] for s in third} == {str(primary)}
    assert len(first) == len(second) == 2
    assert [s['duration'] for s in second] == [8,8]
    assert [s['freeze'] for s in second] == [.5,0]
    assert all(call[3] is False for call in selector.calls if call[0]=='score')


def test_single_trailer_preserves_source_timings_and_narration(tmp_path):
    primary, _secondary, shots, selector = setup(tmp_path)
    plan = choose_from_sources(selector,[primary],shots,'A character.',14,False)
    assert plan == [{'start':10,'duration':7,'freeze':.5,'source':str(primary)},
                    {'start':40,'duration':7,'freeze':0,'source':str(primary)}]


@pytest.mark.parametrize('failure',['relevance','missing_shots','missing_file','missing_features','choose'])
def test_optional_failures_use_primary_without_failing_review(tmp_path, failure):
    primary, secondary, shots, selector = setup(tmp_path, (.2,.8))
    if failure == 'relevance':
        selector.fail_score.add(str(secondary))
    elif failure == 'missing_shots':
        del shots[str(secondary)]
    elif failure == 'missing_file':
        secondary.unlink()
    elif failure == 'missing_features':
        del selector.features[str(secondary)]
    else:
        selector.fail_choose.add(str(secondary))
    plan = choose_from_sources(selector,[primary,secondary],shots,'A story.',18,False)
    assert {shot['source'] for shot in plan} == {str(primary)}
    assert sum(shot['duration'] for shot in plan) == 18


def test_similarity_failure_has_no_minimum_relevance_gate(tmp_path):
    primary, secondary, shots, selector = setup(tmp_path, (-.8,-.9))
    selector.fail_score.update([str(primary),str(secondary)])
    plan = choose_from_sources(selector,[primary,secondary],shots,'A loose visual description.',12,False)
    assert {shot['source'] for shot in plan} == {str(primary)}


def test_intro_keeps_one_choice_per_film_and_soft_fallback(tmp_path):
    primary, secondary, shots, selector = setup(tmp_path, (.2,.8))
    result = best_intro_sources(selector,[[primary,secondary],[primary]],shots,'A hook.',True)
    assert result == [secondary,primary]
    selector.fail_score.add(str(secondary))
    assert best_intro_sources(selector,[[primary,secondary]],shots,'A hook.',True) == [primary]


def test_render_fallback_replaces_only_optional_sources_and_keeps_intro_film_identity(tmp_path):
    from movie_pipeline.source_pool import primary_fallback_plan

    class FallbackSelector:
        def __init__(self):
            self.calls = []
        def choose(self, path, shots, text, seconds, action, maximum_shot=None):
            self.calls.append((path, shots, seconds, maximum_shot))
            return [{'start': shots[0]['start'], 'duration': seconds, 'freeze': 0}]

    primary = [tmp_path/'film0.mp4', tmp_path/'film1.mp4']
    shot_sets = [[{'start':10,'end':20}], [{'start':40,'end':50}]]
    unchanged = {'source':str(primary[0]),'start':12,'duration':3,'film_index':0}
    meme = {'source':str(tmp_path/'meme.jpg'),'image':True,'still':True,'start':0,'duration':3,'film_index':1}
    optional = {'source':str(tmp_path/'film1-extra.mp4'),'start':7,'duration':4,'film_index':1}
    chosen = [unchanged, meme, optional]
    selector = FallbackSelector()
    result = primary_fallback_plan(selector,chosen,primary,shot_sets,'A hook.',True,0)
    assert result[:2] == [unchanged,meme]
    assert result[2]['source'] == str(primary[1])
    assert result[2]['start'] == 40
    assert result[2]['film_index'] == 1
    assert sum(s['duration'] for s in result) == sum(s['duration'] for s in chosen)
    assert selector.calls == [(primary[1],shot_sets[1],4,4)]


def test_render_fallback_keeps_one_held_frame_for_optional_still(tmp_path):
    from movie_pipeline.source_pool import primary_fallback_plan

    class FallbackSelector:
        def choose(self, path, shots, text, seconds, action, maximum_shot=None):
            return [{'start':10,'duration':seconds/2,'freeze':.5},
                    {'start':40,'duration':seconds/2,'freeze':0}]

    primary = tmp_path/'film.mp4'
    chosen = [{'source':str(tmp_path/'extra.mp4'),'start':7,'duration':6.4,'still':True,'freeze':0}]
    result = primary_fallback_plan(FallbackSelector(),chosen,[primary],[[{'start':10,'end':20}]],
                                   'A performance.',False,0)
    assert len(result) == 1
    assert result[0]['duration'] == pytest.approx(6.4)
    assert result[0]['still'] is True
    assert result[0]['freeze'] == 0
    assert result[0]['source'] == str(primary)
    assert result[0]['film_index'] == 0


def test_fresh_second_trailer_is_used_within_a_long_paragraph(tmp_path):
    import numpy as np

    from movie_pipeline.shots import ShotSelector

    primary, secondary, _, _ = setup(tmp_path)
    selected = object.__new__(ShotSelector)
    selected.uses, selected.last_shots, selected.cut_count = {}, {}, 0
    selected.features = {str(primary): object(), str(secondary): object()}
    selected.relevance = lambda *_: np.array([1.0, .8, .7])
    selected.card_kind = lambda *_: None
    shots = {str(p): [{"start": i*20, "end": i*20+8, "motion": 25} for i in range(3)]
             for p in (primary, secondary)}
    plan = choose_from_sources(selected, [primary, secondary], shots, "A chase through the city.", 36, False)
    assert sum(s["duration"] for s in plan) == pytest.approx(36)
    assert {s["source"] for s in plan} == {str(primary), str(secondary)}
    for i, a in enumerate(plan):
        for b in plan[i+1:]:
            if a["source"] == b["source"] and not a.get("still") and not b.get("still"):
                assert min(a["start"]+a["duration"]-a.get("freeze",0), b["start"]+b["duration"]-b.get("freeze",0)) <= max(a["start"], b["start"])+.01


def test_exhausted_trailer_rotates_oldest_scene_despite_strong_hero_match():
    import numpy as np

    from movie_pipeline.shots import ShotSelector

    path = Path("short.mp4")
    selected = object.__new__(ShotSelector)
    selected.uses, selected.last_shots, selected.cut_count = {}, {}, 0
    selected.relevance = lambda *_: np.array([.1, .1, 100.0])
    selected.card_kind = lambda *_: None
    shots = [{"start": i*20, "end": i*20+8, "motion": 25} for i in range(3)]
    for shot in shots:
        selected.reserve(path, shot["start"], 8)
    plan = selected.choose(path, shots, "The story continues.", 16, False)
    assert [cut["start"] for cut in plan] == [0, 20]
