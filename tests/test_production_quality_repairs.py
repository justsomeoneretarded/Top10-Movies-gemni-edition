"""Current-job quality repair orchestration, without cloud or speech generation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from movie_pipeline.core import PipelineError
from movie_pipeline.production_quality import (choose_narrated_draft, fallback_pool_inputs,
    refresh_stale_plan, repair_measured_depth, write_quality_receipts)


def draft():
    return {"sections": [
        {"kind": "intro", "film": 0, "paragraphs": ["Five shows worth watching."]},
        {"kind": "film", "film": 0, "paragraphs": ["Number 1: Example.", "A doctor works a difficult shift."]},
        {"kind": "outro", "film": 0, "paragraphs": ["Like and subscribe for more."]}],
        "package": {"title": "Five Shows"}}


SPEC = SimpleNamespace(target=450, minimum=337.5, maximum=562.5, automatic=True)


def test_measured_shortfall_requests_supported_repair_without_mutating_original():
    original = draft()
    calls = []
    class Writer:
        def repair_depth(self, *args, **kwargs):
            calls.append(kwargs)
            args[3]["sections"][1]["paragraphs"].append("The team must decide who needs help first as patients keep arriving.")
            return args[3]
    repaired, changed, report = repair_measured_depth(Writer(), "title", "brief", [], original, SPEC, 220, 45)
    assert changed and len(calls) == 1
    assert calls[0]["estimated_seconds"] == 220 and calls[0]["max_repairs"] == 2
    assert report["changed_film_indices"] == [0]
    assert len(original["sections"][1]["paragraphs"]) == 2
    assert repaired["sections"][0] == original["sections"][0]


@pytest.mark.parametrize("bad_kind", ["drop_film", "change_intro", "change_rank", "empty_speech"])
def test_repair_cannot_drop_reviews_or_overwrite_verified_non_review_speech(bad_kind):
    original = draft()
    class Writer:
        def repair_depth(self, *args, **kwargs):
            result = args[3]
            if bad_kind == "drop_film": result["sections"].pop(1)
            if bad_kind == "change_intro": result["sections"][0]["paragraphs"] = ["An unrelated new hook."]
            if bad_kind == "change_rank": result["sections"][1]["paragraphs"][0] = "Number 2: Wrong."
            if bad_kind == "empty_speech": result["sections"][1]["paragraphs"] = []
            return result
    result, changed, _ = repair_measured_depth(Writer(), "t", "b", [], original, SPEC, 220, 45)
    assert result is original and not changed


def test_unavailable_repair_retains_complete_draft_and_only_records_exception_class():
    class Writer:
        def repair_depth(self, *args, **kwargs):
            raise RuntimeError("potentially-sensitive transport body")
    original = draft()
    result, changed, report = repair_measured_depth(Writer(), "t", "b", [], original, SPEC, 220, 45)
    assert result is original and not changed
    assert report["reason"] == "RuntimeError"
    assert "sensitive" not in json.dumps(report)


def test_complete_in_range_episode_does_not_trigger_another_editor_call():
    class Writer:
        def repair_depth(self, *args, **kwargs):
            pytest.fail("No timing repair is needed")
    original = draft()
    assert repair_measured_depth(Writer(), "t", "b", [], original, SPEC, 400, 45)[:2] == (original, False)


def test_promoted_hd_decode_fallback_stays_in_same_film_and_preserves_other_primary():
    ahd, asd, bhd, bsd = map(Path, ["a-hd.mp4", "a-sd.mp4", "b-hd.mp4", "b-sd.mp4"])
    pools = [[ahd, asd], [bhd, bsd]]
    trailers = [ahd, bhd]
    shots = [[], []]
    lookup = {str(p): [{"start": 1, "end": 8}] for p in (ahd, asd, bhd, bsd)}
    assert fallback_pool_inputs(pools, trailers, shots, lookup, [asd, bsd], {str(ahd)})
    assert trailers == [asd, bhd] and pools == [[asd], [bhd, bsd]]
    assert shots[0] is lookup[str(asd)]


def test_bad_original_can_fall_back_to_hd_but_missing_alternative_is_not_hidden():
    original, extra = Path("a.mp4"), Path("a-extra.mp4")
    pools, trailers, shots = [[original, extra]], [original], [[]]
    assert fallback_pool_inputs(pools, trailers, shots, {str(extra): []}, [original], {str(original)})
    assert trailers == [extra]
    with pytest.raises(PipelineError):
        fallback_pool_inputs([[original]], [original], [[]], {str(original): []}, [original], {str(original)})


def test_receipts_copy_current_job_only_and_export_actual_spoken_script(tmp_path):
    work, output = tmp_path / "current", tmp_path / "output"
    work.mkdir()
    (work / "audio-review.json").write_text('{"accepted":true,"current":true}')
    old = tmp_path / "old"
    old.mkdir()
    (old / "audio-review.json").write_text('{"accepted":false}')
    write_quality_receipts(work, output, draft(), measured_repairs=[{"status": "retained"}])
    assert json.loads((output / "audio-review.json").read_text())["current"]
    assert "A doctor works" in (output / "final-script.txt").read_text()
    assert json.loads((output / "production-quality-report.json").read_text())["measured_repairs"]


def speech_for(script):
    return {(i, p): Path(f"speech-{i}-{p}.wav")
            for i, section in enumerate(script["sections"])
            for p in range(len(section["paragraphs"]))}


@pytest.mark.parametrize("distance", [10, 11, 30])
def test_complete_substantive_repair_is_kept_even_if_duration_is_equal_or_less_close(distance):
    original, repaired = draft(), draft()
    repaired["sections"][1]["paragraphs"][1] += " Patients arrive faster than the team can treat them."
    previous = (10, original, speech_for(original))
    selected, adopted = choose_narrated_draft(previous, distance, repaired, speech_for(repaired), prefer_quality=True)
    assert adopted and selected[1] is repaired


@pytest.mark.parametrize("missing", ["none", "key"])
def test_incomplete_repaired_speech_never_discards_a_complete_take(missing):
    original, repaired = draft(), draft()
    previous = (10, original, speech_for(original))
    speech = speech_for(repaired)
    if missing == "none":
        speech[(1, 1)] = None
    else:
        del speech[(1, 1)]
    selected, adopted = choose_narrated_draft(previous, 0, repaired, speech, prefer_quality=True)
    assert not adopted and selected is previous


def test_normal_timing_revision_still_prefers_the_closer_complete_draft():
    original, revised = draft(), draft()
    previous = (10, original, speech_for(original))
    assert choose_narrated_draft(previous, 11, revised, speech_for(revised))[0] is previous
    assert choose_narrated_draft(previous, 9, revised, speech_for(revised))[0][1] is revised


def test_already_pruned_extra_still_permits_same_title_fallback():
    original, removed, other = map(Path, ["movie-1.mp4", "movie-1-extra3.mp4", "movie-2.mp4"])
    pools, trailers, shots = [[original], [other]], [original, other], [[], []]
    lookup = {str(original): [{"start": 0, "end": 8}], str(other): []}
    assert fallback_pool_inputs(pools, trailers, shots, lookup, trailers[:],
                                {str(removed)}, affected_films={0})
    assert pools == [[original], [other]] and trailers == [original, other]


def test_stale_montage_replans_only_removed_source_with_stable_film_identity():
    original, removed, other = map(Path, ["movie-1.mp4", "movie-1-extra3.mp4", "movie-2.mp4"])
    calls = []
    class Selector:
        def choose(self, source, shots, text, seconds, action, **kwargs):
            calls.append(source)
            return [{"start": 2, "duration": seconds}]
    keep = {"source": str(other), "start": 4, "duration": 3, "film_index": 1}
    stale = {"source": str(removed), "start": 5, "duration": 4, "still": True}
    image = {"source": "meme.jpg", "image": True, "duration": 2}
    result = refresh_stale_plan(Selector(), [keep, stale, image], [[original], [other]],
        [original, other], [[{"start": 0, "end": 10}], []],
        {str(original): 0, str(removed): 0, str(other): 1}, "context", False, 1)
    assert result[0] is keep and result[2] is image
    assert result[1]["source"] == str(original) and result[1]["film_index"] == 0
    assert result[1]["duration"] == 4 and result[1]["still"]
    assert calls == [original]
