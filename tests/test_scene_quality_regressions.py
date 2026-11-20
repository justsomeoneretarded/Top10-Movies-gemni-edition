"""Regression checks for source identity and coherent scene boundaries."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from movie_pipeline import gemini_editing, moments, previews, review, shots
from movie_pipeline import editorial_policy


def test_same_size_same_duration_source_replacement_invalidates_shot_and_frame_caches(monkeypatch, tmp_path):
    source = tmp_path / "same-name.mp4"
    source.write_bytes(b"AAAA")
    original_stat = source.stat()
    monkeypatch.setattr(shots, "duration", lambda path: 10.0)
    monkeypatch.setenv("ANALYSIS_CACHE", str(tmp_path / "shared"))
    cache = tmp_path / "shots.json"
    old_identity = shots._analysis_identity(source)
    old_shots = [{"start": 2.0, "end": 8.0, "motion": 10.0}]
    hashed_cache = cache.with_name(cache.stem + f".{shots.ANALYSIS_POLICY}-{old_identity['sha256']}.json")
    hashed_cache.write_text(json.dumps(old_shots))
    frames = source.with_suffix(".frames")
    frames.mkdir()
    (frames / "source.json").write_text(json.dumps(old_identity))
    Image.new("RGB", (30, 20), "red").save(frames / "0000.jpg")
    assert shots.analyse(source, cache) == old_shots
    assert shots.shot_frame(source, old_shots, 0).getpixel((10, 10))[0] > 200
    replacement = tmp_path / "replacement.mp4"
    replacement.write_bytes(b"BBBB")
    os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    replacement.replace(source)
    assert source.stat().st_size == original_stat.st_size
    assert source.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert shots._analysis_identity(source)["sha256"] != old_identity["sha256"]
    monkeypatch.setattr(shots, "get_frame", lambda *args: Image.new("RGB", (30, 20), "green"))
    assert shots.shot_frame(source, old_shots, 0).getpixel((10, 10)) == (0, 128, 0)

    class AnalysisRequested(RuntimeError):
        pass

    def analyse_new_bytes(*args):
        raise AnalysisRequested("fresh source must be analysed")

    monkeypatch.setattr(shots, "promo_cutoff", lambda *args: None)
    monkeypatch.setattr(shots.cv2, "VideoCapture", analyse_new_bytes)
    with pytest.raises(AnalysisRequested):
        shots.analyse(source, cache)


def test_complete_verified_scene_can_preserve_more_than_three_connected_shots(monkeypatch):
    source = Path("movie.mp4")
    shot_list = [{"start": i * 2.0, "end": (i + 1) * 2.0, "motion": 20}
                 for i in range(6)]
    scene = {"id": "complete-action", "start": 0.0, "end": 12.0, "kind": "action",
             "energy": .9, "sound": "native", "complete": True, "verified": True,
             "description": "A complete chase from setup to escape.", "interest": .9}
    advice = {"provider": {"model": "verified-test"}, "cache": "test"}
    monkeypatch.setattr(gemini_editing, "analyse_source", lambda path: advice)
    monkeypatch.setattr(gemini_editing, "verify_scenes", lambda *args: [scene])
    monkeypatch.setattr(gemini_editing, "scene_for", lambda *args: scene)
    monkeypatch.setattr(gemini_editing, "soft_rank", lambda *args: (0.0, []))
    monkeypatch.setattr(gemini_editing, "opening_fit", lambda *args: .9)
    monkeypatch.setattr(editorial_policy, "title_genres", lambda film: ["action"])
    monkeypatch.setattr(moments.peak, "envelope", lambda source: np.full(120, -20.0))
    selector = SimpleNamespace(identity_relevance=lambda *args: np.full(6, .5),
                               features={}, card_kind=lambda *args: None)
    rows = moments.candidates(selector, [source], {str(source): shot_list}, {"title": "Movie"})
    assert len(rows) == 1
    assert rows[0]["native_scene_window"] is True
    assert rows[0]["duration"] == 12
    assert rows[0]["coherent_start"] == 0 and rows[0]["coherent_end"] == 12


def test_complete_scene_candidate_rejects_a_promo_card_inside_its_extent(monkeypatch):
    source = Path("movie.mp4")
    shot_list = [{"start": i * 2.0, "end": (i + 1) * 2.0, "motion": 20}
                 for i in range(6)]
    scene = {"id": "scene", "start": 0.0, "end": 12.0, "kind": "action", "energy": .9,
             "sound": "native", "complete": True, "verified": True,
             "description": "Not actually safe when a promo card intervenes.", "interest": .9}
    monkeypatch.setattr(gemini_editing, "analyse_source", lambda path: {"provider": {}, "cache": "test"})
    monkeypatch.setattr(gemini_editing, "verify_scenes", lambda *args: [scene])
    monkeypatch.setattr(gemini_editing, "scene_for", lambda *args: scene)
    monkeypatch.setattr(gemini_editing, "soft_rank", lambda *args: (0.0, []))
    monkeypatch.setattr(editorial_policy, "title_genres", lambda film: ["action"])
    monkeypatch.setattr(moments.peak, "envelope", lambda source: np.full(120, -20.0))
    selector = SimpleNamespace(identity_relevance=lambda *args: np.full(6, .5), features={},
                               card_kind=lambda path, index: "promo" if index == 2 else None)
    assert moments.candidates(selector, [source], {str(source): shot_list}, {"title": "Movie"}) == []


def test_native_dialogue_expands_to_complete_sentence_only_within_its_scene():
    candidate = {"source": "movie.mp4", "start": 4.0, "duration": 5.0}
    cues = [{"start": 3.0, "end": 4.2, "text": "We need"},
            {"start": 4.3, "end": 6.0, "text": "to get out."}]
    result = moments.complete_window(candidate, cues, (1.0, 12.0))
    assert result["start"] == pytest.approx(2.86)
    assert result["start"] + result["duration"] == pytest.approx(9.0)
    assert result["native_verified"] is True
    assert [cue["text"] for cue in result["cues"]] == ["We need", "to get out."]
    assert moments.complete_window(candidate, cues, (4.0, 12.0)) is None


def test_native_silent_candidate_cannot_cross_coherent_extent():
    candidate = {"start": 4.0, "duration": 6.0}
    assert moments.complete_window(candidate, [], (4.0, 9.0)) is None
    assert moments.complete_window(candidate, [], (5.0, 12.0)) is None
    assert moments.complete_window(candidate, [], (4.0, 10.0))["duration"] == 6.0


def test_short_preview_holds_clean_frame_from_its_own_source_not_promo_card():
    source = Path("target-film.trailer-5.mp4")
    lookup = {str(source): [{"start": 2, "end": 4, "motion": 2},
                            {"start": 4, "end": 6, "motion": 40}]}
    selector = SimpleNamespace(relevance=lambda *args: np.array([.1, 1.0]),
                               card_kind=lambda path, index: "promo" if index == 1 else None)
    result = previews.choose_preview(selector, [source], lookup, "Target Film", 4.0, True)
    assert result["source"] == str(source)
    assert result["still"] is True and result["muted"] is True
    assert result["start"] == pytest.approx(3.0)


def test_thumbnail_review_includes_all_ten_finished_options(monkeypatch, tmp_path):
    from movie_pipeline import scenes
    monkeypatch.setattr(scenes, "enabled", lambda: False)
    for i in range(1, 11):
        Image.new("RGB", (128, 72), (i * 15, 20, 20)).save(tmp_path / f"thumbnail-{i:02}.jpg")
    Image.new("RGB", (128, 72), "red").save(tmp_path / "thumbnail-99.jpg")
    rows = review.thumbnail_scores(tmp_path, "action")
    assert [row["file"] for row in rows] == [f"thumbnail-{i:02}.jpg" for i in range(1, 11)]
