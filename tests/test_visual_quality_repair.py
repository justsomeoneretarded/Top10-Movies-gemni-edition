"""Conservative picture repair must preserve speech and the exact frame clock."""
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from movie_pipeline import quality_repair as repair
from movie_pipeline.core import PipelineError, ffmpeg


def event(kind="narration"):
    return {"type": kind, "film": "The Studio (2025)", "section_kind": "review",
            "start": 0.0, "duration": 4.0}


def test_failed_repairs_are_bounded_and_original_parts_are_retained(monkeypatch, tmp_path):
    paths = [tmp_path / f"part-{i}.mp4" for i in range(5)]
    for path in paths:
        path.write_bytes(b"original media")
    monkeypatch.setattr(repair, "_video_info", lambda path: {"seconds": 4.0})
    monkeypatch.setattr(repair, "_black_intervals", lambda *args: [(1.5, 2.2)])
    monkeypatch.setattr(repair, "_classify", lambda *args: {
        "status": "repairable", "reason": "empty_gap_under_continuous_narration",
        "hold": Image.new("RGB", (32, 18), "red")})
    calls = []

    def failed(path, *args):
        calls.append(path)
        raise PipelineError("candidate decode failed")

    monkeypatch.setattr(repair, "_replace_empty_gap", failed)
    report = repair.repair_visual_parts(paths, [event() for _ in paths],
                                        tmp_path / "report", max_repairs=2)
    assert report["repaired"] == []
    assert report["attempted_repairs"] == 2
    assert len(calls) == 2
    assert len(report["unresolved"]) == 2
    assert all(path.read_bytes() == b"original media" for path in paths)
    assert report["uninspected_parts"] == 3


def test_part_identity_mismatch_never_repairs(monkeypatch, tmp_path):
    monkeypatch.setattr(repair, "_replace_empty_gap",
                        lambda *args: pytest.fail("mismatched timeline cannot be repaired"))
    report = repair.repair_visual_parts([tmp_path / "one.mp4"], [], tmp_path / "work")
    assert report["repaired"] == []
    assert report["unresolved"] == [{"reason": "part_timeline_identity_mismatch"}]


@pytest.mark.parametrize("kind", ["trailer-audio", "review-preview", "cold-open"])
def test_native_scene_black_pauses_remain_advisory(monkeypatch, kind):
    def frame(path, point):
        return Image.new("RGB", (320, 180), "black") if 1.5 < point < 2.2 else Image.fromarray(
            np.tile(np.arange(320, dtype=np.uint8), (180, 1)))
    monkeypatch.setattr(repair, "get_frame", frame)
    result = repair._classify(Path("native.mp4"), 1.5, 2.2, (0, 4), event(kind))
    assert result["status"] == "advisory"
    assert result["reason"] == "native_scene_pause_requires_review"


def test_export_inspection_requires_exact_tempo_mapping(monkeypatch):
    monkeypatch.setattr(repair, "_video_info", lambda path: {"seconds": 4.0})
    monkeypatch.setattr(repair, "_black_intervals",
                        lambda *args: pytest.fail("cannot map a repair without export speed"))
    result = repair.inspect_visual_export(Path("finished.mp4"), [event()], {})
    assert result["repairable_parts"] == []
    assert result["status"] == "exact_tempo_unavailable"


def test_export_gap_maps_to_its_same_title_part_at_actual_speed(monkeypatch):
    monkeypatch.setattr(repair, "_video_info", lambda path: {"seconds": 8.0})
    monkeypatch.setattr(repair, "_black_intervals", lambda *args: [(3.0, 3.7)])
    seen = []
    def classify(path, start, end, bounds, row):
        seen.append((bounds, row["film"]))
        return {"status": "repairable", "hold": Image.new("RGB", (2, 2))}
    monkeypatch.setattr(repair, "_classify", classify)
    timeline = [dict(event(), start=0, duration=4), dict(event(), start=4, duration=4, film="The Pitt (2025)")]
    result = repair.inspect_visual_export(Path("finished.mp4"), timeline, {"export_speed": 2})
    assert result["repairable_parts"] == [1]
    assert seen == [((2.0, 4.0), "The Pitt (2025)")]


HAS_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


def media_with_insert(tmp_path, name, image):
    """Actual encoded picture, clear tonal speech stand-in and an internal slate/gap."""
    slate = tmp_path / f"{name}.png"
    image.save(slate)
    target = tmp_path / f"{name}.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30",
           "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
           "-loop", 1, "-framerate", 30, "-i", slate,
           "-filter_complex", "[0:v][2:v]overlay=enable='between(t,1.5,2.2)'[v]",
           "-map", "[v]", "-map", "1:a:0", "-t", 4,
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", 10,
           "-pix_fmt", "yuv420p", "-c:a", "aac", target)
    return target


@pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg is required")
def test_real_empty_gap_repair_keeps_decoded_audio_and_exact_video_clock(tmp_path):
    video = media_with_insert(tmp_path, "empty", Image.new("RGB", (320, 180), "black"))
    before = repair._video_info(video)
    original_audio = repair._audio_digest(video)
    assert repair._black_intervals(video, 20)
    report = repair.repair_visual_parts([video], [event()], tmp_path / "repair", max_repairs=1)
    assert len(report["repaired"]) == 1, report
    assert report["repaired"][0]["film"] == "The Studio (2025)"
    after = repair._video_info(video)
    assert after["frames"] == before["frames"] == 120
    assert (after["width"], after["height"]) == (320, 180)
    assert after["seconds"] == pytest.approx(before["seconds"], abs=.002)
    assert repair._audio_digest(video) == original_audio
    assert repair._black_intervals(video, 20) == []
    assert report["repaired"][0]["audio"] == "decoded_sha256_identical"
    ffmpeg("-xerror", "-err_detect", "explode", "-i", video, "-f", "null", "-")


@pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg is required")
@pytest.mark.parametrize("content", ["dark_texture", "promo_text"])
def test_real_dark_scene_and_text_slate_are_advisories_not_deleted(tmp_path, content):
    if content == "dark_texture":
        pixels = np.zeros((180, 320, 3), dtype=np.uint8)
        pixels[:, 12:24] = 12
        pixels[:, 48:60] = 12
        pixels[:, 96:108] = 12
        pixels[:, 156:168] = 12
        pixels[:, 228:240] = 12
        image = Image.fromarray(pixels)
    else:
        image = Image.new("RGB", (320, 180), "black")
        ImageDraw.Draw(image).text((115, 85), "1 SHIFT", fill="white")
    video = media_with_insert(tmp_path, content, image)
    before = hashlib.sha256(video.read_bytes()).hexdigest()
    assert repair._black_intervals(video, 20), "fixture must exercise black detection"
    report = repair.repair_visual_parts([video], [event()], tmp_path / "review")
    assert report["repaired"] == []
    assert any(row["reason"] == "dark_scene_or_visible_text" for row in report["advisories"]), report
    assert hashlib.sha256(video.read_bytes()).hexdigest() == before


@pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg is required")
def test_real_rejected_candidate_cannot_replace_original_audio_or_picture(monkeypatch, tmp_path):
    video = media_with_insert(tmp_path, "invalid", Image.new("RGB", (320, 180), "black"))
    before = video.read_bytes()
    actual = repair._audio_digest
    calls = []
    def corrupt_digest(path):
        calls.append(path)
        return actual(path) if Path(path) == video else "different-audio"
    monkeypatch.setattr(repair, "_audio_digest", corrupt_digest)
    with pytest.raises(ValueError, match="repair_changed_audio_or_frame_clock"):
        repair._replace_empty_gap(video, 1.5, 2.2, repair.get_frame(video, 1.2), tmp_path / "candidate")
    assert len(calls) == 2
    assert video.read_bytes() == before
    assert not (tmp_path / "candidate" / "candidate.mp4").exists()
