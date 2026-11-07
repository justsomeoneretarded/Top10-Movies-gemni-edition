import numpy as np
from PIL import Image

from movie_pipeline.core import ffmpeg
from movie_pipeline.review import build_review, loudness_spikes, repeated_shots, silent_captions


def test_checks_find_repeats_spikes_and_drift():
    pattern = Image.effect_mandelbrot((320, 180), (-2, -1, 1, 1), 60).convert("RGB")
    other = Image.linear_gradient("L").resize((320, 180)).convert("RGB")
    frames = [(0, pattern), (10, other), (40, pattern)]
    assert repeated_shots(frames, ignore_after=100) == [(0, 40)]
    assert repeated_shots(frames, ignore_after=30) == []
    profile = np.full(100, -30.0)
    profile[50] = -10
    profile[60:70] = -80
    assert loudness_spikes(profile) == [20.0]
    cues = [{"start": 24.0, "end": 27.6, "text": "said over silence"}, {"start": 2.0, "end": 4.0, "text": "fine"}]
    assert [c["text"] for c in silent_captions(cues, profile)] == ["said over silence"]


def test_review_page_is_written(tmp_path, monkeypatch):
    monkeypatch.setenv("SCENE_DESCRIPTIONS", "false")
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30", "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000",
           "-t", 30, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", tmp_path / "final-video.mp4")
    Image.new("RGB", (1280, 720), (90, 20, 20)).save(tmp_path / "thumbnail-01.jpg")
    summary = build_review(tmp_path, [{"start": 1, "end": 3, "text": "Hello."}], [{"start": 0, "title": "Intro"}], "horror", ending_seconds=5)
    page = (tmp_path / "review.html").read_text()
    assert "Contact sheet" in page and "thumbnail-01.jpg" in page and "data:image/jpeg;base64," in page
    assert summary["thumbnails"][0]["file"] == "thumbnail-01.jpg"


def test_intro_and_recap_repeats_are_not_counted():
    pattern = Image.effect_mandelbrot((320, 180), (-2, -1, 1, 1), 60).convert("RGB")
    other = Image.linear_gradient("L").resize((320, 180)).convert("RGB")
    frames = [(0, pattern), (10, other), (40, pattern)]
    assert repeated_shots(frames, ignore_after=100, ignore=[(35, 45)]) == []
    assert repeated_shots(frames, ignore_after=100, ignore=[(0, 5)]) == []
