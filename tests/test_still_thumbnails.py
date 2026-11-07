import numpy as np
import pytest
from PIL import Image, ImageDraw

import movie_pipeline.still_thumbnails as stills
from movie_pipeline import render


def fake_still(path, cache):
    if path == "/missing":
        return None
    image = Image.new("RGB", (1280, 720), (40, 60, 110))
    draw = ImageDraw.Draw(image)
    draw.ellipse((700, 150, 950, 450), fill=(200, 160, 130))
    draw.rectangle((0, 600, 1280, 720), fill=(10, 10, 20))
    return image


@pytest.fixture
def films():
    return [{"title": "Inception", "backdrops": ["/a", "/b"]}, {"title": "Arrival", "backdrops": ["/c"]},
            {"title": "Dune", "backdrops": ["/missing"]}]


def test_three_still_thumbnails_with_headlines_and_netflix_option(monkeypatch, tmp_path, films):
    monkeypatch.setattr(stills, "fetch_still", fake_still)
    monkeypatch.setattr(stills.Cutter, "cut", lambda self, image: None)
    report = stills.build_still_thumbnails(films, ["Mind Benders", "Reality Breaks", "Watch Tonight"], tmp_path, tmp_path / "work")
    assert [item["file"] for item in report] == ["thumbnail-01.jpg", "thumbnail-02.jpg", "thumbnail-03.jpg"]
    assert report[2]["netflix_badge"] is True
    for item in report:
        path = tmp_path / item["file"]
        with Image.open(path) as image:
            assert image.size == (1280, 720)
            pixels = np.asarray(image, dtype=float)
        assert path.stat().st_size < 2_000_000
        assert pixels.std() > 20  # real composition, not a flat placeholder


def test_cutout_option_composites_subject_when_model_succeeds(monkeypatch, tmp_path, films):
    monkeypatch.setattr(stills, "fetch_still", fake_still)
    subject = Image.new("RGBA", (400, 600), (230, 180, 150, 255))
    monkeypatch.setattr(stills.Cutter, "cut", lambda self, image: subject)
    report = stills.build_still_thumbnails(films, ["A", "B", "C"], tmp_path, tmp_path / "work")
    assert report[1]["style"] == "cutout"


def test_missing_stills_fall_back_to_generated_designs(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(stills, "fetch_still", lambda path, cache: None)
    monkeypatch.setattr("movie_pipeline.thumbnails.build_thumbnails", lambda *args, **kwargs: called.setdefault("fallback", True))
    render.thumbnails([], [], ["A", "B", "C"], tmp_path, films=[{"title": "X", "backdrops": ["/x"]}], work=tmp_path / "w")
    assert called == {"fallback": True}


def test_ai_source_setting_skips_stills(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setenv("THUMBNAIL_SOURCE", "ai")
    monkeypatch.setattr(stills, "build_still_thumbnails", lambda *a, **k: called.setdefault("stills", True))
    monkeypatch.setattr("movie_pipeline.thumbnails.build_thumbnails", lambda *args, **kwargs: called.setdefault("ai", True))
    render.thumbnails([], [], ["A", "B", "C"], tmp_path, films=[{"title": "X", "backdrops": ["/x"]}])
    assert called == {"ai": True}
