import json

import pytest
from PIL import Image, ImageDraw

from movie_pipeline import footage, stills_trailer
from movie_pipeline.core import PipelineError, duration, probe

FILM = {"title": "Blocked Film", "year": 2022, "kind": "movie", "id": 7, "trailers": [{"key": "abcdefghijk", "site": "YouTube"}],
        "backdrops": [f"/b{i}.jpg" for i in range(6)]}


def picture(index):
    image = Image.new("RGB", (1920, 1080), (40 + index * 30, 60, 90))
    ImageDraw.Draw(image).ellipse((600, 300, 1300, 800), fill=(200, 180 - index * 10, 40))
    return image


def test_stills_trailer_renders_a_usable_clip(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_RESOLUTION", "720")
    monkeypatch.setattr(stills_trailer, "fetch_original", lambda path, cache: picture(int(path[2])))
    path = stills_trailer.build(FILM, tmp_path / "movie-7-720p.mp4", tmp_path)
    assert 35 <= duration(path) <= 37
    streams = {s["codec_type"]: s for s in probe(path)["streams"]}
    assert streams["video"]["height"] == 720 and "audio" in streams
    assert json.loads(path.with_suffix(".json").read_text())["stills_fallback"] is True


def test_too_few_images_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(stills_trailer, "fetch_original", lambda path, cache: None)
    with pytest.raises(PipelineError, match="usable TMDB images"):
        stills_trailer.build(FILM, tmp_path / "x.mp4", tmp_path)


def test_download_uses_stills_only_after_every_route_fails(monkeypatch, tmp_path):
    routes = []

    def failing(film, root, path, height, options, yt_dlp):
        routes.append(options.get("proxy", "direct"))
        raise PipelineError("No usable trailer for Blocked Film (2022).")
    monkeypatch.setattr(footage, "_download_route", failing)
    monkeypatch.setattr(footage.download_access, "routes", lambda: ["direct", "fallback"])
    monkeypatch.setattr(footage.download_access, "options", lambda base, route: {"proxy": route})
    built = []
    monkeypatch.setattr(stills_trailer, "build", lambda film, path, root: built.append(path) or path)
    import sys
    import types
    monkeypatch.setitem(sys.modules, "yt_dlp", types.ModuleType("yt_dlp"))
    result = footage.download(FILM, tmp_path)
    assert routes == ["direct", "fallback"] and built == [result]
    monkeypatch.setenv("STILLS_FALLBACK", "false")
    with pytest.raises(PipelineError, match="No usable trailer"):
        footage.download(FILM, tmp_path)
