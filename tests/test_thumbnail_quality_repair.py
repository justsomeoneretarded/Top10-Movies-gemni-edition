"""Offline regressions for observable photo discovery and bounded composition repairs."""
import hashlib
import html
import json
import threading
import time
from io import BytesIO

import numpy as np
import pytest
import requests
from PIL import Image, ImageChops, ImageDraw

import movie_pipeline.image_discovery as discovery
import movie_pipeline.thumbnail_studio as studio


def patterned(color, marker=0, size=(1200, 700)):
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((50 + marker * 40, 100, 340 + marker * 40, 600), fill="white")
    draw.ellipse((500, 100 + marker * 10, 850, 500 + marker * 10), fill="black")
    return image


@pytest.fixture(autouse=True)
def clear_discovery_context():
    discovery._LOCAL.report = None
    yield
    discovery._LOCAL.report = None


def begin_report():
    report = {"queries": [], "downloads": [], "counts": {}, "_deadline": time.monotonic() + 60}
    discovery._LOCAL.report = report
    return report


def entry(url="https://images.example.com/odyssey.jpg", page="https://example.com/the-odyssey-2026-matt-damon"):
    return {"murl": url, "purl": page, "t": "The Odyssey 2026 Matt Damon film scene"}


class SearchResponse:
    def __init__(self, body="", status=200):
        self.text = body
        self.status_code = status
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("synthetic offline response")

    def close(self):
        self.closed = True


@pytest.mark.parametrize("quote", ["'", '"'])
def test_metadata_parser_accepts_both_quote_styles_and_entities(quote):
    metadata = html.escape(json.dumps(entry()), quote=True)
    items, info = discovery.parse_metadata(f"<a class='iusc' m={quote}{metadata}{quote}></a>")
    assert items == [entry()]
    assert info == {"metadata_seen": 1, "malformed_metadata": 0}


def test_metadata_parser_reports_malformed_and_bounds_work():
    body = "<a m='not-json'></a>" + "<a m='{}'></a>" * 250
    rows, report = discovery.parse_metadata(body)
    assert rows == []
    assert report["malformed_metadata"] == 1
    assert report["metadata_seen"] == 200


@pytest.mark.parametrize(("response", "expected"), [
    (SearchResponse(status=403), "blocked_or_rate_limited"),
    (SearchResponse(status=429), "blocked_or_rate_limited"),
    (SearchResponse("Please verify you are human"), "challenge"),
    (SearchResponse("<html>No matches</html>"), "no_image_metadata"),
    (SearchResponse("<a m='broken'></a>"), "malformed_metadata"),
])
def test_search_explains_blocked_challenge_empty_and_parser_outcomes(monkeypatch, response, expected):
    monkeypatch.setattr(discovery, "image_queries", lambda film, limit: ["The Odyssey 2026 best scene image"])
    monkeypatch.setattr(discovery.requests, "get", lambda *a, **k: response)
    report = begin_report()
    assert discovery.search({"title": "The Odyssey", "year": 2026, "cast": ["Matt Damon"]}) == []
    assert report["queries"][0]["status"] == expected
    assert response.closed


def test_search_keeps_valid_identity_and_reports_rejections(monkeypatch):
    wrong = entry("https://images.example.com/wrong.jpg", "https://www.themoviedb.org/movie/999-other")
    right = entry()
    duplicate = dict(right)
    body = "".join("<a m='" + html.escape(json.dumps(row), quote=True) + "'></a>"
                   for row in (wrong, right, duplicate))
    monkeypatch.setattr(discovery, "image_queries", lambda film, limit: ["The Odyssey best image"])
    monkeypatch.setattr(discovery.requests, "get", lambda *a, **k: SearchResponse(body))
    report = begin_report()
    film = {"title": "The Odyssey", "year": 2026, "kind": "movie", "id": 123, "cast": ["Matt Damon"]}
    rows = discovery.search(film)
    assert len(rows) == 1 and rows[0]["url"] == right["murl"]
    assert report["queries"][0]["rejections"] == {"identity_not_corroborated": 1, "duplicate_url": 1}


def test_corroboration_rejects_foreign_record_and_wrong_requested_season():
    film = {"title": "The Odyssey", "year": 2026, "id": 123, "kind": "movie",
            "cast": ["Matt Damon"], "imdb_id": "tt1234567"}
    context = "The Odyssey 2026 Matt Damon"
    assert not discovery.corroborates(film, context, "https://www.themoviedb.org/movie/999-the-odyssey")
    assert not discovery.corroborates(film, context, "https://www.imdb.com/title/tt9999999/")
    assert discovery.corroborates(film, context, "https://www.themoviedb.org/movie/123-the-odyssey")
    assert not discovery.corroborates(film, context + " AI-generated concept art", "https://example.com/image")
    show = {"title": "Example Show", "kind": "tv", "requested_season": 3, "cast": ["Known Person"]}
    assert not discovery.corroborates(show, "Example Show season 1 Known Person", "https://example.com/show")


def test_cached_images_are_validated_before_reuse(monkeypatch, tmp_path):
    url = "https://images.example.com/too-small.jpg"
    cache = tmp_path / "cache"
    cache.mkdir()
    target = cache / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".jpg")
    Image.new("RGB", (200, 100), "red").save(target)
    monkeypatch.setattr(discovery.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("offline")))
    report = begin_report()
    assert discovery.read_image(url, cache) is None
    assert not target.exists()
    assert report["counts"]["invalid_cached_image"] == 1
    assert report["downloads"][0]["result"] == "download_ConnectionError"


def test_hash_collision_does_not_erase_different_flat_images():
    red = Image.new("RGB", (1000, 600), "red")
    blue = Image.new("RGB", (1000, 600), "blue")
    assert not discovery._duplicates(red, blue, 0, 0)
    assert discovery._duplicates(red, red.copy(), 0, 0)


def test_discovery_covers_whole_list_and_labels_sparse_publicity_fallback(monkeypatch, tmp_path):
    films = [{"title": f"Production {i}", "backdrops": [f"/{i}.jpg"]} for i in range(8)]
    visited = []
    mutex = threading.Lock()

    def search(film):
        with mutex:
            visited.append(film["title"])
        if film["title"] == "Production 0":
            return [{"url": "https://example.com/web.jpg", "page": "https://example.com/article",
                     "query": "Production 0 best image", "context": "verified production context"}]
        return []

    monkeypatch.setattr(discovery, "search", search)
    monkeypatch.setattr(discovery, "read_image", lambda *a: patterned((240, 40, 40), 0))
    monkeypatch.setattr("movie_pipeline.stills_trailer.fetch_original",
                        lambda path, cache: patterned((20, 80 + int(path[1:-4]) * 15, 180), int(path[1:-4])))
    monkeypatch.setattr(studio, "frame_sharpness", lambda image: 100)
    monkeypatch.setattr(studio, "has_burned_text", lambda image: False)
    monkeypatch.setattr(studio, "face_quality", lambda image: (None, 1, False))
    photos, provenance = discovery.gather_photos(films, tmp_path)
    assert set(visited) == {f["title"] for f in films}
    assert {c.film for c in photos} == set(range(8))
    assert any(c.film == 0 and c.source == "web-image" for c in photos)
    assert any(c.film == 0 and c.source == "publicity-fallback" for c in photos)
    assert all(c.source in {"web-image", "publicity-fallback"} for c in photos)
    assert all("fallback_reason" in c.notes for c in photos if c.source == "publicity-fallback")
    assert json.loads((tmp_path / "image-discovery.json").read_text()) == provenance
    report = json.loads((tmp_path / "image-discovery-report.json").read_text())
    assert len(report["titles"]) == 8 and report["titles"][0]["status"] == "web_and_publicity"
    assert report["accepted"] == len(photos)
    assert report["independently_searched"] == 1


def test_local_quality_failure_does_not_cancel_other_titles(monkeypatch, tmp_path):
    films = [{"title": "Bad", "backdrops": ["/bad"]}, {"title": "Good", "backdrops": ["/good"]}]
    monkeypatch.setattr(discovery, "search", lambda film: [])
    monkeypatch.setattr("movie_pipeline.stills_trailer.fetch_original",
                        lambda path, cache: patterned((200, 80, 80)) if path == "/bad" else patterned((40, 180, 60)))
    monkeypatch.setattr(studio, "frame_sharpness", lambda image: 100)
    monkeypatch.setattr(studio, "has_burned_text", lambda image: False)

    def faces(image):
        if image.getpixel((0, 0))[0] == 200:
            raise RuntimeError("optional detector unavailable")
        return None, 1, False

    monkeypatch.setattr(studio, "face_quality", faces)
    photos, _ = discovery.gather_photos(films, tmp_path)
    assert [c.film for c in photos] == [1]
    reports = json.loads((tmp_path / "image-discovery-report.json").read_text())["titles"]
    assert reports[0]["counts"]["publicity_RuntimeError"] == 1
    assert reports[1]["status"] == "publicity_fallback"


def candidate(color, film, score=3, marker=0, face=True):
    return studio.Candidate(patterned(color, marker), film, "verified-test-still",
                            face=(100, 140, 200, 300) if face else None, score=score,
                            notes={"good_face": face, "face_quality": 1, "source_kind": "publicity-fallback"})


def test_alternative_selection_avoids_reused_picture_when_fresh_is_comparable(monkeypatch):
    old = candidate((180, 80, 50), 0, marker=0)
    fresh = candidate((30, 150, 190), 1, marker=3)
    monkeypatch.setattr(studio, "finish", lambda image, grade: image)
    image, chosen, assessment = studio._alternative_design(
        [old, fresh], [old], [], 3, {"divider": "white"}, "neutral")
    assert chosen == [fresh]
    assert image.size == (1280, 720)
    assert assessment["attempts"] <= 8
    assert assessment["metrics"]["reused_source_images"] == 0
    assert assessment["metrics"]["face_retention"] > .95


def test_alternative_selection_prefers_readable_face_over_flat_empty_source(monkeypatch):
    empty = studio.Candidate(Image.new("RGB", (1200, 700), (80, 80, 80)), 0, "verified-still", score=3)
    expressive = candidate((180, 100, 60), 1)
    monkeypatch.setattr(studio, "finish", lambda image, grade: image)
    _, chosen, assessment = studio._alternative_design(
        [empty, expressive], [], [], 3, {"divider": "white"}, "neutral")
    assert chosen == [expressive]
    assert assessment["metrics"]["phone_contrast"] > 20


def install_build_pool(monkeypatch, photo=True):
    pool = [candidate((160, 50, 40), 0, 9, 0), candidate((40, 80, 170), 0, 8, 1),
            candidate((60, 140, 60), 1, 7, 2), candidate((160, 130, 50), 2, 6, 3)]
    monkeypatch.setattr(studio, "gather", lambda *a, **k: pool)
    monkeypatch.setattr(studio, "rate_with_vision", lambda *a, **k: 0)
    photo_pool = [candidate((120, 50, 170), 1, 4, 4)] if photo else []
    for c in photo_pool:
        c.source = "publicity-fallback"
    monkeypatch.setattr(discovery, "gather_photos", lambda *a, **k: (photo_pool, []))
    return pool


def test_ten_outputs_keep_original_first_two_pixels_and_explain_repairs(monkeypatch, tmp_path):
    pool = install_build_pool(monkeypatch)
    films = [{"title": "One"}, {"title": "Two"}, {"title": "Three"}]
    output, work = tmp_path / "output", tmp_path / "work"
    report = studio.build(films, [], [], [], output, work, title="Three movies", brief="")
    profile = studio.GENRE_PROFILES[studio.detect_genre(films, "Three movies", "")]
    expected = [
        studio.finish(studio.face_crop(pool[0], face_height=.52, face_x=.6, face_y=.42), profile["grade"]),
        studio.finish(studio.panels([pool[1], pool[2], pool[3]], profile["divider"]), profile["grade"]),
    ]
    for number, image in enumerate(expected, 1):
        ImageDraw.Draw(image).rectangle((2, 2, 1277, 717), outline="#d92935", width=5)
        reference = tmp_path / f"expected-{number}.jpg"
        studio.save(image, reference)
        assert (output / f"thumbnail-{number:02}.jpg").read_bytes() == reference.read_bytes()
    assert len(report) == 10
    for number in range(1, 11):
        with Image.open(output / f"thumbnail-{number:02}.jpg") as image:
            assert image.size == (1280, 720)
            red, green, _ = image.getpixel((3, 3))
            assert red > green * 1.5
    sources = json.loads((output / "thumbnail-sources.json").read_text())
    assert len(sources["quality_repairs"]) == 8
    assert sources["discovery"]["status"] == "report_unavailable"
    assert all(row["photo_route"] == ["publicity-fallback"] for row in report[5:])
    assert all(row["quality_review"]["attempts"] <= 8 for row in report[2:])


def test_no_verified_photo_is_nonfatal_and_never_substitutes_video_frame(monkeypatch, tmp_path):
    install_build_pool(monkeypatch, photo=False)
    report = studio.build([{"title": "One"}, {"title": "Two"}, {"title": "Three"}],
                          [], [], [], tmp_path, tmp_path / "work", title="Three movies", brief="")
    assert len(report) == 10
    assert all(row["style"] == "typographic-fallback" and row["sources"] == [] for row in report[5:])
    assert all("no video frames used" in row["fallback_reason"].lower() for row in report[5:])
