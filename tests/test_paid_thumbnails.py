import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from movie_pipeline import gemini, paid_thumbnails as paid


def picture(size=(1280, 720), color="#284364"):
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    for x in range(0, size[0], 45):
        draw.line((x, 0, x + 180, size[1]), fill="#b4a07b", width=9)
    return image


def response(image=None, *, fmt="PNG", mime=None):
    buffer = io.BytesIO()
    (image or picture((2048, 1152), "#81432c")).save(buffer, fmt)
    return {"modelVersion": paid.IMAGE_MODEL, "candidates": [{
        "finishReason": "STOP", "content": {"parts": [{"inlineData": {
            "mimeType": mime or {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[fmt],
            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }}]},
    }]}


def accepted():
    return {key: "high" if key == "confidence" else True for key in paid.REVIEW_SCHEMA["required"]}


class Client:
    vision_model = "gemini-3.8-flash"

    def __init__(self, data=None, verdict=None, error=None):
        self.data = response() if data is None else data
        self.verdict = accepted() if verdict is None else verdict
        self.error = error
        self.requests = []
        self.reviews = []

    def _request(self, model, payload, stage):
        self.requests.append((model, payload, stage))
        if self.error:
            raise self.error
        return self.data

    def generate_json(self, prompt, schema, **kwargs):
        self.reviews.append(kwargs)
        return self.verdict


@pytest.fixture
def trial(monkeypatch):
    monkeypatch.setenv("GEMINI_PAID_TRIAL", "true")
    monkeypatch.setenv("GEMINI_FREE_ONLY", "false")
    monkeypatch.setattr(gemini, "record_event", lambda *args, **kwargs: None)
    image = picture()
    film = {"title": "Verified Drama", "kind": "tv", "id": 123, "year": 2026,
            "genres": ["Drama"], "overview": "Two sisters face a difficult family decision.",
            "backdrops": ["/verified.jpg"]}
    candidate = SimpleNamespace(image=image, film=0, source="tmdb-still",
                                notes={"tmdb_path": "/verified.jpg"})
    return image, candidate, film


@pytest.mark.parametrize("trial_switch,free_switch,expected", [
    ("true", "false", True), ("true", "true", False), ("false", "false", False),
    ("", "false", False), ("true", "", False), ("1", "false", False),
])
def test_both_explicit_switches_required(monkeypatch, trial_switch, free_switch, expected):
    monkeypatch.setenv("GEMINI_PAID_TRIAL", trial_switch)
    monkeypatch.setenv("GEMINI_FREE_ONLY", free_switch)
    assert paid.enabled() is expected


def test_disabled_and_other_options_never_acquire_client(trial, tmp_path, monkeypatch):
    original, candidate, film = trial
    monkeypatch.setattr(gemini, "get_client", lambda: pytest.fail("unexpected paid client"))
    for number in (1, 2, 6, 7, 8, 9, 10):
        image, report = paid.edit(number, original, [candidate], [film], "List", tmp_path)
        assert image is original and report == {}
    monkeypatch.setenv("GEMINI_FREE_ONLY", "true")
    image, report = paid.edit(3, original, [candidate], [film], "List", tmp_path)
    assert image is original and report == {}


@pytest.mark.parametrize("fmt,mime", [
    ("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp"),
])
def test_real_image_decoding_normalizes_to_rgb_16_by_9(fmt, mime):
    decoded = paid.decode_image(response(fmt=fmt, mime=mime), (1280, 720))
    assert decoded.mode == "RGB" and decoded.size == (1280, 720)


@pytest.mark.parametrize("size", [(100, 56), (1024, 1024), (640, 1280), (1280, 300)])
def test_small_or_wrong_shape_images_are_rejected(size):
    with pytest.raises(ValueError):
        paid.decode_image(response(picture(size)), (1280, 720))


def test_invalid_bytes_mime_and_incomplete_output_rejected():
    data = response()
    inline = data["candidates"][0]["content"]["parts"][0]["inlineData"]
    inline["data"] = "not base64"
    with pytest.raises(ValueError, match="base64"):
        paid.decode_image(data, (1280, 720))
    with pytest.raises(ValueError, match="mime_mismatch"):
        paid.decode_image(response(mime="image/jpeg"), (1280, 720))
    data = response()
    data["candidates"][0]["finishReason"] = "MAX_TOKENS"
    with pytest.raises(gemini.GeminiUnavailable):
        paid.decode_image(data, (1280, 720))


def test_multiple_output_images_rejected():
    data = response()
    parts = data["candidates"][0]["content"]["parts"]
    parts.append(parts[0])
    with pytest.raises(ValueError, match="one_generated_image"):
        paid.decode_image(data, (1280, 720))


def test_success_uses_metered_image_transport_and_independent_review(trial, tmp_path, monkeypatch):
    original, candidate, film = trial
    client = Client()
    monkeypatch.setattr(gemini, "get_client", lambda: client)
    image, report = paid.edit(3, original, [candidate], [film], "5 New Shows Worth Watching 2025-26", tmp_path)
    assert image.size == original.size and image is not original
    assert len(client.requests) == len(client.reviews) == 1
    model, payload, stage = client.requests[0]
    assert (model, stage) == (paid.IMAGE_MODEL, "image")
    config = payload["generationConfig"]
    assert config["responseModalities"] == ["TEXT", "IMAGE"]
    assert config["imageConfig"] == {"aspectRatio": "16:9", "imageSize": "2K"}
    assert config["maxOutputTokens"] == 4096
    assert len(payload["contents"][0]["parts"]) == 3
    prompt = payload["contents"][0]["parts"][-1]["text"]
    assert "Verified Drama" in prompt and "Two sisters" in prompt
    assert "Preserve the actual actors" in prompt and "No typography" in prompt
    assert client.reviews[0]["max_output_tokens"] == 2048
    assert len(client.reviews[0]["images"]) == 2
    assert report["image_status"] == "success"
    assert report["image_backend"] == "gemini-reference-image-edit"
    assert report["actual_image_model"] == paid.IMAGE_MODEL
    assert report["reference_images"][0]["url"].endswith("/verified.jpg")
    assert len(report["reference_sha256"]) == len(report["generated_sha256"]) == 64
    artifact = tmp_path / "paid-thumbnails" / "thumbnail-03-candidate.jpg"
    with Image.open(artifact) as saved:
        assert saved.format == "JPEG" and saved.size == (1280, 720)


@pytest.mark.parametrize("change", [
    {"confidence": "uncertain"}, {"same_people": False}, {"same_series_context": False},
    {"no_invented_people_or_events": False}, {"no_unwanted_text_or_logos": False},
    {"clear_at_phone_size": False},
])
def test_any_authenticity_uncertainty_retains_original(trial, tmp_path, monkeypatch, change):
    original, candidate, film = trial
    client = Client(verdict={**accepted(), **change})
    monkeypatch.setattr(gemini, "get_client", lambda: client)
    image, report = paid.edit(4, original, [candidate], [film], "List", tmp_path)
    assert image is original
    assert len(client.requests) == len(client.reviews) == 1
    assert report["image_status"] == "fallback"
    assert report["image_backend"] == "original-source-composition"
    assert report["fallback_reason"] == "authenticity_review_unavailable_or_rejected"


def test_provider_failure_has_no_retry_and_no_secret_in_report(trial, tmp_path, monkeypatch):
    original, candidate, film = trial
    client = Client(error=RuntimeError("sensitive response and secret"))
    monkeypatch.setattr(gemini, "get_client", lambda: client)
    image, report = paid.edit(5, original, [candidate], [film], "List", tmp_path)
    assert image is original and len(client.requests) == 1 and not client.reviews
    assert report["fallback_reason"] == "paid_image_unavailable"
    assert "sensitive" not in str(report) and "secret" not in str(report)


def test_malformed_image_retains_original_without_review(trial, tmp_path, monkeypatch):
    original, candidate, film = trial
    client = Client(data={"candidates": []})
    monkeypatch.setattr(gemini, "get_client", lambda: client)
    image, report = paid.edit(3, original, [candidate], [film], "List", tmp_path)
    assert image is original and len(client.requests) == 1 and not client.reviews
    assert report["fallback_reason"] == "invalid_generated_image"


@pytest.mark.parametrize("source,notes", [
    ("web-image", {"tmdb_path": "/verified.jpg"}),
    ("tmdb-still", {"tmdb_path": "/another-show.jpg"}),
    ("trailer-frame", {}),
])
def test_unverified_references_never_spend(trial, tmp_path, monkeypatch, source, notes):
    original, candidate, film = trial
    candidate.source, candidate.notes = source, notes
    monkeypatch.setattr(gemini, "get_client", lambda: pytest.fail("unverified reference spent"))
    image, report = paid.edit(3, original, [candidate], [film], "List", tmp_path)
    assert image is original and report["fallback_reason"] == "reference_unavailable"


def test_trailer_provenance_preserves_selected_frame_time(trial):
    _, candidate, film = trial
    candidate.source = "trailer-frame"
    candidate.notes = {"trailer_path": "/work/trailers/verified.mp4", "frame_seconds": 12.35}
    provenance = paid._reference(candidate, film)
    assert provenance["trailer_file"] == "verified.mp4"
    assert provenance["frame_seconds"] == 12.35
    assert provenance["tmdb_id"] == 123


def test_all_ten_outputs_and_unaffected_designs_are_preserved(trial, tmp_path, monkeypatch):
    import movie_pipeline.thumbnail_studio as studio

    _, _, film = trial
    films = [{**film, "id": 123 + index, "title": f"Series {index}",
              "backdrops": [f"/series-{index}.jpg"]} for index in range(3)]
    def gather(*args, **kwargs):
        return [studio.Candidate(
            picture(color=("#284364" if index % 2 else "#81432c")),
            index % 3, "tmdb-still", face=(300, 120, 230, 280), score=20 - index,
            notes={"tmdb_path": f"/series-{index % 3}.jpg", "good_face": True},
        ) for index in range(9)]
    monkeypatch.setattr(studio, "gather", gather)
    monkeypatch.setattr(studio, "rate_with_vision", lambda *args, **kwargs: 0)
    monkeypatch.setattr("movie_pipeline.image_discovery.gather_photos", lambda *args: ([], []))
    client = Client()
    monkeypatch.setattr(gemini, "get_client", lambda: client)
    monkeypatch.setenv("GEMINI_PAID_TRIAL", "false")
    baseline = tmp_path / "baseline"
    studio.build(films, [], [], [], baseline, tmp_path / "baseline-work", title="5 New Shows")
    monkeypatch.setenv("GEMINI_PAID_TRIAL", "true")
    output = tmp_path / "paid"
    report = studio.build(films, [], [], [], output, tmp_path / "paid-work", title="5 New Shows")
    assert len(report) == 10 and len(client.requests) == len(client.reviews) == 3
    assert all(row["style"] == "paid-reference-edit" for row in report[2:5])
    for number in (1, 2, 6, 7, 8, 9, 10):
        name = f"thumbnail-{number:02}.jpg"
        assert (baseline / name).read_bytes() == (output / name).read_bytes()
    for number in range(1, 11):
        with Image.open(output / f"thumbnail-{number:02}.jpg") as image:
            assert image.format == "JPEG" and image.size == (1280, 720)
            red, green, blue = image.getpixel((3, 3))
            assert red > green * 1.5
    assert (output / "thumbnail-sources.json").exists()
    assert (tmp_path / "paid-work" / "thumbnail-report.json").exists()
