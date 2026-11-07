import io
import json
from unittest.mock import MagicMock, Mock

import pytest
import requests
from PIL import Image

from movie_pipeline.memes import CATALOG_URL, _rare, prepare_intro_meme


def seed():
    return next(str(i) for i in range(100) if _rare(str(i)))


def response(data, content_type):
    result = MagicMock(status_code=200, headers={"Content-Type": content_type})
    result.__enter__.return_value = result
    result.iter_content.return_value = [data]
    return result


def image_bytes(size=(640, 480)):
    stream = io.BytesIO()
    Image.new("RGB", size, "gray").save(stream, format="PNG")
    return stream.getvalue()


def catalog(url="https://i.imgflip.com/test.jpg", name="Absolute Cinema"):
    return json.dumps({"success": True, "data": {"memes": [{"name": name, "url": url}]}}).encode()


def test_exact_context_and_rare_seed_gate_network(monkeypatch, tmp_path):
    get = Mock(side_effect=AssertionError("Not relevant or rare"))
    monkeypatch.setattr("movie_pipeline.memes.requests.get", get)
    assert prepare_intro_meme("Cinema is great", seed(), tmp_path) is None
    ineligible = next(str(i) for i in range(100) if not _rare(str(i)))
    assert prepare_intro_meme("Absolute masterpiece", ineligible, tmp_path) is None
    get.assert_not_called()


def test_relevant_template_is_downloaded_once_and_cached(monkeypatch, tmp_path):
    get = Mock(side_effect=[response(catalog(), "application/json"), response(image_bytes(), "image/png")])
    monkeypatch.setattr("movie_pipeline.memes.requests.get", get)
    first = prepare_intro_meme("This is absolute cinema.", seed(), tmp_path)
    assert first and first["template"] == "Absolute Cinema"
    assert "Imgflip" in first["attribution"]
    assert prepare_intro_meme("Absolute masterpieces", seed(), tmp_path) == first
    assert get.call_count == 2
    assert get.call_args_list[0].args[0] == CATALOG_URL
    assert all(call.kwargs["allow_redirects"] is False for call in get.call_args_list)


@pytest.mark.parametrize("url", ["http://i.imgflip.com/test.jpg", "https://evil.example/test.jpg", "https://i.imgflip.com.evil.example/test.jpg", "https://user@i.imgflip.com/test.jpg"])
def test_untrusted_image_host_is_not_requested(monkeypatch, tmp_path, url):
    get = Mock(return_value=response(catalog(url), "application/json"))
    monkeypatch.setattr("movie_pipeline.memes.requests.get", get)
    assert prepare_intro_meme("Absolute cinema", seed(), tmp_path) is None
    assert get.call_count == 1


@pytest.mark.parametrize("failure", [requests.Timeout(), requests.ConnectionError()])
def test_network_error_skips_optional_meme(monkeypatch, tmp_path, failure):
    monkeypatch.setattr("movie_pipeline.memes.requests.get", Mock(side_effect=failure))
    assert prepare_intro_meme("Absolute cinema", seed(), tmp_path) is None


def test_small_image_and_corrupt_cache_are_skipped(monkeypatch, tmp_path):
    get = Mock(side_effect=[response(catalog(), "application/json"), response(image_bytes((64,64)), "image/png")])
    monkeypatch.setattr("movie_pipeline.memes.requests.get", get)
    assert prepare_intro_meme("Absolute cinema", seed(), tmp_path) is None
    assert not list(tmp_path.iterdir())
    get.side_effect = [response(catalog(), "application/json"), response(image_bytes(), "image/png")]
    result = prepare_intro_meme("Absolute cinema", seed(), tmp_path)
    from pathlib import Path
    Path(result["path"]).write_bytes(b"corrupted image")
    get.side_effect = AssertionError("Corrupt cached image must skip")
    assert prepare_intro_meme("Absolute cinema", seed(), tmp_path) is None
    assert not list(tmp_path.iterdir())
