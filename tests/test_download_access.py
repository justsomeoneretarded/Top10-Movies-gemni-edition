from pathlib import Path

import pytest

from movie_pipeline import download_access
from movie_pipeline.core import PipelineError
from movie_pipeline.footage import download, trailer_url

BOT = "Sign in to confirm you're not a bot"


@pytest.fixture(autouse=True)
def fresh_routes():
    download_access.reset()
    yield
    download_access.reset()


def info(height=1080):
    return {"title": "Example Official Trailer", "duration": 120, "channel": "Studio", "channel_is_verified": True,
            "formats": [{"width": round(height*16/9), "height": height, "vcodec": "vp9", "filesize": 50_000_000}]}


def fake_downloader(monkeypatch, *, blocked_without_proxy=(), always_blocked=()):
    import yt_dlp
    calls = []

    class FakeYDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def _check(self, url):
            calls.append((url, self.options.get("proxy")))
            if any(part in url for part in always_blocked):
                raise yt_dlp.utils.DownloadError(BOT)
            if not self.options.get("proxy") and any(part in url for part in blocked_without_proxy):
                raise yt_dlp.utils.DownloadError(BOT)

        def extract_info(self, url, **kwargs):
            if url.startswith("ytsearch"):
                return {"entries": []}
            self._check(url)
            return info()

        def download(self, urls):
            self._check(urls[0])
            Path(self.options["outtmpl"].replace("%(ext)s", "mp4")).write_text(urls[0])

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr("movie_pipeline.footage.duration", lambda path: 120)
    monkeypatch.setattr("movie_pipeline.footage.require_disk", lambda path: None)
    monkeypatch.setattr("movie_pipeline.footage.probe",
                        lambda path: {"streams": [{"codec_type": "video", "width": 1920, "height": 1080}]})
    monkeypatch.setenv("VIDEO_RESOLUTION", "1080")
    return calls


def film(*trailers, identifier=1):
    return {"title": "Example", "year": 2025, "kind": "movie", "id": identifier, "trailers": list(trailers)}


def test_trailer_urls_cover_youtube_and_vimeo_only():
    assert trailer_url({"key": "aaaaaaaaaaa"}) == "https://www.youtube.com/watch?v=aaaaaaaaaaa"
    assert trailer_url({"key": "123456789", "site": "Vimeo"}) == "https://vimeo.com/123456789"
    assert trailer_url({"key": "not-an-id", "site": "Vimeo"}) is None
    assert trailer_url({"key": "aaaaaaaaaaa", "site": "Dailymotion"}) is None


def test_bot_block_retries_through_fallback_route_and_remembers_it(monkeypatch, tmp_path):
    monkeypatch.setenv("YTDLP_FALLBACK_PROXY", "socks5h://127.0.0.1:40000")
    calls = fake_downloader(monkeypatch, blocked_without_proxy=("youtube.com",))
    result = download(film({"key": "aaaaaaaaaaa"}), tmp_path)
    assert "aaaaaaaaaaa" in result.read_text()
    assert calls[0][1] is None and calls[-1][1] == "socks5h://127.0.0.1:40000"
    # Later titles go straight to the route that worked.
    calls.clear()
    download(film({"key": "bbbbbbbbbbb"}, identifier=2), tmp_path)
    assert all(proxy == "socks5h://127.0.0.1:40000" for _, proxy in calls)


def test_vimeo_copy_is_used_when_youtube_refuses_every_route(monkeypatch, tmp_path):
    fake_downloader(monkeypatch, always_blocked=("youtube.com",))
    result = download(film({"key": "aaaaaaaaaaa"}, {"key": "123456789", "site": "Vimeo"}), tmp_path)
    assert result.read_text() == "https://vimeo.com/123456789"


def test_blocked_everywhere_still_explains_the_runner_block(monkeypatch, tmp_path):
    monkeypatch.setenv("YTDLP_FALLBACK_PROXY", "socks5h://127.0.0.1:40000")
    fake_downloader(monkeypatch, always_blocked=("youtube.com",))
    with pytest.raises(PipelineError, match="YouTube blocked this runner"):
        download(film({"key": "aaaaaaaaaaa"}), tmp_path)


def test_cookies_are_applied_on_every_route(monkeypatch):
    monkeypatch.setenv("YOUTUBE_COOKIE_FILE", "/tmp/cookies.txt")
    monkeypatch.setenv("YTDLP_FALLBACK_PROXY", "socks5h://127.0.0.1:40000")
    direct = download_access.options({"quiet": True}, "direct")
    fallback = download_access.options({"quiet": True}, "fallback")
    assert direct["cookiefile"] == fallback["cookiefile"] == "/tmp/cookies.txt"
    assert "proxy" not in direct and fallback["proxy"].startswith("socks5h://")
