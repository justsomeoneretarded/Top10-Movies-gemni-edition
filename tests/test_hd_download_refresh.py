"""Bounded HD refreshes preserve accepted media across routes and Actions jobs."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from movie_pipeline import download_access, editorial_policy, footage
from movie_pipeline.core import PipelineError

IDS = ("aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc", "ddddddddddd")
FILM = {"title": "Example", "year": 2025, "kind": "movie", "id": 1,
        "trailers": [{"key": identifier} for identifier in IDS]}


@pytest.fixture
def downloader(monkeypatch, tmp_path):
    import yt_dlp
    state = SimpleNamespace(catalog={"direct": {IDS[0]: 1080}}, actual={}, failures={},
                            downloads=[], inspected=[], successes=[], routes=["direct"],
                            dimensions={"old": 360})

    class FakeYDL:
        def __init__(self, options):
            self.options = options
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def extract_info(self, url, **kwargs):
            route = self.options["test_route"]
            identifier = url[-11:]
            state.inspected.append((route, identifier))
            height = state.catalog.get(route, {}).get(identifier)
            if height is None:
                return None
            if isinstance(height, Exception):
                raise height
            return {"title": "Example Trailer", "duration": 120,
                    "formats": [{"width": round(height*16/9), "height": height, "vcodec": "h264"}]}
        def download(self, urls):
            route = self.options["test_route"]
            identifier = urls[0][-11:]
            state.downloads.append((route, identifier))
            target = Path(self.options["outtmpl"].replace("%(ext)s", "mp4"))
            error = state.failures.get((route, identifier))
            if error:
                target.with_suffix(".part").write_bytes(b"incomplete")
                raise yt_dlp.utils.DownloadError(error)
            token = f"{route}:{identifier}"
            state.dimensions[token] = state.actual.get((route, identifier), state.catalog[route][identifier])
            target.write_text(token)

    def probe(path):
        height = state.dimensions[path.read_text()]
        return {"streams": [{"codec_type": "video", "width": round(height*16/9), "height": height}]}

    def seed(height=360, metadata_height=None, stills=False):
        path = tmp_path/"movie-1-2160p.mp4"
        path.write_text("old")
        state.dimensions["old"] = height
        record_height = metadata_height or height
        path.with_suffix(".json").write_text(json.dumps({
            "source": "https://www.youtube.com/watch?v=zzzzzzzzzzz",
            "quality_selection_version": 2, "source_width": round(record_height*16/9),
            "source_height": record_height, "stills_fallback": stills,
        }))
        return path

    state.seed = seed
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(footage, "duration", lambda _: 120)
    monkeypatch.setattr(footage, "probe", probe)
    monkeypatch.setattr(footage, "require_disk", lambda _: None)
    monkeypatch.setattr(footage, "_LOW_QUALITY_REFRESHES", set())
    monkeypatch.setattr(editorial_policy, "search_queries", lambda *args, **kwargs: [])
    monkeypatch.setattr(download_access, "routes", lambda: state.routes)
    monkeypatch.setattr(download_access, "options", lambda base, route: {**base, "test_route": route})
    monkeypatch.setattr(download_access, "record_route_success", state.successes.append)
    monkeypatch.setattr(download_access, "fallback_proxy", lambda: None)
    monkeypatch.setenv("VIDEO_RESOLUTION", "2160")
    monkeypatch.setenv("GITHUB_RUN_ID", "test-hd")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("STILLS_FALLBACK", "false")
    return state


def test_cached_sd_upgrades_only_after_validation_and_invalidates_own_analyses(downloader, tmp_path):
    path = downloader.seed()
    shots = path.with_suffix(".shots.calm-v6.json")
    dialogue = path.with_suffix(".dialogue.json")
    shots.write_text("stale cuts")
    dialogue.write_text("stale words")
    unrelated = tmp_path/"models-cpu-cache"
    unrelated.write_text("keep model cache")
    assert footage.download(FILM, tmp_path) == path
    assert path.read_text() == f"direct:{IDS[0]}"
    record = json.loads(path.with_suffix(".json").read_text())
    assert record["source_height"] == 1080
    assert record["film_kind"] == "movie" and record["film_id"] == 1
    assert record["quality_refresh_status"] == "upgraded"
    assert not shots.exists() and not dialogue.exists()
    assert unrelated.read_text() == "keep model cache"
    assert not list(tmp_path.glob("*quality-*"))
    assert not list(tmp_path.glob("*candidate*"))


@pytest.mark.parametrize("outcome", ["blocked", "lower", "same"])
def test_failed_or_worse_refresh_retains_original_bytes_and_source(downloader, tmp_path, outcome):
    path = downloader.seed()
    downloader.catalog["direct"] = {IDS[0]: 240 if outcome == "lower" else 360}
    if outcome == "blocked":
        downloader.failures[("direct", IDS[0])] = "Sign in to confirm you're not a bot"
    assert footage.download(FILM, tmp_path) == path
    assert path.read_text() == "old"
    record = json.loads(path.with_suffix(".json").read_text())
    assert record["source"].endswith("zzzzzzzzzzz")
    assert record["source_height"] == 360
    assert not list(tmp_path.glob("*quality-*"))


def test_refresh_marker_survives_new_process_but_new_workflow_attempt_can_try(downloader, tmp_path, monkeypatch):
    path = downloader.seed()
    downloader.catalog["direct"] = {IDS[0]: 360}
    footage.download(FILM, tmp_path)
    record = json.loads(path.with_suffix(".json").read_text())
    assert (record["quality_refresh_run"], record["quality_refresh_attempt"]) == ("test-hd", "1")
    count = len(downloader.downloads)
    monkeypatch.setattr(footage, "_LOW_QUALITY_REFRESHES", set())  # another Actions job/process
    footage.download(FILM, tmp_path)
    assert len(downloader.downloads) == count
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    footage.download(FILM, tmp_path)
    assert len(downloader.downloads) == count+1
    assert json.loads(path.with_suffix(".json").read_text())["quality_refresh_attempt"] == "2"


def test_first_sd_download_is_not_retried_by_the_producer_in_same_attempt(downloader, tmp_path, monkeypatch):
    downloader.catalog["direct"] = {IDS[0]: 360}
    path = footage.download(FILM, tmp_path)
    assert json.loads(path.with_suffix(".json").read_text())["quality_refresh_status"] == "limited_source_available"
    count = len(downloader.downloads)
    monkeypatch.setattr(footage, "_LOW_QUALITY_REFRESHES", set())
    assert footage.download(FILM, tmp_path) == path
    assert len(downloader.downloads) == count


def test_usable_sd_does_not_end_route_discovery_when_hd_route_is_available(downloader, tmp_path):
    downloader.routes = ["direct", "fallback"]
    downloader.catalog = {"direct": {IDS[0]: 360}, "fallback": {IDS[0]: 1080}}
    path = footage.download(FILM, tmp_path)
    assert downloader.downloads == [("direct", IDS[0]), ("fallback", IDS[0])]
    assert path.read_text() == f"fallback:{IDS[0]}"
    assert downloader.successes == ["fallback"]


def test_second_route_failure_keeps_first_routes_usable_sd(downloader, tmp_path):
    downloader.routes = ["direct", "fallback"]
    downloader.catalog = {"direct": {IDS[0]: 360}, "fallback": {IDS[0]: 1080}}
    downloader.failures[("fallback", IDS[0])] = "network unavailable"
    path = footage.download(FILM, tmp_path)
    assert path.read_text() == f"direct:{IDS[0]}"
    assert downloader.successes == ["direct"]


def test_actual_media_failures_share_three_attempt_limit_across_routes(downloader, tmp_path):
    downloader.routes = ["direct", "fallback"]
    downloader.catalog = {"direct": {key: 2160 for key in IDS},
                          "fallback": {IDS[0]: 2160}}
    downloader.failures = {("direct", key): "transport unavailable" for key in IDS}
    with pytest.raises(PipelineError, match="No usable trailer"):
        footage.download(FILM, tmp_path)
    assert len(downloader.downloads) == footage.MAX_QUALITY_DOWNLOADS == 3
    assert all(route == "direct" for route, _ in downloader.downloads)
    assert not list(tmp_path.iterdir())


def test_failed_attempts_in_first_route_leave_only_remaining_budget_for_fallback(downloader, tmp_path):
    downloader.routes = ["direct", "fallback"]
    downloader.catalog = {"direct": {IDS[0]: 2160, IDS[1]: 2160},
                          "fallback": {IDS[0]: 1080, IDS[1]: 2160}}
    downloader.failures = {("direct", IDS[0]): "temporary", ("direct", IDS[1]): "temporary"}
    path = footage.download(FILM, tmp_path)
    assert len(downloader.downloads) == 3
    assert path.read_text().startswith("fallback:")
    assert not list(tmp_path.glob("*candidate*"))


def test_cached_sd_survives_disk_reserve_without_network_attempt(downloader, tmp_path, monkeypatch):
    path = downloader.seed()
    def reserve(_):
        raise PipelineError("disk reserve")
    monkeypatch.setattr(footage, "require_disk", reserve)
    assert footage.download(FILM, tmp_path) == path
    assert path.read_text() == "old"
    assert not downloader.downloads
    assert json.loads(path.with_suffix(".json").read_text())["quality_refresh_status"] == "retained_disk_reserve"


@pytest.mark.parametrize("failure", ["incumbent-media", "incumbent-record", "candidate-record"])
def test_atomic_upgrade_rollback_preserves_untouched_or_backed_up_incumbent(tmp_path, monkeypatch, failure):
    target, attempt = tmp_path/"movie.mp4", tmp_path/"candidate.mp4"
    target.write_bytes(b"original media")
    target.with_suffix(".json").write_text('{"source":"old"}')
    attempt.write_bytes(b"validated HD media")
    blocked = {"incumbent-media": target, "incumbent-record": target.with_suffix(".json"),
               "candidate-record": attempt.with_suffix(".json")}[failure]
    original_replace = Path.replace
    def replace(self, destination):
        if self == blocked:
            raise OSError("injected rename failure")
        return original_replace(self, destination)
    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(OSError, match="injected rename failure"):
        footage._install_candidate(attempt, target, {"source": "new"})
    assert target.read_bytes() == b"original media"
    assert json.loads(target.with_suffix(".json").read_text()) == {"source": "old"}


def test_actual_cached_dimensions_override_an_incorrect_hd_sidecar(downloader, tmp_path):
    path = downloader.seed(height=360, metadata_height=1080)
    footage.download(FILM, tmp_path)
    assert downloader.downloads == [("direct", IDS[0])]
    assert path.read_text() == f"direct:{IDS[0]}"


def test_valid_cached_hd_is_reused_without_optional_network_work(downloader, tmp_path):
    path = downloader.seed(height=1080)
    assert footage.download(FILM, tmp_path) == path
    assert not downloader.downloads
