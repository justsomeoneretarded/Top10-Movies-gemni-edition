import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from movie_pipeline.core import PipelineError
from movie_pipeline.footage import download
from movie_pipeline.video_settings import (
    export_size,
    require_disk,
    resolution_from_brief,
)
from scripts.cleanup_work import clean_work


def test_export_defaults_and_brief_override(monkeypatch):
    monkeypatch.delenv("VIDEO_RESOLUTION", raising=False)
    assert export_size() == (1920, 1080)
    assert resolution_from_brief("Cinematic, 4K please") == 2160
    assert resolution_from_brief("Use 1440p") == 1440
    assert resolution_from_brief("Use 720p") == 720
    monkeypatch.setenv("VIDEO_RESOLUTION", "4k")
    assert export_size() == (3840, 2160)
    monkeypatch.setenv("VIDEO_RESOLUTION", "bad")
    with pytest.raises(PipelineError, match="VIDEO_RESOLUTION"):
        export_size()


def test_working_disk_reserve_independent_of_artifact_expiry(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_RESOLUTION", "2160")
    monkeypatch.setattr("movie_pipeline.video_settings.shutil.disk_usage", lambda path: SimpleNamespace(free=2_000_000_000))
    with pytest.raises(PipelineError, match="artifact expiry"):
        require_disk(tmp_path, 540)
    monkeypatch.setattr("movie_pipeline.video_settings.shutil.disk_usage", lambda path: SimpleNamespace(free=10_000_000_000))
    require_disk(tmp_path, 540)


def test_cleanup_keeps_models_output_and_speech_but_not_footage_or_parts(tmp_path):
    root = tmp_path / "work"
    for name in ("footage/a.mp4", "models/weights.bin", "jobs/a/part-001.mp4", "jobs/a/render-001/shot-001.mp4",
                 "jobs/a/speech/voice.wav", "jobs/a/script.json", "jobs/a/music.wav"):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"data")
    output = tmp_path / "output/final-video.mp4"
    output.parent.mkdir()
    output.write_bytes(b"finished")
    report = clean_work(root)
    assert report["jobs"] == 1
    assert (root / "jobs/a/speech/voice.wav").exists()
    assert (root / "jobs/a/script.json").exists()
    assert (root / "models/weights.bin").exists()
    assert output.exists()
    assert not (root / "footage").exists()
    assert not list((root / "jobs").rglob("*.mp4"))
    assert not (root / "jobs/a/music.wav").exists()


def test_cleanup_limits_checkpoint_count_and_size(monkeypatch, tmp_path):
    import os
    for i in range(4):
        path = tmp_path / "jobs" / str(i) / "script.json"
        path.parent.mkdir(parents=True)
        path.write_text("test")
        os.utime(path, (i+100, i+100))
    assert clean_work(tmp_path)["jobs"] == 2
    assert {p.name for p in (tmp_path / "jobs").iterdir()} == {"2", "3"}
    monkeypatch.setattr("scripts.cleanup_work.MAX_CHECKPOINT_BYTES", 1)
    assert clean_work(tmp_path)["jobs"] == 0


def test_workflow_never_caches_footage_and_expires_artifact_after_two_days():
    data = yaml.load(Path(".github/workflows/generate-video.yml").read_text(), Loader=yaml.BaseLoader)
    steps = data["jobs"]["produce"]["steps"]
    caches = [s for s in steps if s.get("uses", "").startswith("actions/cache")]
    assert all("footage" not in s["with"]["path"] for s in caches)
    cleanup = next(i for i,s in enumerate(steps) if s.get("run") == "python -m scripts.cleanup_work")
    save = next(i for i,s in enumerate(steps) if s.get("uses") == "actions/cache/save@v4")
    assert cleanup < save
    assert steps[cleanup]["if"] == "always()"
    upload = next(s for s in steps if s.get("uses", "").startswith("actions/upload-artifact"))
    assert upload["with"]["retention-days"] == "2"


def test_download_requests_target_height_and_records_actual_resolution(monkeypatch, tmp_path):
    import yt_dlp
    options = []
    class FakeYDL:
        def __init__(self, opts):
            self.options = opts
            options.append(opts)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def extract_info(self, *args, **kwargs):
            return {"title": "Example Official Trailer", "duration": 120, "channel": "Studio"}
        def download(self, urls):
            Path(self.options["outtmpl"].replace("%(ext)s", "mp4")).write_bytes(b"media")
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr("movie_pipeline.footage.duration", lambda path: 120)
    monkeypatch.setattr("movie_pipeline.footage.require_disk", lambda path: None)
    monkeypatch.setattr("movie_pipeline.footage.probe", lambda path: {"streams": [{"codec_type": "video", "width": 1920, "height": 1080}]})
    monkeypatch.setenv("VIDEO_RESOLUTION", "2160")
    path = download({"title": "Example", "year": 2025, "kind": "movie", "id": 1, "trailers": [{"key": "abcdefghijk"}]}, tmp_path)
    assert path.name == "movie-1-2160p.mp4"
    assert "height<=2160" in options[-1]["format"]
    assert "vcodec^=avc1" not in options[-1]["format"]
    assert options[-1]["max_filesize"] == 800_000_000
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["source_height"] == 1080 and metadata["requested_height"] == 2160


def test_failed_download_cleans_partial_streams(monkeypatch, tmp_path):
    import yt_dlp
    class FakeYDL:
        def __init__(self, opts):
            self.options = opts
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def extract_info(self, *args, **kwargs):
            return {"title": "Example Official Trailer", "duration": 120}
        def download(self, urls):
            Path(self.options["outtmpl"].replace("%(ext)s", "f313.webm.part")).write_bytes(b"partial")
            raise yt_dlp.utils.DownloadError("Temporary failure")
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr("movie_pipeline.footage.require_disk", lambda path: None)
    monkeypatch.setenv("VIDEO_RESOLUTION", "2160")
    with pytest.raises(PipelineError, match="No usable trailer"):
        download({"title": "Example", "year": 2025, "kind": "movie", "id": 1, "trailers": [{"key": "abcdefghijk"}]}, tmp_path)
    assert not list(tmp_path.iterdir())
