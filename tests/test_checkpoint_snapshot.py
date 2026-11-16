import hashlib
import io
import json
import os
import tarfile
import wave
from pathlib import Path, PurePosixPath

import pytest

from scripts import checkpoint_snapshot as checkpoint

JOB = "a" * 20
BASE = Path("work/jobs") / JOB


def write(root, relative, data):
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return destination


def json_file(root, relative, value):
    return write(root, relative, json.dumps(value).encode())


def speech(root):
    recording = io.BytesIO()
    with wave.open(recording, "wb") as stream:
        stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x01\x00" * 800)
    raw = recording.getvalue()
    relative = BASE / "speech" / ("b" * 20 + ".wav")
    write(root, relative, raw)
    json_file(root, BASE / "speech-checks" / ("c" * 20 + ".verified.json"),
              {"audio_sha256": hashlib.sha256(raw).hexdigest(),
               "cues": [{"start": 0, "end": .1, "text": "Verified words."}], "stage": "base-vad"})
    return relative, raw


def members(archive):
    with tarfile.open(archive, "r:gz") as stream:
        return {item.name: stream.extractfile(item).read() for item in stream if item.isfile()}


def test_checkpoint_keeps_reusable_inputs_and_only_hash_verified_speech(tmp_path):
    relative, raw = speech(tmp_path)
    json_file(tmp_path, BASE / "script.json", {"sections": [{"paragraphs": ["Original words."]}]})
    json_file(tmp_path, BASE / "input.json", {"title": "Episode", "voice_model": "same voice"})
    json_file(tmp_path, BASE / "script-sections" / ("d" * 20 + ".json"), {"paragraphs": ["Warm section."]})
    write(tmp_path, BASE / "speech" / ("e" * 20 + ".wav"), raw + b"unverified")
    forbidden = [
        BASE / "speech" / ("b" * 20 + ".raw.wav"),
        BASE / "speech" / ("b" * 20 + ".dry.wav"),
        BASE / "final-video.mp4", BASE / "soundtrack.wav", BASE / "cookies.txt",
        Path("work/footage/trailer.mp4"), Path("work/models/model.bin"),
        Path(".cache/pip/wheel.whl"), Path("work/voices/warguy.wav"),
    ]
    for item in forbidden:
        write(tmp_path, item, b"must stay private and untouched")
    archive = tmp_path / "snapshot.tar.gz"

    manifest = checkpoint.create_snapshot(tmp_path, archive)
    saved = members(archive)

    assert str(relative) in saved
    assert saved[str(relative)] == raw
    assert str(BASE / "script.json") in saved
    assert str(BASE / "script-sections" / ("d" * 20 + ".json")) in saved
    assert not any(str(item) in saved for item in forbidden)
    assert str(BASE / "speech" / ("e" * 20 + ".wav")) not in saved
    assert manifest["skipped"]["unverified_audio"] == 1
    assert archive.stat().st_size <= checkpoint.MAX_BYTES
    for item in forbidden:
        assert (tmp_path / item).read_bytes() == b"must stay private and untouched"


def test_known_secrets_signed_urls_and_credentials_are_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-secret-that-must-not-survive")
    json_file(tmp_path, BASE / "input.json", {
        "title": "test-secret-that-must-not-survive",
        "api_key": "remove entire field",
        "nested": {"cookie": "session value", "safe": "https://host/file?sig=private-signature&x=1"},
        "source": "https://name:password@host/path",
    })
    write(tmp_path, Path("work/runner-diagnostics/production.log"),
          b"Authorization: Bearer xyz-private-value\nhttps://host/file?token=private-url\n"
          b"test-secret-that-must-not-survive\nhf_abcdefghijklmno12345\n")
    archive = tmp_path / "snapshot.tar.gz"
    checkpoint.create_snapshot(tmp_path, archive)
    saved = members(archive)
    all_bytes = b"\n".join(saved.values())
    for secret in (b"test-secret-that-must-not-survive", b"remove entire field",
                   b"session value", b"private-signature", b"name:password",
                   b"xyz-private-value", b"private-url", b"hf_abcdefghijklmno12345"):
        assert secret not in all_bytes
    parsed = json.loads(saved[str(BASE / "input.json")])
    assert "api_key" not in parsed and "cookie" not in parsed["nested"]


def test_symlinks_and_unstable_files_are_skipped(tmp_path, monkeypatch):
    outside = tmp_path / "outside.json"
    outside.write_text('{"private":"outside"}')
    destination = tmp_path / BASE / "script.json"
    destination.parent.mkdir(parents=True)
    destination.symlink_to(outside)
    archive = tmp_path / "snapshot.tar.gz"
    checkpoint.create_snapshot(tmp_path, archive)
    assert str(BASE / "script.json") not in members(archive)

    destination.unlink()
    destination.write_text('{"version":1}')
    original_read = checkpoint.os.read
    changed = False

    def torn_read(descriptor, count):
        nonlocal changed
        value = original_read(descriptor, count)
        if not changed:
            changed = True
            destination.write_text('{"version":200}')
        return value

    monkeypatch.setattr(checkpoint.os, "read", torn_read)
    with pytest.raises(ValueError, match="unstable"):
        checkpoint.regular_read(tmp_path, PurePosixPath(str(BASE / "script.json")), 1024)


def test_atomic_snapshot_failure_leaves_previous_archive(tmp_path, monkeypatch):
    archive = tmp_path / "snapshot.tar.gz"
    archive.write_bytes(b"previous usable snapshot")

    def fail(*args, **kwargs):
        raise OSError("simulated disk pressure")

    monkeypatch.setattr(checkpoint, "add_bytes", fail)
    with pytest.raises(OSError):
        checkpoint.create_snapshot(tmp_path, archive)
    assert archive.read_bytes() == b"previous usable snapshot"
    assert not list(tmp_path.glob(".checkpoint-*"))


def test_restore_reuses_verified_speech_without_overwriting_warm_files(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    relative, raw = speech(source)
    json_file(source, BASE / "script.json", {"version": "checkpoint"})
    json_file(target, BASE / "script.json", {"version": "fresh warm data"})
    archive = tmp_path / "snapshot.tar.gz"
    checkpoint.create_snapshot(source, archive)

    count = checkpoint.restore_snapshot(target, archive)

    assert count == 2
    assert (target / relative).read_bytes() == raw
    assert json.loads((target / BASE / "script.json").read_text())["version"] == "fresh warm data"
    assert not (target / "work/runner-diagnostics").exists()


@pytest.mark.parametrize("name,kind", [
    ("../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE),
    ("./work/jobs/" + JOB + "/script.json", tarfile.REGTYPE),
    ("work/jobs/" + JOB + "/script.json", tarfile.SYMTYPE),
    ("work/jobs/" + JOB + "/script.json", tarfile.LNKTYPE),
    ("work/jobs/" + JOB + "/script.json", tarfile.FIFOTYPE),
    ("work/voices/warguy.wav", tarfile.REGTYPE),
])
def test_restore_rejects_unsafe_archive_before_writing(tmp_path, name, kind):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        item = tarfile.TarInfo(name)
        item.type = kind
        if kind == tarfile.REGTYPE:
            item.size = 1
        stream.addfile(item, io.BytesIO(b"x") if item.size else None)
    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises((ValueError, KeyError)):
        checkpoint.restore_snapshot(target, archive)
    assert not list(target.iterdir())


def test_restore_validates_all_hashes_before_publishing_any_file(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    json_file(source, BASE / "input.json", {"title": "episode"})
    json_file(source, BASE / "script.json", {"paragraphs": ["complete"]})
    archive = tmp_path / "snapshot.tar.gz"
    checkpoint.create_snapshot(source, archive)
    original = members(archive)
    original[str(BASE / "script.json")] = b'{"corrupted":true}'
    with tarfile.open(archive, "w:gz") as stream:
        for name, data in original.items():
            checkpoint.add_bytes(stream, name, data)
    with pytest.raises(ValueError, match="member_hash"):
        checkpoint.restore_snapshot(target, archive)
    assert not list(target.iterdir())


def test_restore_refuses_destination_symlink(tmp_path):
    source, target, outside = tmp_path / "source", tmp_path / "target", tmp_path / "outside"
    for folder in (source, target, outside):
        folder.mkdir()
    json_file(source, BASE / "input.json", {"title": "episode"})
    archive = tmp_path / "snapshot.tar.gz"
    checkpoint.create_snapshot(source, archive)
    (target / "work").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe_destination"):
        checkpoint.restore_snapshot(target, archive)
    assert not list(outside.iterdir())


def test_payload_budget_and_diagnostic_tail_are_bounded(tmp_path):
    for index in range(4):
        json_file(tmp_path, BASE / "script-sections" / (f"{index:020x}.json"),
                  {"paragraphs": [os.urandom(160000).hex()]})
    write(tmp_path, Path("work/runner-diagnostics/production.log"), b"long line " * 100000 + b"\nlast complete line\n")
    archive = tmp_path / "snapshot.tar.gz"

    manifest = checkpoint.create_snapshot(tmp_path, archive, max_bytes=1024 * 1024)

    assert archive.stat().st_size <= 1024 * 1024
    assert manifest["selected_bytes"] <= 3 * 1024 * 1024 // 4
    assert manifest["skipped"].get("byte_limit", 0) > 0
    data = members(archive).get("work/runner-diagnostics/production.log", b"")
    assert len(data) <= 512 * 1024
    if data:
        assert data.endswith(b"last complete line\n")


def test_cloud_signatures_and_encoded_credential_query_keys_are_scrubbed():
    result = checkpoint.redact(
        "https://host/file?X-Amz-Signature=amz-private&X-Amz-Credential=credential-private"
        "&X-Goog-Signature=goog-private&lsig=lsig-private&%74oken=encoded-private&v=public-video")
    for value in ("amz-private", "credential-private", "goog-private", "lsig-private", "encoded-private"):
        assert value not in result
    assert "v=public-video" in result
