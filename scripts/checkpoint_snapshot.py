"""Bounded, allowlisted production checkpoints; never archive caches or credentials.

Snapshots are best effort. They contain reusable JSON and independently hash-bound
verified speech, not finished media. Publication uses an atomic rename; restore
validates every member before writing and never overwrites existing warm files.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import tarfile
import tempfile
import time
import wave
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

MAX_BYTES = 128 * 1024 * 1024
MAX_FILES = 512
JSON_BYTES = 2 * 1024 * 1024
WAV_BYTES = 32 * 1024 * 1024
DIAGNOSTIC_BYTES = 2 * 1024 * 1024
JOB = re.compile(r"[0-9a-f]{20,64}")
HASH = re.compile(r"[0-9a-f]{64}")
TOKEN = re.compile(r"(?:gh[pousr]_[A-Za-z0-9_]{12,}|github_pat_[A-Za-z0-9_]{12,}|hf_[A-Za-z0-9]{12,}|[0-9]{7,12}:[A-Za-z0-9_-]{25,}|eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,})")
SECRET_KEY = re.compile(r"token|secret|password|passwd|cookie|api.?key|private.?key|authorization|credential", re.I)
ROOT_JSON = {"input.json", "research.json", "script.json", "quality-report.json",
             "duration-report.json", "speech-verification.json", "audio-review.json",
             "comparison-audio-review.json"}
JSON_DIRS = {"script-sections", "availability", "speech-checks", "export-audio-checks",
             "comparison-audio-checks"}
DIAGNOSTICS = {"production.log", "resources.jsonl"}


def redact(text: str) -> str:
    """Mask known secret values and credentials commonly embedded in URLs/logs."""
    for name, value in os.environ.items():
        if SECRET_KEY.search(name) and 4 <= len(value) <= 65536:
            text = text.replace(value, "[REDACTED]")
    text = TOKEN.sub("[REDACTED]", text)
    text = re.sub(r"(?i)(https?://)[^\s/@:]+:[^\s/@]+@", r"\1[REDACTED]@", text)
    text = re.sub(r"([?&]([^=&\s\"'<>]{1,128})=)[^&\s\"'<>]+",
                  lambda match: match[1] + "[REDACTED]" if re.search(
                      r"token|key|sig|credential|auth|password|secret|cookie|session|jwt|policy|code",
                      unquote(match[2]), re.I) else match[0], text)
    text = re.sub(r"(?i)\b(authorization\s*[:=]\s*(?:bearer\s+)?|cookie\s*[:=]\s*)[^\r\n]+",
                  r"\1[REDACTED]", text)
    return text


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()
                if isinstance(key, str) and not SECRET_KEY.search(key)}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, str):
        return redact(value)
    return value


def allowed_job_file(relative: PurePosixPath) -> bool:
    parts = relative.parts
    if len(parts) < 4 or parts[:2] != ("work", "jobs") or not JOB.fullmatch(parts[2]):
        return False
    tail = parts[3:]
    if len(tail) == 1:
        return tail[0] in ROOT_JSON
    if len(tail) != 2:
        return False
    directory, name = tail
    if directory in JSON_DIRS:
        if re.fullmatch(r"[0-9a-f]{20}\.wav", name) and directory.endswith("checks"):
            return True
        return (name == "speech-verification.json" or
                bool(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,119}\.json", name))) and not SECRET_KEY.search(name)
    if directory == "speech":
        return bool(re.fullmatch(r"[0-9a-f]{20}(?:-\d+)?(?:-pace-v3-[0-9.]+)?\.(?:wav|captions\.json)", name))
    return False


def safe_parts(relative: PurePosixPath) -> bool:
    return (not relative.is_absolute() and bool(relative.parts)
            and all(part not in {"", ".", ".."} and "\\" not in part
                    and not any(ord(char) < 32 for char in part) for part in relative.parts))


def regular_read(workspace: Path, relative: PurePosixPath, limit: int, *, tail=False) -> bytes:
    """Open each component without following links and detect replaced/torn files."""
    if not safe_parts(relative):
        raise ValueError("unsafe_path")
    directory = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in relative.parts[:-1]:
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = following
        descriptor = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or (not tail and before.st_size > limit):
                raise ValueError("oversize_or_nonregular")
            if tail and before.st_size > limit:
                os.lseek(descriptor, before.st_size - limit, os.SEEK_SET)
            chunks, remaining = [], limit + 1
            while remaining:
                block = os.read(descriptor, min(1024 * 1024, remaining))
                if not block:
                    break
                chunks.append(block)
                remaining -= len(block)
            data = b"".join(chunks)
            after = os.fstat(descriptor)
            current = os.stat(relative.name, dir_fd=directory, follow_symlinks=False)
            identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
            if len(data) > limit or identity(before) != identity(after) or identity(after) != identity(current):
                raise ValueError("unstable")
            if tail and before.st_size > limit:
                # Drop a partial leading line instead of leaking a token suffix.
                data = data.split(b"\n", 1)[1] if b"\n" in data else b""
            return data
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def candidates(workspace: Path):
    jobs = workspace / "work" / "jobs"
    if jobs.is_symlink() or not jobs.is_dir() or (workspace / "work").is_symlink():
        return []
    result = []
    # The pipeline run key is hexadecimal. Never walk arbitrary cache/media trees.
    with os.scandir(jobs) as entries:
        job_names = sorted(entry.name for entry in entries
                           if JOB.fullmatch(entry.name) and entry.is_dir(follow_symlinks=False))[:128]
    for job in job_names:
        root = jobs / job
        for name in sorted(ROOT_JSON):
            relative = PurePosixPath("work", "jobs", job, name)
            result.append(relative)
        for folder in sorted(JSON_DIRS | {"speech"}):
            directory = root / folder
            if directory.is_symlink() or not directory.is_dir():
                continue
            with os.scandir(directory) as entries:
                names = sorted(entry.name for entry in entries if entry.is_file(follow_symlinks=False))[:4096]
            for name in names:
                relative = PurePosixPath("work", "jobs", job, folder, name)
                if allowed_job_file(relative):
                    result.append(relative)
    return sorted(result, key=lambda item: (0 if item.name.endswith(".verified.json") else
                                           2 if item.suffix == ".wav" else 1, str(item)))


def verified_hashes(relative: PurePosixPath, data) -> set[str]:
    if not isinstance(data, dict):
        return set()
    if relative.name.endswith(".verified.json"):
        value = data.get("audio_sha256")
        cues = data.get("cues")
        if isinstance(value, str) and HASH.fullmatch(value) and isinstance(cues, list) and cues:
            return {value}
    return set()


def add_bytes(archive, name: str, data: bytes):
    member = tarfile.TarInfo(name)
    member.size, member.mode, member.mtime = len(data), 0o600, 0
    archive.addfile(member, io.BytesIO(data))


def create_snapshot(workspace: Path, output: Path, *, max_bytes=MAX_BYTES):
    workspace = workspace.resolve()
    if not 65536 <= max_bytes <= MAX_BYTES:
        raise ValueError("invalid_budget")
    output.parent.mkdir(parents=True, exist_ok=True)
    selected, skipped, hashes = [], Counter(), set()
    payload = 0
    budget = max_bytes - min(2 * 1024 * 1024, max_bytes // 4)
    job_budget = budget - min(DIAGNOSTIC_BYTES, budget // 8)
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tar.gz", dir=output.parent)
    os.close(descriptor)
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=1, format=tarfile.USTAR_FORMAT) as archive:
            for relative in candidates(workspace):
                if len(selected) >= MAX_FILES:
                    skipped["file_limit"] += 1
                    continue
                try:
                    raw = regular_read(workspace, relative, WAV_BYTES if relative.suffix == ".wav" else JSON_BYTES)
                    if relative.suffix == ".wav":
                        if hashlib.sha256(raw).hexdigest() not in hashes:
                            skipped["unverified_audio"] += 1
                            continue
                        with wave.open(io.BytesIO(raw), "rb") as recording:
                            if recording.getnframes() <= 0 or recording.getcomptype() != "NONE":
                                raise ValueError("invalid_audio")
                        data = raw
                    else:
                        parsed = clean_json(json.loads(raw))
                        data = json.dumps(parsed, ensure_ascii=False, separators=(",", ":")).encode()
                    if payload + len(data) > job_budget:
                        skipped["byte_limit"] += 1
                        continue
                    add_bytes(archive, str(relative), data)
                    payload += len(data)
                    selected.append({"path": str(relative), "bytes": len(data),
                                     "sha256": hashlib.sha256(data).hexdigest()})
                    if relative.suffix != ".wav":
                        hashes.update(verified_hashes(relative, parsed))
                except FileNotFoundError:
                    continue
                except (OSError, ValueError, TypeError, RecursionError, wave.Error, EOFError):
                    skipped["unsafe_unstable_or_invalid"] += 1
            diagnostic_remaining = min(DIAGNOSTIC_BYTES, max(0, budget - payload))
            for name in sorted(DIAGNOSTICS):
                if diagnostic_remaining <= 0:
                    break
                relative = PurePosixPath("work", "runner-diagnostics", name)
                try:
                    raw = regular_read(workspace, relative, min(512 * 1024, diagnostic_remaining), tail=True)
                    data = redact(raw.decode("utf-8", errors="replace")).encode()[:diagnostic_remaining]
                    add_bytes(archive, str(relative), data)
                    payload += len(data)
                    diagnostic_remaining -= len(data)
                    selected.append({"path": str(relative), "bytes": len(data),
                                     "sha256": hashlib.sha256(data).hexdigest()})
                except FileNotFoundError:
                    continue
                except (OSError, ValueError):
                    skipped["diagnostic_unavailable"] += 1
            manifest = {"version": 1, "created_unix": int(time.time()),
                        "run_id": re.sub(r"[^0-9]", "", os.getenv("GITHUB_RUN_ID", ""))[:24],
                        "run_attempt": re.sub(r"[^0-9]", "", os.getenv("GITHUB_RUN_ATTEMPT", ""))[:8],
                        "commit": os.getenv("GITHUB_SHA", "") if re.fullmatch(r"[0-9a-f]{40}", os.getenv("GITHUB_SHA", "")) else "",
                        "selected_count": len(selected), "selected_bytes": payload,
                        "files": selected, "skipped": dict(skipped),
                        "scope": "allowlisted-job-json-and-hash-verified-speech; bounded-redacted-diagnostics"}
            add_bytes(archive, "checkpoint-manifest.json", json.dumps(manifest, separators=(",", ":")).encode())
        if os.stat(temporary).st_size > max_bytes:
            raise ValueError("archive_budget")
        os.replace(temporary, output)
        return manifest
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def destination_parent(workspace: Path, relative: PurePosixPath) -> Path:
    current = workspace
    for component in relative.parts[:-1]:
        current = current / component
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            pass
        if current.is_symlink() or not current.is_dir():
            raise ValueError("unsafe_destination")
    return current


def restore_snapshot(workspace: Path, archive_path: Path):
    """Prevalidate names, types, sizes, hashes and speech bindings before writing."""
    workspace = workspace.resolve()
    if archive_path.stat().st_size > MAX_BYTES:
        raise ValueError("archive_budget")
    with tarfile.open(archive_path, "r:gz") as archive:
        members, names, total = [], set(), 0
        for member in archive:
            members.append(member)
            if len(members) > MAX_FILES + len(DIAGNOSTICS) + 1:
                raise ValueError("member_limit")
            relative = PurePosixPath(member.name)
            diagnostic = relative.parts[:2] == ("work", "runner-diagnostics") and len(relative.parts) == 3 and relative.name in DIAGNOSTICS
            if (member.name in names or str(relative) != member.name or not member.isfile() or not safe_parts(relative)
                    or (member.name != "checkpoint-manifest.json" and not diagnostic and not allowed_job_file(relative))
                    or member.size < 0 or member.size > WAV_BYTES):
                raise ValueError("invalid_member")
            names.add(member.name)
            if ((member.name == "checkpoint-manifest.json" and member.size > JSON_BYTES)
                    or (allowed_job_file(relative) and relative.suffix == ".json" and member.size > JSON_BYTES)
                    or (diagnostic and member.size > DIAGNOSTIC_BYTES)):
                raise ValueError("member_budget")
            total += member.size
            if total > MAX_BYTES:
                raise ValueError("archive_budget")
        source = archive.extractfile("checkpoint-manifest.json")
        manifest = json.load(source) if source else None
        if not isinstance(manifest, dict) or manifest.get("version") != 1 or not isinstance(manifest.get("files"), list):
            raise ValueError("invalid_manifest")
        expected = {}
        for entry in manifest["files"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or entry["path"] in expected:
                raise ValueError("invalid_manifest")
            expected[entry["path"]] = entry
        if set(expected) != names - {"checkpoint-manifest.json"}:
            raise ValueError("manifest_members")
        hashes = set()
        for member in members:
            if member.name == "checkpoint-manifest.json":
                continue
            data = archive.extractfile(member).read(member.size + 1)
            entry = expected[member.name]
            if len(data) != entry.get("bytes") or hashlib.sha256(data).hexdigest() != entry.get("sha256"):
                raise ValueError("member_hash")
            relative = PurePosixPath(member.name)
            if allowed_job_file(relative) and relative.suffix == ".json":
                parsed = json.loads(data)
                if clean_json(parsed) != parsed:
                    raise ValueError("credential_in_json")
                hashes.update(verified_hashes(relative, parsed))
        for member in members:
            if member.name.endswith(".wav") and expected[member.name]["sha256"] not in hashes:
                raise ValueError("unverified_audio")
        restored = 0
        for member in members:
            relative = PurePosixPath(member.name)
            # Diagnostics are evidence, never application inputs on a rerun.
            if not allowed_job_file(relative):
                continue
            parent = destination_parent(workspace, relative)
            target = parent / relative.name
            if target.exists() or target.is_symlink():
                continue
            data = archive.extractfile(member).read(member.size + 1)
            descriptor, temporary = tempfile.mkstemp(prefix=".restore-", dir=parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                # Hard-link publishes atomically without replacing an existing file.
                try:
                    os.link(temporary, target, follow_symlinks=False)
                    restored += 1
                except FileExistsError:
                    pass
            finally:
                os.unlink(temporary)
        return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "restore"))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "create":
            if args.output is None:
                parser.error("create requires --output")
            manifest = create_snapshot(args.workspace, args.output)
            print(json.dumps({"selected_count": manifest["selected_count"], "selected_bytes": manifest["selected_bytes"]}))
        else:
            if args.archive is None:
                parser.error("restore requires --archive")
            print(json.dumps({"restored_count": restore_snapshot(args.workspace, args.archive)}))
    except (OSError, ValueError, TypeError, KeyError, tarfile.TarError, RecursionError):
        # Never print arbitrary path names, file contents, command lines or env values.
        print('{"checkpoint":"unavailable"}')
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
