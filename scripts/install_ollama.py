"""Pinned, CPU-only Ollama installation for ephemeral Ubuntu amd64 runners.

Ollama publishes a combined Linux archive, not a separate CPU release. The
v0.40.14 packaging puts CPU files in lib/ollama and GPU files in cuda_v*/vulkan
subdirectories. Exclude those directories while extracting, before they consume
runner disk. See https://github.com/ollama/ollama/blob/v0.40.14/scripts/build_linux.sh
and https://github.com/ollama/ollama/blob/v0.40.14/llama/server/CMakePresets.json.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

VERSION = "0.40.14"
ARCHIVE_SHA256 = "993e83d08c3d13d17c55009a3e2664c8a25445e644aec3a68bef5abb6abce67d"
ARCHIVE_URL = (
    f"https://github.com/ollama/ollama/releases/download/v{VERSION}/"
    "ollama-linux-amd64.tar.zst"
)
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT = 600
MAX_DOWNLOAD_BYTES = 2 * 1024**3
GPU_EXCLUDES = (
    "lib/ollama/cuda_v*",
    "./lib/ollama/cuda_v*",
    "lib/ollama/vulkan",
    "./lib/ollama/vulkan",
)


def download_archive(archive: Path) -> None:
    """Retry whole file downloads; never concatenate retried stdout into tar."""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        archive.unlink(missing_ok=True)
        try:
            subprocess.run(
                [
                    "curl", "--fail", "--show-error", "--location",
                    "--connect-timeout", "20", "--max-time", str(DOWNLOAD_TIMEOUT),
                    "--max-filesize", str(MAX_DOWNLOAD_BYTES),
                    "--output", str(archive), ARCHIVE_URL,
                ],
                check=True,
                timeout=DOWNLOAD_TIMEOUT + 30,
                stdout=sys.stderr,
            )
            with archive.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != ARCHIVE_SHA256:
                raise RuntimeError("Ollama archive checksum did not match the pinned release")
            return
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            archive.unlink(missing_ok=True)
            print(f"Ollama download attempt {attempt}/{DOWNLOAD_ATTEMPTS}: {error}", file=sys.stderr)
            if attempt == DOWNLOAD_ATTEMPTS:
                raise RuntimeError("Could not download the verified Ollama release") from error
            time.sleep(2 ** attempt)


def extract_payload(archive: Path, destination: Path) -> None:
    destination.mkdir()
    subprocess.run(
        [
            "tar", "--zstd", "--extract", "--file", str(archive),
            "--directory", str(destination), "--no-same-owner",
            *(f"--exclude={pattern}" for pattern in GPU_EXCLUDES),
        ],
        check=True,
        timeout=300,
        stdout=sys.stderr,
    )


def validate_payload(payload: Path) -> None:
    binary = payload / "bin" / "ollama"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError("Ollama archive did not contain its executable")
    if not (payload / "lib" / "ollama" / "llama-server").is_file():
        raise RuntimeError("Ollama archive did not contain the CPU runtime")
    result = subprocess.run(
        [str(binary), "--version"],
        check=True,
        timeout=15,
        capture_output=True,
        text=True,
    )
    if not re.search(rf"(?<![\d.]){re.escape(VERSION)}(?![\d.])", result.stdout + result.stderr):
        raise RuntimeError("Installed Ollama version did not match the pinned release")


def installed_bin(root: Path) -> Path | None:
    current = root / "current"
    if not current.is_symlink():
        return None
    payload = current.resolve()
    if payload.parent != (root / "releases").resolve():
        raise RuntimeError("Ollama current link is outside its owned releases directory")
    try:
        if (payload / ".release-sha256").read_text().strip() != ARCHIVE_SHA256:
            return None
        validate_payload(payload)
    except (OSError, subprocess.SubprocessError, RuntimeError):
        return None
    return current / "bin"


def persist_path(binary_dir: Path) -> None:
    github_path = os.environ.get("GITHUB_PATH")
    if github_path:
        with open(github_path, "a", encoding="utf-8") as destination:
            destination.write(f"{binary_dir}\n")


def install_ollama(runner_temp: str | Path | None = None) -> Path:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise RuntimeError("The pinned CPU Ollama installer requires Linux amd64")
    for tool in ("curl", "tar", "zstd"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"Required Ollama installation tool is missing: {tool}")
    temporary_root = Path(runner_temp or os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()).resolve()
    root = temporary_root / "movie-ranks-ollama"
    if root.is_symlink():
        raise RuntimeError("Refusing to install through a symlinked Ollama directory")
    root.mkdir(parents=True, exist_ok=True)
    releases = root / "releases"
    if releases.is_symlink():
        raise RuntimeError("Refusing to install through a symlinked releases directory")
    releases.mkdir(exist_ok=True)
    if (root / "current").exists() and not (root / "current").is_symlink():
        raise RuntimeError("Ollama current path is not an owned symlink")

    binary_dir = installed_bin(root)
    if binary_dir is not None:
        persist_path(binary_dir)
        return binary_dir

    # Staging and final directories share a filesystem: rename, never a second
    # expanded copy. TemporaryDirectory removes only this invocation's files.
    with tempfile.TemporaryDirectory(prefix=".install-", dir=root) as temporary:
        stage = Path(temporary)
        archive = stage / "ollama.tar.zst"
        payload = stage / "payload"
        download_archive(archive)
        extract_payload(archive, payload)
        validate_payload(payload)
        (payload / ".release-sha256").write_text(ARCHIVE_SHA256 + "\n", encoding="utf-8")
        archive.unlink()
        release = releases / f"v{VERSION}-{uuid.uuid4().hex}"
        link = root / f".current-{uuid.uuid4().hex}"
        moved = False
        activated = False
        try:
            payload.rename(release)
            moved = True
            link.symlink_to(Path("releases") / release.name, target_is_directory=True)
            os.replace(link, root / "current")
            activated = True
        finally:
            link.unlink(missing_ok=True)
            if moved and not activated:
                shutil.rmtree(release)
    binary_dir = root / "current" / "bin"
    persist_path(binary_dir)
    return binary_dir


def existing_install(runner_temp: str | Path | None = None) -> Path | None:
    """Probe a verified prior installation without downloading or extracting."""
    temporary_root=Path(runner_temp or os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()).resolve()
    root=temporary_root/"movie-ranks-ollama"
    if root.is_symlink() or (root/"releases").is_symlink():
        raise RuntimeError("Refusing a symlinked Ollama installation directory")
    if not root.is_dir():
        return None
    binary_dir=installed_bin(root)
    if binary_dir is not None:
        persist_path(binary_dir)
    return binary_dir

def main(arguments=()) -> int:
    arguments=list(arguments)
    if arguments not in ([],["--existing"]):
        print("Usage: python -m scripts.install_ollama [--existing]",file=sys.stderr)
        return 2
    try:
        if arguments:
            binary_dir=existing_install()
            if binary_dir is None:
                return 3
        else:
            binary_dir=install_ollama()
        print(binary_dir)
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"Ollama installation failed: {error}",file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
