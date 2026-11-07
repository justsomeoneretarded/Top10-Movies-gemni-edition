"""Optional YouTube session cookies, stored outside caches and artifacts."""
from __future__ import annotations

import os
from http.cookiejar import LoadError, MozillaCookieJar
from pathlib import Path

from .core import PipelineError


def prepare_cookies(contents: str, directory: Path) -> Path | None:
    if not contents.strip():
        return None
    if len(contents.encode()) >= 48_000:
        raise PipelineError("YOUTUBE_COOKIES must be smaller than 48 KB. Export cookies for YouTube only.")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "youtube-cookies.txt"
    # Private even if the parent directory is shared. Never print cookie contents.
    path.unlink(missing_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(contents.strip() + "\n")
    try:
        jar = MozillaCookieJar(str(path))
        jar.load(ignore_discard=True, ignore_expires=True)
        if not any(c.domain.lstrip(".") == "youtube.com" or c.domain.endswith(".youtube.com") for c in jar):
            raise LoadError("No YouTube cookies")
    except (LoadError, OSError, ValueError) as exc:
        path.unlink(missing_ok=True)
        raise PipelineError("YOUTUBE_COOKIES must contain a valid Netscape-format cookies.txt export for YouTube, not JSON or base64.") from exc
    return path
