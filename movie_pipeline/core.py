from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


class PipelineError(RuntimeError):
    """An actionable failure; never silently substitute another movie or voice."""


def run(args: list, **kwargs) -> str:
    result = subprocess.run([str(x) for x in args], check=False, text=True, capture_output=True, **kwargs)
    if result.returncode:
        # Commands never contain credentials. Do not dump arbitrary environment variables.
        raise PipelineError(f"{args[0]} failed: {result.stderr[-2400:]}")
    return result.stdout


def notice(title: str, message: str, level: str = "notice"):
    """A GitHub Actions annotation (visible on the run page) when running in Actions; plain log otherwise."""
    text = " ".join(str(message).split()) if level != "error" else str(message)
    if os.getenv("GITHUB_ACTIONS") == "true":
        clean = text[:3000].replace("%", "%25").replace("\r", "").replace("\n", "%0A")
        print(f"::{level} title={title}::{clean}", flush=True)
    else:
        print(f"{title}: {text}", flush=True)


def ffmpeg(*args):
    return run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args])


def probe(path: Path) -> dict:
    return json.loads(run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path]))


def duration(path: Path) -> float:
    return float(probe(path)["format"]["duration"])


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # numpy numbers and booleans (from image analysis) are written as plain values, never a crash.
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                              default=lambda value: value.item() if hasattr(value, "item") else str(value)))
    tmp.replace(path)


def key(*values) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:20]


def fold_accents(text: str) -> str:
    """'Shōgun' -> 'Shogun', 'Amélie' -> 'Amelie': accents never decide whether two titles match."""
    import unicodedata
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", fold_accents(text).casefold()))


@dataclass(frozen=True)
class FilmInput:
    title: str
    year: int


def parse_movies(text: str) -> list[FilmInput]:
    items = [x.strip() for x in re.split(r"[;\n]+", text) if x.strip()]
    if not 2 <= len(items) <= 15:
        raise PipelineError("Enter 2–15 movies/shows, separated by newlines or semicolons.")
    result = []
    for item in items:
        match = re.fullmatch(r"(.+?)\s*\((\d{4})\)", item)
        if not match:
            match = re.fullmatch(r"(.+?)\s+(\d{4})", item)
        if not match or not 1888 <= int(match[2]) <= 2100:
            raise PipelineError(f"Use Title (Year), for example Arrival (2016). Invalid entry: {item!r}")
        result.append(FilmInput(match[1].strip(), int(match[2])))
    if len({(normalized(x.title), x.year) for x in result}) != len(result):
        raise PipelineError("The movie list contains duplicates.")
    return result


def narrator_from_brief(brief: str) -> str:
    # Negated instructions must not select the forbidden voice.
    text = brief.casefold().replace("’", "'")
    text = re.sub(r"\b(?:do not|don't|never|avoid|not)\s+(?:use\s+)?(?:the\s+)?(?:female|woman|kit laser|warguy)(?:\s+(?:audio|voice|narrator))?", "", text)
    aliases = {"female": r"(?:\b(?:use|select|choose)\s+(?:the\s+)?(?:female|woman)\b|\b(?:female|woman)\s+(?:audio|voice|narrator|narration)\b|\bnarrator\s*[:=]\s*female\b|^female$)", "kit-laser": r"\bkit\s*laser\b", "warguy": r"\bwar\s*guy\b"}
    chosen = [name for name, pattern in aliases.items() if re.search(pattern, text)]
    if len(chosen) > 1:
        raise PipelineError("Specify one narrator: Warguy, Kit Laser, or Female.")
    return chosen[0] if chosen else "warguy"


def selected_narrator(brief: str) -> str:
    """An explicit workflow choice wins over narrator wording in the brief."""
    value = normalized(os.getenv("NARRATOR", ""))
    if not value:
        return narrator_from_brief(brief)
    choices = {"warguy": "warguy", "war guy": "warguy", "kit laser": "kit-laser", "female": "female"}
    if value not in choices:
        raise PipelineError("Choose Warguy, Kit Laser, or Female in the narrator selector.")
    return choices[value]


def selected_content_type() -> str | None:
    """Return the explicit workflow medium; absent means legacy automatic lookup."""
    value = normalized(os.getenv("VIDEO_TYPE", ""))
    if not value:
        return None
    choices = {"movies": "movie", "movie": "movie", "shows": "tv", "show": "tv", "series": "tv", "tv": "tv"}
    if value not in choices:
        raise PipelineError("Choose Movies or Shows in the content-type selector.")
    return choices[value]


def voice_reference(narrator: str, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for extension in ("wav", "mp3", "ogg", "opus"):
        path = root / f"{narrator}.{extension}"
        if path.exists():
            if not 5 <= duration(path) <= 30:
                raise PipelineError("Voice reference must be a clean 5–30 second recording.")
            return path
    name = "VOICE_" + narrator.upper().replace("-", "_") + "_B64"
    encoded = os.getenv(name, "")
    if not encoded:
        raise PipelineError(f"Missing {name}. Run the one-time private voice setup described in README.")
    try:
        data = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise PipelineError(f"{name} is not valid base64.") from exc
    path = root / f"{narrator}.ogg"
    path.write_bytes(data)
    if not 5 <= duration(path) <= 30:
        raise PipelineError(f"Invalid reference duration for {narrator}.")
    return path


def timestamp(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def write_srt(path: Path, cues: list[dict]):
    blocks = []
    previous = 0.0
    for cue in sorted(cues, key=lambda c: c["start"]):
        start = max(float(cue["start"]), previous)
        end = float(cue["end"])
        text = " ".join(cue["text"].split()).strip()
        if end <= start or not text:
            continue
        from .subtitles import two_lines
        text = two_lines(text)
        blocks.append(f"{len(blocks)+1}\n{timestamp(start)} --> {timestamp(end)}\n{text}\n")
        previous = end
    path.write_text("\n".join(blocks), encoding="utf-8")
