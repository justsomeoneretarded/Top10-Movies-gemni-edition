"""A short reaction intermission, with the spoken joke and its image chosen together.

Small, credited reaction assets are bundled so an unavailable image service cannot
break a finished video. GIFs keep their original moving frames; stills stay still.
"""
from __future__ import annotations

import base64
import hashlib
import io
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw

from .core import PipelineError, duration, ffmpeg, key
from .video_settings import export_size

ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "memes"
MAX_ASSET_BYTES = 2_000_000

# Credits record where a reaction came from, rather than claiming ownership,
# permission from its subjects, or a new license for the original screen footage.
REACTIONS = (
    {
        "theme": "high-five",
        "asset": "self-high-five.gif.b64",
        "sha256": "c23d7a9452a4fd1523d154d502e5072ae1f379137bedfe40c8412995e5ffa447",
        "animated": True,
        "headline": "HIGH FIVE?",
        "source": "https://github.com/milsyobtaf/gifs/blob/primary/30rock-self-highfive.gif",
        "attribution": "Reaction: Liz Lemon self high five (30 Rock); GIF source: milsyobtaf/gifs, 30rock-self-highfive.gif.",
        "jokes": (
            "Like and subscribe. Virtual high five. Zero awkward eye contact.",
            "Like and subscribe. That high five costs less than another terrible sequel.",
            "Hit like and subscribe. I'll handle both sides of the high five.",
        ),
    },
    {
        "theme": "logical-cat",
        "asset": "spock-cat.gif.b64",
        "sha256": "74cb2c9cd4033bc38e54cb3a556a609d0276fcfd4448921f0b7e1c38eb2ec372",
        "animated": True,
        "headline": "ONLY LOGICAL.",
        "source": "https://github.com/tlberglund/animated-gifs/blob/master/spock-stroking-cat.gif",
        "attribution": "Reaction: Spock stroking a cat (Star Trek); GIF source: tlberglund/animated-gifs, spock-stroking-cat.gif.",
        "jokes": (
            "Like and subscribe. It's only logical. Even the cat agrees.",
            "Like and subscribe. Excellent movie taste is logical. Petting the cat is optional.",
        ),
    },
    {
        "theme": "crying-cat",
        "asset": "crying-thumbs-up-cat.jpg.b64",
        "sha256": "25b475c1744a886c979a59692f196e8df4dc931d66c4e7401fe4e19b0ea50868",
        "animated": False,
        "headline": "PLEASE. SUBSCRIBE.",
        "source": "https://github.com/Pratikshya32/cat-meme-cam/blob/main/memes.js",
        "attribution": "Reaction: crying thumbs-up cat; image source: Pratikshya32/cat-meme-cam, memes.js (thumbsup).",
        "jokes": (
            "Like and subscribe. This cat believes in my movie career. Please don't tell him.",
            "Like and subscribe. My mom thinks I'm unemployed. The cat is starting to agree.",
            "Like and subscribe. The cat says he's fine. My subscriber count says otherwise.",
        ),
    },
)

from .reaction_library import expand_reactions
REACTIONS = expand_reactions(REACTIONS)


def reaction_choices() -> list[dict]:
    """Descriptions let the writer pair a fresh line with the actual visual."""
    return [{"theme": r["theme"], "description": r["description"]} for r in REACTIONS]


def _number(*values: str) -> int:
    return int(hashlib.sha256("\0".join(values).encode()).hexdigest()[:12], 16)


def _preferred(title: str, brief: str, seed: str) -> int:
    """Reproducible variety across all reactions; genre does not lock an asset."""
    return _number("cta-variety-v2", title, brief, seed) % len(REACTIONS)


def _validate(data: bytes, animated: bool):
    if not data or len(data) > MAX_ASSET_BYTES:
        raise ValueError("Invalid reaction asset size")
    with Image.open(io.BytesIO(data)) as picture:
        if min(picture.size) < 180 or picture.width * picture.height > 8_000_000:
            raise ValueError("Invalid reaction asset dimensions")
        frames = getattr(picture, "n_frames", 1)
        if animated and (picture.format != "GIF" or frames < 2):
            raise ValueError("Reaction GIF is not animated")
        picture.seek(frames // 2)
        picture.load()


def _asset(reaction: dict, work: Path) -> Path:
    encoded = ASSET_ROOT / reaction["asset"]
    # Bound before decoding, including whitespace in the reviewable text asset.
    if encoded.stat().st_size > MAX_ASSET_BYTES * 1.5:
        raise ValueError("Reaction asset is too large")
    data = base64.b64decode(b"".join(encoded.read_bytes().split()), validate=True)
    if hashlib.sha256(data).hexdigest() != reaction["sha256"]:
        raise ValueError("Reaction asset checksum mismatch")
    _validate(data, reaction["animated"])
    work.mkdir(parents=True, exist_ok=True)
    decoded = work / reaction["asset"].removesuffix(".b64")
    if not decoded.exists() or decoded.read_bytes() != data:
        decoded.write_bytes(data)
    return decoded.resolve()


def prepare_cta(title: str, brief: str, seed: str, work: Path, *, suggestion: dict | None = None) -> dict:
    """Choose a matching joke and validated local reaction before synthesizing it.

    The seed makes reruns reproducible. A missing/corrupt preferred asset tries the
    other bundled reactions with their own matching joke; all assets missing uses
    a movie-background intermission. No HTTP request or image model is needed.
    """
    suggestion = suggestion if isinstance(suggestion, dict) else {}
    proposed = " ".join(str(suggestion.get("text", "")).split())
    matching = next((i for i, r in enumerate(REACTIONS) if r["theme"] == suggestion.get("theme")), None)
    usable = (matching is not None and 3 <= len(proposed.split()) <= 24
              and re.search(r"\blike\b", proposed, re.I) and re.search(r"\bsubscribe\b", proposed, re.I))
    preferred = matching if usable else _preferred(title, brief, seed)
    order = [preferred] + [i for i in range(len(REACTIONS)) if i != preferred]
    for index in order:
        reaction = REACTIONS[index]
        try:
            path = _asset(reaction, work)
        except (OSError, ValueError, Image.DecompressionBombError):
            continue
        text = (proposed if usable and index == matching else
                reaction["jokes"][_number(seed, reaction["theme"]) % len(reaction["jokes"])])
        return {name: reaction[name] for name in ("theme", "animated", "headline", "source", "attribution")} | {
            "path": str(path), "text": text,
        }
    print("Optional reaction assets unavailable; using a film-background intermission.", flush=True)
    return {
        "theme": "movie-night", "animated": False, "path": None,
        "headline": "SAVE MOVIE NIGHT.",
        "text": "Like and subscribe. Protect your next movie night from terrible recommendations.",
        "source": "", "attribution": "",
    }


def cta_variants(plan: dict) -> list[dict]:
    """Offer at most three matching lines while retaining the selected reaction.

    Try the exact original plan first. Later choices change only its spoken text,
    using the same reaction's existing jokes in a stable order. This gives speech
    recovery a different short line without changing the image, caption, voice,
    or credit. Unknown/custom reactions retain their original line alone.
    """
    reaction = next((item for item in REACTIONS if item["theme"] == plan.get("theme")), None)
    if reaction is not None:
        jokes = reaction["jokes"]
    elif plan.get("theme") == "movie-night":
        jokes = (
            "Like and subscribe. Bad movie nights have claimed enough victims.",
            "Like and subscribe. Your next movie night deserves a happy ending.",
        )
    else:
        jokes = ()
    choices = [dict(plan)]
    seen = {plan["text"]}
    for text in jokes:
        if text in seen:
            continue
        choices.append({**plan, "text": text})
        seen.add(text)
        if len(choices) == 3:
            break
    return choices


def _caption(path: Path, headline: str):
    """A small caption in the safe area, leaving the reaction itself in charge."""
    from .render import HEIGHT, WIDTH, font

    picture = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(picture)
    face = font(32)
    width = min(WIDTH - 112, int(draw.textlength(headline, font=face)) + 48)
    left = (WIDTH - width) // 2
    top, bottom = HEIGHT - 110, HEIGHT - 44
    draw.rounded_rectangle((left, top, left+width, bottom), radius=8, fill=(9, 12, 17, 215))
    draw.text((WIDTH // 2, (top+bottom)//2-1), headline, font=face, anchor="mm", fill=(250, 250, 250))
    picture.save(path)


def _reaction_video(path: Path, animated: bool, seconds: float, target: Path):
    """Fit the whole face/gesture, with its soft duplicate filling the widescreen."""
    from .render import FPS

    width, height = export_size()
    inputs = ["-stream_loop", "-1", "-i", path] if animated else ["-loop", "1", "-framerate", FPS, "-i", path]
    # Blur the small backdrop before enlarging it; the sharp foreground keeps the
    # source's complete composition instead of losing a hand/face to a 16:9 crop.
    graph = (
        "[0:v]split=2[back][front];"
        "[back]scale=320:180:force_original_aspect_ratio=increase,crop=320:180,"
        f"gblur=sigma=12,scale={width}:{height},eq=brightness=-0.12:saturation=0.65[bg];"
        f"[front]scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,"
        f"fps={FPS},setpts=PTS-STARTPTS[out]"
    )
    ffmpeg(*inputs, "-filter_complex_threads", "1", "-filter_complex", graph,
           "-map", "[out]", "-frames:v", math.ceil(seconds*FPS), "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-pix_fmt", "yuv420p", "-threads", "2", target)


def render_cta(plan: dict, audio: Path, output: Path, work: Path, background: dict | Path | None = None) -> dict:
    """Render the chosen reaction for the complete narration, preserving GIF motion.

    Uses the shared speech renderer, including its lossless intermediate audio.
    A trailer still remains available if an optional reaction cannot be decoded.
    Return the actual source shot for the timeline. Reaction provenance is only
    returned when that reaction reached the video, never when a fallback did.
    """
    from .render import speech_video

    work.mkdir(parents=True, exist_ok=True)
    seconds = duration(audio)
    caption = work / "cta-caption.png"
    _caption(caption, plan["headline"])
    reaction_path = Path(plan["path"]) if plan.get("path") else None
    prepared = work / f"reaction-{key(str(reaction_path), seconds, export_size())}.mp4"
    if reaction_path is not None:
        try:
            _reaction_video(reaction_path, bool(plan.get("animated")), seconds, prepared)
            speech_video(prepared, audio, [{"start": 0, "duration": seconds}], output,
                         caption, work / "speech", label_seconds=seconds)
        except (OSError, PipelineError, ValueError):
            print("Optional reaction could not render; retaining the film-background intermission.", flush=True)
        else:
            animated = bool(plan.get("animated"))
            return {"source": str(reaction_path), "start": 0, "duration": seconds,
                    "image": True, "animation": animated, "still": not animated,
                    "attribution": plan.get("attribution", ""),
                    "credit_source": plan.get("source", "")}
        finally:
            prepared.unlink(missing_ok=True)
    backdrop = Path(background["source"]) if isinstance(background, dict) else (Path(background) if background else None)
    if backdrop is not None and backdrop.exists():
        source = backdrop
        start = float(background.get("start", 0)) if isinstance(background, dict) else duration(source)*0.3
        shot = {"source": str(source), "start": max(0, start), "duration": seconds,
                "image": False, "animation": False, "still": True}
    else:
        # An otherwise complete video should still export if every optional
        # reaction and trailer backdrop is unavailable.
        source = work / "neutral-background.png"
        Image.new("RGB", (1280, 720), (12, 16, 23)).save(source)
        shot = {"source": str(source), "start": 0, "duration": seconds,
                "image": True, "animation": False, "still": True}
    speech_video(source, audio, [shot], output, caption, work / "speech", label_seconds=seconds)
    return {**shot, "attribution": "", "credit_source": ""}
