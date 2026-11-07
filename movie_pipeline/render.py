from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .core import PipelineError, duration, ffmpeg, probe, run, save_json
from .soundtrack import music_bed
from .timing import DurationSpec, final_tempo
from .video_settings import export_size

WIDTH, HEIGHT, FPS = 1280, 720, 30
LOSSLESS_AUDIO = ("-ar", "48000", "-ac", "2", "-channel_layout", "stereo",
                  "-c:a", "alac", "-sample_fmt", "s32p")
AUDIO_ALIGNMENT_TOLERANCE = .12
FOOTAGE_GRADE = "eq=contrast=1.015:saturation=1.015"
# Genre looks for all footage (VIDEO_GENRE is set by the pipeline from the title, brief and TMDB genres).
GENRE_GRADES = {
    "horror": "eq=contrast=1.09:saturation=0.76:brightness=-0.025:gamma=0.96,colorbalance=bs=0.05:bm=0.02:rs=-0.02:rh=0.02",
    "thriller": "eq=contrast=1.06:saturation=0.86:brightness=-0.01,colorbalance=bs=0.04:bm=0.01",
    "action": "eq=contrast=1.06:saturation=1.08,colorbalance=bs=0.03:rh=0.03:gh=0.01",
    "scifi": "eq=contrast=1.05:saturation=1.02,colorbalance=bs=0.05:gs=0.01:bh=0.02",
    "comedy": "eq=contrast=1.02:saturation=1.1:brightness=0.01",
    "family": "eq=contrast=1.02:saturation=1.12:brightness=0.015",
    "adventure": "eq=contrast=1.05:saturation=1.06,colorbalance=rm=0.03:gm=0.01:bh=-0.02",
    "romance": "eq=contrast=1.02:saturation=1.03:brightness=0.01,colorbalance=rm=0.03:rh=0.02:bh=-0.02",
    "fantasy": "eq=contrast=1.05:saturation=1.06,colorbalance=bs=0.04:rh=0.02:bh=0.02",
    "drama": "eq=contrast=1.03:saturation=0.97",
}


def footage_grade() -> str:
    return GENRE_GRADES.get(os.getenv("VIDEO_GENRE", "").strip().lower(), FOOTAGE_GRADE)


def _hold_motion(frames: int, live_frames: int, width: int, height: int, style: str) -> str:
    """Slow zoom over the frozen portion (at most 3.5%), without altering time.

    Uses the perspective filter, which samples between pixels (bicubic), so the frame scales
    continuously. zoompan snaps its crop to whole pixels, which made held stills shimmer and
    flicker by up to a pixel from frame to frame.
    """
    seconds = (frames-live_frames)/FPS
    if seconds < 1.5:
        return "null"  # A tiny timing hold should not turn into a quick zoom.
    amount = min(.035, seconds*.012)
    if live_frames:
        # A trailing freeze must start at the moving footage's existing framing.
        style = "zoom_in"
    progress = f"min(1,max(0,(in-{live_frames})/{max(1, frames-live_frames-1)}))"
    # Ease in and out (smoothstep) so the move starts and settles without a jolt.
    eased = f"({progress})*({progress})*(3-2*({progress}))"
    zoom = f"({1+amount}-{amount}*{eased})" if style == "zoom_out" else f"(1+{amount}*{eased})"
    mx, my = f"(W*(1-1/{zoom})/2)", f"(H*(1-1/{zoom})/2)"
    return (f"format=yuv444p,perspective=x0='{mx}':y0='{my}':x1='W-{mx}':y1='{my}':"
            f"x2='{mx}':y2='H-{my}':x3='W-{mx}':y3='H-{my}':interpolation=cubic:eval=frame,format=yuv420p")


FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
ACCENT = (255, 190, 40)


def brand_font(kind: str, size: int):
    """Bundled OFL fonts: 'display' (Anton), 'title' (Bebas Neue), 'body'/'body-medium' (Montserrat)."""
    files = {"display": "Anton-Regular.ttf", "title": "BebasNeue-Regular.ttf",
             "body": "Montserrat-Variable.ttf", "body-medium": "Montserrat-Variable.ttf"}
    path = FONT_DIR / files.get(kind, files["body"])
    if not path.exists():
        return font(size)
    face = ImageFont.truetype(str(path), size)
    if kind.startswith("body"):
        try:
            face.set_variation_by_name("SemiBold" if kind == "body-medium" else "ExtraBold")
        except (OSError, ValueError):
            pass
    return face


def font(size: int):
    custom = os.getenv("FONT_PATH", "")
    if custom and Path(custom).exists():
        return ImageFont.truetype(custom, size)
    if (FONT_DIR / "Montserrat-Variable.ttf").exists():
        return brand_font("body", size)
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def wrapped(draw, text, face, width):
    lines = []
    current = ""
    for word in text.split():
        trial = (current + " " + word).strip()
        if draw.textbbox((0, 0), trial, font=face)[2] > width and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def label_image(path: Path, title: str, rank: int | None, year: int | None):
    """Rank reveal card in the lower third: large accented rank, condensed title, year."""
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    if rank is None:
        image.save(path)
        return
    draw = ImageDraw.Draw(image)
    rank_face = brand_font("display", 104)
    rank_text = f"#{rank}"
    rank_width = draw.textlength(rank_text, font=rank_face)
    left, bottom = 56, 668
    title_left = left + 44 + round(rank_width) + 30
    max_width = WIDTH - 56 - title_left - 34
    words = title.upper()
    size = 66
    while size > 34 and draw.textlength(words, font=brand_font("title", size)) > max_width:
        size -= 2
    lines = wrapped(draw, words, brand_font("title", size), max_width)
    while len(lines) > 2 and size > 26:
        size -= 2
        lines = wrapped(draw, words, brand_font("title", size), max_width)
    title_face = brand_font("title", size)
    line_height = round(size * 0.98)
    content_height = line_height * len(lines) + (30 if year is not None else 0)
    top = max(512, bottom - max(132, content_height + 34))
    content_width = max((draw.textlength(line, font=title_face) for line in lines), default=0)
    if year is not None:
        content_width = max(content_width, draw.textlength(str(year), font=brand_font("body-medium", 22)))
    right = min(WIDTH - 56, round(title_left + content_width + 34))
    # Soft shadow, dark glass plate, accent bar.
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((left+4, top+6, right+4, bottom+6), radius=16, fill=(0, 0, 0, 120))
    from PIL import ImageFilter
    image.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(8)))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((left, top, right, bottom), radius=16, fill=(9, 11, 16, 228), outline=(255, 255, 255, 38), width=1)
    draw.rounded_rectangle((left, top, left+10, bottom), radius=5, fill=ACCENT + (255,))
    middle = (top + bottom) // 2
    hash_width = draw.textlength("#", font=rank_face)
    draw.text((left + 34, middle), "#", font=rank_face, anchor="lm", fill=ACCENT + (255,))
    draw.text((left + 34 + hash_width, middle), str(rank), font=rank_face, anchor="lm", fill=(250, 250, 252, 255))
    divider = title_left - 16
    draw.line((divider, top + 24, divider, bottom - 24), fill=(255, 255, 255, 60), width=2)
    y = middle - content_height // 2
    for line in lines:
        draw.text((title_left, y), line, font=title_face, fill=(250, 250, 252, 255))
        y += line_height
    if year is not None:
        draw.text((title_left, y + 2), str(year), font=brand_font("body-medium", 22), fill=(190, 196, 206, 255))
    image.save(path)


def _label_graph(input_index: int, seconds: float, width: int, height: int) -> str:
    """One consistent eased slide/fade animation, measured on the whole clip clock."""
    end = max(.1, seconds)
    delay = min(.12, end*.10)
    enter = min(.42, end*.30)
    leave = min(.34, end*.25)
    out_start = max(delay+enter, end-leave)
    # PNG has a full transparent canvas, so moving by one frame width fully hides it.
    x = (f"if(lt(t,{delay}),-{width},"
         f"if(lt(t,{delay+enter}),-{width}*pow(1-(t-{delay})/{enter},3),"
         f"if(lt(t,{out_start}),0,-{width}*pow((t-{out_start})/{leave},3))))")
    return (
        f"[{input_index}:v]scale={width}:{height}:flags=lanczos,format=rgba,"
        f"fade=t=in:st={delay}:d={enter}:alpha=1,"
        f"fade=t=out:st={out_start}:d={leave}:alpha=1[rank];"
        f"[v][rank]overlay=x='{x}':y=0:enable='between(t,{delay},{end})':eof_action=pass[out]"
    )


def subscribe_image(path: Path):
    """A quiet footer over reaction footage, leaving the meme as the focal point."""
    image = Image.new("RGBA", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((365, 608, 915, 672), radius=9,
                           fill=(10, 12, 16, 225), outline=(255, 255, 255, 45), width=1)
    draw.text((398, 626), "LIKE  +  SUBSCRIBE", font=font(27), fill=(248, 249, 251))
    # Small bell line icon rather than a full-screen faux interface.
    draw.arc((857, 623, 881, 649), 180, 360, fill=(240, 242, 247), width=2)
    draw.line((857,636,854,652,884,652,881,636), fill=(240, 242, 247), width=2)
    draw.ellipse((866, 655, 872, 661), fill=(240, 242, 247))
    image.save(path)


def embed_chapters(path: Path, chapters: list[dict], total: float, work: Path):
    def escaped(text):
        return str(text).replace("\\", "\\\\").replace("\n", " ").replace("\r", " ").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#")
    lines = [";FFMETADATA1"]
    for index, chapter in enumerate(chapters):
        start = round(chapter["start"]*1000)
        end = round((chapters[index+1]["start"] if index+1 < len(chapters) else total)*1000)
        if end > start:
            lines.extend(["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={end}", f"title={escaped(chapter['title'])}"])
    metadata = work / "chapters.ffmeta"
    metadata.write_text("\n".join(lines)+"\n")
    target = work / "chaptered.mp4"
    ffmpeg("-i", path, "-f", "ffmetadata", "-i", metadata, "-map", "0", "-map_metadata", "1", "-map_chapters", "1", "-c", "copy", "-movflags", "+faststart", target)
    target.replace(path)


def concat_file(paths: list[Path], listing: Path, durations: list[float] | None = None):
    # Paths are generated internally; apostrophe escaping follows FFmpeg concat syntax.
    entries = []
    for index, path in enumerate(paths):
        entries.append("file '" + str(path.resolve()).replace("'", "'\\''") + "'\n")
        if durations is not None:
            entries.append(f"duration {durations[index]:.9f}\n")
    listing.write_text("".join(entries))


def speech_video(source: Path, audio: Path, chosen: list[dict], output: Path, label: Path | None, work: Path, *, label_seconds: float = 3.5):
    work.mkdir(parents=True, exist_ok=True)
    width, height = export_size()
    total_frames = math.ceil(duration(audio) * FPS)
    remaining = total_frames
    clips = []
    for index, shot in enumerate(chosen):
        frames = remaining if index == len(chosen)-1 else min(remaining, max(1, round(shot["duration"] * FPS)))
        if frames <= 0:
            continue
        out = work / f"shot-{index:03}.mp4"
        filters = f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,crop={width}:{height},setsar=1,fps={FPS}"
        if not shot.get("image") and not shot.get("animation"):
            filters += ","+footage_grade()
        hold_frames = min(max(0, round(shot.get("freeze", 0)*FPS)), max(0, frames-1))
        if shot.get("still"):
            filters += f",trim=end_frame=1,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop={frames+1}"
        elif hold_frames:
            filters += f",trim=end_frame={frames-hold_frames},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop={hold_frames+1}"
        else:
            # A selected window can approach the source's last decoded frame. Hold
            # that last picture to cover narration instead of shortening the export.
            filters += f",setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop={frames+1}"
        if (shot.get("still") or hold_frames) and not shot.get("animation"):
            live_frames = 0 if shot.get("still") else frames-hold_frames
            filters += ","+_hold_motion(frames, live_frames, width, height, shot.get("still_motion", "zoom_in"))
        if shot.get("animation"):
            inputs = ["-stream_loop", "-1", "-i", shot["source"]]
        elif shot.get("image"):
            inputs = ["-loop", "1", "-framerate", FPS, "-i", shot["source"]]
        else:
            inputs = ["-ss", shot["start"], "-i", shot.get("source", source)]
        ffmpeg(*inputs, "-vf", filters+f",setpts=N/({FPS}*TB)", "-frames:v", frames, "-r", FPS,
               "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-threads", "2", out)
        clips.append(out)
        remaining -= frames
        if remaining <= 0:
            break
    if remaining > 0:
        raise PipelineError("Shot plan is shorter than its narration.")
    listing = work / "concat.txt"
    concat_file(clips, listing)
    silent = work / "silent.mp4"
    ffmpeg("-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", silent)
    seconds = total_frames / FPS
    inputs = ["-i", silent, "-i", audio]
    if label:
        inputs += ["-loop", "1", "-framerate", FPS, "-i", label]
        graph = f"[0:v]setpts=PTS-STARTPTS[v];{_label_graph(2, min(label_seconds, seconds), width, height)}"
        video_options = ["-filter_complex", graph, "-map", "[out]", "-c:v", "libx264",
                         "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", FPS, "-threads", "2"]
    else:
        video_options = ["-map", "0:v", "-c:v", "copy"]
    # Lossless intermediate audio avoids an independent AAC encoder delay at every
    # paragraph/preview boundary. Only the finished upload is encoded to AAC.
    ffmpeg(*inputs, *video_options, "-map", "1:a", "-af", "asetpts=PTS-STARTPTS,apad",
           "-t", seconds, *LOSSLESS_AUDIO, output)
    for clip in clips:
        clip.unlink(missing_ok=True)
    silent.unlink(missing_ok=True)


def trailer_video(source: Path, start: float, seconds: float, output: Path, *,
                  label: Path | None = None, label_seconds: float = 3.5, muted: bool = False,
                  still: bool = False, still_motion: str | None = None, reveal: bool = False,
                  audio: Path | None = None, reveal_style: str = "punch"):
    """A narration-free preview, optionally carrying the animated rank introduction."""
    frames = max(FPS, round(seconds * FPS))
    actual = frames / FPS
    width, height = export_size()
    inputs = ["-ss", start, "-i", source]
    has_audio = not muted and any(s.get("codec_type") == "audio" for s in probe(source).get("streams", []))
    audio_index = 0
    label_index = 1
    if has_audio and audio is not None:
        # Replacement audio for this window (for example, dialogue with the trailer score lowered).
        inputs += ["-i", audio]
        audio_index = 1
        label_index = 2
    elif not has_audio:
        inputs += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        audio_index = 1
        label_index = 2
    hold = "trim=end_frame=1,setpts=PTS-STARTPTS," if still else ""
    motion = _hold_motion(frames, 0, width, height, still_motion)+"," if still and still_motion else ""
    graph = (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height},setsar=1,{footage_grade()},setpts=PTS-STARTPTS,fps={FPS},"
        f"{hold}tpad=stop_mode=clone:stop={frames},{motion}{_reveal_filter(width, height, reveal_style) if reveal else ''}setpts=N/({FPS}*TB)[v];"
        f"[{audio_index}:a]asetpts=PTS-STARTPTS,afade=t=in:d=0.08,"
        f"afade=t=out:st={max(0,actual-0.15)}:d=0.15,loudnorm=I=-18:TP=-2:LRA=9,apad[a]"
    )
    video = "[v]"
    if label:
        inputs += ["-loop", "1", "-framerate", FPS, "-i", label]
        graph += ";"+_label_graph(label_index, min(label_seconds, actual), width, height)
        video = "[out]"
    ffmpeg(*inputs, "-filter_complex", graph, "-map", video, "-map", "[a]", "-t", actual,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", FPS, "-threads", "2",
           *LOSSLESS_AUDIO, output)


REVEAL_STYLES = ("zoom_in", "zoom_out")


def reveal_style_for(position: int, rank: int) -> str:
    """Alternate restrained, centered zooms without flashes or camera shake."""
    return REVEAL_STYLES[position % len(REVEAL_STYLES)]


GENRE_ENTRY = {"horror": ("black", 0.45), "thriller": ("black", 0.3), "drama": ("black", 0.35),
               "romance": ("black", 0.35), "documentary": ("black", 0.3)}


def _reveal_filter(width: int, height: int, style: str = "zoom_in") -> str:
    """Gently zoom across the preview. Slower genres also rise out of black (a short dip between
    entries); action and comedy keep a straight cut for pace."""
    entry = GENRE_ENTRY.get(os.getenv("VIDEO_GENRE", "").strip().lower())
    dip = f"fade=t=in:st=0:d={entry[1]}:color={entry[0]}," if entry else ""
    return _hold_motion(5*FPS, 0, width, height, style)+","+dip


def j_cut(part: Path, source: Path, source_start: float, *, lead: float = 0.6, gain: float = 0.55):
    """Let the next trailer clip's own sound rise under the last ``lead`` seconds of ``part``.

    The audio is taken from just before ``source_start``, so it continues seamlessly into
    the clip that follows. Video is copied; only the audio is remixed.
    """
    streams = probe(part).get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    # The clip's video length rules: the rebuilt audio is padded or trimmed to exactly match it,
    # so the final assembly never sees an audio/video length mismatch.
    length = float(video.get("duration") or duration(part))
    lead = min(lead, max(0.0, source_start), length * 0.4)
    if lead < 0.2 or not any(s.get("codec_type") == "audio" for s in probe(source).get("streams", [])):
        return False
    temp = part.with_name(part.stem + ".jcut" + part.suffix)
    delay = int((length - lead) * 1000)
    ffmpeg("-i", part, "-ss", source_start - lead, "-t", lead, "-i", source, "-filter_complex",
           f"[1:a]aresample=48000,aformat=channel_layouts=stereo,afade=t=in:d={lead},volume={gain},adelay={delay}:all=1[lead];"
           f"[0:a]aresample=48000,aformat=channel_layouts=stereo[base];"
           f"[base][lead]amix=inputs=2:duration=longest:normalize=0,apad=whole_dur={length:.4f},atrim=end={length:.4f}[a]",
           "-map", "0:v", "-map", "[a]", "-c:v", "copy", *LOSSLESS_AUDIO, temp)
    temp.replace(part)
    return True


OVERLAY_HEIGHT = 50  # tracker and up-next pill share one compact plate (in 1280x720 units)


def _overlay_plate(width: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Dark glass plate with the accent bar, the same look as the rank/title box."""
    canvas = Image.new("RGBA", (width, OVERLAY_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((0, 0, width - 1, OVERLAY_HEIGHT - 1), radius=12, fill=(9, 11, 16, 210),
                           outline=(255, 255, 255, 38), width=1)
    draw.rounded_rectangle((0, 0, 7, OVERLAY_HEIGHT - 1), radius=4, fill=ACCENT + (255,))
    return canvas, draw


def watermark_image(path: Path, text: str = "MOVIE RANKS"):
    """Small channel stamp for the top right corner of every video: white, slightly translucent."""
    face = brand_font("display", 26)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    spaced = " ".join(text.upper())  # light letter spacing
    width = round(probe.textlength(spaced, font=face)) + 16
    canvas = Image.new("RGBA", (width, 44), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((9, 24), spaced, font=face, anchor="lm", fill=(0, 0, 0, 90))  # soft shadow
    draw.text((8, 22), spaced, font=face, anchor="lm", fill=(255, 255, 255, 175))
    canvas.save(path)
    return canvas.size


def tracker_image(path: Path, total: int, current: int):
    """Compact countdown tracker (10 9 8 [7] 6 ... 1): done ranks dim, the current one in an accent circle."""
    size, gap = 30, 6
    width = 22 + total * size + (total - 1) * gap + 14
    canvas, draw = _overlay_plate(width)
    face = brand_font("display", 23)
    for slot, rank in enumerate(range(total, 0, -1)):
        x = 22 + slot * (size + gap) + size / 2
        y = OVERLAY_HEIGHT / 2
        if rank == current:
            draw.ellipse((x - size / 2, y - size / 2, x + size / 2, y + size / 2), fill=ACCENT + (255,))
            draw.text((x, y + 1), str(rank), font=face, anchor="mm", fill=(0, 0, 0))
        else:
            draw.text((x, y + 1), str(rank), font=face, anchor="mm", fill=(255, 255, 255, 85 if rank > current else 225))
    canvas.save(path)
    return canvas.size


def up_next_image(path: Path, rank: int, title: str | None):
    """'UP NEXT  #4  TITLE' on the tracker's plate; number one's reads 'UP NEXT  #1' with no title."""
    label_face, rank_face, title_face = brand_font("body-medium", 17), brand_font("display", 28), brand_font("title", 25)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    text = (title or "").upper()
    while probe.textlength(text, font=title_face) > 420:
        text = text.rstrip("…")[:-1].rstrip() + "…"
    width = round(22 + probe.textlength("UP NEXT", font=label_face) + 14 + probe.textlength(f"#{rank}", font=rank_face)
                  + (12 + probe.textlength(text, font=title_face) if text else 0) + 20)
    canvas, draw = _overlay_plate(width)
    middle, x = OVERLAY_HEIGHT / 2, 22
    draw.text((x, middle), "UP NEXT", font=label_face, anchor="lm", fill=(255, 255, 255, 190))
    x += probe.textlength("UP NEXT", font=label_face) + 14
    draw.text((x, middle + 1), f"#{rank}", font=rank_face, anchor="lm", fill=ACCENT + (255,))
    x += probe.textlength(f"#{rank}", font=rank_face) + 12
    if text:
        draw.text((x, middle + 1), text, font=title_face, anchor="lm", fill=(250, 250, 252))
    canvas.save(path)
    return canvas.size


def number_one_tease_image(path: Path, line: str = "STAY FOR NUMBER ONE"):
    """Full-frame overlay for the 3-second #1 teaser in the intro: a huge '#1' and a short line."""
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    import numpy as np
    ramp = np.clip(1 - np.linspace(0, 1, WIDTH) / 0.62, 0, 1) ** 1.3 * 190  # dark on the left, clear by 62%
    image.putalpha(Image.fromarray(np.repeat(ramp[None, :], HEIGHT, axis=0).astype(np.uint8)))
    draw = ImageDraw.Draw(image)
    face = brand_font("display", 300)
    draw.text((70, HEIGHT / 2 - 40), "#1", font=face, anchor="lm", fill=ACCENT + (255,), stroke_width=8, stroke_fill=(0, 0, 0))
    small = brand_font("title", 54)
    draw.text((78, HEIGHT / 2 + 150), line.upper(), font=small, anchor="lm", fill=(255, 255, 255), stroke_width=4, stroke_fill=(0, 0, 0))
    image.save(path)


def end_screen_image(path: Path, background: Image.Image | None, headline: str = "THANKS FOR WATCHING"):
    """Calm, uncluttered end-screen plate: blurred dark film frame and a small headline.

    The middle and lower areas stay clear for YouTube end-screen elements
    (video suggestions and the subscribe button) added in YouTube Studio.
    """
    from PIL import ImageEnhance, ImageFilter
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (8, 10, 14))
    if background is not None:
        frame = background.convert("RGB")
        scale = max(WIDTH / frame.width, HEIGHT / frame.height)
        frame = frame.resize((round(frame.width*scale), round(frame.height*scale)), Image.LANCZOS)
        left, top = (frame.width-WIDTH)//2, (frame.height-HEIGHT)//2
        frame = frame.crop((left, top, left+WIDTH, top+HEIGHT)).filter(ImageFilter.GaussianBlur(22))
        canvas = ImageEnhance.Brightness(frame).enhance(0.38)
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    face = brand_font("display", 58)
    draw.text((WIDTH//2, 92), headline, font=face, anchor="mm", fill=(250, 250, 252, 255))
    draw.rounded_rectangle((WIDTH//2-60, 132, WIDTH//2+60, 138), radius=3, fill=ACCENT + (255,))
    draw.text((WIDTH//2, 172), "More picks are waiting for you", font=brand_font("body-medium", 26), anchor="mm",
              fill=(205, 210, 220, 255))
    image = canvas.convert("RGBA")
    image.alpha_composite(overlay)
    image.convert("RGB").save(path)


def end_screen_video(image: Path, seconds: float, output: Path):
    """A silent, slowly drifting end screen with a lossless silent audio track."""
    frames = max(FPS, round(seconds * FPS))
    width, height = export_size()
    graph = (f"[0:v]scale={width}:{height}:flags=lanczos,setsar=1,fps={FPS},trim=end_frame=1,setpts=PTS-STARTPTS,"
             f"tpad=stop_mode=clone:stop={frames},"
             +_hold_motion(frames, 0, width, height, "zoom_in")+","
             +f"setpts=N/({FPS}*TB)[v];[1:a]apad[a]")
    ffmpeg("-loop", "1", "-framerate", FPS, "-i", image, "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
           "-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-frames:v", frames, "-t", frames/FPS,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", FPS, "-threads", "2",
           *LOSSLESS_AUDIO, output)


def _audio_stream(path: Path) -> dict:
    metadata = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_fmt,sample_rate,channels,channel_layout,bits_per_raw_sample,extradata,extradata_size",
        "-show_data", "-of", "json", path]))
    streams = metadata.get("streams", [])
    if not streams:
        raise PipelineError(f"No audio stream is available in {path.name}.")
    return streams[0]


def _canonical_audio(stream: dict) -> bool:
    # ALAC codec configuration is global in MP4. Identical codec names/rates
    # alone are insufficient: 16-bit packets cannot use a 24-bit stream header.
    return (stream.get("codec_name") == "alac" and stream.get("sample_fmt") == "s32p"
            and str(stream.get("sample_rate")) == "48000" and stream.get("channels") == 2
            and stream.get("channel_layout") == "stereo" and str(stream.get("bits_per_raw_sample")) == "24"
            and bool(stream.get("extradata")) and stream.get("extradata_size", 0) > 0)


def _normalized_audio_parts(parts: list[Path], work: Path) -> list[Path]:
    """Keep video bytes while making every audio codec configuration identical."""
    if not parts:
        raise PipelineError("Cannot assemble a video without media parts.")
    result, streams = [], []
    for index, value in enumerate(parts):
        path = Path(value)
        stream = _audio_stream(path)
        if not _canonical_audio(stream):
            normalized = work / f"part-{index:04}.mp4"
            ffmpeg("-xerror", "-err_detect", "explode", "-i", path, "-map", "0:v:0", "-map", "0:a:0",
                   "-c:v", "copy", "-af", "asetpts=PTS-STARTPTS", "-t", duration(path), *LOSSLESS_AUDIO, normalized)
            path, stream = normalized, _audio_stream(normalized)
        result.append(path)
        streams.append(stream)
    # A foreign ALAC encoder can use different predictor/configuration bytes
    # despite matching PCM parameters. Reencode that uncommon group uniformly.
    if len({stream.get("extradata") for stream in streams}) != 1:
        for index, path in enumerate(result):
            normalized = work / f"uniform-{index:04}.mp4"
            ffmpeg("-xerror", "-err_detect", "explode", "-i", path, "-map", "0:v:0", "-map", "0:a:0",
                   "-c:v", "copy", "-af", "asetpts=PTS-STARTPTS", "-t", duration(path), *LOSSLESS_AUDIO, normalized)
            result[index], streams[index] = normalized, _audio_stream(normalized)
    if not all(_canonical_audio(stream) for stream in streams) or len({stream.get("extradata") for stream in streams}) != 1:
        raise PipelineError("Intermediate audio codec configurations remain incompatible after lossless normalization.")
    return result


MAX_REPAIR_SECONDS = 3.0


def repair_part_timing(path: Path, work: Path) -> Path | None:
    """Make a clip's audio exactly as long as its video (pad with silence or trim).

    Small audio/video length differences can come from any rendering step; they are
    repaired instead of failing the whole video. Larger differences (over three seconds)
    mean something is really missing and are left for the strict check to report.
    """
    metadata = probe(path)
    video = next((s for s in metadata.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in metadata.get("streams", []) if s.get("codec_type") == "audio"), None)
    try:
        video_seconds = float(video["duration"])
        audio_seconds = float(audio["duration"]) if audio else 0.0
    except (TypeError, KeyError, ValueError):
        return None
    if not math.isfinite(video_seconds) or video_seconds <= 0 or abs(video_seconds - audio_seconds) > MAX_REPAIR_SECONDS:
        return None
    repaired = work / f"repaired-{path.stem}.mp4"
    if audio is None:
        ffmpeg("-i", path, "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "copy", "-t", f"{video_seconds:.4f}", *LOSSLESS_AUDIO, repaired)
    else:
        ffmpeg("-i", path, "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy", "-af",
               f"asetpts=PTS-STARTPTS,aresample=48000,apad=whole_dur={video_seconds:.4f},atrim=end={video_seconds:.4f}",
               *LOSSLESS_AUDIO, repaired)
    from .core import notice
    notice("Clip timing repaired", f"{path.name}: audio {audio_seconds:.2f}s adjusted to the video's {video_seconds:.2f}s.", "warning")
    return repaired


def _strict_audio_decode(path: Path):
    # Audio-only decoding is inexpensive even for long 4K exports. Treat packet
    # corruption as an error before a final encode can conceal it behind music.
    ffmpeg("-xerror", "-err_detect", "explode", "-i", path, "-map", "0:a:0", "-vn", "-f", "null", "-")


def validate_audio_video_duration(path: Path, expected_seconds: float | None = None, *, metadata: dict | None = None) -> float:
    """Require complete, aligned audio/video streams, including the ending hold."""
    metadata = probe(path) if metadata is None else metadata
    video = next((stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        raise PipelineError(f"Both audio and video streams are required in {path.name}.")
    try:
        video_start, audio_start = float(video.get("start_time", 0)), float(audio.get("start_time", 0))
        video_seconds, audio_seconds = float(video["duration"]), float(audio["duration"])
        total = float(metadata["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(f"Cannot verify complete audio/video duration in {path.name}.") from exc
    values = (video_start, audio_start, video_seconds, audio_seconds, total)
    if not all(math.isfinite(value) for value in values) or min(video_seconds, audio_seconds, total) <= 0:
        raise PipelineError(f"Invalid audio/video duration in {path.name}.")
    tolerance = AUDIO_ALIGNMENT_TOLERANCE
    if (abs(video_start-audio_start) > tolerance or
            abs((video_start+video_seconds)-(audio_start+audio_seconds)) > tolerance):
        raise PipelineError(f"Audio/video timing is incomplete in {path.name}: video {video_seconds:.3f}s, audio {audio_seconds:.3f}s.")
    if expected_seconds is not None and (not math.isfinite(expected_seconds) or expected_seconds <= 0 or
            any(abs(value-expected_seconds) > tolerance for value in (video_seconds, audio_seconds, total))):
        raise PipelineError(f"Export streams do not cover the complete planned edit in {path.name}.")
    return total




def measure_loudness(path: Path) -> dict:
    """Integrated loudness (LUFS) and true peak of a file's first audio stream."""
    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0",
                             "-af", "loudnorm=I=-14:TP=-1:LRA=11:print_format=json", "-f", "null", "-"],
                            capture_output=True, text=True, check=False)
    match = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", result.stderr, re.S)
    if result.returncode or not match:
        raise PipelineError("Could not measure the export's loudness.")
    data = json.loads(match[-1])
    return {"integrated": float(data["input_i"]), "true_peak": float(data["input_tp"]), "range": float(data["input_lra"])}


def normalize_loudness(path: Path, seconds: float, work: Path, target: float = -14.0) -> dict:
    """Linear gain to YouTube's playback loudness with a -1 dBTP ceiling; the video stream is copied."""
    measured = measure_loudness(path)
    if not math.isfinite(measured["integrated"]):
        return {"measured": measured, "gain_db": 0.0}
    gain = max(-12.0, min(12.0, target - measured["integrated"]))
    if abs(gain) < 0.5:
        return {"measured": measured, "gain_db": 0.0}
    adjusted = work / "loudness-adjusted.mp4"
    ffmpeg("-i", path, "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy",
           "-af", f"volume={gain:.2f}dB,alimiter=limit=0.891:attack=5:release=60:level=0",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-t", seconds, "-movflags", "+faststart", adjusted)
    adjusted.replace(path)
    return {"measured": measured, "gain_db": round(gain, 2)}


def finish(parts: list[Path], output: Path, work: Path, action: bool, spec: DurationSpec | None = None, *, ending_hold: float = 3.0, music_theme: str | None = None,
           timeline: list[dict] | None = None, top_title: str | None = None,
           overlays: list[tuple[Path, float, float]] | None = None) -> tuple[float, float]:
    spec = spec or DurationSpec()
    work.mkdir(parents=True, exist_ok=True)
    report = {"status": "pending", "canonical_audio": {"codec": "alac", "sample_format": "s32p",
              "bits_per_sample": 24, "sample_rate": 48000, "channels": 2, "channel_layout": "stereo"},
              "part_count": len(parts), "normalized_part_count": 0,
              "concatenated_strict_decode": "pending", "final_strict_decode": "pending"}

    def record(stage, **values):
        report.update(stage=stage, **values)
        save_json(work / "audio-assembly.json", report)

    def failed(exc):
        record(report["stage"], status="failed", error=type(exc).__name__, detail=str(exc)[-500:])

    record("normalizing")
    listing = work / "final-concat.txt"
    # Independent H.264 encodes may use different B-frame reorder delays.
    # Stream-copying their packets into one MP4 can produce backwards DTS.
    # Decode the parts sequentially in the existing final encode instead;
    # keep only the compatible, lossless audio joined before that encode.
    with tempfile.TemporaryDirectory(prefix="audio-normalized-", dir=work) as directory:
        normalization = Path(directory)
        combined_audio = normalization / "joined-audio.m4a"
        try:
            normalized = _normalized_audio_parts(parts, normalization)
            lengths = []
            for position, path in enumerate(normalized):
                metadata = probe(path)
                try:
                    validate_audio_video_duration(path, metadata=metadata)
                except PipelineError:
                    repaired = repair_part_timing(path, normalization)
                    if repaired is None:
                        raise
                    path = normalized[position] = repaired
                    metadata = probe(path)
                    validate_audio_video_duration(path, metadata=metadata)
                # Container durations round to milliseconds; accumulating those
                # across many parts can shift captions and narration. Use the
                # precise stream ends. An audio tail or VFR video track can
                # legitimately extend a frame beyond the other stream's header.
                lengths.append(max(float(stream["duration"])+max(0, float(stream.get("start_time", 0)))
                                   for stream in metadata["streams"] if stream["codec_type"] in {"audio", "video"}))
            original = sum(lengths)
            record("concatenating", assembly_mode="direct-video-encode",
                   normalized_part_count=sum(Path(old) != new for old, new in zip(parts, normalized)),
                   part_durations=lengths)
            concat_file(normalized, listing, durations=lengths)
            ffmpeg("-xerror", "-f", "concat", "-safe", "0", "-i", listing,
                   "-map", "0:a:0", "-vn", "-c:a", "copy", combined_audio)
            record("checking-concatenated-audio", concatenated_strict_decode="running")
            _strict_audio_decode(combined_audio)
            record("checking-concatenated-duration", concatenated_strict_decode="passed")
            metadata = probe(combined_audio)
            audio = next(stream for stream in metadata["streams"] if stream["codec_type"] == "audio")
            audio_start, audio_seconds = float(audio.get("start_time", 0)), float(audio["duration"])
            if (not all(math.isfinite(value) for value in (audio_start, audio_seconds))
                    or abs(audio_start) > AUDIO_ALIGNMENT_TOLERANCE
                    or abs(audio_start+audio_seconds-original) > AUDIO_ALIGNMENT_TOLERANCE):
                raise PipelineError("Joined narration does not cover the complete planned edit.")
            record("timing", joined_audio_seconds=audio_seconds)
            speed = final_tempo(original, spec, ending_hold)
            if not spec.automatic and not spec.minimum <= original/speed+ending_hold <= spec.maximum:
                print("Keeping the complete video at a natural pace outside the soft runtime target.", flush=True)
            final_duration = original/speed + ending_hold
            music = work / "music.wav"
            music_bed(music, final_duration + 1, action, **({"theme": music_theme} if music_theme else {}))
            effects = None
            if timeline:
                from .sound_design import effects_track, gain_curve, shape_music
                shape_music(music, gain_curve(timeline, speed, final_duration + 1, original/speed, top_title))
                effects = work / "effects.wav"
                record("sound-design", effect_events=effects_track(effects, timeline, speed, final_duration + 1))
            mix_inputs = "[voice][duck][fx]amix=inputs=3" if effects else "[voice][duck]amix=inputs=2"
            overlay_inputs, overlay_graph, last = [], "", "base"
            first_overlay = 4 if effects else 3
            out_width, out_height = export_size()
            factor = out_width / WIDTH  # overlay images are drawn in 1280x720 units
            for number, (image, begin, end, *corner) in enumerate(overlays or []):
                # Small plates on the final clock, fading in and out: the countdown tracker and the
                # 'up next' pill sit lower left, where the rank/title box appears (default: top right).
                overlay_inputs += ["-loop", "1", "-framerate", FPS, "-i", image]
                start, stop = begin / speed, end / speed
                fade = min(0.25, max(0.05, (stop - start) / 4))
                if corner and corner[0] == "lower-left":
                    x, y = f"{round(56 * factor)}", f"H-h-{round(52 * factor)}"
                else:
                    x, y = f"W-w-{round(36 * factor)}", f"{round(30 * factor)}"
                overlay_graph += (f"[{first_overlay + number}:v]format=rgba,scale=trunc(iw*{factor:.4f}/2)*2:-2,"
                                  f"fade=t=in:st={start:.3f}:d={fade:.3f}:alpha=1,"
                                  f"fade=t=out:st={max(start, stop - fade):.3f}:d={fade:.3f}:alpha=1[p{number}];"
                                  f"[{last}][p{number}]overlay=x={x}:y={y}:eof_action=pass:"
                                  f"enable='between(t,{start:.3f},{stop:.3f})'[o{number}];")
                last = f"o{number}"
            graph = (
                # Honor legitimate display gaps first, then rebuild the clock.
                # Packet reorder delays never become final output timestamps.
                f"[0:v]fps=fps={FPS}:start_time=0,setpts=N/({FPS}*TB*{speed}),fps={FPS},tpad=stop_mode=clone:stop_duration={ending_hold},fade=t=out:st={final_duration-1.2}:d=1.2[base];"
                + overlay_graph + f"[{last}]null[v];"
                # Preserve declared silence before changing the sample clock.
                f"[1:a]asetpts=PTS-STARTPTS,aresample=48000:async=1:first_pts=0:min_hard_comp=0.0001,atempo={speed},"
                f"asetpts=N/SR/TB,apad=pad_dur={ending_hold},asplit=2[voice][control];"
                "[2:a]equalizer=f=220:t=q:w=0.8:g=-3,equalizer=f=2400:t=q:w=0.7:g=-4[bed];"
                "[bed][control]sidechaincompress=threshold=0.025:ratio=6:attack=15:release=300[duck];"
                + ("[3:a]asetpts=PTS-STARTPTS[fx];" if effects else "")
                # Rebuild the sample clock after mixing: the muxer then always receives
                # strictly increasing audio timestamps, whatever FFmpeg build is installed.
                + f"{mix_inputs}:duration=longest:normalize=0,aresample=48000:async=1:first_pts=0,asetpts=N/SR/TB,"
                  f"volume=1.26,alimiter=limit=0.84:attack=5:release=50:level=0:latency=1,afade=t=out:st={final_duration-1.2}:d=1.2,asetpts=N/SR/TB[a]"
            )
            record("encoding", planned_duration=final_duration, music_theme=music_theme or ("action" if action else "drama"))
            for source, name in ((combined_audio, "edit-without-music.wav"), (music, "background-only.wav")):
                ffmpeg("-i", source, "-t", min(20, original), "-c:a", "pcm_s16le", work / name)
            ffmpeg("-xerror", "-err_detect", "explode", "-f", "concat", "-safe", "0", "-i", listing,
                   "-i", combined_audio, "-i", music, *(["-i", effects] if effects else []), *overlay_inputs,
                   "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
                   "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", FPS,
                   "-threads", "2", "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", final_duration, output)
            record("loudness", loudness=normalize_loudness(output, final_duration, work))
            record("checking-final-audio", final_strict_decode="running")
            _strict_audio_decode(output)
            metadata = probe(output)
            record("checking-final-duration", final_strict_decode="passed", final_stream_durations={
                stream["codec_type"]: stream.get("duration") for stream in metadata.get("streams", [])
                if stream.get("codec_type") in ("audio", "video")})
            actual_duration = validate_audio_video_duration(output, final_duration)
        except Exception as exc:
            failed(exc)
            raise
        for name, start, length in (("finished-mix.wav", 0, min(20, actual_duration)),
                                    ("ending-music-only.wav", max(0, actual_duration-ending_hold), ending_hold)):
            if length > 0:
                ffmpeg("-ss", start, "-i", output, "-t", length, "-vn", "-ar", "48000",
                       "-ac", "2", "-c:a", "pcm_s16le", work / name)
        record("complete", status="passed", final_duration=actual_duration)
        music.unlink(missing_ok=True)
        (work / "effects.wav").unlink(missing_ok=True)
        return actual_duration, speed



def thumbnails(trailers: list[Path], shots: list[list[dict]], phrases: list[str], output: Path, **options):
    """Thumbnails from official TMDB stills when available; generated/local designs remain the fallback."""
    from .thumbnails import build_thumbnails
    films = options.pop("films", None)
    def completed(result):
        from .thumbnail_fallback import complete
        complete(output, films, Path(options.get("work") or output / ".thumbnail-work"))
        return result
    if films and os.getenv("THUMBNAIL_SOURCE", "stills").lower() != "ai":
        from .still_thumbnails import build_still_thumbnails
        from .thumbnail_studio import build as build_studio
        work = Path(options.get("work") or output / ".thumbnail-work")
        try:
            return completed(build_studio(films, trailers, shots, phrases, output, work))
        except Exception as exc:  # noqa: BLE001 -- the headline designs remain available
            import traceback

            from .core import notice
            print(f"Studio thumbnails unavailable ({type(exc).__name__}: {exc}); using headline still designs.", flush=True)
            notice("Thumbnail studio fell back", f"{type(exc).__name__}: {str(exc)[:300]}\n{traceback.format_exc()[-1200:]}", "warning")
        try:
            return completed(build_still_thumbnails(films, phrases, output, work))
        except Exception as exc:  # noqa: BLE001 -- stills are preferred, never required
            print(f"Still-based thumbnails unavailable ({type(exc).__name__}); using the generated designs.", flush=True)
    return completed(build_thumbnails(trailers, shots, phrases, output, **options))
