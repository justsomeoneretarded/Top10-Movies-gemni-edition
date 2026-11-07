"""Readable synchronized like/subscribe/bell interaction, with brief UI sounds."""
from __future__ import annotations

import math
import re
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .core import duration, ffmpeg
from .render import FPS, LOSSLESS_AUDIO, brand_font


def action_times(text: str, cues: list[dict], seconds: float) -> list[float]:
    times = []
    for word, fallback in (("like", .15), ("subscribe", .4), ("bell", .7)):
        cue = next((cue for cue in cues if re.search(r"\b"+word+r"\b", cue["text"], re.IGNORECASE)), None)
        if cue:
            match = re.search(r"\b"+word+r"\b", cue["text"], re.IGNORECASE)
            # Captions are short phrases, so use the word's proportional location.
            when = cue["start"]+(cue["end"]-cue["start"])*match.start()/max(1,len(cue["text"]))
        else:
            when = seconds*fallback
        times.append(min(max(.1, when), max(.1, seconds-.4)))
    times[1] = max(times[0]+.15, times[1])
    times[2] = min(max(times[1]+.45, times[2]), max(.1, seconds-.2))
    return times


UI_WIDTH, UI_HEIGHT = 1000, 200
SCALE = 2  # draw at 2x, then downsample for smooth edges
RED, BLUE, GREY = (230, 33, 23), (62, 166, 255), (72, 76, 86)


def _ease_out_back(x: float) -> float:
    x = min(1.0, max(0.0, x))
    c = 1.70158
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


def _ease_in_out(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def _pop(t: float, at: float, length: float = 0.35, amount: float = 0.35) -> float:
    """1 + a damped bump just after a click."""
    x = (t - at) / length
    return 1.0 if not 0 <= x <= 1 else 1 + amount * math.sin(math.pi * x) * (1 - x)


def _thumb(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float, filled: bool, color):
    s = size / 100
    def pt(x, y):
        return (cx + (x - 50) * s, cy + (y - 50) * s)
    cuff = [pt(8, 44), pt(26, 44), pt(26, 92), pt(8, 92)]
    hand = [pt(32, 46), pt(46, 44), pt(56, 14), pt(64, 10), pt(72, 16), pt(70, 40), pt(90, 40),
            pt(96, 48), pt(90, 88), pt(84, 92), pt(32, 92)]
    width = max(2, round(7 * s))
    if filled:
        draw.polygon(cuff, fill=color)
        draw.polygon(hand, fill=color)
    else:
        draw.line(cuff + [cuff[0]], fill=color, width=width, joint="curve")
        draw.line(hand + [hand[0]], fill=color, width=width, joint="curve")


def _bell_layer(size: int, filled: bool, color) -> Image.Image:
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    s = size / 100
    width = max(2, round(7 * s))
    body = [(22 * s, 74 * s), (28 * s, 64 * s), (28 * s, 44 * s), (36 * s, 24 * s), (50 * s, 18 * s),
            (64 * s, 24 * s), (72 * s, 44 * s), (72 * s, 64 * s), (78 * s, 74 * s)]
    if filled:
        d.polygon(body, fill=color)
    else:
        d.line(body + [body[0]], fill=color, width=width, joint="curve")
    d.ellipse((43 * s, 76 * s, 57 * s, 90 * s), fill=color)
    d.ellipse((46 * s, 9 * s, 54 * s, 17 * s), fill=color)
    return layer


def _cursor(draw: ImageDraw.ImageDraw, x: float, y: float, size: float, pressed: bool):
    k = size / 40 * (0.85 if pressed else 1.0)
    shape = [(0, 0), (0, 34), (9, 26), (16, 40), (22, 37), (15, 24), (27, 24)]
    points = [(x + px * k, y + py * k) for px, py in shape]
    draw.polygon([(px + 2 * k, py + 2 * k) for px, py in points], fill=(0, 0, 0, 110))
    draw.polygon(points, fill=(255, 255, 255, 255), outline=(10, 10, 12, 255))


def interaction_frame(t: float, times: list[float]) -> Image.Image:
    """One frame of the animated like / subscribe / bell sequence (transparent RGBA)."""
    like_at, subscribe_at, bell_at = times
    W, H = UI_WIDTH * SCALE, UI_HEIGHT * SCALE
    image = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    appear = like_at - 0.6
    leave = bell_at + 1.1
    entrance = _ease_out_back((t - appear) / 0.45)
    exit_progress = _ease_in_out((t - leave) / 0.35)
    if t < appear or exit_progress >= 1:
        return image.resize((UI_WIDTH, UI_HEIGHT), Image.LANCZOS)
    offset = (1 - entrance) * H * 0.9 + exit_progress * H
    alpha = max(0.0, min(1.0, (t - appear) / 0.2)) * (1 - exit_progress)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    top = 30 * SCALE + offset
    plate = (40 * SCALE, top, (UI_WIDTH - 40) * SCALE, top + 140 * SCALE)
    draw.rounded_rectangle((plate[0] + 6, plate[1] + 10, plate[2] + 6, plate[3] + 10), radius=70 * SCALE, fill=(0, 0, 0, 90))
    draw.rounded_rectangle(plate, radius=70 * SCALE, fill=(18, 18, 22, 236), outline=(255, 255, 255, 40), width=2 * SCALE)
    cy = top + 70 * SCALE

    # Like button: thumbs-up that fills blue and pops, with a particle burst.
    like_x = 140 * SCALE
    liked = t >= like_at
    size = 74 * SCALE * _pop(t, like_at)
    _thumb(draw, like_x, cy, size, liked, BLUE + (255,) if liked else (240, 240, 244, 255))
    burst = (t - like_at) / 0.5
    if 0 <= burst <= 1:
        for k in range(10):
            angle = 2 * math.pi * k / 10
            radius = (40 + 60 * burst) * SCALE
            dot = (1 - burst) * 7 * SCALE
            px, py = like_x + math.cos(angle) * radius, cy + math.sin(angle) * radius
            color = (BLUE if k % 2 else RED) + (round(255 * (1 - burst)),)
            draw.ellipse((px - dot, py - dot, px + dot, py + dot), fill=color)

    # Subscribe button: red pill, ripple on click, becomes grey "SUBSCRIBED" with a check.
    sub_center = 500 * SCALE
    subscribed = t >= subscribe_at + 0.12
    press = 0.94 if subscribe_at - 0.05 <= t <= subscribe_at + 0.1 else 1.0
    bw, bh = 330 * SCALE * press * _pop(t, subscribe_at + 0.12, 0.3, 0.08), 84 * SCALE * press
    box = (sub_center - bw / 2, cy - bh / 2, sub_center + bw / 2, cy + bh / 2)
    draw.rounded_rectangle(box, radius=bh / 2, fill=(GREY if subscribed else RED) + (255,))
    ripple = (t - subscribe_at) / 0.55
    if 0 <= ripple <= 1:
        r = ripple * 190 * SCALE
        draw.ellipse((sub_center - r, cy - r * 0.45, sub_center + r, cy + r * 0.45),
                     outline=(255, 255, 255, round(170 * (1 - ripple))), width=4 * SCALE)
    label = "SUBSCRIBED" if subscribed else "SUBSCRIBE"
    face = brand_font("display", 40 * SCALE)
    text_width = draw.textlength(label, font=face)
    text_x = sub_center - text_width / 2 + (16 * SCALE if subscribed else 0)
    draw.text((text_x, cy), label, font=face, anchor="lm", fill=(255, 255, 255, 255))
    if subscribed:
        cx0 = text_x - 34 * SCALE
        draw.line([(cx0, cy), (cx0 + 10 * SCALE, cy + 11 * SCALE), (cx0 + 28 * SCALE, cy - 12 * SCALE)],
                  fill=(255, 255, 255, 255), width=6 * SCALE, joint="curve")

    # Bell: swings with a damped wobble and sends out ring waves once tapped.
    bell_x = 860 * SCALE
    rung = t >= bell_at
    bell = _bell_layer(round(84 * SCALE), rung, (255, 214, 64, 255) if rung else (240, 240, 244, 255))
    since = t - bell_at
    angle = 22 * math.exp(-since * 3.2) * math.sin(since * 26) if 0 <= since <= 1.6 else 0
    bell = bell.rotate(angle, resample=Image.BICUBIC, center=(bell.width / 2, bell.height * 0.12))
    layer.alpha_composite(bell, (round(bell_x - bell.width / 2), round(cy - bell.height / 2)))
    if 0 <= since <= 0.9:
        for wave_index in range(2):
            phase = since / 0.9 - wave_index * 0.2
            if 0 <= phase <= 1:
                r = (52 + 40 * phase) * SCALE
                fade = round(200 * (1 - phase))
                for sign in (-1, 1):
                    start_angle = -40 if sign > 0 else 140
                    draw.arc((bell_x - r, cy - r, bell_x + r, cy + r), start_angle, start_angle + 80,
                             fill=(255, 214, 64, fade), width=4 * SCALE)

    # Cursor glides between the three buttons and presses each one.
    targets = [(like_x + 18 * SCALE, cy + 14 * SCALE, like_at), (sub_center + 120 * SCALE, cy + 10 * SCALE, subscribe_at),
               (bell_x + 12 * SCALE, cy + 16 * SCALE, bell_at)]
    start_point = (W * 0.98, top + 160 * SCALE, appear + 0.25)
    path = [start_point] + targets
    cursor = None
    for (x0, y0, t0), (x1, y1, t1) in zip(path, path[1:]):
        travel_start = max(t0 + 0.12, t1 - 0.45)
        if t <= t1:
            k = _ease_in_out((t - travel_start) / max(0.05, t1 - travel_start)) if t >= travel_start else 0.0
            cursor = (x0 + (x1 - x0) * k, y0 + (y1 - y0) * k, abs(t - t1) <= 0.08)
            break
    if cursor is None and t <= leave:
        x, y, last = targets[-1]
        cursor = (x, y, abs(t - last) <= 0.08)
    if cursor and t >= appear + 0.25:
        _cursor(draw, cursor[0], cursor[1], 44 * SCALE, cursor[2])

    if alpha < 1:
        layer.putalpha(layer.getchannel("A").point(lambda v: round(v * alpha)))
    image.alpha_composite(layer)
    return image.resize((UI_WIDTH, UI_HEIGHT), Image.LANCZOS)


def _ui_sounds(times: list[float], seconds: float) -> np.ndarray:
    """Original synthesized UI sounds: a soft pop (like), a click (subscribe), a two-tone chime (bell)."""
    rate = 48000
    samples = np.zeros((round(seconds * rate), 2), dtype=np.float32)

    def place(sound, when):
        start = round(when * rate)
        stop = min(len(samples), start + len(sound))
        if stop > start:
            samples[start:stop] += sound[:stop - start, None]

    t = np.arange(round(0.12 * rate)) / rate
    pop = 0.06 * np.sin(2 * np.pi * (420 + 900 * t / 0.12) * t) * np.exp(-t / 0.03)
    click = 0.05 * np.sign(np.sin(2 * np.pi * 2400 * t)) * np.exp(-t / 0.006)
    t2 = np.arange(round(0.9 * rate)) / rate
    chime = (0.035 * (np.sin(2 * np.pi * 1568 * t2) + 0.6 * np.sin(2 * np.pi * 2349 * t2)
                      + 0.25 * np.sin(2 * np.pi * 3136 * t2)) * np.exp(-t2 / 0.28))
    place(pop.astype(np.float32), times[0])
    place(click.astype(np.float32), times[1])
    place(chime.astype(np.float32), times[2])
    place(chime.astype(np.float32) * 0.5, times[2] + 0.16)
    return samples


def add_cta_animation(video: Path, text: str, cues: list[dict], work: Path):
    work.mkdir(parents=True, exist_ok=True)
    seconds = duration(video)
    times = action_times(text, cues, seconds)
    for frame in range(math.ceil(seconds*FPS)):
        interaction_frame(frame/FPS, times).save(work / f"ui-{frame:04}.png")
    samples = _ui_sounds(times, seconds)
    sound = work / "interaction.wav"
    with wave.open(str(sound), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes((samples*32767).astype("<i2").tobytes())
    temporary = work / "with-interaction.mp4"
    # Scale the overlay relative to the output, preserving readability at 4K.
    visible_start = max(0, times[0]-.7)
    visible_end = min(seconds, times[-1]+1.6)
    graph = (f"[1:v][0:v]scale2ref=w=iw*0.62:h=ow*{UI_HEIGHT}/{UI_WIDTH}[ui][base];"
             f"[base][ui]overlay=(W-w)/2:H-h-H*0.045:shortest=1:enable='between(t,{visible_start},{visible_end})'[v]")
    mixed, silent = work / "mixed.wav", work / "interaction-video.mp4"
    try:
        # FFmpeg 6 amix duration=first can end early with unequal input packet
        # sizes. Both inputs cover the full edit: drain both and mix independently,
        # then mux it without -shortest or an output -t. Never conceal missing
        # samples by padding the result after the fact.
        audio_graph = (
            "[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,asetpts=N/SR/TB[voice];"
            "[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,asetpts=N/SR/TB[sfx];"
            "[voice][sfx]amix=inputs=2:duration=longest:normalize=0[a]"
        )
        ffmpeg("-i", video, "-i", sound, "-filter_complex", audio_graph,
               "-map", "[a]", "-vn", "-c:a", "pcm_s24le", mixed)
        ffmpeg("-i", video, "-framerate", FPS, "-i", work / "ui-%04d.png",
               "-filter_complex_threads", "1", "-filter_complex", graph,
               "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-threads", "2", "-frames:v", round(seconds*FPS), silent)
        ffmpeg("-i", silent, "-i", mixed, "-map", "0:v", "-map", "1:a",
               "-c:v", "copy", *LOSSLESS_AUDIO, temporary)
        from .render import _strict_audio_decode, validate_audio_video_duration
        _strict_audio_decode(temporary)
        validate_audio_video_duration(temporary, seconds)
        temporary.replace(video)
    finally:
        for image in work.glob("ui-*.png"):
            image.unlink(missing_ok=True)
        sound.unlink(missing_ok=True)
        mixed.unlink(missing_ok=True)
        silent.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
