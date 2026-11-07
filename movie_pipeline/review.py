"""One-page quality report (review.html) so the video can be checked in a minute before upload.

Contents: a contact sheet of the whole video, and automatic checks for black frames,
the same shot shown twice far apart, sudden loudness spikes, captions that sit on
silence (drift), and a score for every thumbnail (vision model when available, plus
contrast at phone size). Every check is advisory: the report never blocks delivery.
"""
from __future__ import annotations

import base64
import html
import io
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .core import duration
from .render import brand_font


def _frames(video: Path, step: float, width: int = 320) -> list[tuple[float, Image.Image]]:
    """Frames every ``step`` seconds, decoded in one pass at low resolution."""
    height = round(width * 9 / 16 / 2) * 2
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vf",
                             f"fps=1/{step},scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            capture_output=True, timeout=900, check=False)
    size = width * height * 3
    count = len(result.stdout) // size
    return [(index * step, Image.frombytes("RGB", (width, height), result.stdout[index * size:(index + 1) * size]))
            for index in range(count)]


def _dhash(image: Image.Image) -> int | None:
    gray = np.asarray(image.convert("L").resize((9, 8)), dtype=np.int16)
    if gray.std() < 6 or gray.mean() < 14:
        return None
    return sum(int(bit) << i for i, bit in enumerate((gray[:, 1:] > gray[:, :-1]).reshape(-1)))


def black_frames(video: Path) -> list[tuple[float, float]]:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(video), "-vf", "scale=320:-2,blackdetect=d=0.4:pix_th=0.08",
                             "-an", "-f", "null", "-"], capture_output=True, text=True, timeout=900, check=False)
    return [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", result.stderr)]


def repeated_shots(frames: list[tuple[float, Image.Image]], ignore_after: float, gap: float = 25.0,
                   ignore: list[tuple[float, float]] | None = None) -> list[tuple[float, float]]:
    """The same shot shown twice far apart, outside the intro, cold open and outro recap
    (those deliberately preview or recap clips from the reviews)."""
    seen: list[tuple[float, int]] = []
    repeats = []
    for when, image in frames:
        if when >= ignore_after:
            break
        if any(low <= when <= high for low, high in ignore or []):
            continue
        value = _dhash(image)
        if value is None:
            continue
        match = next((earlier for earlier, old in seen if when - earlier >= gap and (value ^ old).bit_count() <= 4), None)
        if match is not None and not any(abs(match - a) < 3 and abs(when - b) < 3 for a, b in repeats):
            repeats.append((match, when))
        seen.append((when, value))
    return repeats


def loudness_profile(video: Path, hop: float = 0.4) -> np.ndarray:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "8000",
                             "-f", "s16le", "-"], capture_output=True, timeout=900, check=False)
    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768
    size = int(8000 * hop)
    if samples.size < size:
        return np.zeros(0)
    frames = samples[: samples.size // size * size].reshape(-1, size)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-6)


def loudness_spikes(profile: np.ndarray, hop: float = 0.4, margin: float = 10.0) -> list[float]:
    if not profile.size:
        return []
    active = profile[profile > -50]
    if not active.size:
        return []
    typical = float(np.median(active))
    spikes = [i * hop for i, value in enumerate(profile) if value > typical + margin]
    merged = []
    for when in spikes:
        if not merged or when - merged[-1] > 3:
            merged.append(when)
    return merged


def silent_captions(cues: list[dict], profile: np.ndarray, hop: float = 0.4) -> list[dict]:
    """Captions shown while (almost) nothing is audible: a sign of caption drift."""
    if not profile.size:
        return []
    floor = float(np.median(profile[profile > -60])) - 22 if (profile > -60).any() else -60
    drifted = []
    for cue in cues:
        low, high = int(float(cue["start"]) / hop), int(float(cue["end"]) / hop) + 1
        window = profile[low:high]
        if window.size >= 2 and float(np.mean(window > floor)) < 0.35:
            drifted.append(cue)
    return drifted


def contact_sheet(frames: list[tuple[float, Image.Image]], columns: int = 6, limit: int = 60) -> Image.Image:
    step = max(1, len(frames) // limit)
    chosen = frames[::step][:limit]
    if not chosen:
        return Image.new("RGB", (320, 180))
    width, height = chosen[0][1].size
    rows = (len(chosen) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * (height + 22)), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)
    face = brand_font("body", 15)
    for index, (when, image) in enumerate(chosen):
        x, y = (index % columns) * width, (index // columns) * (height + 22)
        sheet.paste(image, (x, y))
        draw.text((x + 6, y + height + 3), f"{int(when // 60)}:{int(when % 60):02}", font=face, fill=(220, 220, 220))
    return sheet


def _data_uri(image: Image.Image, quality: int = 72) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def thumbnail_scores(output: Path, genre: str) -> list[dict]:
    from . import scenes
    from .thumbnail_studio import GENRE_PROFILES, phone_check
    profile = GENRE_PROFILES.get(genre, {})
    rows = []
    for path in sorted(output.glob("thumbnail-0*.jpg")):
        with Image.open(path) as image:
            image = image.convert("RGB")
            verdict = scenes.rate_thumbnail(image, profile.get("label", ""), profile.get("hint", "")) if scenes.enabled() else None
            rows.append({"file": path.name, "image": image.copy(), "phone_contrast": round(phone_check(image), 1),
                         "vision": verdict})
    return rows


def _time(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02}"


def build_review(output: Path, cues: list[dict], chapters: list[dict], genre: str, *, ending_seconds: float = 20.0,
                 stills_films: list[str] | None = None, montage: list[tuple[float, float]] | None = None) -> dict:
    video = output / "final-video.mp4"
    length = duration(video)
    frames = _frames(video, 2.0)
    blacks = [(a, b) for a, b in black_frames(video) if a < length - ending_seconds]
    repeats = repeated_shots(frames, length - ending_seconds, ignore=montage)
    profile = loudness_profile(video)
    spikes = loudness_spikes(profile)
    drift = silent_captions(cues, profile)
    thumbs = thumbnail_scores(output, genre)
    summary = {"seconds": round(length, 1), "black_frames": len(blacks), "repeated_shots": len(repeats),
               "loudness_spikes": len(spikes), "captions_on_silence": len(drift),
               "thumbnails": [{k: v for k, v in row.items() if k != "image"} for row in thumbs]}

    def status(count: int, warn: int = 1) -> str:
        return "ok" if count < warn else "check"

    stills_films = stills_films or []
    summary["stills_fallback"] = stills_films
    items = [
        ("Films shown with stills (no trailer could be downloaded)", status(len(stills_films)),
         ", ".join(html.escape(name) for name in stills_films) or "none"),
        ("Black frames", status(len(blacks)), ", ".join(f"{_time(a)}–{_time(b)}" for a, b in blacks[:12]) or "none"),
        ("Same shot shown twice inside the reviews", status(len(repeats)), ", ".join(f"{_time(a)} and {_time(b)}" for a, b in repeats[:12]) or "none"),
        ("Loudness spikes", status(len(spikes), 3), ", ".join(_time(t) for t in spikes[:15]) or "none"),
        ("Captions over silence (drift)", status(len(drift), 3),
         "; ".join(f"{_time(float(c['start']))} “{html.escape(str(c['text'])[:50])}”" for c in drift[:8]) or "none"),
    ]
    thumb_cards = "".join(
        f"<figure><img src='{_data_uri(row['image'].resize((480, 270)), 80)}' alt='{row['file']}'>"
        f"<figcaption><b>{row['file']}</b><br>phone contrast {row['phone_contrast']}"
        + (f"<br>genre fit {row['vision']['genre_fit']}/10 · score {row['vision']['score']}/10"
           f"{' · <span class=bad>logo/text seen</span>' if row['vision'].get('logo') else ''}" if row['vision'] else "")
        + "</figcaption></figure>" for row in thumbs)
    chapter_list = "".join(f"<li>{_time(float(c['start']))} {html.escape(str(c['title']))}</li>" for c in chapters)
    rows = "".join(f"<tr><td>{name}</td><td class='{state}'>{state}</td><td>{detail}</td></tr>" for name, state, detail in items)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Video review</title><style>
:root{{--bg:#111214;--fg:#eceae6;--muted:#9a9894;--ok:#5fc27e;--warn:#f0b43c;--card:#1b1c1f}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;padding:24px 16px;max-width:1100px;margin:auto}}
h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:17px;margin:28px 0 10px}}p.sub{{color:var(--muted);margin:0 0 16px}}
table{{border-collapse:collapse;width:100%}}td{{padding:8px 10px;border-bottom:1px solid #2a2b2f;vertical-align:top}}
td.ok{{color:var(--ok);font-weight:600}}td.check{{color:var(--warn);font-weight:600}}
.thumbs{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}}
figure{{margin:0;background:var(--card);border-radius:8px;overflow:hidden}}figure img{{width:100%;display:block}}
figcaption{{padding:8px 10px;font-size:13px;color:var(--muted)}}.bad{{color:#ff6b6b}}
img.sheet{{width:100%;border-radius:8px}}ul{{columns:2;padding-left:18px}}
</style></head><body>
<h1>Video review</h1><p class="sub">{_time(length)} · genre {html.escape(genre)} · checks are automatic and advisory</p>
<h2>Checks</h2><table>{rows}</table>
<h2>Thumbnails</h2><div class="thumbs">{thumb_cards}</div>
<h2>Chapters</h2><ul>{chapter_list}</ul>
<h2>Contact sheet</h2><img class="sheet" src="{_data_uri(contact_sheet(frames))}" alt="Frames across the whole video">
</body></html>"""
    (output / "review.html").write_text(page, encoding="utf-8")
    print(f"Review report: {summary['black_frames']} black, {summary['repeated_shots']} repeated, "
          f"{summary['loudness_spikes']} spikes, {summary['captions_on_silence']} drifting captions.", flush=True)
    return summary
