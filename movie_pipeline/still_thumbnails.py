"""Thumbnails from official TMDB stills: recognizable films, faces and bold short text.

Three options for YouTube Studio's thumbnail test:
1. Hero: number one's strongest still, face-aware framing, headline on the dark side.
2. Cutout: a person cut out of number one's still (BiRefNet when available) over
   darkened stills from the next picks, with a large headline.
3. Netflix-style: an alternative still with the red ribbon "N" badge (an editorial
   style choice, never a claim of Netflix availability).
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from .core import PipelineError, key, save_json
from .render import ACCENT, HEIGHT, WIDTH, brand_font, wrapped

TMDB_IMAGE = "https://image.tmdb.org/t/p/w1280"
CUTOUT_MODELS = ("birefnet-portrait", "birefnet-general", "u2net_human_seg", "u2net")


def fetch_still(path: str, cache: Path) -> Image.Image | None:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{key('tmdb-still', path)}.jpg"
    if not target.exists():
        try:
            response = requests.get(TMDB_IMAGE + path, timeout=30)
            response.raise_for_status()
            Image.open(io.BytesIO(response.content)).verify()
            target.write_bytes(response.content)
        except (requests.RequestException, OSError, ValueError):
            return None
    try:
        return Image.open(target).convert("RGB")
    except OSError:
        return None


_FACES = None


def faces(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Face boxes in the image's own coordinates (OpenCV Haar cascade; no identity)."""
    global _FACES
    if _FACES is None:
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        _FACES = cv2.CascadeClassifier(str(path)) if path.exists() else False
    if not _FACES:
        return []
    scale = 640 / image.width
    small = cv2.cvtColor(np.asarray(image.resize((640, max(1, round(image.height * scale))))), cv2.COLOR_RGB2GRAY)
    found = list(_FACES.detectMultiScale(small, 1.1, 5, minSize=(28, 28)))
    if not found:
        # Dramatic stills often show faces at an angle: try profile detection, both directions.
        profile = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_profileface.xml"))
        if not profile.empty():
            found = list(profile.detectMultiScale(small, 1.1, 5, minSize=(28, 28)))
            flipped = profile.detectMultiScale(cv2.flip(small, 1), 1.1, 5, minSize=(28, 28))
            found += [(small.shape[1] - x - w, y, w, h) for x, y, w, h in flipped]
    return [(round(x / scale), round(y / scale), round(w / scale), round(h / scale)) for x, y, w, h in found]


def still_score(image: Image.Image) -> float:
    gray = cv2.cvtColor(np.asarray(image.resize((480, 270))), cv2.COLOR_RGB2GRAY)
    face_area = max((w * h for _, _, w, h in faces(image)), default=0) / (image.width * image.height)
    sharpness = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 800) / 800
    contrast = min(float(gray.std()) / 60, 1)
    exposure = 1 - abs(float(gray.mean()) - 110) / 110
    return face_area * 10 + sharpness * 0.8 + contrast * 0.6 + exposure * 0.4


def best_stills(film: dict, cache: Path, limit: int = 4) -> list[Image.Image]:
    images = [image for image in (fetch_still(path, cache) for path in film.get("backdrops", [])[:limit]) if image]
    return sorted(images, key=still_score, reverse=True)


def grade(image: Image.Image) -> Image.Image:
    image = ImageEnhance.Contrast(image).enhance(1.14)
    image = ImageEnhance.Color(image).enhance(1.18)
    return image.filter(ImageFilter.UnsharpMask(radius=2, percent=70, threshold=2))


def cover(image: Image.Image, focus_x: float = 0.5, zoom: float = 1.0) -> Image.Image:
    """Fill 1280x720, placing horizontal focus point ``focus_x`` (0..1 of the source) where possible."""
    scale = max(WIDTH / image.width, HEIGHT / image.height) * zoom
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    left = round(min(max(0, resized.width * focus_x - WIDTH * 0.5), resized.width - WIDTH))
    top = (resized.height - HEIGHT) // 2
    return resized.crop((left, top, left + WIDTH, top + HEIGHT))


def framed(image: Image.Image, face_side: str = "right") -> Image.Image:
    """Put the main face on one side, leaving the other side for the headline."""
    boxes = faces(image)
    if not boxes:
        return cover(image, 0.5)
    x, _, w, _ = max(boxes, key=lambda b: b[2] * b[3])
    center = (x + w / 2) / image.width
    target = 0.68 if face_side == "right" else 0.32
    zoom = 1.18
    # Shift the crop so the face lands near the target third.
    return cover(image, min(1, max(0, center + (0.5 - target) / zoom)), zoom)


def vignette(canvas: Image.Image, side: str | None = "left", strength: int = 215) -> Image.Image:
    shade = Image.new("L", canvas.size, 0)
    draw = ImageDraw.Draw(shade)
    for i in range(WIDTH):
        if side == "left":
            alpha = max(0, strength * (1 - i / (WIDTH * 0.62)))
        elif side == "right":
            alpha = max(0, strength * (1 - (WIDTH - i) / (WIDTH * 0.62)))
        else:
            alpha = 0
        draw.line((i, 0, i, HEIGHT), fill=round(alpha))
    edge = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(edge).rectangle((0, HEIGHT - 170, WIDTH, HEIGHT), fill=120)
    edge = edge.filter(ImageFilter.GaussianBlur(70))
    mask = Image.fromarray(np.maximum(np.asarray(shade), np.asarray(edge)))
    dark = Image.new("RGBA", canvas.size, (0, 0, 0, 255))
    dark.putalpha(mask)
    return Image.alpha_composite(canvas.convert("RGBA"), dark)


def headline(canvas: Image.Image, phrase: str, box: tuple[int, int, int, int], *, align: str = "left"):
    """Huge condensed caps, black stroke, last word in the accent colour."""
    words = " ".join(phrase.upper().split()[:4]) or "WATCH THESE"
    left, top, width, height = box
    draw = ImageDraw.Draw(canvas)
    size = 168
    while size > 56:
        face = brand_font("display", size)
        lines = wrapped(draw, words, face, width)
        line_height = round(size * 1.02)
        if len(lines) <= 3 and line_height * len(lines) <= height and all(draw.textlength(l, font=face) <= width for l in lines):
            break
        size -= 4
    face = brand_font("display", size)
    lines = wrapped(draw, words, face, width)
    line_height = round(size * 1.02)
    y = top + (height - line_height * len(lines)) // 2
    accent_line = len(lines) - 1
    for index, line in enumerate(lines):
        line_width = draw.textlength(line, font=face)
        x = left if align == "left" else left + (width - line_width) / 2
        fill = ACCENT if index == accent_line and len(lines) > 1 else (255, 255, 255)
        if index == accent_line and len(lines) == 1 and " " in line:
            head, last = line.rsplit(" ", 1)
            draw.text((x, y), head + " ", font=face, fill=(255, 255, 255), stroke_width=8, stroke_fill=(0, 0, 0))
            x += draw.textlength(head + " ", font=face)
            line, fill = last, ACCENT
        draw.text((x, y), line, font=face, fill=fill, stroke_width=8, stroke_fill=(0, 0, 0))
        y += line_height


def badge(canvas: Image.Image, text: str):
    draw = ImageDraw.Draw(canvas)
    face = brand_font("display", 44)
    width = draw.textlength(text, font=face)
    draw.rounded_rectangle((38, 30, 38 + width + 40, 98), radius=12, fill=ACCENT)
    draw.text((58, 64), text, font=face, anchor="lm", fill=(12, 12, 14))


def netflix_badge(canvas: Image.Image):
    from .thumbnails import netflix_badge as ribbon
    ribbon(canvas)


class Cutter:
    def __init__(self):
        self.session = None
        self.model = None
        self.tried = False

    def cut(self, image: Image.Image, crop: bool = True) -> Image.Image | None:
        if not self.tried:
            self.tried = True
            try:
                from rembg import new_session
            except ImportError:
                return None
            for name in CUTOUT_MODELS:
                try:
                    self.session = new_session(name, providers=["CPUExecutionProvider"])
                    self.model = name
                    break
                except Exception:  # noqa: BLE001 -- try the next free model
                    continue
        if self.session is None:
            return None
        try:
            from rembg import remove
            result = remove(image, session=self.session).convert("RGBA")
        except Exception:  # noqa: BLE001 -- optional cutout
            return None
        alpha = np.asarray(result.getchannel("A"))
        coverage = float((alpha > 128).mean())
        bbox = result.getchannel("A").point(lambda v: 255 if v > 128 else 0).getbbox()
        if not bbox or not 0.04 <= coverage <= 0.65 or (bbox[3] - bbox[1]) < image.height * 0.45:
            return None
        return result.crop(bbox) if crop else result


def hero(still: Image.Image, phrase: str, count: int) -> Image.Image:
    canvas = vignette(grade(framed(still, "right")), "left")
    headline(canvas, phrase, (52, 130, 600, 520))
    badge(canvas, f"TOP {count}")
    return canvas


def cutout_design(subject: Image.Image | None, background: list[Image.Image], fallback: Image.Image,
                  phrase: str, count: int) -> tuple[Image.Image, str]:
    if subject is None:
        canvas = vignette(grade(framed(fallback, "right")), "left", 230)
        headline(canvas, phrase, (52, 150, 620, 500))
        badge(canvas, f"TOP {count}")
        return canvas, "still-only"
    canvas = Image.new("RGBA", (WIDTH, HEIGHT), (10, 10, 14, 255))
    panels = background[:2] or [fallback]
    panel_width = WIDTH // len(panels)
    for index, panel in enumerate(panels):
        full = cover(panel)
        offset = (WIDTH - panel_width) // 2
        piece = full.crop((offset, 0, offset + panel_width, HEIGHT))
        piece = ImageEnhance.Brightness(piece.filter(ImageFilter.GaussianBlur(6))).enhance(0.45)
        canvas.paste(piece, (index * panel_width, 0))
    subject = ImageEnhance.Contrast(subject).enhance(1.1)
    subject = ImageOps.contain(subject, (760, round(HEIGHT * 1.08)), Image.LANCZOS)
    x, y = WIDTH - subject.width - 10, HEIGHT - subject.height + 20
    glow = Image.new("RGBA", subject.size, ACCENT + (0,))
    glow.putalpha(subject.getchannel("A").filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.GaussianBlur(12)))
    canvas.alpha_composite(glow, (x, y))
    canvas.alpha_composite(subject, (x, y))
    canvas = vignette(canvas, "left", 170)
    headline(canvas, phrase, (52, 140, 600, 500))
    badge(canvas, f"TOP {count}")
    return canvas, "cutout"


def netflix_style(still: Image.Image, phrase: str) -> Image.Image:
    canvas = vignette(grade(framed(still, "right")), None)
    shade = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(shade).rectangle((0, HEIGHT - 260, WIDTH, HEIGHT), fill=(0, 0, 0, 150))
    canvas = Image.alpha_composite(canvas, shade.filter(ImageFilter.GaussianBlur(40)))
    headline(canvas, phrase, (52, HEIGHT - 285, 760, 230))
    netflix_badge(canvas)
    return canvas


def save(canvas: Image.Image, target: Path):
    image = canvas.convert("RGB")
    for quality in (92, 85, 78, 70):
        image.save(target, quality=quality, optimize=True)
        if target.stat().st_size < 2_000_000:
            return


def build_still_thumbnails(films: list[dict], phrases: list[str], output: Path, work: Path) -> list[dict]:
    cache = work / "stills"
    ranked = [best_stills(film, cache) for film in films[:3]]
    if not ranked or not ranked[0]:
        raise PipelineError("No TMDB still is available for number one.")
    count = len(films)
    phrases = (list(phrases) + ["WATCH THESE", "DON'T MISS", "NEXT WATCH"])[:3]
    top = ranked[0]
    others = [stills[0] for stills in ranked[1:] if stills]
    report = []

    canvas = hero(top[0], phrases[0], count)
    save(canvas, output / "thumbnail-01.jpg")
    report.append({"file": "thumbnail-01.jpg", "style": "hero-still", "film": films[0]["title"]})

    cutter = Cutter()
    subject = cutter.cut(top[0])
    canvas, method = cutout_design(subject, others, top[1] if len(top) > 1 else top[0], phrases[1], count)
    save(canvas, output / "thumbnail-02.jpg")
    report.append({"file": "thumbnail-02.jpg", "style": method, "cutout_model": cutter.model})

    alternative = top[1] if len(top) > 1 else (others[0] if others else top[0])
    save(netflix_style(alternative, phrases[2]), output / "thumbnail-03.jpg")
    report.append({"file": "thumbnail-03.jpg", "style": "netflix-style", "netflix_badge": True})
    save_json(work / "thumbnail-report.json", report)
    print(f"Thumbnails built from official TMDB stills ({method} option 2).", flush=True)
    return report
