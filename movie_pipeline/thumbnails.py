"""Complete cinematic scenes, candidate selection, and local editorial fallback."""
from __future__ import annotations

import io
import random
import time
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from .core import PipelineError, key, save_json
from .render import HEIGHT, WIDTH, font, wrapped
from .shots import get_frame

PALETTES = [("#071327", "#136dba", "#ff8739"), ("#130b2a", "#7931c1", "#32eedb"), ("#200b12", "#ac2334", "#ffcf43")]


def default_concepts(title: str, brief: str, count: int) -> list[dict]:
    if count < 1:
        raise ValueError("At least one film is required for thumbnails")
    topic = (title + " " + brief).casefold()
    if any(w in topic for w in ("horror", "haunted", "scary", "terror")):
        themes = ["terrified adult close-up, ominous silhouette in a blue shadowy corridor",
                  "frightened adult on the right, haunted house and red moon, clear dark space on the left",
                  "two tense adults inside a rain-soaked car, threatening silhouette in a foggy forest"]
    elif any(w in topic for w in ("action", "fight", "chase")):
        themes = ["determined adult action hero in an intense close-up, sparks and a blurred chase behind",
                  "adult action hero on the right, explosive urban pursuit, dark clear space on the left",
                  "two tense adult protagonists beside an armored vehicle, smoke and dramatic backlight"]
    else:
        themes = ["expressive adult protagonist close-up, mysterious cinematic architecture and deep atmosphere",
                  "intrigued adult on the right, surreal fractured space, clear dark space on the left",
                  "two adult protagonists, a glowing horizon and atmospheric cinematic depth"]
    return [{"background_prompt": f"Complete photorealistic widescreen cinematic scene: {theme}. Coherent natural lighting, realistic skin, strong focal point, exaggerated emotional intensity, no text, no lettering, no logos. Illustrative genre artwork, not a claimed movie still.",
             "subject_indices": [i % count] if i == 0 else list(dict.fromkeys([i % count, (i+1) % count])),
             "layout": layout} for i, (theme, layout) in enumerate(zip(themes, ("hero", "duo", "diagonal")))]


def normalize_concepts(concepts, title: str, brief: str, count: int) -> list[dict]:
    defaults = default_concepts(title, brief, count)
    if not isinstance(concepts, list):
        return defaults
    for i, concept in enumerate(concepts[:3]):
        if not isinstance(concept, dict):
            continue
        prompt = concept.get("background_prompt")
        indices = concept.get("subject_indices")
        if isinstance(prompt, str) and 20 <= len(prompt) <= 1200:
            defaults[i]["background_prompt"] = prompt
        if isinstance(indices, list):
            clean = list(dict.fromkeys(n for n in indices if isinstance(n, int) and not isinstance(n, bool) and 0 <= n < count))[:2]
            if clean:
                defaults[i]["subject_indices"] = clean
        if concept.get("layout") in ("hero", "duo", "diagonal"):
            defaults[i]["layout"] = concept["layout"]
    return defaults


def netflix_badge(canvas: Image.Image):
    """Small editorial red ribbon N; never an availability verification."""
    mark = Image.new("RGBA", (125, 175))
    draw = ImageDraw.Draw(mark)
    draw.polygon([(0,0),(36,0),(36,158),(0,164)], fill="#ad0610")
    draw.polygon([(78,0),(114,0),(114,164),(78,158)], fill="#ad0610")
    draw.polygon([(0,0),(36,0),(114,164),(78,158)], fill="#e50914")
    canvas.alpha_composite(mark, (40,30))


class Pollinations:
    """Anonymous Classic API only: no keys, billing, or paid fallback."""
    def __init__(self, cache: Path):
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.last_request = None
        self.unavailable = False

    def generate(self, prompt: str, seed: int) -> Image.Image | None:
        path = self.cache / f"{key('pollinations-scene-v2', prompt, seed)}.jpg"
        if path.exists():
            try:
                with Image.open(path) as image:
                    return image.convert("RGB")
            except OSError:
                path.unlink(missing_ok=True)
        if self.unavailable:
            return None
        if self.last_request is not None:
            time.sleep(max(0, 45 - (time.monotonic()-self.last_request)))
        self.last_request = time.monotonic()
        try:
            # Prompts are public to this external service; never include tokens or private references.
            url = "https://image.pollinations.ai/prompt/" + quote(prompt, safe="")
            with requests.get(url, params={"width": 1024, "height": 576, "seed": seed, "safe": "true"}, timeout=(10, 55), stream=True) as response:
                if response.status_code != 200 or not response.headers.get("Content-Type", "").lower().startswith("image/"):
                    raise ValueError("Provider unavailable or rate limited")
                data = bytearray()
                for chunk in response.iter_content(65536):
                    data.extend(chunk)
                    if len(data) > 10_000_000:
                        raise ValueError("Image response too large")
            with Image.open(io.BytesIO(data)) as image:
                if min(image.size) < 256 or image.width*image.height > 16_000_000:
                    raise ValueError("Invalid image dimensions")
                result = image.convert("RGB")
            result.save(path, quality=95)
            return result
        except (requests.RequestException, OSError, ValueError):
            self.unavailable = True
            print("Free thumbnail background service unavailable; using original local artwork for remaining concepts.", flush=True)
            return None


def local_background(variant: int, seed: int) -> Image.Image:
    """Original procedural artwork; never a raw trailer frame or a paid model call."""
    dark, mid, accent = PALETTES[variant]
    from PIL import ImageColor
    colors = [np.array(ImageColor.getrgb(c), dtype=float) for c in (dark, mid)]
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    glow = np.clip(1 - np.sqrt(((xx-WIDTH*.7)/WIDTH)**2 + ((yy-HEIGHT*.35)/HEIGHT)**2)*1.6, 0, 1)
    pixels = colors[0][None, None, :] + glow[:, :, None]*(colors[1]-colors[0])[None, None, :]
    canvas = Image.fromarray(pixels.astype(np.uint8)).convert("RGBA")
    lights = Image.new("RGBA", canvas.size)
    draw = ImageDraw.Draw(lights)
    rng = random.Random(seed)
    for _ in range(28):
        x, y = rng.randrange(WIDTH), rng.randrange(HEIGHT)
        length = rng.randrange(35, 250)
        draw.line((x, y, x+length, y-length//2), fill=accent, width=rng.randrange(1, 4))
    for radius in (90, 155, 240):
        draw.ellipse((850-radius, 260-radius, 850+radius, 260+radius), outline=accent, width=3)
    return Image.alpha_composite(canvas, lights.filter(ImageFilter.GaussianBlur(2)))


def subject_frame(path: Path, shots: list[dict], variant: int) -> Image.Image:
    # Prefer sharp, well-lit frames containing a face, instead of the most chaotic frame.
    candidates = sorted(shots, key=lambda s: s["motion"], reverse=True)[:8]
    face_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    faces = cv2.CascadeClassifier(str(face_path)) if face_path.exists() else None
    ranked = []
    for shot in candidates:
        try:
            frame = get_frame(path, (shot["start"]+shot["end"])/2)
        except PipelineError:
            continue
        gray = cv2.cvtColor(np.array(frame.resize((480, 270))), cv2.COLOR_RGB2GRAY)
        found = faces.detectMultiScale(gray, 1.1, 4) if faces is not None else []
        face_area = max((w*h for _, _, w, h in found), default=0)/(480*270)
        sharpness = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1000)/1000
        score = face_area*12 + sharpness - abs(float(gray.mean())-115)/255
        ranked.append((score, frame))
    ranked.sort(key=lambda x: x[0], reverse=True)
    if not ranked:
        raise PipelineError(f"No readable thumbnail reference frames in {path.name}.")
    return ranked[min(variant, len(ranked)-1)][1]


class Cutouts:
    def __init__(self, use_model: bool):
        self.use_model = use_model
        self.session = None
        self.checked = False

    def extract(self, image: Image.Image) -> tuple[Image.Image, str]:
        if self.use_model:
            try:
                from rembg import new_session, remove
                if not self.checked:
                    self.checked = True
                    self.session = new_session("u2net", providers=["CPUExecutionProvider"])
                if self.session is not None:
                    result = remove(image.convert("RGB"), session=self.session).convert("RGBA")
                    if result.getchannel("A").getextrema() == (0, 255) and result.getchannel("A").getbbox():
                        return result.crop(result.getchannel("A").getbbox()), "rembg-u2net"
            except Exception:  # noqa: BLE001 -- optional model/download failure must not discard the completed video.
                self.session = None
                self.checked = True
                print("AI subject extraction unavailable; using local foreground segmentation.", flush=True)
                self.use_model = False
        small = ImageOps.contain(image.convert("RGB"), (512, 512))
        rgb = np.array(small)
        height, width = rgb.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        try:
            cv2.grabCut(rgb, mask, (width//10, height//20, width*8//10, height*9//10), np.zeros((1,65)), np.zeros((1,65)), 3, cv2.GC_INIT_WITH_RECT)
            alpha = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
        except cv2.error:
            alpha = np.zeros((height, width), np.uint8)
        if np.count_nonzero(alpha) < width*height*0.02:
            # Soft editorial portrait mask when foreground/background cannot be separated.
            soft = Image.new("L", (width, height))
            ImageDraw.Draw(soft).ellipse((width*.08, height*.02, width*.92, height*.98), fill=255)
        else:
            soft = Image.fromarray(alpha)
        result = image.convert("RGBA")
        result.putalpha(soft.resize(image.size).filter(ImageFilter.GaussianBlur(1)))
        return result.crop(result.getchannel("A").getbbox()), "local-segmentation"


def paste_subject(canvas: Image.Image, subject: Image.Image, box: tuple, accent: str, tilt: float):
    subject = ImageEnhance.Contrast(subject).enhance(1.2)
    subject = ImageEnhance.Color(subject).enhance(1.15)
    subject = ImageOps.contain(subject, (box[2], box[3]), method=Image.Resampling.LANCZOS)
    subject = subject.rotate(tilt, resample=Image.Resampling.BICUBIC, expand=True)
    x, y = box[0]+(box[2]-subject.width)//2, box[1]+box[3]-subject.height
    silhouette = Image.new("RGBA", subject.size, accent)
    silhouette.putalpha(subject.getchannel("A").filter(ImageFilter.MaxFilter(13)).filter(ImageFilter.GaussianBlur(9)))
    canvas.alpha_composite(silhouette, (x, y))
    canvas.alpha_composite(subject, (x, y))


def select_concepts(raw):
    """Pick one plan per style by explicit composition cues, not prose length."""
    if not isinstance(raw, list) or len(raw) < 6:
        return raw
    selected = []
    for offset, cues in ((0, ("close-up", "expressive", "lighting", "depth")),
                         (2, ("right", "left", "empty", "lighting")),
                         (4, ("two", "upper-left", "lighting", "depth"))):
        def score(concept, cues=cues):
            if not isinstance(concept, dict):
                return -100
            prompt = str(concept.get("background_prompt", "")).casefold()
            return sum(cue in prompt for cue in cues) - sum(prompt == c.get("background_prompt", "").casefold() for c in selected)*10
        best = max(raw[offset:offset+2], key=score)
        selected.append(best if isinstance(best, dict) else {})
    return selected


def candidate_quality(image: Image.Image) -> float:
    """Cheap visual checks at phone size; not a face/anatomy or CTR judge."""
    rgb = np.asarray(image.convert("RGB").resize((320, 180)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    contrast = min(float(gray.std()) / 55, 1.0)
    sharpness = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 500, 1.0)
    clipped = float(np.mean((gray < 8) | (gray > 247)))
    return round(contrast * .55 + sharpness * .35 + (1-clipped) * .1, 5)


def scene_prompt(plan: dict, variant: int) -> str:
    placement = ("One dominant expressive adult subject, face fully inside frame, cinematic close-up.",
                 "Expressive adult subject on the right; left 45 percent simple and dark for a headline.",
                 "At most two expressive adults; keep upper-left corner simple for editorial branding.")[variant]
    return (plan["background_prompt"] + " " + placement +
            " Complete coherent 16:9 photorealistic cinematic scene, directional lighting, strong depth,"
            " realistic skin texture, striking emotional stakes, one clear focal point readable at phone size."
            " No collage, no borders, no typography, no letters, no logos, no watermarks requested."
            " Illustrative artwork inspired by the video's genre; do not invent a movie scene or casting claim.")


def choose_scene(provider, plan: dict, variant: int, seed: int, previous: list) -> tuple:
    candidates = []
    for attempt in range(2):
        image = provider.generate(scene_prompt(plan, variant), seed + attempt) if provider else None
        if image is None:
            break
        signature = np.asarray(image.convert("RGB").resize((32, 18)), dtype=float)
        duplicate = any(float(np.mean(np.abs(signature-old))) < 4 for old in previous)
        score = candidate_quality(image) - (1 if duplicate else 0)
        candidates.append((score, attempt, image, signature, duplicate))
    if not candidates:
        return None, [], None
    best = max(candidates, key=lambda c: c[0])
    # An essentially blank image or duplicate should not displace the local composition.
    usable = best[0] >= .15 and not best[4]
    if usable:
        previous.append(best[3])
    return (best[2] if usable else None,
            [{"seed": seed+c[1], "visual_score": c[0], "duplicate": c[4]} for c in candidates],
            seed+best[1] if usable else None)


def build_thumbnails(trailers: list[Path], shots: list[list[dict]], phrases: list[str], output: Path, *, concepts=None, title="", brief="", work: Path | None = None, use_api=False, use_cutouts=False):
    plans = normalize_concepts(concepts, title, brief, len(trailers))
    output.mkdir(parents=True, exist_ok=True)
    work = work or output / ".thumbnail-work"
    work.mkdir(parents=True, exist_ok=True)
    provider = Pollinations(work / "backgrounds") if use_api else None
    cutouts = Cutouts(use_cutouts)
    report = []
    previous = []
    for variant, plan in enumerate(plans):
        seed = int(key(title, plan, variant)[:8], 16) % 2_000_000_000
        backdrop, candidates, selected_seed = choose_scene(provider, plan, variant, seed, previous)
        generated = backdrop is not None
        if generated:
            # Preserve the full background aspect ratio, including provider branding.
            canvas = Image.new("RGBA", (WIDTH, HEIGHT), PALETTES[variant][0])
            backdrop = ImageOps.contain(backdrop, canvas.size).convert("RGBA")
            canvas.alpha_composite(backdrop, ((WIDTH-backdrop.width)//2, (HEIGHT-backdrop.height)//2))
        else:
            canvas = local_background(variant, seed)
        # Reserve the provider's bottom-right corner from subsequent compositing.
        corner = canvas.crop((WIDTH-200, HEIGHT-75, WIDTH, HEIGHT)) if generated else None
        indices = plan["subject_indices"]
        if plan["layout"] == "hero":
            indices = indices[:1]
            boxes = [(180, -25, 1080, 765)] if variant == 0 else [(610, 15, 690, 690)]
            text_width = 590
        elif plan["layout"] == "duo":
            boxes = [(0, 35, 570, 670), (710, 25, 570, 680)]
            text_width = 1100
        else:
            boxes = [(540, -20, 760, 745), (180, 100, 480, 590)]
            text_width = 560
        methods = []
        for index, box in ([] if generated else zip(indices, boxes)):
            try:
                frame = subject_frame(trailers[index], shots[index], variant)
            except PipelineError:
                print("Thumbnail reference frame unavailable; retaining the designed background and layout.", flush=True)
                methods.append("reference-unavailable")
                continue
            subject, method = cutouts.extract(frame)
            paste_subject(canvas, subject, box, PALETTES[variant][2], (-5 if variant == 2 else (3 if variant == 1 else 0)))
            methods.append(method)
        if variant == 1:
            # One bold editorial option; two alternatives keep the cinematic imagery uncluttered.
            shade = Image.new("RGBA", canvas.size)
            draw = ImageDraw.Draw(shade)
            if generated or plan["layout"] != "duo":
                text_width = 500
                draw.rectangle((0, 180, 590, HEIGHT), fill=(0,0,0,175))
            else:
                draw.rectangle((0, 450, WIDTH, HEIGHT), fill=(0,0,0,155))
            canvas = Image.alpha_composite(canvas, shade)
            draw = ImageDraw.Draw(canvas)
            phrase = " ".join(phrases[variant].upper().split()[:5])
            size = 104
            while size > 20:
                lines = wrapped(draw, phrase, font(size), text_width)
                if len(lines) <= (2 if plan["layout"] == "duo" else 3) and all(draw.textbbox((0,0), line, font=font(size))[2] <= text_width for line in lines):
                    break
                size -= 2
            y = HEIGHT-len(lines)*(size+8)-40 if plan["layout"] == "duo" else max(215, 450-len(lines)*(size+8)//2)
            for line in lines:
                draw.text((55,y), line, font=font(size), fill="white", stroke_width=5, stroke_fill="black")
                y += size+8
            draw.rounded_rectangle((40, 30, 225, 100), radius=12, fill="#e50914")
            draw.text((60, 42), f"TOP {len(trailers)}", font=font(32), fill="white")
        branded = variant == 2
        if branded:
            netflix_badge(canvas)
        if corner is not None:
            canvas.paste(corner, (WIDTH-200, HEIGHT-75))
        target = output / f"thumbnail-{variant+1:02}.jpg"
        canvas.convert("RGB").save(target, quality=91, optimize=True)
        if target.stat().st_size >= 2_000_000:
            canvas.convert("RGB").save(target, quality=80, optimize=True)
        report.append({"file": target.name, "candidates": candidates, "selected_seed": selected_seed, "composition": "complete-generated-scene" if generated else "local-cutout", "background": "pollinations-classic" if generated else "original-local-artwork", "cutouts": methods, "concept": plan, "style": "bold-text" if variant == 1 else "cinematic-minimal", "netflix_badge": branded})
    save_json(work / "thumbnail-report.json", report)
