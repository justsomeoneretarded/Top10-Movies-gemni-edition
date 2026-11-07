from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import subprocess
import unicodedata
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .core import PipelineError, duration, save_json

MAX_MOVING_SECONDS = 18.0  # long, uninterrupted stretches of a trailer read as one coherent scene
# Longest moving shot per delivery: hype lines cut quickly, tense or warm lines breathe.
PACE_LIMITS = {"hype": 7.0, "normal": 12.0, "dramatic": 16.0, "calm": 14.0}
ACTION_PACE_LIMITS = {"hype": 6.0, "normal": 10.0, "dramatic": 14.0, "calm": 12.0}


def pace_limit(delivery: str, action: bool = False) -> float:
    return (ACTION_PACE_LIMITS if action else PACE_LIMITS).get(delivery, PACE_LIMITS["normal"])
HOLD_COOLDOWN_SECONDS = 30.0  # freeze-frame holds are rare: running footage reads better
HOLD_COOLDOWN_BEATS = 3


def _zscore(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    spread = float(values.std())
    return (values - float(values.mean())) / spread if spread > 1e-6 else np.zeros_like(values)


def moving_limit(selector) -> float:
    """Keep moving excerpts near twelve seconds after a global tempo correction."""
    try:
        speed = float(getattr(selector, "playback_speed", 1))
    except (TypeError, ValueError):
        speed = 1.0
    if not math.isfinite(speed):
        speed = 1.0
    return MAX_MOVING_SECONDS if speed >= 1 else MAX_MOVING_SECONDS*max(.85, speed)-.04


def frame_text(image: Image.Image) -> str:
    """Optional bounded OCR of a candidate title card; no service or new AI model."""
    command = shutil.which("tesseract")
    if not command:
        return ""
    buffer = io.BytesIO()
    image = image.copy().convert("RGB")
    image.thumbnail((1280,720))
    image.save(buffer,format="PNG")
    try:
        result = subprocess.run([command,"stdin","stdout","--psm","6"],input=buffer.getvalue(),
                                capture_output=True,check=False,timeout=4)
        return result.stdout.decode("utf-8",errors="ignore")[:4000] if result.returncode == 0 else ""
    except (OSError,subprocess.TimeoutExpired):
        return ""


PROMO_TEXT = re.compile(
    r"subscri|playlist|insider|facebook|twitter|instagram|tiktok|youtube|follow\s+us|watch\s+more|click|"
    r"channel|www\.|\.com\b|https?:|hashtag|fandango|get\s+tickets|tickets\s+(?:on\s+sale|now)|pre-?order|"
    r"more\s+trailers|latest\s+(?:movie|trailers)|step\s+inside", re.IGNORECASE)


def promo_cutoff(path: Path, length: float, *, start_fraction: float = 0.62, step: float = 1.5) -> float | None:
    """Where a trailer's promotional end slate begins (subscribe/playlist/social/website/ticket text).

    Scans only the final part of the trailer with local OCR. Returns None when no
    promo text is found or OCR is unavailable.
    """
    if not shutil.which("tesseract") or length <= 0:
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        for tick in np.arange(length * start_fraction, length - 0.2, step):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(tick) * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if PROMO_TEXT.search(frame_text(image)):
                return float(tick)
    finally:
        cap.release()
    return None


def matches_title_card(text: str, title: str) -> bool:
    def words(value):
        plain = "".join(char for char in unicodedata.normalize("NFKD",value.casefold()) if not unicodedata.combining(char))
        return re.findall(r"[a-z0-9]+",plain)
    # A supplied release year is metadata, while a numeric title such as 1917
    # or 1923 must remain meaningful title text.
    title = re.sub(r"\s*\((?:19|20)\d{2}\)\s*$","",title)
    expected = [word for word in words(title) if word not in {"the","a","an","and","of","to","in","on","for","with","at","by"}]
    found = set(words(text))
    if not expected:
        return False
    if len(expected) == 1 and len(expected[0]) < 3:
        return any(words(line) == words(title) for line in text.splitlines())
    return sum(word in found for word in expected)/len(expected) >= (1 if len(expected) <= 2 else .8)


def content_core(gray: np.ndarray) -> np.ndarray:
    """The picture without black letterbox/pillarbox bars, then its centre (75% x 85%).

    The same shot appears letterboxed in one trailer and full-frame or re-cropped in another;
    fingerprinting only the central content lets those copies match."""
    rows = np.where(gray.mean(axis=1) > 18)[0]
    cols = np.where(gray.mean(axis=0) > 18)[0]
    if len(rows) > gray.shape[0] * .4 and len(cols) > gray.shape[1] * .4:
        gray = gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = gray.shape
    return gray[round(h * .075):round(h * .925), round(w * .125):round(w * .875)]


def scene_hash(image: Image.Image) -> int | None:
    gray = cv2.cvtColor(np.array(image.resize((320,180))),cv2.COLOR_RGB2GRAY)
    if float(gray.std()) < 8 or float(gray.mean()) < 12:
        return None
    small = cv2.resize(content_core(gray),(9,8),interpolation=cv2.INTER_AREA)
    return sum(int(bit)<<index for index,bit in enumerate((small[:,1:]>small[:,:-1]).reshape(-1)))


CARD_TEXT = re.compile(r"(rated\s+(?:g|pg|pg-13|r|nc-17)\b|\bmpa\b|motion picture association|approved for|"
                       r"(?:appropriate|all) audiences|restricted|under 17|coming soon|only in (?:theaters|theatres|cinemas)|"
                       r"in (?:theaters|theatres|cinemas)|now streaming|streaming (?:on|soon)|presents|production|"
                       r"a film by|from the (?:director|producer|creators?)|academy award|winner|nominee|\bimax\b|"
                       r"official (?:trailer|teaser)|copyright|all rights reserved)", re.IGNORECASE)


def text_card_kind(text: str) -> str | None:
    """'rating', 'promo' or 'text' for rating cards, release/studio cards and other text slates."""
    if re.search(r"rated\s+(?:g|pg|pg-13|r|nc-17)\b|\bmpa\b|approved for|audiences|restricted|under 17", text, re.IGNORECASE):
        return "rating"
    if CARD_TEXT.search(text):
        return "promo"
    if len(re.findall(r"\b[A-Za-z]{3,}\b", text)) >= 6:
        return "text"
    return None


def audio_envelope(path: Path, rate: int = 10) -> np.ndarray:
    """Loudness in dB at ``rate`` frames per second (empty when the file has no audio)."""
    try:
        result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1",
                                 "-ar", "8000", "-f", "s16le", "-"], capture_output=True, timeout=180, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return np.zeros(0)
    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768
    hop = 8000 // rate
    if samples.size < hop:
        return np.zeros(0)
    frames = samples[: samples.size // hop * hop].reshape(-1, hop)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-5)


def punch(envelope: np.ndarray, start: float, end: float, rate: int = 10) -> float:
    """0-1: the biggest sudden jump in loudness inside the shot (stingers, bangs, screams, hits)."""
    if not envelope.size:
        return 0.0
    best = 0.0
    for index in range(max(0, int(start * rate)), min(envelope.size, int(end * rate) + 1)):
        before = envelope[max(0, index - rate): index]
        if before.size:
            best = max(best, float(envelope[index] - np.median(before)))
    return float(np.clip(best / 24, 0, 1))


def analyse(path: Path, cache: Path) -> list[dict]:
    """Sample cuts and motion at 2fps; preserve source timestamps, excluding end slates.

    Each shot also records ``punch`` (sudden loudness jump), ``peak`` (punch and motion
    combined: the trailer's scare/impact beats) and ``text_card`` for rating, release or
    studio cards found by OCR, which never reach the edit.
    """
    cache = cache.with_name(cache.stem + ".calm-v6.json")
    length = duration(path)
    # Cross-run cache (restored by the workflow): the same video file, recognised by its name,
    # size and length, is never analysed twice.
    shared = Path(os.getenv("ANALYSIS_CACHE", "work/analysis-cache")) / f"{path.stem}-{path.stat().st_size}-{length:.2f}.calm-v6.json"
    if cache.exists():
        shots = json.loads(cache.read_text())
        if not shared.exists():
            try:
                shared.parent.mkdir(parents=True, exist_ok=True)
                save_json(shared, shots)
            except OSError:
                pass
        return shots
    if shared.exists():
        try:
            shots = json.loads(shared.read_text())
            cache.write_text(json.dumps(shots))
            return shots
        except (OSError, ValueError):
            pass
    # Studio channel end slates (subscribe, playlists, social links) can run longer than the
    # default final 10%; OCR finds where they begin so they never reach the edit.
    end = length * 0.90
    slate = promo_cutoff(path, length)
    if slate is not None and slate < end:
        print(f"End slate detected in {path.name} at {slate:.1f}s of {length:.1f}s; footage after it is excluded.", flush=True)
        end = max(slate - 0.5, length * 0.3)
    cap = cv2.VideoCapture(str(path))
    previous = None
    previous_hist = None
    boundaries = [max(2.0, length * 0.04)]
    samples = []
    sampled = {}
    for tick in np.arange(boundaries[0], end, 0.5):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(tick) * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
        motion = float(np.abs(gray.astype(float) - previous.astype(float)).mean()) if previous is not None else 0
        hsv = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(histogram, histogram)
        correlation = cv2.compareHist(previous_hist, histogram, cv2.HISTCMP_CORREL) if previous_hist is not None else 1
        # Large movement alone is not a scene cut. Require an appearance change.
        if motion > 38 and correlation < .65 and tick - boundaries[-1] >= 1.5:
            boundaries.append(float(tick))
        samples.append((float(tick), motion, float(gray.mean()), float(gray.std())))
        # Keep a small JPEG of every sampled frame so card checks never seek the (often 4K) file again.
        ok, encoded = cv2.imencode(".jpg", cv2.resize(frame, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            sampled[float(tick)] = encoded
        previous = gray
        previous_hist = histogram
    boundaries.append(end)
    cap.release()
    shots = []

    def dark(sample) -> bool:
        # Trailer fades to black (and near-black title gaps) must never reach the edit.
        return sample[2] < 14 or (sample[2] < 24 and sample[3] < 7)

    for start, end in pairwise(boundaries):
        # Long uncut shots are subdivided to provide useful choices.
        for point in np.arange(start, end - 1.2, 12.0):
            stop = min(float(point + 12.0), end)
            metrics = [s for s in samples if point <= s[0] < stop]
            if not metrics or np.mean([s[2] for s in metrics]) < 15 or np.mean([s[3] for s in metrics]) < 12:
                continue
            # Split the window around dark samples; keep each lit stretch of at least 1.2 s.
            runs, current = [], []
            for sample in metrics:
                if dark(sample):
                    if current:
                        runs.append((current, sample[0]))
                    current = []
                else:
                    current.append(sample)
            if current:
                runs.append((current, stop))
            for run, limit in runs:
                begin = float(point) if run[0] is metrics[0] else run[0][0]
                finish = min(limit - 0.25, stop) if limit < stop else stop
                if finish - begin >= 1.2:
                    shots.append({"start": begin, "end": finish, "motion": float(np.mean([s[1] for s in run]))})
    if not shots:
        raise PipelineError(f"No usable shots detected in {path.name}.")
    envelope = audio_envelope(path)
    for shot in shots:
        shot["punch"] = round(punch(envelope, shot["start"], shot["end"]), 3)
        if envelope.size:
            low, high = int(shot["start"] * 10), min(envelope.size, int(shot["end"] * 10) + 1)
            if high > low:
                shot["punch_at"] = round((low + int(np.argmax(np.diff(envelope[max(0, low - 1):high], prepend=envelope[max(0, low - 1)])))) / 10, 2)
        shot["peak"] = round(0.6 * shot["punch"] + 0.4 * min(shot["motion"], 40) / 40, 3)
        kind = None
        middle = (shot["start"] + shot["end"]) / 2
        tick = min(sampled, key=lambda value: abs(value - middle)) if sampled else None
        if tick is not None:
            frames_dir = path.with_suffix(".frames")
            frames_dir.mkdir(exist_ok=True)
            (frames_dir / f"{shots.index(shot):04}.jpg").write_bytes(sampled[tick].tobytes())
            image = cv2.imdecode(sampled[tick], cv2.IMREAD_COLOR)
            small = cv2.cvtColor(cv2.resize(image, (160, 90)), cv2.COLOR_BGR2GRAY)
            # Rating, release and studio cards sit on dark or plain backgrounds; only those are read.
            if float(small.mean()) < 80 or float(small.std()) < 40:
                kind = text_card_kind(frame_text(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
        if kind:
            shot["text_card"] = kind
    cards = sum(1 for shot in shots if shot.get("text_card"))
    if cards:
        print(f"{cards} rating/promo/text cards found in {path.name}; they are kept out of the edit.", flush=True)
    save_json(cache, shots)
    try:
        shared.parent.mkdir(parents=True, exist_ok=True)
        save_json(shared, shots)
    except OSError:
        pass  # the cross-run cache is a speed-up only
    return shots


def shot_frame(path: Path, shots: list[dict], index: int) -> Image.Image:
    """The shot's middle frame, from analysis's saved copy when present (no seek into the video)."""
    saved = path.with_suffix(".frames") / f"{index:04}.jpg"
    if saved.exists():
        try:
            return Image.open(saved).convert("RGB")
        except OSError:
            pass
    shot = shots[index]
    return get_frame(path, (float(shot["start"]) + float(shot["end"])) / 2)


def get_frame(path: Path, seconds: float) -> Image.Image:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise PipelineError(f"Cannot read frame from {path.name} at {seconds:.1f}s.")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


class SigLIP:
    """SigLIP 2 image/text features through OpenCLIP (independent of the transformers version
    the voice model pins), or through transformers when that is new enough."""

    def __init__(self, torch, backend: str, model, preprocess, tokenizer):
        self.torch, self.backend, self.model, self.preprocess, self.tokenizer = torch, backend, model, preprocess, tokenizer

    def _norm(self, values):
        if not isinstance(values, self.torch.Tensor):
            values = values.pooler_output
        return values / values.norm(dim=-1, keepdim=True)

    def images(self, images):
        with self.torch.inference_mode():
            if self.backend == "open_clip":
                batch = self.torch.stack([self.preprocess(image.convert("RGB")) for image in images])
                return self._norm(self.model.encode_image(batch))
            return self._norm(self.model.get_image_features(**self.preprocess(images=images, return_tensors="pt")))

    def text(self, text: str):
        with self.torch.inference_mode():
            if self.backend == "open_clip":
                return self._norm(self.model.encode_text(self.tokenizer([text])))
            inputs = self.preprocess(text=[text], return_tensors="pt", padding="max_length", max_length=64, truncation=True)
            return self._norm(self.model.get_text_features(**inputs))


def load_siglip(torch) -> SigLIP | None:
    errors = []
    name = os.getenv("SIGLIP_OPEN_CLIP", "hf-hub:timm/ViT-B-16-SigLIP2")
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(name)
        print(f"SigLIP 2 loaded through OpenCLIP ({name}).", flush=True)
        return SigLIP(torch, "open_clip", model.eval(), preprocess, open_clip.get_tokenizer(name))
    except Exception as exc:  # noqa: BLE001 -- try transformers next
        errors.append(f"OpenCLIP {type(exc).__name__}: {str(exc)[:120]}")
    try:
        from transformers import AutoModel, AutoProcessor
        hf_name = os.getenv("SIGLIP_MODEL", "google/siglip2-base-patch16-224")
        return SigLIP(torch, "transformers", AutoModel.from_pretrained(hf_name).eval(), AutoProcessor.from_pretrained(hf_name), None)
    except Exception as exc:  # noqa: BLE001 -- CLIP matching remains available
        errors.append(f"transformers {type(exc).__name__}: {str(exc)[:120]}")
    from .core import notice
    notice("SigLIP 2 unavailable", "Matching footage with CLIP. " + " | ".join(errors), "warning")
    return None


class ShotSelector:
    """Soft CLIP relevance, stable shots, and occasional face holds; all CPU."""
    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # SigLIP 2 matches narration to frames more accurately; CLIP stays for title-card and
        # character checks whose thresholds are tuned to it.
        self.siglip = self.siglip_processor = None
        self.siglip_features = {}
        self.scene_embeddings = {}
        if os.getenv("SHOT_MATCH_MODEL", "siglip2").lower() != "clip":
            self.siglip = load_siglip(torch)
            self.siglip_processor = self.siglip
        self.features = {}
        self.uses = {}
        self.last_shots = {}
        self.cut_count = 0
        self.faces = {}
        self.cards = {}
        self.card_texts = {}
        self.titles = {}
        self._scene_hashes = {}
        self._scene_ranges = {}
        self._visual_history = []
        self.people = {}
        self.held = set()
        self._source_intervals = {}
        self._history_clock = 0

    def index(self, path: Path, shots: list[dict]):
        vectors = []
        siglip_vectors = []
        face_flags = []
        small_frames = []
        face_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(str(face_path)) if face_path.exists() else None
        for offset in range(0, len(shots), 8):
            batch = shots[offset:offset+8]
            images = []
            for shot in batch:
                try:
                    image = get_frame(path, (shot["start"]+shot["end"])/2)
                except PipelineError:
                    image = Image.new("RGB", (320, 180), "gray")
                images.append(image)
                small_frames.append(image.resize((960,540)))
                gray = cv2.cvtColor(np.array(image.resize((480, 270))), cv2.COLOR_RGB2GRAY)
                detected = detector.detectMultiScale(gray, 1.1, 4) if detector is not None else []
                face_flags.append(any(w*h >= 1000 for _, _, w, h in detected))
            inputs = self.processor(images=images, return_tensors="pt")
            with self.torch.inference_mode():
                values = self.model.get_image_features(**inputs)
                if not isinstance(values, self.torch.Tensor):
                    values = values.pooler_output
                vectors.append(values / values.norm(dim=-1, keepdim=True))
            if self.siglip is not None:
                try:
                    siglip_vectors.append(self.siglip.images(images))
                except Exception as exc:  # noqa: BLE001
                    print(f"SigLIP 2 image features failed ({type(exc).__name__}); using CLIP for {path.name}.", flush=True)
                    siglip_vectors = None
                    self.siglip = None
        self.features[str(path)] = self.torch.cat(vectors)
        if siglip_vectors:
            self.siglip_features[str(path)] = self.torch.cat(siglip_vectors)
        from .scenes import load as load_scenes
        scenes = load_scenes(path, len(shots))
        if scenes:
            self.scene_embeddings[str(path)] = np.asarray(scenes["embeddings"], dtype=np.float32)
        self.faces[str(path)] = face_flags
        self._scene_ranges[str(path)] = shots
        self._scene_hashes[str(path)] = [scene_hash(frame) for frame in small_frames]
        self._index_cards(path,shots,small_frames)

    def _index_cards(self, path: Path, shots: list[dict], images: list[Image.Image]):
        """Conservatively flag title/promo cards with the already-loaded CLIP model."""
        try:
            if getattr(self,"_card_text_features",None) is None:
                prompts = ["A movie or television show title card: large title lettering on a plain background.",
                           "A production company, TV channel or streaming service logo on a plain background.",
                           "A cinematic scene with people, places, action, or a dramatic landscape."]
                inputs = self.processor(text=prompts,return_tensors="pt",padding=True,truncation=True)
                with self.torch.inference_mode():
                    vector = self.model.get_text_features(**inputs)
                    if not isinstance(vector,self.torch.Tensor):
                        vector = vector.pooler_output
                    self._card_text_features = vector/vector.norm(dim=-1,keepdim=True)
            with self.torch.inference_mode():
                scores = (self.features[str(path)] @ self._card_text_features.T).numpy()
            self.cards[str(path)] = [
                ("promo" if row[1] > row[0] else "title")
                if max(row[0],row[1]) >= .27 and max(row[0],row[1])-row[2] >= .055 else None
                for row in scores]
        except (AttributeError,RuntimeError,TypeError,ValueError):
            scores = None
            self.cards[str(path)] = [None]*len(self.features[str(path)])
        texts = [""]*len(shots)
        title = getattr(self,"titles",{}).get(str(path),"")
        if title:
            candidates = []
            for index,image in enumerate(images):
                gray = cv2.cvtColor(np.array(image.resize((160,90))),cv2.COLOR_RGB2GRAY)
                plain = float(np.mean(gray < 60)) > .55 and .01 < float(np.mean(gray > 140)) < .4
                score = max(scores[index][0],scores[index][1])-scores[index][2] if scores is not None else -1
                if self.cards[str(path)][index] or plain or score > -.035:
                    candidates.append((score+(.04 if plain else 0),index))
            for _score,index in sorted(candidates,reverse=True)[:12]:
                texts[index] = frame_text(images[index])
                if matches_title_card(texts[index],title):
                    self.cards[str(path)][index] = "title"
        self.card_texts[str(path)] = texts
        for index, shot in enumerate(shots):
            if shot.get("text_card") and self.cards[str(path)][index] != "title":
                self.cards[str(path)][index] = "promo"
        for index,kind in enumerate(self.cards[str(path)]):
            if kind:
                self.reserve(path,shots[index]["start"],shots[index]["end"]-shots[index]["start"],"card-exclusion")

    def card_kind(self, path: Path, index: int) -> str | None:
        values = getattr(self,"cards",{}).get(str(path),[])
        return values[index] if index < len(values) else None

    def note_visual_beat(self, seconds: float, *, still: bool = False):
        """Include displayed previews/cards in the hold cooldown, in edit order."""
        if still:
            self._moving_beats_since_hold = self._moving_seconds_since_hold = 0
        else:
            self._moving_beats_since_hold = getattr(self, "_moving_beats_since_hold", HOLD_COOLDOWN_BEATS)+1
            self._moving_seconds_since_hold = getattr(self, "_moving_seconds_since_hold", HOLD_COOLDOWN_SECONDS)+seconds

    def relevance(self, path: Path, text: str):
        """How well each shot fits a narration sentence, on CLIP's familiar scale (about 0.2-0.35).

        SigLIP 2 image-text similarity is blended with similarity to the AI scene
        descriptions when available; both are normalised per trailer.
        """
        signals = []
        if self.siglip is not None and str(path) in self.siglip_features:
            try:
                vector = self.siglip.text(text.lower())
                with self.torch.inference_mode():
                    signals.append((0.6, (self.siglip_features[str(path)] @ vector.T).squeeze(-1).numpy()))
            except Exception:  # noqa: BLE001 -- fall back to CLIP below
                signals = []
        if signals and str(path) in self.scene_embeddings:
            try:
                from .scenes import embed
                sentence = embed([text])[0]
                values = self.scene_embeddings[str(path)] @ sentence
                known = np.linalg.norm(self.scene_embeddings[str(path)], axis=1) > .01
                if known.any():
                    values = np.where(known, values, float(values[known].mean()))
                    signals.append((0.4, values))
            except Exception:  # noqa: BLE001 -- image matching alone is fine
                pass
        if signals:
            total = sum(weight for weight, _ in signals)
            combined = sum(weight * _zscore(values) for weight, values in signals) / total
            return (0.25 + 0.03 * combined).astype(np.float32)
        return self.identity_relevance(path, text)

    def identity_relevance(self, path: Path, text: str):
        """Raw CLIP similarity: the scale that title-card and character-hold thresholds expect."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
        with self.torch.inference_mode():
            vector = self.model.get_text_features(**inputs)
            if not isinstance(vector, self.torch.Tensor):
                vector = vector.pooler_output
            vector = vector / vector.norm(dim=-1, keepdim=True)
            relevance = (self.features[str(path)] @ vector.T).squeeze(-1).numpy()
        return relevance

    def _interval_history(self):
        history = getattr(self, "_source_intervals", None)
        if not isinstance(history, dict):
            history = {}
            self._source_intervals = history
        return history

    def reserve(self, path: Path, start: float, seconds: float, kind: str = "narration"):
        """Share consumed/reserved source windows across previews and every paragraph."""
        if not all(math.isfinite(float(value)) for value in (start, seconds)) or seconds <= 0:
            return
        self._history_clock = int(getattr(self, "_history_clock", 0))+1
        ranges = getattr(self,"_scene_ranges",{}).get(str(path),[])
        hashes = getattr(self,"_scene_hashes",{}).get(str(path),[])
        visual = getattr(self,"_visual_history",[])
        for index,shot in enumerate(ranges):
            if kind == "card-exclusion":
                break  # excluded cards are never shown, so they are not part of what viewers have seen
            if index < len(hashes) and hashes[index] is not None and start < shot["end"] and start+seconds > shot["start"]:
                visual.append((hashes[index],self._history_clock,str(path)))
        self._visual_history = visual
        self._interval_history().setdefault(str(path), []).append(
            {"start": max(0.0, float(start)), "end": max(0.0, float(start))+float(seconds),
             "order": self._history_clock, "kind": kind})

    def unused_windows(self, path: Path, start: float, end: float, gap: float = .18) -> list[tuple[float, float]]:
        """Subtract consumed windows, including a small frame-safety separation."""
        remaining = [(float(start), float(end))] if end > start else []
        for interval in self._interval_history().get(str(path), []):
            beginning, stop = interval["start"]-gap, interval["end"]+gap
            pieces = []
            for low, high in remaining:
                if stop <= low or beginning >= high:
                    pieces.append((low, high))
                else:
                    if beginning > low:
                        pieces.append((low, min(high, beginning)))
                    if stop < high:
                        pieces.append((max(low, stop), high))
            remaining = pieces
        return remaining

    def visual_freshness(self, path: Path, start: float, seconds: float) -> float:
        ranges = getattr(self,"_scene_ranges",{}).get(str(path),[])
        hashes = getattr(self,"_scene_hashes",{}).get(str(path),[])
        relevant = [hashes[i] for i,shot in enumerate(ranges) if i < len(hashes) and hashes[i] is not None
                    and start < shot["end"] and start+seconds > shot["start"]]
        history = getattr(self,"_visual_history",[])
        if not relevant:
            return 1.0
        return sum(not any((value^old).bit_count() <= 6 for old,_order,_path in history) for value in relevant)/len(relevant)

    def freshness(self, path: Path, start: float, seconds: float) -> float:
        if seconds <= 0:
            return 0.0
        return sum(end-beginning for beginning, end in self.unused_windows(path, start, start+seconds, 0))/seconds

    def fresh_seconds(self, path: Path, shots: list[dict]) -> float:
        windows = sorted((float(shot["start"]), float(shot["end"])) for shot in shots if shot["end"] > shot["start"])
        merged = []
        for start, end in windows:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return sum((high-low)*self.visual_freshness(path,low,high-low) for start,end in merged for low,high in self.unused_windows(path,start,end))

    def reuse_penalty(self, path: Path, start: float, seconds: float) -> float:
        """Strong recent-overlap cost, fading only after other source windows play."""
        clock = int(getattr(self, "_history_clock", 0))
        cost = 0.0
        for entry in self._interval_history().get(str(path), []):
            overlap = max(0, min(start+seconds, entry["end"])-max(start,entry["start"]))
            if overlap:
                age = max(0, clock-entry["order"])
                cost += overlap/max(seconds,.1) * (2 if age < 4 else .6/(1+age/8))
        return cost

    def visual_reuse_penalty(self, path: Path, start: float, seconds: float) -> float:
        """Keep a scene's cooldown when the other trailer contains the same image."""
        ranges = getattr(self,"_scene_ranges",{}).get(str(path),[])
        hashes = getattr(self,"_scene_hashes",{}).get(str(path),[])
        relevant = [hashes[i] for i,shot in enumerate(ranges) if i < len(hashes) and hashes[i] is not None
                    and start < shot["end"] and start+seconds > shot["start"]]
        clock = int(getattr(self,"_history_clock",0))
        costs = []
        from .editorial_policy import source_identity
        film = getattr(self, "source_films", {}).get(str(path), source_identity(path))
        for value in relevant:
            for old,order,source in getattr(self,"_visual_history",[]):
                if (value^old).bit_count() <= 6:
                    age = max(0,clock-order)
                    # The same image under a different film's review is never acceptable.
                    other_film = getattr(self, "source_films", {}).get(str(source), source_identity(source)) != film
                    costs.append(5.0 if other_film else (2 if age < 4 else .6/(1+age/8)))
        return max(costs,default=0.0)

    def hook_score(self, path: Path, shots: list[dict], prompt: str, action: bool) -> float:
        try:
            relevance = self.relevance(path, prompt)
        except (RuntimeError, ValueError, KeyError):
            relevance = np.zeros(len(shots))
        # Optional second sources with unseen footage outrank an exhausted hero shot.
        allowed = [i for i in range(len(shots)) if self.card_kind(path,i) is None] or list(range(len(shots)))
        values = [float(relevance[i]) + (min(shot["motion"],40)/200 if action else 0)
                  - self.reuse_penalty(path,shot["start"],shot["end"]-shot["start"])
                  - self.visual_reuse_penalty(path,shot["start"],shot["end"]-shot["start"])
                  for i,shot in enumerate(shots) if i in allowed]
        return max(values,default=0.0)

    def choose(self, path: Path, shots: list[dict], text: str, seconds: float, action: bool, maximum_shot: float | None = None) -> list[dict]:
        if not shots or seconds <= 0:
            return []
        # Similarity is a soft preference only: weak matches never reject a shot or a video.
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()] or [text]
        weights = np.cumsum([max(1, len(part.split())) for part in sentences])
        remaining = seconds
        chosen = []
        last_index = self.last_shots.get(str(path), -1)
        explicit_limit = maximum_shot is not None
        # The current line's delivery can shorten shots (hype) or allow longer holds (dramatic, calm).
        maximum_shot = min(moving_limit(self), maximum_shot or MAX_MOVING_SECONDS, float(getattr(self, "pace_limit", MAX_MOVING_SECONDS) or MAX_MOVING_SECONDS))
        allowed = [i for i in range(len(shots)) if self.card_kind(path,i) is None] or list(range(len(shots)))
        actor_names = getattr(self, "people", {}).get(str(path), [])
        about_person = bool(re.search(r"\b(actor|actress|performance|plays|portrays|starring|character)\b", text, re.IGNORECASE)) or any(name.casefold() in text.casefold() for name in actor_names)
        flags = getattr(self, "faces", {}).get(str(path), [False]*len(shots))
        moving_beats = getattr(self, "_moving_beats_since_hold", HOLD_COOLDOWN_BEATS)
        moving_seconds = getattr(self, "_moving_seconds_since_hold", HOLD_COOLDOWN_SECONDS)
        hold_count = getattr(self, "_editorial_hold_count", 0)
        relevance_cache = {}
        while remaining > 0.01:
            hold_ready = (not explicit_limit and moving_beats >= HOLD_COOLDOWN_BEATS
                          and moving_seconds >= HOLD_COOLDOWN_SECONDS)
            position = min(len(sentences)-1, int(np.searchsorted(weights, (seconds-remaining)/seconds * weights[-1], side="right")))
            if position not in relevance_cache:
                try:
                    relevance_cache[position] = self.relevance(path, sentences[position])
                except (RuntimeError, ValueError, KeyError):
                    relevance_cache[position] = np.zeros(len(shots))
            relevance = relevance_cache[position]
            candidates = []
            for index, shot in enumerate(shots):
                if index not in allowed:
                    continue
                used = self.uses.get((str(path), index), 0)
                raw_start, raw_end = float(shot["start"]), float(shot["end"])
                fragments = [(low,high) for low,high in self.unused_windows(path,raw_start,raw_end)
                             if high-low >= min(5,remaining)]
                # Exhaustion is allowed. Revisit the least-recently seen interval rather
                # than resetting the semantic hero at every new narration paragraph.
                for start, end in fragments or [(raw_start,raw_end)]:
                    available = max(.1,end-start)
                    unseen = bool(fragments)
                    virgin = self.freshness(path,raw_start,raw_end-raw_start) >= .999
                    visual_fresh = self.visual_freshness(path,start,available) >= .999
                    tier = 2 if virgin and visual_fresh else (1 if unseen and not virgin else 0)
                    if not unseen:
                        # Once exhausted, rotate the least recently used window
                        # before considering semantic preference again.
                        tier = -max((entry["order"] for entry in self._interval_history().get(str(path), [])
                                     if start < entry["end"] and end > entry["start"]), default=0)
                    score = float(relevance[index]) - used*.04
                    score += min(available,maximum_shot)*.035-min(shot["motion"],60)/600
                    if about_person and hold_ready and flags[index]:
                        score += .22
                    score -= self.reuse_penalty(path,start,min(remaining,available,maximum_shot))
                    score -= self.visual_reuse_penalty(path,start,min(remaining,available,maximum_shot))
                    if index == last_index:
                        score -= .35
                    # Story order: picking up right where the previous excerpt of this trailer ended
                    # plays the trailer's own sequence of scenes, which reads as one coherent segment.
                    previous_end = getattr(self, "last_end", {}).get(str(path))
                    if previous_end is not None and previous_end - .18 <= start <= previous_end + 1.0:
                        score += .8  # A connected sequence can beat a marginal relevance advantage.
                    elif previous_end is not None and previous_end + 1.0 < start <= previous_end + 5:
                        score += .2
                    elif previous_end is None:
                        # A title's first excerpt opens on its strongest footage: the best comes first.
                        score += .35 * min(1.0, float(shot.get("peak", 0)))
                    candidates.append((tier,score,-index,index,start,end,not unseen))
            # When no hold is due, prefer readable source windows over tiny
            # fresh fragments, rather than padding every short shot with a freeze.
            readable = [row for row in candidates if row[5]-row[4] >= min(5, remaining)]
            if readable and not hold_ready:
                candidates = readable
            if all(row[-1] for row in candidates) and any(row[3] != last_index for row in candidates):
                candidates = [row for row in candidates if row[3] != last_index]
            _tier,_score,_order,index,start,end,reused = max(candidates,key=lambda row: row[:3])
            shot = shots[index]
            available = max(.1,end-start)
            portrait = about_person and hold_ready and flags[index] and remaining >= 3
            detail = (hold_ready and not portrait and moving_beats > HOLD_COOLDOWN_BEATS
                      and moving_seconds >= HOLD_COOLDOWN_SECONDS + 4
                      and shot["motion"] <= 18 and remaining >= 3)
            still = portrait or detail
            if still and 7 < remaining < 8.5:
                portrait = detail = still = False
            beat_limit = maximum_shot
            # Balance a short tail into two readable beats instead of extending
            # a moving excerpt beyond twelve seconds or adding an extra freeze.
            if remaining > beat_limit and remaining-beat_limit < 3:
                beat_limit = remaining/2
            # A short window may hold its last frame briefly (1.5 s at most), never a long freeze-frame.
            freeze = 0 if still or not hold_ready else min(1.5, max(0, min(7,remaining)-available))
            hold_length = remaining if remaining <= 7 else 5.5
            length = min(remaining, hold_length if still else available+freeze, beat_limit)
            if not still and hold_ready and 0 < remaining-length < 2.5 and remaining <= maximum_shot:
                extra = min(remaining-length, max(0, 7-freeze))
                freeze += extra
                length += extra
            freeze = min(freeze,max(0,length-.1))
            chosen_start = (start+end)/2 if still else start
            cut = {"start":chosen_start,"duration":length,"freeze":freeze,"still":bool(still),
                   "visual_reason":"portrait-hold" if portrait else ("detail-hold" if detail else
                        ("spaced-source-reuse" if reused else "fresh-script-match"))}
            if still or freeze:
                cut["still_motion"] = ("zoom_in", "zoom_out")[hold_count % 2]
                hold_count += 1
                moving_beats = moving_seconds = 0
            else:
                moving_beats += 1
                moving_seconds += length
            chosen.append(cut)
            # A portrait hold consumes that visual beat; motion reserves just its
            # live source span, leaving the rest of a long scene available later.
            self.reserve(path,start if still else chosen_start,
                         available if still else max(.1,min(available,length-freeze)),
                         "portrait" if portrait else ("detail-hold" if detail else "narration"))
            self.cut_count += 1
            if not still:
                if not hasattr(self, "last_end"):
                    self.last_end = {}
                self.last_end[str(path)] = chosen_start + max(.1, length - freeze)
            self.uses[(str(path), index)] = self.uses.get((str(path), index), 0)+1
            remaining -= length
            last_index = index
        self.last_shots[str(path)] = last_index
        self._moving_beats_since_hold = moving_beats
        self._moving_seconds_since_hold = moving_seconds
        self._editorial_hold_count = hold_count
        return chosen


def intro_montage(selector, trailers: list[Path], shot_sets: list[list[dict]], paragraph_seconds: list[float], brief: str, action: bool) -> list[list[dict]]:
    """Readable multi-title hook, reducing coverage when narration is too short."""
    count = len(trailers)
    required = min(math.ceil(count/2), max(1, int(sum(paragraph_seconds)/2.5)))
    frame_counts = [math.ceil(seconds*30) for seconds in paragraph_seconds]
    if not count or not frame_counts or sum(frame_counts) < required:
        raise PipelineError("Intro audio is too short to show the required title coverage.")
    # Give each hook image roughly three seconds, retaining feasible multi-title coverage.
    slot_frames = 90 if action else 105
    slots = [max(1, math.ceil(frames/slot_frames)) for frames in frame_counts]
    while sum(slots) < required:
        options = [i for i, frames in enumerate(frame_counts) if slots[i] < frames]
        i = max(options, key=lambda i: frame_counts[i]/slots[i])
        slots[i] += 1
    prompt = ("A cinematic action highlight: a fight, chase, stunt, explosion, or dramatic physical action. " if action else "A striking cinematic visual with a compelling character or dramatic environment. ") + brief
    ranked = sorted(range(count), key=lambda i: selector.hook_score(trailers[i], shot_sets[i], prompt, action), reverse=True)
    active = ranked[:min(count, sum(slots))]
    plans = []
    cursor = 0
    for frames, slot_count in zip(frame_counts, slots):
        plan = []
        quotient, remainder = divmod(frames, slot_count)
        for slot in range(slot_count):
            index = active[cursor % len(active)]
            cursor += 1
            seconds = (quotient + (slot < remainder))/30
            cuts = selector.choose(trailers[index], shot_sets[index], prompt, seconds, action, maximum_shot=seconds)
            plan.extend({**cut, "source": str(trailers[index]), "film_index": index} for cut in cuts)
        plans.append(plan)
    covered = {shot["film_index"] for plan in plans for shot in plan}
    if len(covered) < required:
        print("Short intro uses fewer titles to preserve readable pacing.", flush=True)
    return plans
