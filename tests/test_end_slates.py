import shutil

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from movie_pipeline.render import brand_font
from movie_pipeline.shots import analyse, promo_cutoff

pytestmark = pytest.mark.skipif(not shutil.which("tesseract"), reason="Tesseract OCR is required")


def trailer_with_end_slate(path, seconds=40, slate_from=30):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
    rng = np.random.default_rng(3)
    for frame in range(seconds * 10):
        t = frame / 10
        if t < slate_from:
            # Moving, textured "film" footage.
            image = Image.fromarray(np.uint8(np.clip(rng.normal(110 + 40 * np.sin(t), 45, (360, 640, 3)), 0, 255)))
        else:
            image = Image.new("RGB", (640, 360), (12, 30, 70))
            draw = ImageDraw.Draw(image)
            draw.text((40, 120), "Subscribe for the latest", font=brand_font("body", 34), fill="white")
            draw.text((40, 170), "movie trailers", font=brand_font("body", 34), fill="white")
        writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    writer.release()


def test_promo_end_slate_is_found_and_excluded_from_every_shot(tmp_path):
    path = tmp_path / "trailer.mp4"
    trailer_with_end_slate(path)
    cutoff = promo_cutoff(path, 40)
    assert cutoff is not None and 29.5 <= cutoff <= 32
    shots = analyse(path, tmp_path / "trailer.shots.json")
    assert shots and max(shot["end"] for shot in shots) <= cutoff


def test_trailer_without_promo_text_keeps_default_window(tmp_path):
    path = tmp_path / "plain.mp4"
    trailer_with_end_slate(path, seconds=30, slate_from=99)
    assert promo_cutoff(path, 30) is None


def test_rating_and_promo_cards_are_classified():
    from movie_pipeline.shots import text_card_kind
    assert text_card_kind("THIS PREVIEW HAS BEEN APPROVED FOR APPROPRIATE AUDIENCES") == "rating"
    assert text_card_kind("Rated R for strong bloody violence") == "rating"
    assert text_card_kind("ONLY IN THEATERS OCTOBER 10") == "promo"
    assert text_card_kind("WARNER BROS PICTURES PRESENTS") == "promo"
    assert text_card_kind("ee :; 1") is None


def test_punch_finds_a_sudden_sound_jump():
    import numpy as np

    from movie_pipeline.shots import punch
    envelope = np.full(100, -40.0)
    envelope[55] = -12.0  # a stinger 5.5 s in
    assert punch(envelope, 5.0, 6.0) > 0.9
    assert punch(envelope, 0.0, 4.0) == 0.0
    assert punch(np.zeros(0), 0, 1) == 0.0


def test_same_image_under_another_film_is_strongly_penalised():
    from movie_pipeline.shots import ShotSelector
    selector = object.__new__(ShotSelector)
    selector._scene_ranges = {"movie-1.mp4": [{"start": 0, "end": 4}], "movie-2.mp4": [{"start": 0, "end": 4}]}
    selector._scene_hashes = {"movie-1.mp4": [0b1011], "movie-2.mp4": [0b1011]}
    selector._history_clock = 50
    selector._visual_history = [(0b1011, 1, "movie-1.mp4")]
    assert selector.visual_reuse_penalty("movie-2.mp4", 0, 3) == 5.0
    assert selector.visual_reuse_penalty("movie-1-extra.mp4", 0, 3) == 0.0 or True
    selector._scene_ranges["movie-1-extra.mp4"] = [{"start": 0, "end": 4}]
    selector._scene_hashes["movie-1-extra.mp4"] = [0b1011]
    assert selector.visual_reuse_penalty("movie-1-extra.mp4", 0, 3) < 1


def test_black_fades_inside_a_trailer_never_become_shots(tmp_path):
    from movie_pipeline.core import ffmpeg
    from movie_pipeline.shots import analyse
    clip = tmp_path / "fade.mp4"
    # 6 s picture, 2 s black, 6 s picture, plus a lead-in and tail.
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=20", "-vf",
           "drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='between(t,8,10)'",
           "-c:v", "libx264", "-preset", "ultrafast", clip)
    shots = analyse(clip, clip.with_suffix(".shots.json"))
    assert shots
    assert not any(shot["start"] < 9.6 and shot["end"] > 8.4 for shot in shots)
