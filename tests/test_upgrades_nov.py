"""Sizzle-reel intro, channel watermark, phrase-level voice, and stills instead of repeated footage."""
from pathlib import Path

from PIL import Image

from movie_pipeline.previews import SIZZLE_GENRES, sizzle_plan
from movie_pipeline.render import speech_video, watermark_image
from movie_pipeline.voice import phrases


class Selector:
    def __init__(self):
        self.reserved = []

    def card_kind(self, path, index):
        return "title" if index == 0 else None

    def reserve(self, path, start, seconds, kind):
        self.reserved.append((str(path), start, seconds))


def test_sizzle_opens_on_the_best_moment_and_rotates_titles():
    a, b = Path("a.mp4"), Path("b.mp4")
    lookup = {str(a): [{"start": 0, "end": 5, "peak": 1.0}, {"start": 10, "end": 14, "peak": 0.4}, {"start": 20, "end": 24, "peak": 0.2}],
              str(b): [{"start": 0, "end": 5, "peak": 0.1}, {"start": 30, "end": 34, "peak": 0.9}, {"start": 40, "end": 44, "peak": 0.5}]}
    selector = Selector()
    plan = sizzle_plan(selector, [[a], [b]], lookup, seconds=10.4, shot_seconds=2.6)
    assert [cut["film_index"] for cut in plan][:3] == [1, 0, 1]  # b's 0.9 beats a's best usable 0.4 (a's 1.0 is a title card)
    assert plan[0]["start"] >= 30 and not any(cut["source"] == "a.mp4" and cut["start"] < 5 for cut in plan)
    assert len(selector.reserved) == len(plan) and {"action", "adventure"} <= SIZZLE_GENRES and "comedy" not in SIZZLE_GENRES
    lengths = [cut["duration"] for cut in plan]
    assert lengths[0] >= lengths[1] >= 2.2 and sum(lengths) <= 10.4 + 1e-6 and all(cut["muted"] for cut in plan)


def test_watermark_is_small_and_translucent(tmp_path):
    width, height = watermark_image(tmp_path / "w.png")
    image = Image.open(tmp_path / "w.png")
    assert width < 320 and height == 44 and image.getchannel("A").getextrema()[1] < 255


def test_phrases_keep_context_across_reveal_and_following_sentences():
    parts = ["Number 5: White Chicks.", "Two agents go undercover.", "It gets weird fast.", "Then weirder."]
    assert phrases(parts, reveal=True) == ["Number 5: White Chicks. Two agents go undercover. It gets weird fast.", "Then weirder."]


def test_an_official_still_renders_with_a_smooth_zoom(tmp_path):
    still = tmp_path / "still.jpg"
    Image.new("RGB", (1920, 1080), (120, 80, 40)).save(still)
    audio = tmp_path / "voice.wav"
    import subprocess
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=220:d=3", "-ar", "48000", str(audio)], check=True)
    out = tmp_path / "part.mp4"
    speech_video(still, audio, [{"source": str(still), "image": True, "start": 0.0, "duration": 3.0, "still": True,
                                 "freeze": 0.0, "still_motion": "zoom_in"}], out, None, tmp_path / "render")
    assert out.stat().st_size > 1000


def test_accented_titles_match_their_plain_spelling():
    from movie_pipeline.core import fold_accents, normalized
    assert fold_accents("Shōgun") == "Shogun" and normalized("Shōgun") == normalized("SHOGUN")


def test_letterboxed_and_full_frame_copies_of_a_shot_match():
    import numpy as np
    from movie_pipeline.shots import scene_hash
    rng = np.random.default_rng(3)
    picture = (rng.random((90, 160)) * 255).astype(np.uint8)
    picture = np.kron(picture, np.ones((8, 8), dtype=np.uint8))[:720, :1280]
    full = Image.fromarray(np.stack([picture] * 3, axis=2))
    boxed = Image.new("RGB", (1280, 720))
    boxed.paste(full.resize((1280, 534)), (0, 93))  # the same shot letterboxed to 2.4:1
    assert (scene_hash(full) ^ scene_hash(boxed)).bit_count() <= 6


def test_extras_need_half_new_footage():
    from movie_pipeline.secondary_footage import novelty_from_hashes
    primary = [0x1111111111111111, 0x3333333333333333, 0x5555555555555555, 0x7777777777777777]
    novel = [0xEEEEEEEEEEEEEEEE, 0xCCCCCCCCCCCCCCCC, 0xAAAAAAAAAAAAAAAA]
    assert not novelty_from_hashes(primary, [*primary, *novel[:3], primary[0], primary[1]])["distinct"]
