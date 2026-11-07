"""Viewer-retention features: number one teaser, varied reveals, up-next pill, either/or pinned comment."""
import subprocess

from PIL import Image

from movie_pipeline.cli import number_one_tease_shot
from movie_pipeline.render import REVEAL_STYLES, _reveal_filter, number_one_tease_image, reveal_style_for, up_next_image
from movie_pipeline.script import versus_question


def test_reveals_alternate_only_gentle_zooms():
    for count in range(3, 12):
        styles = [reveal_style_for(position, rank) for position, rank in enumerate(range(count, 0, -1))]
        assert set(styles) == {"zoom_in", "zoom_out"}
        assert all(a != b for a, b in zip(styles, styles[1:])), styles
        assert len(set(styles)) == 2


def test_every_reveal_style_renders(tmp_path):
    for style in REVEAL_STYLES:
        out = tmp_path / f"{style}.mp4"
        graph = f"testsrc2=s=640x360:r=30:d=1,{_reveal_filter(640, 360, style)}format=yuv420p"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", graph, "-y", str(out)], check=True)
        assert out.stat().st_size > 1000


def test_tease_avoids_number_ones_own_review_moments_and_cards():
    shots = [{"start": 0, "end": 4, "peak": 0.9}, {"start": 10, "end": 15, "peak": 0.8},
             {"start": 20, "end": 24, "peak": 0.3}, {"start": 30, "end": 33, "peak": 1.0}]
    chosen = number_one_tease_shot(shots, lambda i: i == 3, taken=[(0, 5)])
    assert chosen and 10 <= chosen["start"] and chosen["end"] <= 15
    assert abs(chosen["end"] - chosen["start"] - 3.0) < 1e-6
    assert number_one_tease_shot(shots, lambda i: True, taken=[]) is None


def test_up_next_and_tease_images(tmp_path):
    width, height = up_next_image(tmp_path / "next.png", 4, "A Very Long Title That Must Be Shortened To Fit The Pill")
    assert width <= 700 and height == 50
    secret, _ = up_next_image(tmp_path / "one.png", 1, None)
    assert secret < width
    number_one_tease_image(tmp_path / "tease.png")
    assert Image.open(tmp_path / "tease.png").size == (1280, 720)


def test_pinned_question_is_either_or():
    shows = [{"title": "A Knight of the Seven Kingdoms", "kind": "tv"}, {"title": "Wonder Man", "kind": "tv"}]
    assert versus_question(shows, "10 New Shows of 2026") == \
        "Which 2026 show are you watching first: A Knight of the Seven Kingdoms or Wonder Man?"
    assert versus_question([{"title": "Heat"}, {"title": "Ronin"}], "Best heist movies") == \
        "Which movie are you watching first: Heat or Ronin?"
