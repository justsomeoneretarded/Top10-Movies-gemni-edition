import numpy as np
import pytest

from movie_pipeline.core import duration, ffmpeg, probe
from movie_pipeline.cta_animation import _ui_sounds, add_cta_animation, interaction_frame


def test_animation_states_progress_through_like_subscribe_and_bell():
    times = [1.0, 2.0, 3.0]
    frames = [np.asarray(interaction_frame(t, times), dtype=float) for t in (0.2, 0.95, 1.3, 2.4, 3.3, 5.0)]
    assert frames[0][..., 3].max() == 0          # hidden before it slides in
    assert frames[1][..., 3].max() > 200         # visible plate
    assert np.abs(frames[2] - frames[1]).mean() > 1   # like pressed
    assert np.abs(frames[3] - frames[2]).mean() > 1   # subscribed
    assert np.abs(frames[4] - frames[3]).mean() > 1   # bell rung
    assert frames[5][..., 3].max() == 0          # gone after the exit


def test_ui_sounds_land_on_each_click():
    sound = _ui_sounds([0.5, 1.0, 1.5], 3)
    energy = lambda start: float(np.abs(sound[int(start*48000):int((start+0.05)*48000)]).max())
    assert energy(0.5) > 0.01 and energy(1.0) > 0.01 and energy(1.5) > 0.01
    assert energy(0.2) == 0


def test_cta_render_keeps_full_video_and_audio(tmp_path):
    source = tmp_path / "clip.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30", "-f", "lavfi",
           "-i", "sine=frequency=330:sample_rate=48000", "-t", 4,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "alac", "-sample_fmt", "s32p", source)
    add_cta_animation(source, "Like, subscribe and hit the bell.",
                      [{"start": .3, "end": 2.5, "text": "Like, subscribe and hit the bell."}], tmp_path / "cta")
    assert duration(source) == pytest.approx(4, abs=.05)
    assert any(stream["codec_type"] == "audio" for stream in probe(source)["streams"])
    ffmpeg("-xerror", "-i", source, "-f", "null", "-")
