"""Steady pacing, shared hold spacing, and real rendered motion on frozen frames."""
import shutil
import wave
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from movie_pipeline.core import duration, ffmpeg
from movie_pipeline.previews import outro_recap_plan
from movie_pipeline.render import speech_video
from movie_pipeline.shots import ShotSelector, get_frame


def selector(paths, *, faces=False, speed=1):
    selected = object.__new__(ShotSelector)
    selected.uses, selected.last_shots, selected.cut_count = {}, {}, 0
    selected.relevance = lambda *_: np.ones(24)
    selected.faces = {str(path): [faces]*24 for path in paths}
    selected.playback_speed = speed
    return selected


def scenes():
    return [{"start": i*20, "end": i*20+8, "motion": 8} for i in range(24)]


@pytest.mark.parametrize("faces", [False, True])
def test_more_holds_are_spaced_across_paragraphs_and_different_reviews(faces):
    paths = [Path("first.mp4"), Path("second.mp4")]
    selected = selector(paths, faces=faces)
    cuts = []
    for path in paths:
        for _ in range(2):
            cuts += selected.choose(path, scenes(), "The actor gives this character a great performance.", 30, False)
    assert sum(cut["duration"] for cut in cuts) == pytest.approx(120)
    holds = [i for i, cut in enumerate(cuts) if cut["still"] or cut["freeze"]]
    assert len(holds) >= 3
    assert len({cuts[i]["still_motion"] for i in holds}) >= 2
    for previous, current in pairwise(holds):
        moving = cuts[previous+1:current]
        assert len(moving) >= 2
        assert sum(cut["duration"] for cut in moving) >= 14
    assert all(2 <= cuts[i]["duration"] <= 12 for i in holds)


def test_title_card_or_frozen_preview_delays_the_next_narrated_hold():
    path = Path("movie.mp4")
    selected = selector([path], faces=True)
    selected.note_visual_beat(1.5, still=True)
    selected.note_visual_beat(3.5)
    cuts = selected.choose(path, scenes(), "The actor gives a great performance.", 70, False)
    first_hold = next(i for i, cut in enumerate(cuts) if cut["still"])
    assert sum(cut["duration"] for cut in cuts[:first_hold])+3.5 >= 30  # holds are rare: 30 s of running footage first
    assert first_hold+1 >= 2


@pytest.mark.parametrize("speed", [.85, 1, 1.18])
@pytest.mark.parametrize("action", [False, True])
def test_moving_excerpts_stay_under_twelve_seconds_after_final_tempo(speed, action):
    path = Path("movie.mp4")
    selected = selector([path], speed=speed)
    selected.reserve(path, 0, 1000, "previous-review")  # Exercise footage exhaustion too.
    cuts = selected.choose(path, scenes(), "The story takes a turn.", 40, action)
    assert sum(cut["duration"] for cut in cuts) == pytest.approx(40)
    assert all((cut["duration"]-cut["freeze"])/speed <= 12 for cut in cuts if not cut["still"])
    plans = outro_recap_plan(selected, [[path]], {str(path): scenes()}, [8, 10], "See our other picks.")
    assert all(cut["duration"]/speed <= 12 for plan in plans for cut in plan)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")
@pytest.mark.parametrize("style", ["zoom_in", "zoom_out", "pan_left", "trailing-freeze"])
def test_gentle_still_motion_renders_without_audio_or_timing_changes(monkeypatch, tmp_path, style):
    monkeypatch.setenv("VIDEO_RESOLUTION", "320x180")
    image = Image.new("RGB", (320, 180), "#27354c")
    ImageDraw.Draw(image).rectangle((100, 55, 220, 125), fill="white")
    picture, source, audio = tmp_path/"scene.png", tmp_path/"source.mp4", tmp_path/"voice.wav"
    image.save(picture)
    ffmpeg("-loop", "1", "-framerate", 30, "-i", picture, "-t", 4,
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", source)
    rate, seconds = 48000, 3.2
    time = np.arange(round(rate*seconds))/rate
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((np.sin(2*np.pi*900*time)*4000).astype("<i2").tobytes())
    shot = {"start": 0, "duration": seconds, "still": style != "trailing-freeze",
            "still_motion": style if style != "trailing-freeze" else "pan_left",
            "freeze": 2.2 if style == "trailing-freeze" else 0}
    output = tmp_path/"rendered.mp4"
    speech_video(source, audio, [shot], output, None, tmp_path/"render")

    def subject(at):
        frame = np.asarray(get_frame(output, at))
        _y, x = np.where(frame.min(axis=2) > 220)
        return x.max()-x.min(), x.mean()

    early, late = subject(.15), subject(3.05)
    if style == "zoom_out":
        assert late[0] < early[0]-1
    else:
        assert late[0] > early[0]+1
        assert abs(late[1]-early[1]) <= 2  # Chroma/crop rounding at this small test resolution.
    if style == "trailing-freeze":
        assert subject(.85)[0] == pytest.approx(subject(1.03)[0], abs=1)
    assert duration(output) == pytest.approx(seconds, abs=.04)
    samples_path = tmp_path/"decoded.wav"
    ffmpeg("-i", output, "-vn", "-ac", 1, "-ar", rate, "-c:a", "pcm_s16le", samples_path)
    with wave.open(str(samples_path), "rb") as stream:
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
    assert len(samples)/rate == pytest.approx(seconds, abs=.04)
    for low, high in [(.03, .2), (2.9, 3.15)]:
        window = samples[round(low*rate):round(high*rate)].astype(float)
        frequency = np.fft.rfftfreq(len(window), 1/rate)[np.argmax(np.abs(np.fft.rfft(window)))]
        assert frequency == pytest.approx(900, abs=8)
