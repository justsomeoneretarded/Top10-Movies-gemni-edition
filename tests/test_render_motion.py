"""Real media regressions for readable rank motion and intact audio boundaries."""
from __future__ import annotations

import shutil
import subprocess
import wave

import numpy as np
import pytest
from PIL import Image

from movie_pipeline.core import duration, ffmpeg, probe
from movie_pipeline.render import finish, label_image, speech_video, trailer_video
from movie_pipeline.shots import get_frame
from movie_pipeline.timing import DurationSpec

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")


def solid_video(path, seconds=4):
    ffmpeg("-f", "lavfi", "-i", "color=c=0x506070:size=1280x720:rate=30", "-t", seconds,
           "-c:v", "libx264", "-preset", "ultrafast", path)


def marker_audio(path, seconds=2):
    rate = 48000
    t = np.arange(round(rate*seconds))/rate
    audio = .1*np.sin(2*np.pi*320*t)
    audio[t < .15] = .35*np.sin(2*np.pi*1700*t[t < .15])
    audio[t > seconds-.15] = .35*np.sin(2*np.pi*2300*t[t > seconds-.15])
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((audio*32767).astype("<i2").tobytes())


def decoded_audio(path):
    result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                             "-ar", "48000", "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype="<f4")


def dominant_frequency(samples, start, end):
    section = samples[round(start*48000):round(end*48000)]
    spectrum = np.abs(np.fft.rfft(section*np.hanning(len(section))))
    return np.fft.rfftfreq(len(section), 1/48000)[np.argmax(spectrum)]


def test_rank_enters_holds_and_leaves_over_silent_preview(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_RESOLUTION", "720")
    source, label, output = (tmp_path/name for name in ("source.mp4", "rank.png", "preview.mp4"))
    solid_video(source)
    label_image(label, "Mission: Impossible — The Final Reckoning", 4, 2025)
    trailer_video(source, 0, 4, output, label=label, label_seconds=3.5)
    before = np.asarray(get_frame(output, .03), dtype=float)
    held = np.asarray(get_frame(output, 1.1), dtype=float)
    gone = np.asarray(get_frame(output, 3.8), dtype=float)
    assert np.abs(before-gone).mean() < 1.0
    assert np.abs(held[540:680]-before[540:680]).mean() > 8
    # The picture above the lower third stays unobstructed.
    assert np.abs(held[:500]-before[:500]).mean() < 1.0
    assert duration(output) == pytest.approx(4, abs=.05)
    audio = next(s for s in probe(output)["streams"] if s["codec_type"] == "audio")
    assert audio["codec_name"] == "alac"
    assert np.abs(decoded_audio(output)).max() < 1e-5


def test_label_does_not_disappear_at_first_source_cut(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_RESOLUTION", "720")
    source, audio, label, output = (tmp_path/name for name in ("source.mp4", "speech.wav", "rank.png", "speech.mp4"))
    solid_video(source)
    marker_audio(audio, 4)
    label_image(label, "A Complete Film Title", 2, 2026)
    speech_video(source, audio, [{"start": 0, "duration": .8}, {"start": 1, "duration": 3.2}],
                 output, label, tmp_path/"render", label_seconds=3.5)
    held = np.asarray(get_frame(output, 2.2), dtype=float)
    gone = np.asarray(get_frame(output, 3.8), dtype=float)
    assert np.abs(held[540:680]-gone[540:680]).mean() > 8
    assert np.abs(held[:500]-gone[:500]).mean() < 1.0


def test_full_export_keeps_first_and_last_audio_markers(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_RESOLUTION", "720")
    source, audio = tmp_path/"source.mp4", tmp_path/"voice.wav"
    solid_video(source)
    marker_audio(audio)
    parts = []
    for index in range(2):
        output = tmp_path/f"speech-{index}.mp4"
        speech_video(source, audio, [{"start": 0, "duration": 2}], output, None, tmp_path/f"render-{index}")
        parts.append(output)
    output = tmp_path/"final.mp4"
    seconds, speed = finish(parts, output, tmp_path, False, spec=DurationSpec(7, 0, 20, automatic=True))
    samples = decoded_audio(output)
    assert speed == 1
    assert seconds == pytest.approx(7, abs=.08)
    assert dominant_frequency(samples, .025, .125) == pytest.approx(1700, abs=20)
    assert dominant_frequency(samples, 3.86, 3.96) == pytest.approx(2300, abs=20)
    assert next(s for s in probe(output)["streams"] if s["codec_type"] == "audio")["codec_name"] == "aac"


def test_animated_reaction_stays_animated_for_full_narration(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_RESOLUTION", "720")
    reaction, audio = tmp_path/"reaction.gif", tmp_path/"voice.wav"
    red = Image.new("RGB", (320, 180), "red")
    blue = Image.new("RGB", (320, 180), "blue")
    red.save(reaction, save_all=True, append_images=[blue], duration=300, loop=0)
    marker_audio(audio, 1.8)
    output = tmp_path/"reaction.mp4"
    speech_video(reaction, audio, [{"source": str(reaction), "start": 0, "duration": 1.8,
                                   "animation": True, "image": True}], output, None, tmp_path/"render")
    first = np.asarray(get_frame(output, .1), dtype=float)
    second = np.asarray(get_frame(output, .4), dtype=float)
    repeated = np.asarray(get_frame(output, 1.3), dtype=float)
    assert first[:, :, 0].mean() > 240 and first[:, :, 2].mean() < 10
    assert second[:, :, 2].mean() > 240 and second[:, :, 0].mean() < 10
    assert np.abs(first-repeated).mean() < 1
    assert duration(output) == pytest.approx(1.8, abs=.05)


def test_intro_and_outro_do_not_get_a_generic_countdown_banner(tmp_path):
    label = tmp_path/"intro.png"
    label_image(label, "Top 5 Movies You Need to Watch", None, None)
    with Image.open(label) as image:
        assert image.getchannel("A").getbbox() is None
