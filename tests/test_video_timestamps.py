"""Final assembly with independently encoded clips and different B-frame delays."""
import json
import shutil
import subprocess
from itertools import pairwise

import numpy as np
import pytest

from movie_pipeline import render
from movie_pipeline.core import ffmpeg, probe
from movie_pipeline.shots import get_frame
from movie_pipeline.timing import DurationSpec

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")


def test_legitimate_video_timestamp_gaps_are_preserved_before_clock_reset(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_RESOLUTION", "320x180")
    source = tmp_path/"gapped-video.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=4.6",
           "-f", "lavfi", "-i", "sine=frequency=900:sample_rate=48000:duration=5",
           "-vf", "settb=1/30,setpts='N+12*gte(N,60)'", "-fps_mode", "passthrough",
           "-c:v", "libx264", "-preset", "veryfast", *render.LOSSLESS_AUDIO, source)
    video = next(stream for stream in probe(source)["streams"] if stream["codec_type"] == "video")
    assert int(video["nb_frames"]) == 138
    output = tmp_path/"final.mp4"
    actual, speed = render.finish([source], output, tmp_path, False, DurationSpec(automatic=True))
    assert speed == 1 and actual == pytest.approx(8, abs=.04)
    video = next(stream for stream in probe(output)["streams"] if stream["codec_type"] == "video")
    assert int(video["nb_frames"]) == 240
    render.validate_audio_video_duration(output, 8)


@pytest.mark.parametrize("count,speed", [(6, 1), (63, 1), (63, .85), (63, 1.18)])
def test_mixed_reorder_delays_preserve_every_scene_and_narration_boundary(monkeypatch, tmp_path, count, speed):
    monkeypatch.setenv("VIDEO_RESOLUTION", "320x180")
    palette = ["red", "blue", "yellow", "magenta", "cyan", "lime"]
    sizes = [30, 33, 45, 31, 60, 37]
    sources = []
    for index, (color, frames) in enumerate(zip(palette, sizes)):
        path = tmp_path/f"source-{index}.mp4"
        ffmpeg("-f", "lavfi", "-i", f"color=c={color}:size=320x180:rate=30",
               "-f", "lavfi", "-i", f"sine=frequency={500+index*200}:sample_rate=48000",
               "-t", frames/30, "-c:v", "libx264", "-bf", 0 if index % 2 == 0 else 3,
               "-x264-params", "b-adapt=0", "-preset", "veryfast", "-pix_fmt", "yuv420p",
               *render.LOSSLESS_AUDIO, path)
        sources.append(path)
    parts = [sources[index % len(sources)] for index in range(count)]
    lengths = [float(next(stream for stream in probe(path)["streams"]
                         if stream["codec_type"] == "video")["duration"]) for path in parts]
    listing = tmp_path/"input.txt"
    render.concat_file(parts, listing, durations=lengths)
    packets = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
        "-select_streams", "v:0", "-show_packets", "-show_entries", "packet=dts", "-of", "json",
    ]))["packets"]
    clocks = [int(packet["dts"]) for packet in packets]
    assert any(second < first for first, second in pairwise(clocks))

    original = sum(lengths)
    requested = original/speed+3
    # Production no longer retimes footage to meet runtime. Inject legacy/custom
    # tempo here to retain assembly coverage at every supported audio clock.
    monkeypatch.setattr(render, "final_tempo", lambda *args, **kwargs: speed)
    spec = DurationSpec(requested, requested, requested, automatic=speed == 1)
    output = tmp_path/"final.mp4"
    actual, chosen_speed = render.finish(parts, output, tmp_path, False, spec=spec)
    assert chosen_speed == pytest.approx(speed)
    assert actual == pytest.approx(requested, abs=.06)
    render.validate_audio_video_duration(output, requested)
    video = next(stream for stream in probe(output)["streams"] if stream["codec_type"] == "video")
    if speed == 1:
        assert int(video["nb_frames"]) == sum(sizes[i % len(sizes)] for i in range(count))+90
        raw = subprocess.check_output([
            "ffmpeg", "-v", "error", "-i", str(output), "-an", "-vf", "scale=8:4",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
        ])
        displayed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4, 8, 3).mean(axis=(1, 2))
        colors = [np.asarray(get_frame(path, .2), dtype=float).mean(axis=(0, 1)) for path in sources]
        expected = np.concatenate([np.repeat([colors[i % len(sources)]], sizes[i % len(sizes)], axis=0)
                                   for i in range(count)])
        assert np.abs(displayed[:len(expected)]-expected).max() < 4
    audio = subprocess.check_output([
        "ffmpeg", "-v", "error", "-xerror", "-i", str(output), "-vn", "-ac", "1", "-ar", "48000", "-f", "f32le", "-",
    ])
    samples = np.frombuffer(audio, dtype="<f4")
    for index in sorted({0, 1, count//2, count-1}):
        start = sum(lengths[:index])/speed
        reference = np.asarray(get_frame(parts[index], .2), dtype=float)
        current = np.asarray(get_frame(output, start+.3), dtype=float)
        assert np.abs(reference-current).mean() < 3
        window = samples[round((start+.15)*48000):round((start+.4)*48000)]
        spectrum = np.abs(np.fft.rfft(window*np.hanning(len(window))))
        frequency = np.fft.rfftfreq(len(window), 1/48000)[np.argmax(spectrum)]
        assert frequency == pytest.approx(500+(index % len(sources))*200, abs=10)
    ffmpeg("-xerror", "-err_detect", "explode", "-i", output, "-f", "null", "-")
    report = json.loads((tmp_path/"audio-assembly.json").read_text())
    assert report["status"] == "passed" and report["assembly_mode"] == "direct-video-encode"
    assert not (tmp_path/"combined.mp4").exists()
    assert not list(tmp_path.glob("audio-normalized-*"))
