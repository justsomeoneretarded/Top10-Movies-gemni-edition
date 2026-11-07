"""Exercise real FFmpeg media assembly without downloading models or movie footage."""
import shutil

import pytest
from PIL import Image

from movie_pipeline.core import duration, ffmpeg, probe
from movie_pipeline.render import (
    label_image,
    music_bed,
    speech_video,
    thumbnails,
    trailer_video,
)
from movie_pipeline.shots import analyse

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")


def test_render_preserves_speech_duration_and_produces_thumbnails(tmp_path):
    source = tmp_path / "source.mp4"
    audio = tmp_path / "voice.wav"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", 6, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", source)
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000", "-t", 2.125, audio)
    label = tmp_path / "label.png"
    label_image(label, "Arrival", 1, 2016)
    target = tmp_path / "speech.mp4"
    speech_video(source, audio, [{"start": 1, "duration": 1.1}, {"start": 3, "duration": 1.1}], target, label, tmp_path / "render")
    assert abs(duration(target) - duration(audio)) < 0.09
    assert any(s["codec_type"] == "audio" for s in probe(target)["streams"])
    shots = analyse(source, tmp_path / "shots.json")
    assert shots
    thumbnails([source, source], [shots, shots], ["MIND BENDING", "WATCH THESE", "EXPECT THE UNEXPECTED"], tmp_path)
    for i in range(1, 4):
        path = tmp_path / f"thumbnail-{i:02}.jpg"
        assert Image.open(path).size == (1280, 720)
        assert path.stat().st_size < 2_000_000
    insert = tmp_path / "insert.mp4"
    trailer_video(source, 1, 2.5, insert)
    assert abs(duration(insert) - 2.5) < 0.1
    music_bed(tmp_path / "music.wav", 3.125, True)
    assert abs(duration(tmp_path / "music.wav") - 3.125) < 0.001


@pytest.mark.parametrize("source_seconds, brief", [(480, ""), (330, "Make a 5-minute video")])
def test_final_export_respects_requested_duration_with_valid_audio(tmp_path, source_seconds, brief):
    from movie_pipeline.render import finish
    from movie_pipeline.timing import requested_duration
    spec = requested_duration(brief)
    # A repeated one-second synthetic scene exercises both default and five-minute exports,
    # sidechain music mix, loudness conversion, and final duration policy.
    seed = tmp_path / "seed.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=0x16243a:size=1280x720:rate=30", "-f", "lavfi", "-i", "sine=frequency=180:sample_rate=48000", "-t", 1, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", seed)
    long_source = tmp_path / "long.mp4"
    ffmpeg("-stream_loop", source_seconds-1, "-i", seed, "-t", source_seconds, "-c", "copy", long_source)
    target = tmp_path / "final.mp4"
    seconds, _speed = finish([long_source], target, tmp_path, False, spec=spec)
    assert spec.minimum-0.5 <= seconds <= spec.maximum+0.5
    if brief:
        assert abs(seconds-(source_seconds+3)) < .1
        assert _speed == 1
    metadata = probe(target)
    audio = next(s for s in metadata["streams"] if s["codec_type"] == "audio")
    assert audio["sample_rate"] == "48000"
    assert audio["channels"] == 2
    ffmpeg("-xerror", "-i", target, "-f", "null", "-")


def test_mixed_sources_and_freeze_keep_narration_in_sync(tmp_path):
    import numpy as np

    from movie_pipeline.shots import get_frame

    moving = tmp_path / "moving.mp4"
    blue = tmp_path / "blue.mp4"
    audio = tmp_path / "voice.wav"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30", "-t", 6, "-c:v", "libx264", "-preset", "ultrafast", moving)
    ffmpeg("-f", "lavfi", "-i", "color=c=blue:size=640x360:rate=30", "-t", 3, "-c:v", "libx264", "-preset", "ultrafast", blue)
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000", "-t", 2.4, audio)
    target = tmp_path / "montage.mp4"
    speech_video(moving, audio, [
        {"start": 3, "duration": 1.2, "freeze": 0.4},
        {"source": str(blue), "start": 1, "duration": 1.2},
    ], target, None, tmp_path / "render")
    assert abs(duration(target)-duration(audio)) < 0.09
    held_a = np.asarray(get_frame(target, 0.9), dtype=float)
    held_b = np.asarray(get_frame(target, 1.1), dtype=float)
    assert np.abs(held_a-held_b).mean() < 1.0
    earlier = np.asarray(get_frame(target, 0.3), dtype=float)
    assert np.abs(earlier-held_a).mean() > 2
    next_film = np.asarray(get_frame(target, 1.6), dtype=float)
    assert next_film[:, :, 2].mean() > 240
    assert next_film[:, :, 0].mean() < 10
    ffmpeg("-xerror", "-i", target, "-f", "null", "-")


def test_4k_still_frame_export_and_embedded_chapters(monkeypatch, tmp_path):
    import json

    from movie_pipeline.core import run
    from movie_pipeline.render import embed_chapters
    monkeypatch.setenv('VIDEO_RESOLUTION', '2160')
    source = tmp_path/'source.mp4'
    audio = tmp_path/'audio.wav'
    ffmpeg('-f','lavfi','-i','testsrc2=size=640x360:rate=30','-t',2,'-c:v','libx264','-preset','ultrafast',source)
    ffmpeg('-f','lavfi','-i','sine=frequency=220:sample_rate=48000','-t',.6,audio)
    label = tmp_path/'label.png'
    label_image(label, 'Synthetic 4K test', 1, 2026)
    target = tmp_path/'4k.mp4'
    speech_video(source,audio,[{'start':.8,'duration':.6,'still':True}],target,label,tmp_path/'render')
    video = next(s for s in probe(target)['streams'] if s['codec_type']=='video')
    assert (video['width'],video['height']) == (3840,2160)
    embed_chapters(target,[{'start':0,'title':'Intro = test #1; safe'},{'start':.3,'title':'Outro'}],duration(target),tmp_path)
    chapters = json.loads(run(['ffprobe','-v','error','-show_chapters','-of','json',target]))['chapters']
    assert len(chapters)==2
    assert chapters[0]['tags']['title']=='Intro = test #1; safe'


def test_ending_holds_picture_three_seconds_after_speech(monkeypatch, tmp_path):
    import numpy as np

    from movie_pipeline.render import finish
    from movie_pipeline.shots import get_frame
    from movie_pipeline.timing import DurationSpec
    monkeypatch.setenv('VIDEO_RESOLUTION', '720')
    source = tmp_path/'source.mp4'
    ffmpeg('-f','lavfi','-i','testsrc2=size=1280x720:rate=30','-f','lavfi','-i',
           'sine=frequency=220:sample_rate=48000','-t',2,'-c:v','libx264','-preset','ultrafast','-c:a','aac',source)
    target = tmp_path/'final.mp4'
    seconds,speed = finish([source],target,tmp_path,False,spec=DurationSpec(5,4.9,5.1))
    assert speed == 1
    assert seconds == pytest.approx(5,abs=.1)
    a=np.asarray(get_frame(target,2.2),dtype=float)
    b=np.asarray(get_frame(target,3.4),dtype=float)
    assert np.abs(a-b).mean()<1
    ffmpeg('-xerror','-i',target,'-f','null','-')


def test_outside_runtime_window_still_exports_complete_video(monkeypatch,tmp_path):
    from movie_pipeline.render import finish
    from movie_pipeline.timing import DurationSpec
    monkeypatch.setenv('VIDEO_RESOLUTION','720')
    source=tmp_path/'source.mp4'
    ffmpeg('-f','lavfi','-i','color=c=blue:size=1280x720:rate=30','-f','lavfi','-i',
           'sine=frequency=220:sample_rate=48000','-t',2,'-c:v','libx264','-preset','ultrafast','-c:a','aac',source)
    target=tmp_path/'final.mp4'
    seconds,speed=finish([source],target,tmp_path,False,spec=DurationSpec(480,360,600))
    assert speed == 1
    assert seconds == pytest.approx(5, abs=.08)
    assert any(stream['codec_type']=='audio' for stream in probe(target)['streams'])
    ffmpeg('-xerror','-i',target,'-f','null','-')


def test_countdown_tracker_overlay_appears_only_in_its_window(monkeypatch, tmp_path):
    import numpy as np

    from movie_pipeline.render import finish, tracker_image
    from movie_pipeline.shots import get_frame
    from movie_pipeline.timing import DurationSpec
    monkeypatch.setenv('VIDEO_RESOLUTION', '720')
    source = tmp_path/'source.mp4'
    ffmpeg('-f', 'lavfi', '-i', 'color=c=black:size=1280x720:rate=30', '-f', 'lavfi', '-i',
           'sine=frequency=220:sample_rate=48000', '-t', 4, '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', source)
    tracker = tmp_path/'tracker.png'
    width, height = tracker_image(tracker, 5, 3)
    finish([source], tmp_path/'final.mp4', tmp_path, False, spec=DurationSpec(4, 0, 20, automatic=True),
           overlays=[(tracker, 1.0, 2.5)])
    inside = np.asarray(get_frame(tmp_path/'final.mp4', 1.8), dtype=float)
    outside = np.asarray(get_frame(tmp_path/'final.mp4', 3.2), dtype=float)
    corner = (slice(30, 30+height), slice(1280-36-width, 1280-36))
    assert inside[corner].max() > 150 and outside[corner].max() < 40


def test_j_cut_mixes_trailer_sound_under_the_end_of_a_part(tmp_path):
    import numpy as np

    from movie_pipeline.render import j_cut
    part = tmp_path/'part.mov'
    # Audio deliberately 0.33 s shorter than the video, as narration parts can be.
    ffmpeg('-f', 'lavfi', '-i', 'color=c=black:size=320x180:rate=30:duration=3', '-f', 'lavfi', '-i',
           'anullsrc=r=48000:cl=stereo', '-t', 3, '-af', 'atrim=end=2.67', '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'alac', part)
    source = tmp_path/'trailer.mp4'
    ffmpeg('-f', 'lavfi', '-i', 'color=c=red:size=320x180:rate=30', '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000',
           '-t', 6, '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', source)
    assert j_cut(part, source, 3.0, lead=0.6)
    from movie_pipeline.render import validate_audio_video_duration
    validate_audio_video_duration(part, metadata=probe(part))     # lengths now match exactly
    raw = tmp_path/'a.raw'
    ffmpeg('-i', part, '-f', 's16le', '-ac', '1', '-ar', '8000', raw)
    samples = np.frombuffer(raw.read_bytes(), dtype=np.int16).astype(float)
    quiet = np.abs(samples[:8000]).max()
    assert quiet < 50                                  # first second silent
    tail = np.abs(samples[-4800:]).max()
    assert tail > 300 and tail > 10 * max(quiet, 1)   # trailer sound rising in the last 0.6 s (level varies by FFmpeg build)


@pytest.mark.parametrize("audio_seconds, repaired", [(3.6, True), (0.6, False)])
def test_final_mix_repairs_small_clip_timing_gaps_but_not_missing_audio(monkeypatch, tmp_path, audio_seconds, repaired):
    from movie_pipeline.core import PipelineError
    from movie_pipeline.render import LOSSLESS_AUDIO, finish
    from movie_pipeline.timing import DurationSpec
    monkeypatch.setenv('VIDEO_RESOLUTION', '720')
    good, bad = tmp_path/'good.mp4', tmp_path/'bad.mp4'
    for path, seconds in ((good, 4), (bad, audio_seconds)):
        ffmpeg('-f', 'lavfi', '-i', 'testsrc2=size=1280x720:rate=30:duration=4', '-f', 'lavfi', '-i', 'sine=frequency=220:sample_rate=48000',
               '-t', 4, '-af', f'atrim=end={seconds}', '-c:v', 'libx264', '-preset', 'ultrafast', *LOSSLESS_AUDIO, path)
    target = tmp_path/'final.mp4'
    if repaired:
        seconds, _ = finish([good, bad], target, tmp_path/'work', False, spec=DurationSpec(8, 0, 30, automatic=True))
        assert seconds > 7.5
    else:
        with pytest.raises(PipelineError, match='timing is incomplete'):
            finish([good, bad], target, tmp_path/'work', False, spec=DurationSpec(8, 0, 30, automatic=True))
