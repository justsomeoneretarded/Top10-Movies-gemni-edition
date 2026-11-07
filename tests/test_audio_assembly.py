"""Real codec regressions for mixed preview, narration and reaction audio."""
from __future__ import annotations

import json
import shutil
import subprocess
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from movie_pipeline import render
from movie_pipeline.core import PipelineError, duration, ffmpeg
from movie_pipeline.timing import DurationSpec

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg is required')
RATE = 48000


@pytest.fixture
def source(monkeypatch, tmp_path):
    monkeypatch.setenv('VIDEO_RESOLUTION', '320x180')
    path = tmp_path / 'source.mp4'
    ffmpeg('-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=30:duration=5',
           '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=5',
           '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
           '-c:a', 'aac', '-b:a', '128k', path)
    return path


def tone(path, frequency, codec='pcm_s16le', seconds=.8):
    ffmpeg('-f', 'lavfi', '-i', f'sine=frequency={frequency}:sample_rate={RATE}',
           '-t', seconds, '-ac', '1', '-c:a', codec, path)
    return path


def decoded(path):
    result = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-err_detect', 'explode',
        '-i', str(path), '-map', '0:a:0', '-ac', '1', '-ar', str(RATE), '-f', 'f32le', '-'],
        capture_output=True, check=True)
    assert not result.stderr
    return np.frombuffer(result.stdout, dtype='<f4')


def video_hash(path):
    result = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:v:0',
        '-c:v', 'copy', '-f', 'hash', '-hash', 'sha256', '-'], capture_output=True, check=True)
    return result.stdout


def frequency(samples, start, end):
    selected = samples[round(start*RATE):round(end*RATE)]
    spectrum = np.abs(np.fft.rfft(selected*np.hanning(len(selected))))
    return np.fft.rfftfreq(len(selected), 1/RATE)[int(np.argmax(spectrum))]


def voice_part(footage, audio, output, work, **shot):
    render.speech_video(footage, audio, [{'start': 0, 'duration': duration(audio), **shot}],
                        output, None, work)
    return output


def test_mixed_preview_pcm16_pcm24_float_and_reaction_keep_every_audio_sample(source, tmp_path):
    preview, trailer = tmp_path/'preview.mp4', tmp_path/'trailer.mp4'
    render.trailer_video(source, 0, 1, preview, muted=True)
    speech = voice_part(source, tone(tmp_path/'voice.wav', 900), tmp_path/'speech.mp4', tmp_path/'speech')
    render.trailer_video(source, 1, 1, trailer)
    gif = tmp_path/'reaction.gif'
    Image.new('RGB', (160, 90), 'red').save(gif, save_all=True,
        append_images=[Image.new('RGB', (160, 90), 'blue')], duration=200, loop=0)
    reaction = voice_part(gif, tone(tmp_path/'float.wav', 1320, 'pcm_f32le'),
        tmp_path/'reaction.mp4', tmp_path/'reaction', source=str(gif), image=True, animation=True)
    ending = voice_part(source, tone(tmp_path/'pcm24.wav', 1760, 'pcm_s24le'),
                        tmp_path/'ending.mp4', tmp_path/'ending')
    parts = [preview, speech, trailer, reaction, ending]
    streams = [render._audio_stream(part) for part in parts]
    assert all(render._canonical_audio(stream) for stream in streams)
    assert len({stream['extradata'] for stream in streams}) == 1
    assert np.abs(decoded(preview)).max() == 0

    # Decode the mixed stream copy, not just each homogeneous component alone.
    listing, combined = tmp_path/'mixed.txt', tmp_path/'mixed.mp4'
    render.concat_file(parts, listing)
    ffmpeg('-f', 'concat', '-safe', '0', '-i', listing, '-c', 'copy', combined)
    expected = np.concatenate([decoded(part) for part in parts])
    assert np.array_equal(decoded(combined), expected)

    output = tmp_path/'final.mp4'
    seconds, speed = render.finish(parts, output, tmp_path, False, DurationSpec(automatic=True))
    assert speed == 1
    assert seconds == pytest.approx(7.4, abs=.06)
    samples = decoded(output)
    for start, end, wanted in [(1.1, 1.7, 900), (1.95, 2.6, 440), (2.9, 3.5, 1320), (3.7, 4.3, 1760)]:
        assert frequency(samples, start, end) == pytest.approx(wanted, abs=10)
    assert render.validate_audio_video_duration(output, 7.4) == pytest.approx(7.4, abs=.06)
    report = json.loads((tmp_path/'audio-assembly.json').read_text())
    assert report['status'] == 'passed' and report['normalized_part_count'] == 0
    assert report['concatenated_strict_decode'] == report['final_strict_decode'] == 'passed'


def test_foreign_and_legacy_audio_are_normalized_while_video_bytes_stay_unchanged(source, tmp_path):
    preview = tmp_path/'preview.mp4'
    render.trailer_video(source, 0, 1, preview, muted=True)
    speech = voice_part(source, tone(tmp_path/'voice.wav', 900), tmp_path/'speech.mp4', tmp_path/'speech')
    legacy, foreign = tmp_path/'legacy16.mp4', tmp_path/'foreign.mp4'
    ffmpeg('-i', speech, '-c:v', 'copy', '-c:a', 'alac', '-sample_fmt', 's16p', legacy)
    ffmpeg('-i', source, '-t', 1, '-map', '0:v:0', '-map', '0:a:0', '-c:v', 'copy',
           '-c:a', 'aac', '-ar', '44100', '-ac', '1', foreign)
    assert render._audio_stream(legacy)['bits_per_raw_sample'] == '16'
    normalize = tmp_path/'normalize'
    normalize.mkdir()
    originals = [preview, legacy, foreign]
    prepared = render._normalized_audio_parts(originals, normalize)
    assert prepared[0] == preview
    for old, new in zip(originals, prepared):
        assert video_hash(old) == video_hash(new)
        assert render._canonical_audio(render._audio_stream(new))
    assert np.array_equal(decoded(legacy), decoded(prepared[1]))
    output = tmp_path/'final.mp4'
    seconds, _ = render.finish(originals, output, tmp_path, False, DurationSpec(automatic=True))
    assert seconds == pytest.approx(5.8, abs=.08)
    assert frequency(decoded(output), 1.1, 1.6) == pytest.approx(900, abs=10)
    report = json.loads((tmp_path/'audio-assembly.json').read_text())
    assert report['normalized_part_count'] == 2
    assert not list(tmp_path.glob('audio-normalized-*'))


def test_corrupt_alac_packets_are_rejected_before_final_music_can_hide_them(source, tmp_path, monkeypatch):
    speech = voice_part(source, tone(tmp_path/'voice.wav', 900), tmp_path/'speech.mp4', tmp_path/'speech')
    # FFmpeg versions differ in how they concatenate mismatched ALAC headers.
    # Damage an actual packet instead, keeping the MP4 index and codec header valid.
    packets = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_packets',
        '-show_entries', 'packet=pos,size', '-of', 'json', str(speech),
    ]))['packets']
    packet = next(packet for packet in packets if int(packet['size']) > 32)
    payload = bytearray(speech.read_bytes())
    start, size = int(packet['pos']), int(packet['size'])
    payload[start:start+size] = bytes(size)
    corrupt = tmp_path/'corrupt.mp4'
    corrupt.write_bytes(payload)
    monkeypatch.setattr(render, 'music_bed', Mock(side_effect=AssertionError('Corrupt speech reached the final mix')))
    output = tmp_path/'final.mp4'
    with pytest.raises(PipelineError):
        render.finish([corrupt], output, tmp_path, False, DurationSpec(automatic=True))
    assert not output.exists()
    report = json.loads((tmp_path/'audio-assembly.json').read_text())
    assert report['status'] == 'failed' and report['stage'] == 'checking-concatenated-audio'


def test_real_reported_audio_truncation_cannot_pass_duration_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(render, 'probe', lambda _: {'format': {'duration': '610.8'}, 'streams': [
        {'codec_type': 'video', 'start_time': '0', 'duration': '610.8'},
        {'codec_type': 'audio', 'start_time': '0', 'duration': '127.3'},
    ]})
    with pytest.raises(PipelineError, match='Audio/video timing is incomplete'):
        render.validate_audio_video_duration(tmp_path/'failed-export.mp4')


def test_valid_packet_timestamp_gaps_keep_late_speech_at_its_original_time(monkeypatch, tmp_path):
    monkeypatch.setenv('VIDEO_RESOLUTION', '320x180')
    source = tmp_path/'gapped.mp4'
    ffmpeg('-f', 'lavfi', '-i', 'color=c=0x16243a:size=320x180:rate=30:duration=20',
           '-f', 'lavfi', '-i', 'sine=frequency=180:sample_rate=48000:duration=19.6',
           '-af', "asetpts='PTS+gte(T,10)*0.4/TB'",
           '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'alac',
           '-sample_fmt', 's32p', '-ac', '2', source)
    # Explicitly insert a 400 ms presentation gap; do not depend on AAC loop
    # behavior, which differs between FFmpeg versions. Samples remain intact.
    assert duration(source) == pytest.approx(20, abs=.04)
    assert len(decoded(source))/RATE == pytest.approx(19.6, abs=.01)
    output = tmp_path/'final.mp4'
    seconds, speed = render.finish([source], output, tmp_path, False, DurationSpec(automatic=True))
    assert speed == 1
    assert seconds == pytest.approx(23, abs=.06)
    assert render.validate_audio_video_duration(output, 23) == pytest.approx(23, abs=.06)
    assert frequency(decoded(output), 19.75, 19.95) == pytest.approx(180, abs=10)


def test_title_card_hold_near_source_end_preserves_the_frame_and_silent_audio(source, tmp_path):
    output = tmp_path/'title-card.mp4'
    render.trailer_video(source, 4.9, 1.5, output, muted=True, still=True)
    frames = []
    for position in (.15, 1.2):
        result = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(position), '-i', str(output),
            '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True)
        frames.append(np.frombuffer(result.stdout, dtype=np.uint8).astype(float))
    assert np.abs(frames[0]-frames[1]).mean() < 1
    assert render.validate_audio_video_duration(output, 1.5) == pytest.approx(1.5, abs=.05)
    assert np.abs(decoded(output)).max() == 0


def test_finish_with_timeline_adds_effects_and_reaches_youtube_loudness(source, tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSITION_FX", "true")
    preview = tmp_path/'preview.mp4'
    render.trailer_video(source, 0, 2, preview, muted=True)
    speech = voice_part(source, tone(tmp_path/'voice.wav', 900, seconds=3), tmp_path/'speech.mp4', tmp_path/'speech')
    timeline = [{'start': 0, 'duration': 2, 'type': 'review-preview', 'shots': [{'muted': True}]},
                {'start': 2, 'duration': 3, 'type': 'narration', 'section_kind': 'film', 'film': 'Top'}]
    output = tmp_path/'final.mp4'
    seconds, speed = render.finish([preview, speech], output, tmp_path, False, DurationSpec(automatic=True),
                                   timeline=timeline, top_title='Top')
    assert seconds == pytest.approx(8, abs=.1)
    report = json.loads((tmp_path/'audio-assembly.json').read_text())
    assert report['status'] == 'passed'
    assert report['effect_events'] == 2
    loudness = render.measure_loudness(output)
    assert loudness['integrated'] == pytest.approx(-14, abs=1.5)
    assert loudness['true_peak'] <= -0.5
    # The narration tone stays dominant while it plays.
    assert frequency(decoded(output), 2.5, 4.5) == pytest.approx(900, abs=10)
