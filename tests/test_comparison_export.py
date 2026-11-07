"""Real media checks for same-timeline A/B output and clean background beds."""

import cv2
import numpy as np
import pytest
import soundfile as sf

from movie_pipeline.augmentation import FrameAugmenter, augment_audio, export_augmented
from movie_pipeline.core import duration, ffmpeg, probe
from movie_pipeline.cta_animation import add_cta_animation, interaction_frame
from movie_pipeline.people import insert_trailer_hold
from movie_pipeline.soundtrack import music_bed, narration_rate, theme_for
from movie_pipeline.voice import paced_take


def test_visual_transform_is_deterministic_and_temporally_stable():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (90, 160, 3), dtype=np.uint8)
    first, second = FrameAugmenter(90, 160), FrameAugmenter(90, 160)
    assert np.array_equal(first(image), second(image))
    assert np.array_equal(first(image), first(image))
    assert not np.array_equal(first(image), image)
    occupied = np.concatenate((first.first, first.second))
    assert len(set(occupied)) == len(occupied)
    assert len(occupied)/image.shape[0]/image.shape[1] == pytest.approx(.01, abs=.001)
    assert first(image).shape == image.shape


@pytest.mark.parametrize("theme", ["action", "drama", "romance", "mystery"])
def test_music_is_continuous_across_blocks_and_chord_changes(tmp_path, theme):
    target = tmp_path / "music.wav"
    music_bed(target, 19.3, theme=theme)
    samples, rate = sf.read(target)
    assert rate == 48000 and len(samples) == round(19.3*rate)
    assert np.isfinite(samples).all()
    assert np.max(np.abs(np.diff(samples[:, 0]))) < .015
    assert np.max(np.abs(samples)) < .5
    assert .003 < np.sqrt(np.mean(samples[rate:-rate]**2)) < .025
    assert np.max(np.abs(samples[:10])) < .0001
    assert np.max(np.abs(samples[-10:])) < .0001


def test_theme_and_pacing_follow_topic_not_incidental_brief_words():
    assert theme_for("Top 5 Romance Movies", "Use action footage") == "romance"
    assert narration_rate(theme_for("Top 5 Action Movies")) == 1.08
    assert narration_rate(theme_for("Top 5 Drama Shows")) == 1


def test_audio_overlap_preserves_length_and_silence(tmp_path):
    rate = 16000
    time = np.arange(rate*18)/rate
    signal = .08*np.sin(2*np.pi*220*time)
    signal[rate*9:rate*11] = 0
    source, target = tmp_path / "source.wav", tmp_path / "modified.wav"
    sf.write(source, np.column_stack((signal, signal)), rate, subtype="PCM_24")
    augment_audio(source, target)
    changed, actual_rate = sf.read(target)
    assert actual_rate == rate and len(changed) == len(signal)
    assert np.isfinite(changed).all() and np.abs(changed).max() < .2
    assert np.sqrt(np.mean(changed[rate:rate*7]**2)) > .025
    assert np.abs(changed[rate*9+rate//2:rate*10+rate//2]).max() < 1e-5
    # Check that overlap boundaries do not create clicks.
    assert np.max(np.abs(np.diff(changed[:, 0]))) < .03


def test_portrait_replaces_time_without_repeating_or_shortening_scene():
    plan = [{"start": 10, "duration": 8}, {"start": 40, "duration": 8}]
    result = insert_trailer_hold(plan, {"source": "trailer.mp4", "start": 14, "still": True, "image": False}, 5.5)
    assert sum(cut["duration"] for cut in result) == 16
    assert result[1]["start"] == 15.5
    assert plan[0]["start"] == 10


def test_comparison_and_cta_real_render_keep_full_video_and_audio(tmp_path):
    source, target = tmp_path / "clean.mp4", tmp_path / "augmented.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30", "-f", "lavfi",
           "-i", "sine=frequency=330:sample_rate=48000", "-t", 3,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "alac", "-sample_fmt", "s32p", source)
    add_cta_animation(source, "Like and subscribe.", [{"start": .3, "end": 1.5, "text": "Like and subscribe."}], tmp_path / "cta")
    report = export_augmented(source, target, tmp_path / "augmentation")
    assert report["frames"] == 90
    for path in (source, target):
        assert duration(path) == pytest.approx(3, abs=.05)
        video = next(s for s in probe(path)["streams"] if s["codec_type"] == "video")
        assert (video["width"], video["height"]) == (320, 180)
        ffmpeg("-xerror", "-i", path, "-f", "null", "-")
    cap = cv2.VideoCapture(str(target))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 90
    cap.release()
    assert not list((tmp_path / "augmentation").glob("*.wav"))
    assert not np.array_equal(np.asarray(interaction_frame(0, [.5, 1, 2])),
                              np.asarray(interaction_frame(2.5, [.5, 1, 2])))


def test_paced_narration_keeps_pitch_and_complete_duration(tmp_path):
    source = tmp_path / "voice.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", 4, source)
    result = paced_take(source, 1.15)
    signal, rate = sf.read(result)
    assert len(signal)/rate == pytest.approx(4/1.15, abs=.04)
    window = signal[rate:2*rate]
    frequency = np.fft.rfftfreq(len(window), 1/rate)[np.argmax(np.abs(np.fft.rfft(window)))]
    assert frequency == pytest.approx(440, abs=3)


def test_actual_reference_voice_remains_identifiable_after_comparison(tmp_path):
    from pathlib import Path

    from movie_pipeline.audio_review import verify_export_audio
    from movie_pipeline.render import finish, speech_video
    from movie_pipeline.timing import DurationSpec

    reference = Path(__file__).resolve().parents[1] / 'assets/voices/warguy.ogg'
    source, speech, part = tmp_path/'scene.mp4', tmp_path/'voice.wav', tmp_path/'part.mp4'
    ffmpeg('-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=30', '-t', 8,
           '-c:v', 'libx264', '-preset', 'ultrafast', source)
    ffmpeg('-i', reference, '-t', 8, '-ar', 48000, '-ac', 1, speech)
    import os
    old = os.environ.get('VIDEO_RESOLUTION')
    os.environ['VIDEO_RESOLUTION'] = '320x180'
    try:
        speech_video(source, speech, [{'start':0, 'duration':8}], part, None, tmp_path/'part-work')
        clean, modified = tmp_path/'clean.mp4', tmp_path/'modified.mp4'
        finish([part], clean, tmp_path/'finish', True, DurationSpec(11, 1, 30, automatic=True), music_theme='action')
        export_augmented(clean, modified, tmp_path/'comparison')
        events = [{'type':'narration', 'start':0, 'duration':8, 'text':'Verified reference recording', 'source_audio':str(speech)}]
        for video in (clean, modified):
            report = verify_export_audio(video, events, 1, None, tmp_path)
            assert report['accepted'], report
            assert report['checks'][0]['method'] == 'verified-recording'
    finally:
        if old is None:
            os.environ.pop('VIDEO_RESOLUTION', None)
        else:
            os.environ['VIDEO_RESOLUTION'] = old


def test_character_hold_is_a_frame_from_the_same_trailer_and_ambiguity_is_skipped():
    from pathlib import Path
    from types import SimpleNamespace

    from movie_pipeline.people import trailer_still_for
    source = Path("film-trailer.mp4")
    film = {"title":"Film", "cast_profiles":[{"name":"Jane Example", "character":"Detective Lane"},
                                           {"name":"John Other", "character":"Inspector"}]}
    selector = SimpleNamespace(faces={str(source):[True, True]}, card_kind=lambda *args: None,
                               reuse_penalty=lambda *args: 0,
                               relevance=lambda path, text: [.6, .3] if "Jane Example" in text else [.3, .6])
    lookup = {str(source):[{"start":10, "end":18}, {"start":30, "end":38}]}
    hold = trailer_still_for("Detective Lane has to choose.", film, selector, [source], lookup)
    assert hold['source'] == str(source) and hold['start'] == 14
    assert hold['still'] and hold['image'] is False
    assert trailer_still_for("An unknown actor.", film, selector, [source], lookup) is None
    selector.relevance = lambda *args: [.3, .3]
    assert trailer_still_for("Jane Example plays the lead.", film, selector, [source], lookup) is None


def test_trailer_audio_breaks_use_whole_utterances_and_distinct_windows():
    from movie_pipeline.cli import choose_beat
    cues = [{'start':20, 'end':22, 'text':'We need to get'},
            {'start':22.1, 'end':24, 'text':'out of here now.'},
            {'start':40, 'end':42, 'text':'This is our last chance.'}]
    shots = [{'start':10, 'end':60, 'motion':8}]
    first = choose_beat(cues, shots, 100, True, maximum=7)
    assert first['text'] == 'We need to get out of here now.'
    second = choose_beat(cues, shots, 100, True, maximum=7, excluded=[first])
    assert second['text'] == 'This is our last chance.'
    assert second['start'] > first['end']
    assert choose_beat(cues[:2], shots, 100, True, maximum=3) is None


def test_augmented_audio_does_not_change_protected_narration_samples(tmp_path):
    rate = 16000
    time = np.arange(rate*11)/rate
    samples = np.column_stack((.1*np.sin(2*np.pi*220*time), .1*np.sin(2*np.pi*330*time)))
    source, target = tmp_path/'original.wav', tmp_path/'protected.wav'
    sf.write(source, samples, rate, subtype='PCM_24')
    original, _ = sf.read(source)
    augment_audio(source, target, protected_intervals=[(2, 9)])
    changed, _ = sf.read(target)
    assert len(changed) == len(original)
    assert np.max(np.abs(changed[rate*2:rate*9]-original[rate*2:rate*9])) < 2e-7
    assert np.max(np.abs(changed[:rate]-original[:rate])) > .001
