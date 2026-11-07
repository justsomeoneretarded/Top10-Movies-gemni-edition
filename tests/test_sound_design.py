import json
import subprocess
import wave

import numpy as np
import pytest

from movie_pipeline.render import measure_loudness, normalize_loudness
from movie_pipeline.sound_design import (CONTROL_RATE, MUSIC_LEVELS, delivery_speed, effect_events, effects_track,
                                         gain_curve, impact, shape_music, whoosh)

TIMELINE = [
    {"start": 0, "duration": 5, "type": "cold-open"},
    {"start": 5, "duration": 10, "type": "narration", "section_kind": "intro"},
    {"start": 15, "duration": 5, "type": "review-preview", "shots": [{"muted": True}]},
    {"start": 20, "duration": 10, "type": "narration", "section_kind": "film", "film": "Top Pick"},
    {"start": 30, "duration": 4, "type": "trailer-audio"},
    {"start": 34, "duration": 6, "type": "narration", "section_kind": "outro"},
]


def test_music_is_louder_between_narration_and_nearly_off_under_trailer_dialogue():
    curve = gain_curve(TIMELINE, 1.0, 45, 40, top_title="Top Pick")
    at = lambda seconds: curve[int(seconds * CONTROL_RATE)]
    assert at(2) == pytest.approx(MUSIC_LEVELS["cold-open"], rel=.02)
    assert at(10) == pytest.approx(MUSIC_LEVELS["narration"], rel=.02)
    assert at(17.5) == pytest.approx(MUSIC_LEVELS["review-preview"], rel=.02)
    assert at(25) == pytest.approx(MUSIC_LEVELS["narration-top"], rel=.02)
    assert at(32) == pytest.approx(MUSIC_LEVELS["trailer-audio"], rel=.05)
    assert at(43) == pytest.approx(MUSIC_LEVELS["ending"], rel=.02)
    # Ramps instead of jumps.
    assert np.max(np.abs(np.diff(curve))) < 0.2


def test_curve_follows_final_tempo():
    curve = gain_curve(TIMELINE, 1.25, 40, 32)
    # Preview at 15-20s plays at 12-16s after a 1.25x tempo change.
    assert curve[int(14 * CONTROL_RATE)] == pytest.approx(MUSIC_LEVELS["review-preview"], rel=.02)


def test_rank_reveals_get_whoosh_then_impact(monkeypatch):
    monkeypatch.setenv("TRANSITION_FX", "true")
    events = effect_events(TIMELINE, 1.0)
    names = [(round(t, 2), name) for t, name, _ in events]
    assert names[0] == (0.02, "impact")
    assert (14.68, "whoosh") in names and (15.5, "impact") in names
    assert any(name == "whoosh" and 33 < t < 34 for t, name in names)  # outro transition


def test_effects_are_finite_and_peak_normalized():
    for sound in (whoosh(), impact()):
        assert np.isfinite(sound).all()
        assert sound.shape[1] == 2
        assert 0.6 < float(np.max(np.abs(sound))) <= 1.0  # panned whoosh peaks lower per channel


def test_effects_track_and_music_shaping_write_valid_audio(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSITION_FX", "true")
    fx = tmp_path / "fx.wav"
    assert effects_track(fx, TIMELINE, 1.0, 45) == 4
    with wave.open(str(fx)) as stream:
        data = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2)
    assert np.abs(data[int(15.5*48000):int(16*48000)]).max() > 3000
    assert np.abs(data[int(8*48000):int(9*48000)]).max() == 0
    music = tmp_path / "music.wav"
    tone = (np.sin(np.linspace(0, 2*np.pi*220*45, 45*48000))*3000).astype("<i2")
    with wave.open(str(music), "wb") as stream:
        stream.setnchannels(2), stream.setsampwidth(2), stream.setframerate(48000)
        stream.writeframes(np.column_stack((tone, tone)).tobytes())
    shape_music(music, gain_curve(TIMELINE, 1.0, 45, 40))
    with wave.open(str(music)) as stream:
        shaped = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2)
    loud = np.abs(shaped[2*48000:3*48000]).max()
    quiet = np.abs(shaped[31*48000:32*48000]).max()
    assert loud > 8 * quiet


def test_delivery_shapes_pace_within_safe_bounds():
    assert delivery_speed(1.08, "hype") == pytest.approx(1.134)
    assert delivery_speed(1.0, "dramatic") == 0.94
    assert delivery_speed(1.3, "hype") == 1.18
    assert delivery_speed(1.0, "unknown") == 1.0


def test_export_is_normalized_to_youtube_loudness(tmp_path):
    source = tmp_path / "quiet.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=black:s=320x180:r=30:d=12",
                    "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000:duration=12",
                    "-af", "volume=-3dB", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source)], check=True)
    before = measure_loudness(source)["integrated"]
    assert before < -20
    report = normalize_loudness(source, 12, tmp_path)
    after = measure_loudness(source)
    assert report["gain_db"] > 5
    assert after["integrated"] == pytest.approx(-14, abs=1.5)
    assert after["true_peak"] <= -0.5


def test_each_genre_gets_its_own_reveal_sounds(monkeypatch):
    monkeypatch.setenv("TRANSITION_FX", "true")
    from movie_pipeline.sound_design import GENRE_SOUNDS, effect_events, effects_track
    timeline = [{"type": "cold-open", "start": 0}, {"type": "review-preview", "start": 10}]
    horror = effect_events(timeline, 1.0, "horror")
    assert ("stinger" in [name for _, name, _ in horror]) and ("subdrop" in [name for _, name, _ in horror])
    riser = next(when for when, name, _ in horror if name == "riser")
    assert abs(riser + 1.6 - 10.5) < 1e-6           # the riser ends exactly on the reveal hit
    comedy = [name for _, name, _ in effect_events(timeline, 1.0, "comedy")]
    assert "pop" in comedy and "stinger" not in comedy
    assert {"horror", "comedy", "family", "romance", "action", "adventure", "fantasy", "scifi", "thriller"} <= set(GENRE_SOUNDS)


def test_effects_track_renders_every_genre_sound(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSITION_FX", "true")
    from movie_pipeline.sound_design import GENRE_SOUNDS, effects_track
    timeline = [{"type": "cold-open", "start": 0}, {"type": "review-preview", "start": 3}]
    for genre in GENRE_SOUNDS:
        monkeypatch.setenv("VIDEO_GENRE", genre)
        assert effects_track(tmp_path / f"{genre}.wav", timeline, 1.0, 6.0) >= 3


def test_generated_genre_score_replaces_the_bundled_track(tmp_path, monkeypatch):
    import wave

    import numpy as np
    import soundfile as sf

    from movie_pipeline import soundtrack
    score = tmp_path / "score.wav"
    t = np.arange(48000 * 30) / 48000
    sf.write(score, np.column_stack([0.2 * np.sin(2 * np.pi * 110 * t)] * 2), 48000)
    monkeypatch.setattr(soundtrack, "GENERATED_SCORE", score)
    assert "ACE-Step" in soundtrack.music_credit("drama")
    out = tmp_path / "bed.wav"
    soundtrack.music_bed(out, 40, theme="drama")
    with wave.open(str(out)) as stream:
        assert stream.getnframes() == 48000 * 40
    monkeypatch.setattr(soundtrack, "GENERATED_SCORE", tmp_path / "missing.wav")
    assert "Scott Buckley" in soundtrack.music_credit("drama")


def test_no_automatic_genre_music_for_an_unspecified_episode(monkeypatch):
    from movie_pipeline.soundtrack import theme_for
    monkeypatch.setenv("VIDEO_GENRE", "horror")
    assert theme_for("8 movies to watch") == "neutral"
    assert theme_for("8 horror movies") == "horror"
    assert theme_for("8 movies", "Romance recommendations") == "romance"


def test_transition_fx_are_opt_in(monkeypatch):
    monkeypatch.delenv("TRANSITION_FX", raising=False)
    assert effect_events(TIMELINE, 1.0) == []


def test_optional_reveal_sound_follows_announcement_not_native_scene(monkeypatch):
    from movie_pipeline.sound_design import effect_events
    monkeypatch.setenv("TRANSITION_FX", "true")
    announcement = {"start": 10, "duration": 3, "type": "narration",
                    "section_kind": "film", "rank_announcement": True}
    preview = {"start": 13, "duration": 8, "type": "review-preview"}
    assert effect_events([announcement, preview], 1) == effect_events([announcement], 1)
    assert effect_events([announcement], 1)
