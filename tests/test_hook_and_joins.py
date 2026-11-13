"""The review hook is the movie's most intense moment, and scene sound never drops dead at a cut."""
from pathlib import Path
import subprocess

import numpy as np
import pytest

import movie_pipeline.moments as moments
from movie_pipeline import intensity, sound_design


def test_loud_punchy_window_scores_above_quiet_talk():
    env = np.concatenate([np.full(100, -40.0), np.full(100, -14.0), np.full(100, -38.0)])
    assert intensity.loudness(env, 10, 20) > 0.8 > 0.4 > intensity.loudness(env, 0, 10)


def test_intensity_query_names_the_genre_peak():
    assert "screaming" in intensity.intensity_query("horror")
    assert "argument" in intensity.intensity_query("drama")
    assert intensity.intensity_query("unknown") == intensity.GENERIC


def test_gemini_energy_counts_most_when_verified():
    assert intensity.score(0.2, 0.2, 0.2, 1.0) > intensity.score(0.6, 0.6, 0.6)


def test_hook_is_the_most_intense_short_scene_not_the_most_relevant():
    rows = [{"start": 0, "duration": 8, "score": 1.4, "intensity": 0.2},   # on-topic, calm talk
            {"start": 20, "duration": 7, "score": 0.9, "intensity": 0.95},  # the fight
            {"start": 40, "duration": 18, "score": 1.0, "intensity": 1.0}]  # too long to hit hard
    hooks, rest = moments.hook_order(rows)
    assert hooks[0]["start"] == 20
    assert all(r["duration"] <= moments.HOOK_MAX_SECONDS for r in hooks)
    assert rest[0]["start"] == 0


def test_planner_opens_each_review_on_the_intense_scene(monkeypatch, tmp_path):
    source = Path("trailer.mp4")
    lookup = {str(source): [{"start": 10 * j, "end": 10 * j + 6, "motion": [1, 9, 1, 1, 1][j]} for j in range(5)]}

    class Selector:
        def __init__(self):
            self.used = []

        def identity_relevance(self, path, query):
            if "most intense moment" in query:
                return np.array([.1, .9, .1, .1, .1])
            if "People visibly" in query or "static quiet" in query:
                return np.zeros(5)
            return np.array([.9, .1, .5, .5, .5])  # plot relevance prefers the calm first shot

        def reserve(self, path, start, length, reason):
            self.used.append((str(path), start, start + length))

        def freshness(self, path, start, length):
            return 0 if any(p == str(path) and start < end and start + length > begin for p, begin, end in self.used) else 1
        visual_freshness = freshness

    monkeypatch.setattr(moments, "_enrich", lambda c, *a: {**c, "muted": False, "native_verified": True, "cues": []})
    monkeypatch.setattr("movie_pipeline.core.duration", lambda path: 20)
    monkeypatch.setattr(intensity, "envelope", lambda path: None)
    plan = moments.plan_episode(Selector(), [[source]], lookup, [{"title": "Example", "overview": "x"}],
                                [{"kind": "film", "film": 0, "paragraphs": ["Number one.", "Body."]}],
                                {(0, 0): Path("a.wav"), (0, 1): Path("b.wav")}, {}, None, tmp_path)
    assert plan["previews"][0]["start"] == 10


def test_music_bridges_out_of_a_scene_with_sound():
    timeline = [{"start": 0, "duration": 5, "type": "review-preview", "shots": [{"muted": False}]},
                {"start": 5, "duration": 10, "type": "narration", "section_kind": "film"}]
    curve = sound_design.gain_curve(timeline, 1.0, 15, 15)
    rate = sound_design.CONTROL_RATE
    assert curve[int(5.6 * rate)] > sound_design.MUSIC_LEVELS["narration"] + 0.2
    assert abs(curve[int(10 * rate)] - sound_design.MUSIC_LEVELS["narration"]) < 0.05


def test_joins_l_cut_into_narration_and_fade_before_subscribe(monkeypatch, tmp_path):
    import movie_pipeline.render as render
    from movie_pipeline.cli import smooth_joins
    calls = []
    monkeypatch.setattr(render, "l_cut", lambda part, *a, **k: calls.append(("l", Path(part).name)) or True)
    monkeypatch.setattr(render, "fade_out_part", lambda part, s=0.6: calls.append(("fade", Path(part).name)))
    parts = [tmp_path / n for n in ("clip1.mp4", "talk.mp4", "clip2.mp4", "subscribe.mp4")]
    tails = {str(parts[0]): {"source": "s", "end": 5, "gain_db": 0}, str(parts[2]): {"source": "s", "end": 9, "gain_db": 0}}
    report = smooth_joins(parts, tails, {str(parts[3])})
    assert calls == [("l", "talk.mp4"), ("fade", "clip2.mp4")]
    assert report == {"l_cuts": 1, "fades": 1}


def _rms(path, start, seconds):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(seconds), "-i", str(path), "-vn",
                          "-ac", "1", "-ar", "8000", "-f", "s16le", "-"], capture_output=True, check=True).stdout
    samples = np.frombuffer(raw, dtype=np.int16).astype(float) / 32768
    return 20 * np.log10(np.sqrt(np.mean(samples ** 2)) + 1e-9)


@pytest.mark.skipif(subprocess.run(["which", "ffmpeg"], capture_output=True).returncode, reason="ffmpeg missing")
def test_scene_sound_carries_across_the_cut_without_a_drop(tmp_path, monkeypatch):
    import movie_pipeline.render as render
    monkeypatch.setattr(render, "export_size", lambda: (320, 180))
    monkeypatch.setattr(render, "footage_grade", lambda: "null", raising=False)
    source = tmp_path / "scene.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=320x180:r=30:d=12", "-f", "lavfi",
                    "-i", "sine=f=220:d=12", "-shortest", "-c:v", "libx264", "-c:a", "aac", str(source)], check=True)
    talk = tmp_path / "talk.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=320x180:r=30:d=4", "-f", "lavfi",
                    "-i", "anullsrc=r=48000:cl=stereo", "-t", "4", "-c:v", "libx264", "-c:a", "aac", str(talk)], check=True)
    clip = tmp_path / "clip.mp4"
    gain = render.trailer_video(source, 2, 4, clip, fade_in=0.04, fade_out=0.06)
    assert render.l_cut(talk, source, 6, gain_db=gain)
    before, after = _rms(clip, 3.7, 0.2), _rms(talk, 0.0, 0.15)
    assert abs(before - after) < 3.5          # no step at the cut
    assert _rms(talk, 0.9, 0.2) < after - 6   # then it settles under the narrator
    assert _rms(talk, 2.5, 0.5) < -60         # and is gone
