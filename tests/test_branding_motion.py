"""Motion and sound contract for the original MOVIE RANKS channel entrance."""
from pathlib import Path

import numpy as np
import pytest

from movie_pipeline.branding import IDENT_SECONDS, ident_audio, ident_frame, logo_image


def test_light_entry_moves_and_resolves_without_a_static_fade():
    # Actual pixel changes, not an assertion about a filter-string spelling.
    times = (.0,.23,.47,.73,1.0,1.45,1.9,IDENT_SECONDS)
    frames = [np.asarray(ident_frame(t,320,180)) for t in times]
    assert frames[0].max() < 12
    assert frames[-1].max() == 0
    differences = [np.mean(np.abs(a.astype(float)-b)) for a,b in zip(frames,frames[1:])]
    assert all(value > .5 for value in differences[:5])
    assert differences[-1] > 2
    # Colored filaments live outside the M during entry, then settle away.
    edge = frames[3][:,:80].astype(float)
    settled = frames[-2][:,:80].astype(float)
    assert edge.std() > settled.std()+1
    assert np.mean(frames[-2][:,:,0]-frames[-2][:,:,1].astype(float)) > 3


def test_ident_frame_is_deterministic_native_sized_and_has_no_jitter():
    first = ident_frame(.62,640,360)
    second = ident_frame(.62,640,360)
    assert first.size == (640,360)
    assert np.array_equal(np.asarray(first),np.asarray(second))
    # After the lights resolve, the geometric logo is steady.
    a = np.asarray(ident_frame(1.65,320,180)).astype(float)
    b = np.asarray(ident_frame(1.9,320,180)).astype(float)
    assert np.mean(np.abs(a-b)) < .1


def test_logo_still_is_clean_and_does_not_need_external_assets(tmp_path):
    path = tmp_path/"logo.png"
    result = logo_image(path,640,360)
    assert path.exists() and result.size == (640,360)
    pixels = np.asarray(result)
    assert pixels[:,:,0].max() > 180
    assert (pixels[:,:,0].astype(float)-pixels[:,:,1]).max() > 100


def test_original_sting_has_two_hits_clean_endpoints_and_safe_peak():
    rate = 16000
    audio = ident_audio(rate=rate)
    assert audio.shape == (round(IDENT_SECONDS*rate),2)
    assert np.isfinite(audio).all()
    assert .05 < np.max(np.abs(audio)) <= .65
    assert np.max(np.abs(audio[:round(.22*rate)])) == 0
    assert np.max(np.abs(audio[-1])) == 0
    energy = lambda a,b: float(np.mean(audio[round(a*rate):round(b*rate)]**2))
    assert energy(.24,.39) > energy(.63,.75)
    assert energy(.80,.95) > energy(1.4,1.6)
    # Smooth synthesis must not introduce one-sample clicks.
    assert np.max(np.abs(np.diff(audio,axis=0))) < .1


@pytest.mark.parametrize("seconds",[float("nan"),-1,0])
def test_invalid_technical_ident_duration_is_rejected(seconds):
    with pytest.raises(ValueError):
        ident_audio(seconds)
