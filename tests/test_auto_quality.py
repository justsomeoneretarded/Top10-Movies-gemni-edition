import pytest

from movie_pipeline.video_settings import (
    choose_output_size,
    download_ceiling,
    export_size,
)


@pytest.mark.parametrize('setting,expected', [(None,2160),('true',2160),('false',1080)])
def test_checkbox_default_and_unchecked_false(monkeypatch, setting, expected):
    monkeypatch.delenv('TRY_4K',raising=False)
    if setting is not None:
        monkeypatch.setenv('TRY_4K',setting)
    assert download_ceiling()==expected


@pytest.mark.parametrize('heights,cap,expected', [
    ([2160,2160],2160,(3840,2160)),
    ([2160,1080],2160,(1920,1080)),
    ([1080,1080],2160,(1920,1080)),
    ([2160,1440],2160,(2560,1440)),
    ([1080,720],2160,(1920,1080)),   # never below 1080p
    ([1080,480],2160,(1920,1080)),
    ([2160,2160],1080,(1920,1080)),
])
def test_export_uses_highest_shared_source_quality(heights,cap,expected):
    widths={480:854,720:1280,1080:1920,1440:2560,2160:3840}
    sources=[{'width':widths[h],'height':h} for h in heights]
    assert choose_output_size(sources,cap)==expected


def test_both_dimensions_limit_quality_and_internal_sizes_render(monkeypatch):
    # Low-resolution trailers are enlarged; the video itself never drops below 1080p.
    assert choose_output_size([{'width':1280,'height':1080}],2160)==(1920,1080)
    assert choose_output_size([{'width':320,'height':180}],2160)==(1920,1080)
    assert choose_output_size([{'width':1920,'height':1080},{'width':480,'height':260}],2160)==(1920,1080)
    monkeypatch.setenv('VIDEO_RESOLUTION','640x360')
    assert export_size()==(640,360)


@pytest.mark.parametrize('heights,expected', [
    ([1080, 1080, 1080, 1080, 720], (1920, 1080)),   # one 720p trailer is enlarged 1.5x
    ([1080, 1080, 1080, 1080, 480], (1920, 1080)),   # never below 1080p; the 480p trailer is enlarged
    ([2160, 2160, 2160, 1080, 1080], (2560, 1440)),  # 4K would need 2x; 1440p is native for 3 of 5
    ([1080, 720, 720], (1920, 1080)),                # never below 1080p
])
def test_one_weaker_trailer_no_longer_lowers_the_whole_video(heights, expected):
    widths = {480: 854, 720: 1280, 1080: 1920, 1440: 2560, 2160: 3840}
    assert choose_output_size([{'width': widths[h], 'height': h} for h in heights], 2160) == expected
