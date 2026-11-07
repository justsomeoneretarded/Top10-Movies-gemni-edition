from pathlib import Path

from PIL import Image

from movie_pipeline.cli import validate_package
from movie_pipeline.timing import DurationSpec


def test_valid_complete_package_outside_soft_duration_is_not_rejected(monkeypatch,tmp_path):
    for name in ['final-video.mp4','final-video-augmented.mp4','subtitles.srt','description.txt','recommended-title.txt']:
        (tmp_path/name).write_text('valid test placeholder')
    for i in range(1,11):
        Image.new('RGB',(1280,720),'blue').save(tmp_path/f'thumbnail-{i:02}.jpg')
    monkeypatch.setenv('VIDEO_RESOLUTION','1080')
    monkeypatch.setattr('movie_pipeline.cli.probe', lambda path: {
        'format':{'duration':'120'},'streams':[{'codec_type':'video','width':1920,'height':1080},{'codec_type':'audio'}]})
    assert validate_package(Path(tmp_path), DurationSpec(480,360,600)) == 120


def test_comparison_video_is_only_required_when_enabled(monkeypatch, tmp_path):
    import pytest

    from movie_pipeline.core import PipelineError
    for name in ['final-video.mp4','subtitles.srt','description.txt','recommended-title.txt']:
        (tmp_path/name).write_text('valid test placeholder')
    for i in range(1,11):
        Image.new('RGB',(1280,720),'blue').save(tmp_path/f'thumbnail-{i:02}.jpg')
    monkeypatch.setenv('VIDEO_RESOLUTION','1080')
    monkeypatch.setattr('movie_pipeline.cli.probe', lambda path: {
        'format':{'duration':'120'},'streams':[{'codec_type':'video','width':1920,'height':1080},{'codec_type':'audio'}]})
    monkeypatch.delenv('COMPARISON_EXPORT', raising=False)
    assert validate_package(Path(tmp_path), DurationSpec(automatic=True)) == 120
    monkeypatch.setenv('COMPARISON_EXPORT', 'true')
    with pytest.raises(PipelineError, match='final-video-augmented.mp4'):
        validate_package(Path(tmp_path), DurationSpec(automatic=True))
