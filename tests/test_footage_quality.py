import json
from pathlib import Path

import pytest

from movie_pipeline.core import PipelineError
from movie_pipeline.footage import advertised_quality, download

IDS = ['aaaaaaaaaaa', 'bbbbbbbbbbb', 'ccccccccccc', 'ddddddddddd']


def info(height, *, verified=True):
    return {'title':'Example Official Trailer', 'duration':120, 'channel':'Studio', 'channel_is_verified':verified,
            'formats':[{'width':round(height*16/9), 'height':height, 'vcodec':'vp9', 'filesize':50_000_000}]}


def setup_downloads(monkeypatch, catalog, actual=None, failures=None, discovery=None):
    import yt_dlp
    downloaded = []
    inspected = []
    actual = actual or {key: value['formats'][0]['height'] for key,value in catalog.items()}
    failures = failures or {}

    class FakeYDL:
        def __init__(self, options):
            self.options = options
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def extract_info(self, url, **kwargs):
            if url.startswith('ytsearch'):
                return {'entries':[{'id':key} for key in (discovery or [])]}
            identifier = url[-11:]
            inspected.append(identifier)
            value = catalog[identifier]
            if isinstance(value, Exception):
                raise value
            return value
        def download(self, urls):
            identifier = urls[0][-11:]
            downloaded.append(identifier)
            path = Path(self.options['outtmpl'].replace('%(ext)s','mp4'))
            if identifier in failures:
                path.with_suffix('.f313.webm.part').write_bytes(b'partial')
                raise yt_dlp.utils.DownloadError(failures[identifier])
            path.write_text(identifier)

    def probe(path):
        height = actual[path.read_text()]
        return {'streams':[{'codec_type':'video','width':round(height*16/9),'height':height}]}

    monkeypatch.setattr(yt_dlp,'YoutubeDL',FakeYDL)
    monkeypatch.setattr('movie_pipeline.footage.duration',lambda path:120)
    monkeypatch.setattr('movie_pipeline.footage.probe',probe)
    monkeypatch.setattr('movie_pipeline.footage.require_disk',lambda path:None)
    monkeypatch.setenv('VIDEO_RESOLUTION','2160')
    return downloaded, inspected


def film(ids):
    return {'title':'Example','year':2025,'kind':'movie','id':1,'trailers':[{'key':key} for key in ids]}


def test_inspects_quality_first_and_downloads_only_best_candidate(monkeypatch,tmp_path):
    downloaded, inspected = setup_downloads(monkeypatch,{IDS[0]:info(720),IDS[1]:info(1080)})
    result = download(film(IDS[:2]),tmp_path)
    assert inspected == IDS[:2]
    assert downloaded == [IDS[1]]
    assert result.read_text() == IDS[1]
    assert {path.suffix for path in tmp_path.iterdir()} == {'.mp4','.json'}
    record = json.loads(result.with_suffix('.json').read_text())
    assert record['source_height'] == 1080
    assert record['quality_selection_version'] == 2
    # A validated media/metadata pair can resume without metadata or media downloads.
    assert download(film(IDS[:2]),tmp_path) == result
    assert downloaded == [IDS[1]]


def test_actual_low_resolution_retries_a_better_candidate_without_losing_fallback(monkeypatch,tmp_path):
    downloaded, _ = setup_downloads(monkeypatch,{key:info(2160) for key in IDS[:3]},
                                    {IDS[0]:720,IDS[1]:1080,IDS[2]:2160})
    result = download(film(IDS[:3]),tmp_path)
    assert downloaded == IDS[:3]
    assert result.read_text() == IDS[2]
    assert len(list(tmp_path.glob('*.mp4'))) == 1
    assert not list(tmp_path.glob('*candidate*'))


def test_quality_retries_are_bounded_and_missing_4k_does_not_fail(monkeypatch,tmp_path):
    downloaded, _ = setup_downloads(monkeypatch,{key:info(2160) for key in IDS},dict.fromkeys(IDS,720))
    result = download(film(IDS),tmp_path)
    assert len(downloaded) == 3
    assert json.loads(result.with_suffix('.json').read_text())['source_height'] == 720
    assert len(list(tmp_path.glob('*.mp4'))) == 1


def test_quality_retry_bot_error_preserves_the_usable_trailer(monkeypatch,tmp_path):
    downloaded, _ = setup_downloads(monkeypatch,{key:info(2160) for key in IDS[:2]},
                                    {IDS[0]:720,IDS[1]:2160},
                                    {IDS[1]:"Sign in to confirm you're not a bot"})
    result = download(film(IDS[:2]),tmp_path)
    assert downloaded == IDS[:2]
    assert result.read_text() == IDS[0]
    assert not list(tmp_path.glob('*candidate*'))


def test_no_usable_trailer_still_reports_a_runner_bot_block(monkeypatch,tmp_path):
    setup_downloads(monkeypatch,{IDS[0]:info(2160)},failures={IDS[0]:"Sign in to confirm you're not a bot"})
    with pytest.raises(PipelineError,match='YouTube blocked this runner'):
        download(film(IDS[:1]),tmp_path)
    assert not list(tmp_path.iterdir())


def test_low_quality_curated_trailer_can_be_replaced_by_verified_search_candidate(monkeypatch,tmp_path):
    downloaded, _ = setup_downloads(monkeypatch,{IDS[0]:info(720),IDS[1]:info(2160)},discovery=[IDS[1]])
    result = download(film(IDS[:1]),tmp_path)
    assert downloaded == [IDS[1]]
    assert result.read_text() == IDS[1]
    assert not json.loads(result.with_suffix('.json').read_text())['curated']


def test_genuine_higher_quality_search_trailer_can_come_from_unverified_uploader(monkeypatch,tmp_path):
    downloaded, _ = setup_downloads(monkeypatch,{IDS[0]:info(720),IDS[1]:info(2160,verified=False)},discovery=[IDS[1]])
    result = download(film(IDS[:1]),tmp_path)
    assert downloaded == [IDS[1]]
    assert result.read_text() == IDS[1]


def test_dimension_and_storage_ranking_does_not_overstate_landscape_detail():
    assert advertised_quality({'formats':[{'height':2160,'width':1215}]},2160) == pytest.approx(683.4375)
    assert advertised_quality({'formats':[{'height':2160,'width':3840,'filesize':900_000_000},
                                         {'height':1080,'width':1920}]},2160) == 1080
    assert advertised_quality({'formats':[{'height':2160,'width':3840},{'height':720,'width':1280}]},1080) == 720
