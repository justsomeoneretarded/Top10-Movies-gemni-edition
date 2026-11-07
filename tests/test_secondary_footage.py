import json
from pathlib import Path

import pytest

from movie_pipeline.core import PipelineError, save_json
from movie_pipeline.secondary_footage import (
    download_secondary,
    novelty_from_hashes,
    secondary_candidate_ok,
)

IDS = ['aaaaaaaaaaa','bbbbbbbbbbb','ccccccccccc']
FILM = {'title':'Example Movie','year':2025,'kind':'movie','id':1,
        'trailers':[{'key':identifier} for identifier in IDS]}


def info(title='Example Movie (2025) Fan-Made Trailer', height=1080, verified=False):
    return {'title':title,'duration':120,'channel_is_verified':verified,'channel':'Movie Fans',
            'formats':[{'height':height,'width':round(height*16/9),'vcodec':'vp9'}]}


@pytest.mark.parametrize('title,verified,accepted',[
    ('Example Movie (2025) Fan-Made Trailer',False,True),
    ('Example Movie (2025) Unofficial Trailer',False,True),
    ('Example Movie Fan Edit Trailer',False,True),
    ('Example Movie (2025) Trailer',False,True),
    ('Example Movie Official Trailer',True,True),
    ('Example Movie (2026) Fan-Made Trailer',False,False),
    ('Example Movie (2025) AI-Generated Concept Trailer',True,False),
    ('Example Movie (2025) Reaction Trailer',True,False),
    ('Another Film (2025) Fan-Made Trailer',False,False),
])
def test_secondary_accepts_same_title_fan_edits_but_not_fake_or_wrong_movie(title,verified,accepted):
    assert secondary_candidate_ok(info(title,verified=verified),FILM)==accepted


def test_hash_comparison_rejects_identical_reordered_samples_and_requires_novel_scenes():
    primary = [0x1111111111111111,0x3333333333333333,0x5555555555555555,0x7777777777777777]
    assert not novelty_from_hashes(primary,list(reversed(primary)))['distinct']
    novel = [0xEEEEEEEEEEEEEEEE,0xCCCCCCCCCCCCCCCC,0xAAAAAAAAAAAAAAAA,0x8888888888888888]
    result = novelty_from_hashes(primary,novel)
    assert result['distinct'] and result['novel_fraction']==1
    assert novelty_from_hashes(primary,novel[:2]) is None


def fake_downloads(monkeypatch,tmp_path,*,catalog=None,novel=True,actual_height=1080,fail=False):
    import yt_dlp
    catalog = catalog or {IDS[1]:info()}
    primary=tmp_path/'movie-1-2160p.mp4'
    primary.write_text('primary')
    save_json(primary.with_suffix('.json'),{'source':f'https://www.youtube.com/watch?v={IDS[0]}'})
    downloaded=[]
    inspected=[]
    class FakeYDL:
        def __init__(self,options):
            self.options=options
        def __enter__(self):
            return self
        def __exit__(self,*args):
            pass
        def extract_info(self,url,**kwargs):
            if url.startswith('ytsearch'):
                return {'entries':[{'id':key} for key in catalog]}
            identifier=url[-11:]
            inspected.append(identifier)
            return catalog.get(identifier)
        def download(self,urls):
            downloaded.append(urls[0][-11:])
            target=Path(self.options['outtmpl'].replace('%(ext)s','mp4'))
            if fail:
                target.with_suffix('.f313.webm.part').write_text('partial')
                raise yt_dlp.utils.DownloadError("Sign in to confirm you're not a bot")
            target.write_text(urls[0][-11:])
    monkeypatch.setattr(yt_dlp,'YoutubeDL',FakeYDL)
    monkeypatch.setattr('movie_pipeline.secondary_footage.duration',lambda path:120)
    monkeypatch.setattr('movie_pipeline.secondary_footage.require_disk',lambda path:None)
    monkeypatch.setattr('movie_pipeline.secondary_footage.probe',lambda path:{'streams':[{'codec_type':'video','width':round(actual_height*16/9),'height':actual_height}]})
    monkeypatch.setattr('movie_pipeline.secondary_footage.scene_novelty',lambda *args:{'distinct':novel,'novel_fraction':.5,'novel_frames':9,'sampled_frames':18})
    monkeypatch.setenv('VIDEO_RESOLUTION','1080')
    return primary,downloaded,inspected


def test_second_distinct_fan_edit_is_accepted_and_primary_is_preserved(monkeypatch,tmp_path):
    primary,downloaded,inspected=fake_downloads(monkeypatch,tmp_path)
    result=download_secondary(FILM,tmp_path,primary,(1920,1080))
    assert result and result.name=='movie-1-2160p-extra.mp4'
    assert primary.read_text()=='primary'
    assert IDS[0] not in inspected
    assert downloaded==[IDS[1]]
    record=json.loads(result.with_suffix('.json').read_text())
    assert record['fan_edit'] and record['novelty']['distinct']
    assert download_secondary(FILM,tmp_path,primary,(1920,1080))==result
    assert downloaded==[IDS[1]]


@pytest.mark.parametrize('kwargs',[
    {'novel':False},{'actual_height':720},{'fail':True},
    {'catalog':{IDS[1]:info(height=720)}},
    {'catalog':{IDS[1]:info('Another Film (2025) Fan-Made Trailer')}},
])
def test_optional_failure_duplicate_or_lower_quality_continues_with_primary(monkeypatch,tmp_path,kwargs):
    primary,_downloaded,_inspected=fake_downloads(monkeypatch,tmp_path,**kwargs)
    search_only_film={**FILM,"trailers":[FILM["trailers"][0]]}
    assert download_secondary(search_only_film,tmp_path,primary,(1920,1080)) is None
    assert primary.read_text()=='primary'
    assert len(list(tmp_path.glob('*.mp4')))==1
    assert not list(tmp_path.glob('*candidate*'))


def test_disk_or_missing_metadata_never_blocks_primary(monkeypatch,tmp_path):
    primary,_,_=fake_downloads(monkeypatch,tmp_path)
    monkeypatch.setattr('movie_pipeline.secondary_footage.require_disk',lambda path: (_ for _ in ()).throw(PipelineError('low disk')))
    assert download_secondary(FILM,tmp_path,primary,(1920,1080)) is None
    primary.with_suffix('.json').unlink()
    assert download_secondary(FILM,tmp_path,primary,(1920,1080)) is None
    assert primary.read_text()=='primary'


def test_actual_candidate_downloads_are_bounded(monkeypatch,tmp_path):
    primary,downloaded,_=fake_downloads(monkeypatch,tmp_path,catalog={key:info() for key in IDS[1:]},novel=False)
    assert download_secondary(FILM,tmp_path,primary,(1920,1080)) is None
    assert len(downloaded)==2
    assert not list(tmp_path.glob('*candidate*'))


def test_visual_fingerprints_compare_real_pixel_arrays_and_skip_duplicates(monkeypatch):
    import numpy as np

    from movie_pipeline.secondary_footage import scene_novelty
    class Capture:
        def __init__(self,path):
            self.rng=np.random.default_rng(2 if 'novel' in path else 1)
        def set(self,*args):
            pass
        def read(self):
            return True,self.rng.integers(0,255,(90,160,3),dtype=np.uint8)
        def release(self):
            pass
    monkeypatch.setattr('movie_pipeline.secondary_footage.cv2.VideoCapture',Capture)
    monkeypatch.setattr('movie_pipeline.secondary_footage.duration',lambda path:120)
    assert not scene_novelty(Path('primary.mp4'),Path('copy.mp4'))['distinct']
    assert scene_novelty(Path('primary.mp4'),Path('novel.mp4'))['distinct']


def test_third_trailer_excludes_accepted_ids_and_caches_separately(monkeypatch, tmp_path):
    primary, downloaded, inspected = fake_downloads(monkeypatch, tmp_path, catalog={key:info() for key in IDS[1:]})
    second = download_secondary(FILM, tmp_path, primary, (1920,1080))
    assert second and downloaded == [IDS[1]]
    monkeypatch.setattr('movie_pipeline.secondary_footage._hashes', lambda path: [1,2,3,4])
    monkeypatch.setattr('movie_pipeline.secondary_footage.novelty_from_hashes',
                        lambda old, new: {'distinct':True,'novel_fraction':.5})
    inspected.clear()
    third = download_secondary(FILM, tmp_path, primary, (1920,1080), existing=[second], slot=3)
    assert third and third != second and third.name.endswith('-extra-3.mp4')
    assert downloaded == [IDS[1], IDS[2]]
    assert IDS[0] not in inspected and IDS[1] not in inspected
    assert download_secondary(FILM, tmp_path, primary, (1920,1080), existing=[second], slot=3) == third
    assert downloaded == [IDS[1], IDS[2]]
    assert primary.exists() and second.exists()


def test_third_rejects_footage_already_in_the_union(monkeypatch, tmp_path):
    primary, downloaded, _ = fake_downloads(monkeypatch, tmp_path, catalog={key:info() for key in IDS[1:]})
    second = download_secondary(FILM, tmp_path, primary, (1920,1080))
    monkeypatch.setattr('movie_pipeline.secondary_footage._hashes',
                        lambda path: [0x1111111111111111,0x3333333333333333,0x5555555555555555,0x7777777777777777])
    assert download_secondary(FILM, tmp_path, primary, (1920,1080), existing=[second], slot=3) is None
    assert primary.exists() and second.exists()
    assert not list(tmp_path.glob('*candidate*'))


def test_official_scene_clips_are_accepted_without_the_word_trailer():
    from movie_pipeline.secondary_footage import novelty_from_hashes, secondary_candidate_ok
    film = {"title": "Fast X", "year": 2023, "cast": ["Vin Diesel"], "kind": "movie"}
    clip = {"title": "Fast X (2023) - Rome Bomb Chase Scene | Movieclips", "duration": 150, "channel_is_verified": True,
            "description": "Vin Diesel in Fast X", "channel": "Movieclips"}
    assert secondary_candidate_ok(clip, film) or secondary_candidate_ok({**clip, "description": "Fast X Vin Diesel"}, film)
    for bad in ("Fast X (2023) Full Movie Scene", "Fast X Ending Explained Scene", "Fast X Trailer Reaction",
                "Fast X Behind the Scenes Clip"):
        assert not secondary_candidate_ok({**clip, "title": bad}, film)
    assert not secondary_candidate_ok({**clip, "duration": 12}, film)  # a logo or promo, not a scene
    # A single-scene clip with only three distinct frames counts as new when all three are new.
    primary = [0x1111111111111111, 0x3333333333333333, 0x5555555555555555, 0x7777777777777777]
    novel = [0xEEEEEEEEEEEEEEEE, 0xCCCCCCCCCCCCCCCC, 0xAAAAAAAAAAAAAAAA]
    assert novelty_from_hashes(primary, novel)["distinct"] is True
    assert novelty_from_hashes(primary, [primary[0], *novel[:2]])["distinct"] is False
