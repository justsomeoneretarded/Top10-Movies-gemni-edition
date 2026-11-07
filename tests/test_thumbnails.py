import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import requests
from PIL import Image, ImageDraw

from movie_pipeline.script import Writer
from movie_pipeline.thumbnails import (
    Cutouts,
    Pollinations,
    build_thumbnails,
    normalize_concepts,
)


def response(status=200, content_type="image/png"):
    data = io.BytesIO()
    Image.new("RGB", (640,360), "#254655").save(data, format="PNG")
    result = MagicMock(status_code=status, headers={"Content-Type": content_type})
    result.__enter__.return_value = result
    result.iter_content.return_value = [data.getvalue()]
    return result


def test_background_api_is_free_only_and_caches_images(monkeypatch, tmp_path):
    get = Mock(return_value=response())
    monkeypatch.setattr("movie_pipeline.thumbnails.requests.get", get)
    provider = Pollinations(tmp_path)
    assert provider.generate("Cinematic sky, no people", 12).size == (640,360)
    assert provider.generate("Cinematic sky, no people", 12).size == (640,360)
    assert get.call_count == 1
    assert get.call_args.args[0].startswith("https://image.pollinations.ai/prompt/")
    assert "headers" not in get.call_args.kwargs
    assert "nologo" not in get.call_args.kwargs["params"]


def test_sequential_generation_respects_free_rate_limit(monkeypatch, tmp_path):
    get = Mock(side_effect=[response(), response()])
    monkeypatch.setattr("movie_pipeline.thumbnails.requests.get", get)
    monkeypatch.setattr("movie_pipeline.thumbnails.time.monotonic", Mock(side_effect=[100,110,110]))
    sleep = Mock()
    monkeypatch.setattr("movie_pipeline.thumbnails.time.sleep", sleep)
    provider = Pollinations(tmp_path)
    provider.generate("First cinematic background", 1)
    provider.generate("Second cinematic background", 2)
    sleep.assert_called_once_with(35)


@pytest.mark.parametrize("result", [response(402), response(200, "application/json"), requests.Timeout("unavailable")])
def test_api_failure_switches_to_local_without_paid_retry(monkeypatch, tmp_path, result):
    get = Mock(side_effect=[result])
    monkeypatch.setattr("movie_pipeline.thumbnails.requests.get", get)
    provider = Pollinations(tmp_path)
    assert provider.generate("A cinematic background", 1) is None
    assert provider.generate("Another background", 2) is None
    assert get.call_count == 1


def test_cutout_model_loaded_once_and_foreground_has_transparency(monkeypatch):
    import sys
    image = Image.new("RGBA", (100,100))
    ImageDraw.Draw(image).rectangle((25,5,75,95), fill="red")
    session = Mock()
    module = SimpleNamespace(new_session=session, remove=Mock(return_value=image))
    monkeypatch.setitem(sys.modules, "rembg", module)
    extractor = Cutouts(True)
    first, method = extractor.extract(image)
    extractor.extract(image)
    assert method == "rembg-u2net"
    assert first.size == (51,91)
    assert session.call_count == 1


def test_bad_concepts_cannot_pick_nonexistent_film_indices():
    concepts = normalize_concepts([{"subject_indices": [-1, 99, "0"], "background_prompt": "bad"}], "Action", "Energetic", 2)
    assert len(concepts) == 3
    assert all(0 <= n < 2 for c in concepts for n in c["subject_indices"])


def test_three_redesigned_outputs_with_one_netflix_variant(monkeypatch, tmp_path):
    frame = Image.new("RGB", (640,360), "blue")
    ImageDraw.Draw(frame).ellipse((210,40,430,340), fill="#e3af87")
    monkeypatch.setattr("movie_pipeline.thumbnails.get_frame", lambda *args: frame)
    monkeypatch.setattr("movie_pipeline.thumbnails.subject_frame", lambda *args: frame)
    provider = Mock()
    provider.generate.return_value = Image.new("RGB", (640,360), "#254655")
    monkeypatch.setattr("movie_pipeline.thumbnails.Pollinations", lambda cache: provider)
    build_thumbnails([Path("a.mp4"),Path("b.mp4")], [[{"motion":10}]]*2,
                     ["ACTION PICKS", "WATCH THESE", "THE NEXT WATCH"], tmp_path/"output",
                     title="Top Action Movies", work=tmp_path/"work", use_api=True)
    report = json.loads((tmp_path/"work/thumbnail-report.json").read_text())
    assert [r["netflix_badge"] for r in report] == [False,False,True]
    assert [r["style"] for r in report] == ["cinematic-minimal","bold-text","cinematic-minimal"]
    for index in range(1,4):
        path = tmp_path/"output"/f"thumbnail-{index:02}.jpg"
        with Image.open(path) as image:
            assert image.size == (1280,720)
        assert path.stat().st_size < 2_000_000
    assert len(list((tmp_path/"output").iterdir())) == 3
    with Image.open(tmp_path/"output/thumbnail-03.jpg") as image:
        red, green, _blue = image.getpixel((50,70))
        assert red > 140 and green < 40


def test_qwen_concepts_use_script_and_cache_results(tmp_path):
    writer = Writer(cache_dir=tmp_path)
    writer.ask = Mock(return_value={"concepts": []})
    films = [{"title": "Example", "genres": ["Action"], "overview": "A chase story."}]
    sections = [{"paragraphs": ["Original movie commentary from this video's script."]}]
    result = writer.thumbnail_concepts("Top movies", "Action", films, sections)
    assert len(result) == 3
    assert "Original movie commentary" in writer.ask.call_args.args[0]
    writer.thumbnail_concepts("Top movies", "Action", films, sections)
    assert writer.ask.call_count == 1


def test_qwen_failure_still_produces_three_usable_plans():
    writer = Writer()
    writer.ask = Mock(side_effect=requests.ConnectionError())
    result = writer.thumbnail_concepts("Top movies", "Action", [{"title":"Example"}], [])
    assert len(result) == 3


def test_complete_scenes_do_not_paste_cutouts_and_reject_duplicates(monkeypatch, tmp_path):
    import numpy as np

    from movie_pipeline.thumbnails import choose_scene
    rng = np.random.default_rng(7)
    good = Image.fromarray(rng.integers(20, 230, (360,640,3), dtype=np.uint8))
    provider = Mock()
    provider.generate.side_effect = [Image.new('RGB', (640,360), 'black'), good, good, good]
    previous = []
    plan = normalize_concepts(None, 'Horror', '', 1)[0]
    selected, candidates, seed = choose_scene(provider, plan, 0, 10, previous)
    assert selected is good and seed == 11
    assert len(candidates) == 2
    assert choose_scene(provider, plan, 1, 20, previous)[0] is None
    provider.generate.side_effect = None
    provider.generate.side_effect = [Image.fromarray(np.clip(rng.integers(0, 110, (360,640,3)) + np.array(color), 0, 255).astype(np.uint8)) for color in [(100,0,0),(110,0,0),(0,100,0),(0,110,0),(0,0,100),(0,0,110)]]
    monkeypatch.setattr('movie_pipeline.thumbnails.Pollinations', lambda cache: provider)
    cutouts = Mock(side_effect=AssertionError('Generated scene must not receive cutouts'))
    monkeypatch.setattr('movie_pipeline.thumbnails.subject_frame', cutouts)
    build_thumbnails([Path('a.mp4')], [[]], ['A','WATCH THESE','C'], tmp_path/'out',
                     work=tmp_path/'work', use_api=True)
    report = json.loads((tmp_path/'work/thumbnail-report.json').read_text())
    assert report[0]['composition'] == 'complete-generated-scene'
    assert report[0]['cutouts'] == []


def test_horror_defaults_and_concept_pair_selection():
    from movie_pipeline.thumbnails import select_concepts
    plans = normalize_concepts(None, 'Top 10 Horror Movies', '', 2)
    assert 'terrified' in plans[0]['background_prompt']
    raw = [{'background_prompt': 'Unstructured scene'} for _ in range(6)]
    raw[1] = {'background_prompt': 'Expressive close-up, dramatic lighting and depth'}
    raw[3] = {'background_prompt': 'Subject right, empty left, dramatic lighting'}
    raw[5] = {'background_prompt': 'Two subjects, empty upper-left, lighting and depth'}
    assert select_concepts(raw) == [raw[1],raw[3],raw[5]]
