from pathlib import Path
from types import SimpleNamespace
import pytest
from movie_pipeline.editorial_policy import explicit_genre, search_queries, image_queries, source_identity, clean_review
from movie_pipeline.moments import complete_window
from movie_pipeline.resource_budget import DownloadBudget

def test_unspecified_and_mixed_genres_are_neutral():
    assert explicit_genre("8 Movies of 2026", "A mixed list for your watchlist.") == "neutral"
    assert explicit_genre("8 Shows", "Medical drama, fantasy, sci-fi and horror.") == "neutral"
    assert explicit_genre("8 Horror Movies", "A mix of drama and mystery.") == "horror"

def test_queries_are_per_title_without_network_and_shows_without_year():
    film={"title":"Example Show","year":2023,"kind":"tv","genres":["Comedy"],"networks":["Do Not Search Me"]}
    queries=search_queries(film)
    assert "Example Show best scene" in queries
    assert any("funny scene" in q for q in queries)
    assert all("Do Not Search Me" not in q and "action scene" not in q for q in queries)
    assert any("official trailer" in q for q in search_queries(film,primary=True))
    assert any("best scene image" in q for q in image_queries(film))

@pytest.mark.parametrize("suffix",["","-extra","-extra-3","-extra-4","-extra-5"])
def test_source_identity_survives_all_extra_slots(suffix):
    assert source_identity(Path("movie-42"+suffix+".mp4"))=="movie-42"

def test_filler_removed_without_padding_or_losing_premise():
    paragraphs=["Two brothers get stranded. If you liked First Blood, see this.",
                "You can find it on Prime Video and Apple TV.","Their only route home crosses a flooded town."]
    assert clean_review(paragraphs)==["Two brothers get stranded.","Their only route home crosses a flooded town."]

def test_native_scene_grows_to_complete_dialogue():
    c={"start":10.0,"duration":5.0,"source":"x.mp4"}
    cues=[{"start":9.6,"end":15.5,"text":"We need to leave before the bridge falls."}]
    result=complete_window(c,cues,(0,30))
    assert result and result["start"]<9.6
    assert result["start"]+result["duration"]>15.5
    assert not result["muted"]

def test_incomplete_dialogue_is_not_used_as_an_important_native_beat():
    c={"start":10.0,"duration":5.0,"source":"x.mp4"}
    assert complete_window(c,[{"start":10.1,"end":14.9,"text":"If you think that we"}],(0,30)) is None
    assert complete_window(c,[{"start":9.6,"end":15.5,"text":"A complete thought."}],(9.6,16)) is None

def test_download_reservations_prevent_concurrent_overcommit(monkeypatch,tmp_path):
    monkeypatch.setattr("movie_pipeline.resource_budget.shutil.disk_usage",lambda _:SimpleNamespace(free=3_000_000_000))
    budget=DownloadBudget()
    with budget.claim(tmp_path) as first:
        assert first
        with budget.claim(tmp_path) as second:
            assert not second
    assert budget.pending==0

def test_image_url_filter_rejects_local_targets():
    from movie_pipeline.image_discovery import public_url
    assert not public_url("http://127.0.0.1/private")
    assert not public_url("file:///etc/passwd")
    assert not public_url("https://metadata.google.internal/x")
    assert public_url("https://example.com/picture.jpg")

def test_spoken_availability_removed_but_metadata_remains():
    from movie_pipeline.packaging import refresh_availability
    film={"watch_availability":{"subscription":["Netflix"]}}
    script={"sections":[{"kind":"film","film":0,"paragraphs":["A specific premise.","You can find it on Netflix."],
                         "availability_text":"You can find it on Netflix.","delivery":["dramatic","normal"]}]}
    refresh_availability(script,[film])
    assert script["sections"][0]["paragraphs"]==["A specific premise."]
    assert script["sections"][0]["delivery"]==["dramatic"]
    assert film["watch_availability"]["subscription"]==["Netflix"]

def test_ident_is_original_and_decodable(monkeypatch,tmp_path):
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    from movie_pipeline.branding import render_ident
    from movie_pipeline.core import duration, probe
    monkeypatch.setenv("VIDEO_RESOLUTION","320x180")
    output=tmp_path/"ident.mp4"
    render_ident(output,tmp_path/"ident")
    assert duration(output)==pytest.approx(2.3,abs=.05)
    assert {s["codec_type"] for s in probe(output)["streams"]}=={"audio","video"}

def test_incidental_brief_genre_is_not_episode_direction():
    assert explicit_genre("8 movies", "Use an action scene from the third film.") == "neutral"
    assert explicit_genre("8 movies", "Keep neutral instrumental music, even for the horror title.") == "neutral"
    assert explicit_genre("8 movies", "The video is about horror movies.") == "horror"

def test_complete_dialogue_grows_across_word_aligned_display_chunks():
    c={"start":10.5,"duration":4.5,"source":"x.mp4"}
    cues=[{"start":8,"end":9,"text":"A previous thought."},
          {"start":9.3,"end":11,"text":"We have to"},
          {"start":11.1,"end":14,"text":"get across the bridge"},
          {"start":14.2,"end":16,"text":"before it collapses."}]
    clip=complete_window(c,cues,(0,30))
    assert clip["start"] < 9.3
    assert clip["start"]+clip["duration"] > 16
    assert len(clip["cues"])==3

def test_explicit_season_rejects_contradictory_upload():
    from movie_pipeline.secondary_footage import secondary_candidate_ok
    film={"title":"Example Show","year":2020,"kind":"tv","requested_season":3}
    info={"title":"Example Show Season 2 Best Scene","duration":90}
    assert not secondary_candidate_ok(info,film)
    info["title"]="Example Show Season 3 Best Scene"
    assert secondary_candidate_ok(info,film)

def test_native_cold_open_music_steps_back():
    from movie_pipeline.sound_design import segment_level, MUSIC_LEVELS
    assert segment_level({"type":"cold-open","shots":[{"muted":False}]})==MUSIC_LEVELS["review-preview-dialogue"]

def test_request_launcher_is_narrow_and_reuses_existing_workflow():
    import yaml
    path=Path(".github/workflows/production-request.yml")
    workflow=yaml.load(path.read_text(),Loader=yaml.BaseLoader)
    assert workflow["on"]["push"]["branches"]==["production-requests"]
    assert workflow["on"]["push"]["paths"]==["requests/current.json"]
    assert workflow["jobs"]["produce"]["uses"]=="./.github/workflows/generate-video.yml"
    assert workflow["permissions"]=={"contents":"read"}
    assert "${{" not in workflow["jobs"]["request"]["steps"][-1]["run"]

def test_whole_episode_edit_deletes_only_and_preserves_rank_and_premise():
    from movie_pipeline.script import Writer
    writer=Writer()
    writer.ask=lambda *a,**k: {"remove":["0:0:0","0:0:1","0:0:3"]}
    sections=[{"kind":"film","film":i,"paragraphs":[
        f"Number {i+1}: Example. The remarkable chemistry is significant. A teacher is stranded in space with no memory and must find a way home. It is a real game changer."]} for i in range(3)]
    result=writer.episode_edit("3 movies","",sections)
    assert result[0]["paragraphs"]==["Number 1: Example. A teacher is stranded in space with no memory and must find a way home."]
    assert sections[0]["paragraphs"][0].endswith("game changer.")
    assert all("teacher is stranded" in row["paragraphs"][0] for row in result)

def test_native_planner_uses_extra_pool_and_nonfatal_small_counts(monkeypatch,tmp_path):
    import numpy as np
    import movie_pipeline.moments as moments
    from movie_pipeline.core import duration
    a,b=Path("primary.mp4"),Path("extra-5.mp4")
    lookup={str(p):[{"start":10+j*10,"end":16+j*10,"motion":0} for j in range(5)] for p in (a,b)}
    class Selector:
        def __init__(self): self.used=[]
        def identity_relevance(self,path,query):
            return np.full(5, .8 if path==b else .1) if "People visibly" not in query and "static quiet" not in query else np.zeros(5)
        def reserve(self,path,start,length,reason): self.used.append((str(path),start,start+length))
        def freshness(self,path,start,length):
            return 0 if any(p==str(path) and start<end and start+length>begin for p,begin,end in self.used) else 1
        visual_freshness=freshness
    def enrich(candidate,*args):
        return {**candidate,"muted":False,"native_verified":True,"cues":[]}
    monkeypatch.setattr(moments,"_enrich",enrich)
    monkeypatch.setattr("movie_pipeline.core.duration",lambda path:45)
    result=moments.plan_episode(Selector(),[[a,b]],lookup,[{"title":"Example","overview":"A specific premise"}],
        [{"kind":"film","film":0,"paragraphs":["First.","Second."]}],
        {(0,0):Path("one.wav"),(0,1):Path("two.wav")},{},None,tmp_path)
    assert result["previews"][0]["source"]==str(b)
    assert len(result["inserts"][0])==2
    assert result["report"]["review_counts"]["0"]==3
    assert result["report"]["opening_seconds"]<=3.2

def test_photo_fallback_builds_a_diverse_still_only_pool(monkeypatch,tmp_path):
    from PIL import Image
    from movie_pipeline.image_discovery import gather_photos
    colors={"/a":(10,20,30),"/b":(40,50,60),"/c":(70,80,90)}
    hashes={10:0,40:0xffffffffffffffff,70:0xaaaaaaaaaaaaaaaa}
    monkeypatch.setattr("movie_pipeline.image_discovery.search",lambda film:[])
    monkeypatch.setattr("movie_pipeline.stills_trailer.fetch_original",lambda path,work:Image.new("RGB",(1000,600),colors[path]))
    monkeypatch.setattr("movie_pipeline.thumbnail_studio.face_quality",lambda image:(None,1,False))
    monkeypatch.setattr("movie_pipeline.thumbnail_studio.frame_sharpness",lambda image:100)
    monkeypatch.setattr("movie_pipeline.thumbnail_studio._dhash",lambda image:hashes[image.getpixel((0,0))[0]])
    photos,records=gather_photos([{"title":"Example","year":2026,"backdrops":list(colors)}],tmp_path)
    assert len(photos)==3
    assert all(c.source=="publicity-fallback" for c in photos)
    assert all(r["source_kind"]=="publicity-fallback" and "image.tmdb.org" in r["url"] for r in records)

def test_plain_language_rules_preserve_verified_title_names():
    from movie_pipeline.script import plain_words
    result=plain_words("Journey is a captivating journey.",protected=["Journey"])
    assert result == "Journey is a captivating journey."

def test_source_reservations_are_atomic_and_failure_can_release(tmp_path):
    from movie_pipeline.resource_budget import reserve_source,release_source
    url="https://www.youtube.com/watch?v=abcdefghijk"
    first,second=tmp_path/"a.mp4",tmp_path/"b.mp4"
    assert reserve_source(tmp_path,url,first)
    assert not reserve_source(tmp_path,"https://youtu.be/abcdefghijk",second)
    assert reserve_source(tmp_path,url,first)
    release_source(tmp_path,url,first)
    assert reserve_source(tmp_path,url,second)

def test_verified_keyword_context_broadens_queries_beyond_genre_templates():
    film={"title":"Example","year":2026,"kind":"movie","genres":["Drama"],
          "keywords":[{"name":"desert island"},{"name":"based on novel"}]}
    queries=search_queries(film)
    assert "Example 2026 desert island scene" in queries
    assert "Example 2026 official trailer" in queries
    assert "Example 2026 trailer" in queries
    assert all("based on novel" not in q for q in queries)
    assert len(queries)<=7

def test_external_image_needs_corroboration_beyond_a_popular_name():
    from movie_pipeline.image_discovery import corroborates
    film={"title":"The Odyssey","year":2026,"kind":"movie","cast":["Matt Damon"],"directors":["Christopher Nolan"]}
    assert not corroborates(film,"Assassins Creed Odyssey best image 2026","https://example.com/video-game/odyssey")
    assert not corroborates(film,"The Odyssey 2026 wallpaper","https://example.com/wallpapers")
    assert corroborates(film,"Matt Damon in Christopher Nolan's The Odyssey 2026","https://example.com/movies/odyssey")
    assert corroborates(film,"The Odyssey 2026","https://www.themoviedb.org/movie/12345-the-odyssey")

def test_primary_warmup_concurrency_is_conservative():
    import yaml
    workflow=yaml.load(Path(".github/workflows/generate-video.yml").read_text(),Loader=yaml.BaseLoader)
    assert workflow["jobs"]["footage"]["strategy"]["max-parallel"]=="2"

def test_poster_tagline_and_release_date_are_not_clean_photos(monkeypatch):
    from PIL import Image
    from movie_pipeline.thumbnail_studio import has_burned_text
    monkeypatch.setattr("movie_pipeline.shots.frame_text",lambda image:"DEFY THE GODS 07.17.26")
    assert has_burned_text(Image.new("RGB",(1280,720)))
