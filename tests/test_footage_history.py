"""Footage consumption across paragraphs, previews and alternate trailers."""
from itertools import pairwise
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image, ImageDraw

from movie_pipeline.core import PipelineError
from movie_pipeline.previews import (
    choose_preview,
    choose_title_card,
    cold_open_plan,
    outro_recap_plan,
)
from movie_pipeline.shots import ShotSelector, matches_title_card
from movie_pipeline.source_pool import choose_from_sources


def selector(scores):
    result = object.__new__(ShotSelector)
    result.relevance = Mock(return_value=np.array(scores))
    result.uses = {}
    result.last_shots = {}
    result.cut_count = 0
    return result


def scenes(*starts, length=8):
    return [{"start":start,"end":start+length,"motion":8} for start in starts]


def test_entirely_unseen_scenes_win_before_replaying_high_scoring_hero():
    selected = selector([1,-.8,-.9])
    source = Path("movie.mp4")
    shots = scenes(0,20,40)
    paragraphs = [selected.choose(source,shots,"The story continues.",6,False) for _ in range(3)]
    assert [int(cuts[0]["start"]//20) for cuts in paragraphs] == [0,1,2]
    assert all(cut["duration"] <= 7 for cuts in paragraphs for cut in cuts)


def test_separate_paragraphs_advance_within_a_long_scene():
    selected = selector([1])
    source = Path("movie.mp4")
    first = selected.choose(source,scenes(10,length=24),"A mystery.",6,False)
    second = selected.choose(source,scenes(10,length=24),"The mystery deepens.",6,False)
    assert second[0]["start"] >= first[0]["start"]+first[0]["duration"]
    assert sum(cut["duration"] for cut in second) == 6


def test_preview_and_cold_open_windows_are_consumed_before_narrated_paragraphs():
    selected = selector([1,.1,-.1])
    source = Path("movie.mp4")
    lookup = {str(source):scenes(0,20,40)}
    preview = choose_preview(selected,[source],lookup,"A mystery.",3.5,False)
    opening = cold_open_plan(selected,[[source]],lookup,"A mystery.",False,seconds=5)
    review = selected.choose(source,lookup[str(source)],"A mystery.",6,False)
    assert preview["start"] < 8
    assert opening[0]["start"] >= 20
    assert review[0]["start"] == 40
    assert {entry["kind"] for entry in selected._source_intervals[str(source)]} == {
        "review-preview","cold-open","narration"}


def test_outro_uses_unseen_scene_and_continues_across_paragraphs():
    selected = selector([1,.1])
    source = Path("movie.mp4")
    lookup = {str(source):scenes(0,20,length=12)}
    selected.reserve(source,0,12,"review")
    plans = outro_recap_plan(selected,[[source]],lookup,[4,5],"Watch our other recommendations.")
    assert plans[0][0]["start"] >= 20
    assert plans[1][0]["start"] == pytest.approx(plans[0][0]["start"]+4)
    assert sum(cut["duration"] for plan in plans for cut in plan) == 9


def test_exhausted_short_footage_reuses_spaced_scenes_without_rapid_cuts():
    selected = selector([1,.9])
    source = Path("movie.mp4")
    shots = scenes(0,20)
    plans = [selected.choose(source,shots,"The story.",8,False) for _ in range(6)]
    starts = [cut["start"] for plan in plans for cut in plan]
    assert all(first != second for first,second in pairwise(starts))
    assert all(sum(cut["duration"] for cut in plan) == 8 for plan in plans)
    assert all(3 <= cut["duration"] <= 12 for plan in plans for cut in plan)
    assert any(cut["visual_reason"] == "spaced-source-reuse" for plan in plans for cut in plan)


def test_duplicate_scene_in_other_trailer_is_not_treated_as_new():
    primary,extra = Path("first.mp4"),Path("extra.mp4")
    selected = selector([1,-.9])
    lookup = {str(primary):scenes(0),str(extra):scenes(100,120)}
    selected._scene_ranges = lookup
    selected._scene_hashes = {str(primary):[123],str(extra):[123,0xffffffffffffffff]}
    selected.reserve(primary,0,8,"review")
    assert selected.visual_freshness(extra,100,8) == 0
    assert selected.visual_freshness(extra,120,8) == 1
    assert selected.choose(extra,lookup[str(extra)],"More story.",6,False)[0]["start"] == 120


def test_preview_also_avoids_duplicate_scene_in_other_trailer():
    primary,extra = Path("first.mp4"),Path("extra.mp4")
    selected = selector([1,-.9])
    lookup = {str(primary):scenes(0),str(extra):scenes(100,120)}
    selected._scene_ranges = lookup
    selected._scene_hashes = {str(primary):[123],str(extra):[123,0xffffffffffffffff]}
    selected.reserve(primary,0,8,"review")
    result = choose_preview(selected,[extra],lookup,"The show.",3.5,False)
    assert result["start"] >= 120


def test_fresh_secondary_source_wins_over_exhausted_primary(tmp_path):
    primary,extra = tmp_path/"primary.mp4",tmp_path/"extra.mp4"
    primary.touch()
    extra.touch()
    selected = selector([1])
    lookup = {str(primary):scenes(0),str(extra):scenes(20)}
    selected.features = dict.fromkeys(lookup,object())
    selected.reserve(primary,0,8,"review")
    result = choose_from_sources(selected,[primary,extra],lookup,"The show.",6,False)
    assert {cut["source"] for cut in result} == {str(extra)}


def test_title_and_service_cards_do_not_interrupt_normal_review_or_preview():
    selected = selector([1,.9,-.9])
    source = Path("show.mp4")
    lookup = {str(source):scenes(0,20,40)}
    selected.cards = {str(source):["title","promo",None]}
    assert selected.choose(source,lookup[str(source)],"The story.",6,False)[0]["start"] == 40
    assert choose_preview(selected,[source],lookup,"The show.",3.5,False)["start"] >= 40


def test_short_preview_holds_a_clean_frame_instead_of_crossing_a_title_card():
    selected = selector([1,1,.1])
    source = Path("show.mp4")
    shots = [{"start":10,"end":11.5,"motion":8},
             {"start":11.5,"end":16,"motion":0},
             {"start":30,"end":31.5,"motion":8}]
    selected.cards = {str(source):[None,"title",None]}
    result = choose_preview(selected,[source],{str(source):shots},"The show.",3.5,False)
    assert result["still"] is True
    assert result["muted"] is True
    assert result["start"] == 10.75
    assert result["duration"] == 3.5


def test_card_indexing_ocr_fallback_reads_only_likely_slates(monkeypatch):
    selected = selector([1,1])
    source = Path("show.mp4")
    selected.features = {str(source):[object(),object()]}
    selected.cards = {}
    selected.card_texts = {}
    selected.titles = {str(source):"The Bear"}
    slate = Image.new("RGB",(320,180),"black")
    ImageDraw.Draw(slate).rectangle((80,60,240,95),fill="white")
    read = Mock(return_value="THE BEAR")
    monkeypatch.setattr("movie_pipeline.shots.frame_text",read)
    selected._index_cards(source,scenes(0,20),[slate,Image.new("RGB",(320,180),"gray")])
    assert selected.cards[str(source)] == ["title",None]
    assert read.call_count == 1
    assert selected._source_intervals[str(source)][0]["kind"] == "card-exclusion"


@pytest.mark.parametrize("text,title,expected",[
    ("SHOGUN","Shōgun",True),
    ("1917","1917 (2019)",True),
    ("1923","1923",True),
    ("2001 A SPACE ODYSSEY","2001: A Space Odyssey",True),
    ("THE BEAR\nSTREAMING NOW","The Bear (2022)",True),
    ("MISSION IMPOSSIBLE THE FINAL RECKONING","Mission: Impossible - The Final Reckoning",True),
    ("NETFLIX","The Bear",False),
    ("THE","The Bear",False),
    ("SHOGUN\nHULU ORIGINAL","The Bear",False),
])
def test_title_card_matching_uses_meaningful_normalized_title_words(text,title,expected):
    assert matches_title_card(text,title) is expected


def test_only_own_ocr_verified_title_card_is_returned_for_review_ending():
    selected = selector([1,1])
    source = Path("show.mp4")
    lookup = {str(source):scenes(10,30)}
    selected.cards = {str(source):["promo","title"]}
    selected.card_texts = {str(source):["NETFLIX","THE BEAR"]}
    card = choose_title_card(selected,[source],lookup,"The Bear")
    assert card == {"source":str(source),"start":34,"duration":1.5,"muted":True,"still":True}
    assert selected._source_intervals[str(source)][0]["kind"] == "review-title-card"


def test_missing_or_wrong_title_ocr_skips_optional_card(monkeypatch):
    selected = selector([1])
    source = Path("show.mp4")
    selected.cards = {str(source):["title"]}
    selected.card_texts = {str(source):["NETFLIX"]}
    monkeypatch.setattr("movie_pipeline.previews.duration",Mock(side_effect=PipelineError("No tail")))
    assert choose_title_card(selected,[source],{str(source):scenes(10)},"The Bear") is None


def test_tail_title_slate_can_be_found_without_putting_it_in_normal_shots(monkeypatch):
    selected = selector([1])
    source = Path("show.mp4")
    image = Image.new("RGB",(320,180),"black")
    ImageDraw.Draw(image).rectangle((80,60,240,95),fill="white")
    monkeypatch.setattr("movie_pipeline.previews.duration",lambda path:120)
    read = Mock(return_value=image)
    monkeypatch.setattr("movie_pipeline.previews.get_frame",read)
    monkeypatch.setattr("movie_pipeline.previews.frame_text",lambda image:"SHOGUN")
    lookup = {str(source):scenes(10)}
    card = choose_title_card(selected,[source],lookup,"Shōgun")
    assert card["start"] >= 105
    assert card["still"] is True
    assert read.call_count == 1
    assert lookup[str(source)] == scenes(10)


def test_optional_tail_ocr_is_bounded_and_failures_skip_card(monkeypatch):
    selected = selector([1])
    source = Path("show.mp4")
    image = Image.new("RGB",(320,180),"black")
    ImageDraw.Draw(image).rectangle((80,60,240,95),fill="white")
    monkeypatch.setattr("movie_pipeline.previews.duration",lambda path:120)
    read = Mock(return_value=image)
    monkeypatch.setattr("movie_pipeline.previews.get_frame",read)
    monkeypatch.setattr("movie_pipeline.previews.frame_text",lambda image:"")
    assert choose_title_card(selected,[source],{},"The Bear") is None
    assert read.call_count == 4
