import re

import pytest

from movie_pipeline.packaging import (
    MIDROLLS,
    build_description,
    featured_heading,
    midroll_text,
    topic_hashtags,
)


def test_description_opens_with_a_search_hook_not_narration():
    intro = "Stop scrolling. Your next favorite movie might be here."
    script = {"sections": [{"kind": "intro", "paragraphs": [intro]},
                           {"kind": "film", "paragraphs": ["Number two is a tense chase.", "A later paragraph."]}],
              "package": {"title": "Top 2 Action Movies", "description": "Looking for action movies to watch? Start with these two picks.",
                          "pinned_comment": "Which one would you watch first?"}}
    films = [{"title": "First", "year": 2025, "kind": "movie", "genres": ["Action"], "watch_availability": {"subscription": ["Netflix"]}},
             {"title": "Second", "year": 2024, "kind": "movie", "genres": ["Thriller"]}]
    chapters = [{"start": 0, "title": "Intro"}, {"start": 12.7, "title": "#2 Second"},
                {"start": 80.5, "title": "#1 First"}, {"start": 150, "title": "Outro"}]
    result = build_description(script, films, chapters, ["https://example.org/trailer"], "Music: credit line")
    assert result.startswith("Top 2 Action Movies, featuring First and Second.")
    assert intro not in result and "A later paragraph." not in result
    assert "Looking for action movies to watch?" in result
    assert "00:00 Intro\n00:12 #2 Second\n01:20 #1 First\n02:30 Outro" in result
    assert "Featured movies\n#2 Second (2024)\n#1 First (2025) — Netflix" in result
    assert "Which one would you watch first? Tell me in the comments." in result
    assert "Music: credit line" in result and "https://example.org/trailer" in result
    assert result.rstrip().splitlines()[-1].startswith("#")  # hashtags last
    assert "#ActionMovies" in result and "Film information: TMDB" not in result


def test_writer_hook_is_used_only_when_it_names_the_films():
    films = [{"title": "Sinners", "year": 2025, "kind": "movie"}, {"title": "Weapons", "year": 2025, "kind": "movie"}]
    good = {"package": {"title": "2 Horror Movies", "hook": "2 horror movies from 2025: Sinners and Weapons, and where to stream them."}}
    assert build_description(good, films, [], []).startswith("2 horror movies from 2025: Sinners and Weapons")
    vague = {"package": {"title": "2 Horror Movies", "hook": "Get ready for a night of fear."}}
    assert build_description(vague, films, [], []).startswith("2 Horror Movies, featuring Sinners and Weapons.")


def test_description_removes_legacy_disclaimer_and_names_only_recognised_services():
    script = {"sections": [], "package": {"title": "Best Shows", "description": "Try these shows.\nFilm information: TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB."}}
    films = [{"title": "A Show", "year": 2025, "kind": "tv", "genres": ["Drama"],
              "watch_availability": {"subscription": ["Hulu", "JustWatch TV"], "rent": ["Prime Video"], "checked_at": "2026-10-20T12:00:00Z"}},
             {"title": "B Show", "year": 2024, "kind": "tv", "watch_availability": {"rent": ["Amazon Video"]}}]
    result = build_description(script, films, [], [])
    assert "TMDB" not in result and "JustWatch" not in result
    assert "Featured shows / series" in result
    assert "#1 A Show (2025) — Hulu" in result
    assert "#2 B Show (2024) — rent or buy on Prime Video" in result
    assert "US listings, checked 2026-10-20" in result
    assert "#TVRecommendations" in result


def test_ctas_are_short_and_most_request_like_and_subscribe():
    assert all(len(text.split()) <= 16 for text in MIDROLLS)
    assert sum("like" in text.casefold() and "subscribe" in text.casefold() for text in MIDROLLS) > len(MIDROLLS)/2
    assert midroll_text("Example") == midroll_text("Example")


def test_topic_tags_are_relevant_distinct_and_bounded():
    films = [{"kind": "movie", "genres": ["Horror", "Thriller", "Mystery", "Fantasy"]}]
    tags = topic_hashtags("Top 10 Horror Movies", films)
    assert 3 <= len(tags) <= 5
    assert len(tags) == len(set(tags))
    assert "#HorrorMovies" in tags
    assert "#ActionMovies" not in tags
    assert featured_heading([{"kind": "movie"}, {"kind": "tv"}]) == "Featured movies and shows / series"


@pytest.mark.parametrize("closing", [
    "A clear and casual recommendation.",
    "Like the video, and watch our other recommendations.",
    "Subscribe and watch our other picks.",
    "Like and subscribe, then watch our other videos.",
])
def test_writer_keeps_availability_in_metadata_and_adds_outro_cta(closing):
    from unittest.mock import Mock

    from movie_pipeline.script import CLOSING, Writer
    writer = Writer()
    writer.narration = Mock(side_effect=lambda title, brief, label, facts, target: [
        closing if label == CLOSING else "A clear and casual recommendation."])
    writer.ask = Mock(return_value={"titles": ["Two movies to watch tonight", "Top 2 Movies to Watch"], "description": "Try these two movie picks.",
                                   "thumbnail_texts": ["WATCH THESE", "GREAT MOVIES", "NEXT WATCH"]})
    writer.thumbnail_concepts = Mock(return_value=[])
    films = [{"title": "First", "year": 2025, "status": "Released", "kind": "movie",
              "watch_availability": {"subscription": ["Hulu"]}},
             {"title": "Second", "year": 2024, "status": "Released", "kind": "movie"}]
    result = writer.write("Top 2 Movies", "Keep it relaxed", films, target_seconds=300)
    review = next(section for section in result['sections'] if section['kind'] == 'film' and section['film'] == 0)
    assert not any('Hulu' in p for p in review['paragraphs'])
    assert films[0]['watch_availability']['subscription'] == ['Hulu']
    assert 'US' not in review['paragraphs'][-1]
    outro = ' '.join(result['sections'][-1]['paragraphs']).casefold()
    assert len(re.findall(r"\blike\b", outro)) == 1
    assert len(re.findall(r"\bsubscribe\b", outro)) == 1
    assert "other" in outro
    assert any(word in outro for word in ("picks", "recommendations", "videos"))
    assert result['sections'][1]['film'] == 1  # supplied best-first list plays in countdown order
    assert 'plain everyday search words' in writer.ask.call_args.args[0]


def test_refresh_availability_replaces_old_provider_line_and_keeps_review():
    from movie_pipeline.packaging import refresh_availability
    old = "It's listed on Hulu with a subscription."
    script = {'sections': [{'kind': 'film', 'film': 0, 'paragraphs': ['Original review.', old], 'availability_text': old}]}
    films = [{'watch_availability': {'subscription': ['Netflix']}}]
    refresh_availability(script, films)
    section = script['sections'][0]
    assert section['paragraphs'][0] == 'Original review.'
    assert len(section['paragraphs']) == 1
    assert section['paragraphs'] == ['Original review.']
    assert films[0]['watch_availability']['subscription'] == ['Netflix']
    refresh_availability(script, films)
    assert len(section['paragraphs']) == 1
    refresh_availability(script, [{'watch_availability': {}}])
    assert section['paragraphs'] == ['Original review.']


def test_typed_title_is_kept_and_alternatives_are_offered():
    from unittest.mock import Mock

    from movie_pipeline.script import Writer
    writer = Writer()
    writer.narration = Mock(return_value=["A clear and casual recommendation."])
    writer.ask = Mock(return_value={"titles": ["2 Movies You Need To Watch This Weekend", "Top 2 Movies of 2025 Worth Watching"],
                                   "hook": "x", "description": "d", "thumbnail_texts": ["A B", "C D", "E F"], "tags": ["a"]})
    writer.thumbnail_concepts = Mock(return_value=[])
    films = [{"title": "First", "year": 2025, "status": "Released", "kind": "movie"},
             {"title": "Second", "year": 2024, "status": "Released", "kind": "movie"}]
    package = writer.write("My Own Title For 2 Movies", "", films, target_seconds=300)["package"]
    assert package["title"] == "My Own Title For 2 Movies"
    assert package["title_options"][0] == "My Own Title For 2 Movies" and len(package["title_options"]) == 3
