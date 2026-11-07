from unittest.mock import Mock

import pytest

from movie_pipeline.core import FilmInput, PipelineError
from movie_pipeline.research import Research


def research_for(movies, shows=()):
    research = object.__new__(Research)

    def get(endpoint, **params):
        if endpoint == "search/movie":
            return {"results": movies}
        if endpoint == "search/tv":
            return {"results": list(shows)}
        movie_id = int(endpoint.split("/")[1])
        match = next(x for x in movies + list(shows) if x["id"] == movie_id)
        return {**match, "overview": "An original science fiction premise.", "videos": {"results": [{"site": "YouTube", "type": "Trailer", "key": "example1234"}]}, "credits": {}}

    research.get = Mock(side_effect=get)
    research.wikidata_candidates = Mock(side_effect=AssertionError("An eligible TMDB result must not need independent lookup"))
    return research


def resolve(movies, shows=()):
    return research_for(movies, shows).resolve(FilmInput("Arrival", 2016))


def film(identifier, votes, title="Arrival", year="2016", popularity=0):
    return {"id": identifier, "title": title, "original_title": title, "release_date": year + "-09-01", "vote_count": votes, "popularity": popularity}


def test_first_eligible_result_is_selected():
    assert resolve([film(1, 18000), film(2, 3)])["id"] == 1


def test_api_order_is_not_overridden_by_audience_votes():
    assert resolve([film(2, 3), film(1, 18000)])["id"] == 2


def test_api_order_is_not_overridden_by_popularity():
    assert resolve([film(2, 0, popularity=1), film(1, 0, popularity=500)])["id"] == 2


@pytest.mark.parametrize("votes", [(0, 0), (1, 2), (100, 90)])
def test_multiple_matches_never_fail_for_low_or_similar_votes(votes):
    assert resolve([film(1, votes[0]), film(2, votes[1])])["id"] == 1


def test_zero_vote_upcoming_movies_use_first_eligible_result():
    rows = [film(2, 0, title="The Odyssey", year="2026"), film(1, 0, title="The Odyssey", year="2026")]
    research = research_for(rows)
    assert research.resolve(FilmInput("The Odyssey", 2026))["id"] == 2
    research.wikidata_candidates.assert_not_called()


def test_duplicate_search_records_are_not_ambiguous():
    assert resolve([film(1, 0), film(1, 0)])["id"] == 1


def test_wrong_year_is_filtered_before_selecting_first_result():
    assert resolve([film(2, 20000, year="2015"), film(1, 20)])["id"] == 1


def test_unrelated_title_is_filtered_before_selecting_first_result():
    assert resolve([film(2, 20000, title="A Completely Different Movie"), film(1, 20)])["id"] == 1


def test_first_eligible_near_title_match_is_not_reordered_for_exact_match():
    assert resolve([film(2, 0, title="Arrivals"), film(1, 20000)])["id"] == 2


def test_movie_search_result_keeps_precedence_over_tv_match():
    show = {"id": 2, "name": "Arrival", "original_name": "Arrival", "first_air_date": "2016-01-01", "vote_count": 20000}
    assert resolve([film(1, 0)], [show])["kind"] == "movie"


def test_preferred_tv_result_wins_even_with_fewer_votes_than_movie():
    show = {"id": 2, "name": "Arrival", "original_name": "Arrival", "first_air_date": "2016-01-01", "vote_count": 0}
    research = research_for([film(1, 20000)], [show])
    research.preferred_kind = "tv"
    result = research.resolve(FilmInput("Arrival", 2016))
    assert result["kind"] == "tv"
    assert result["id"] == 2
    assert research.get.call_args_list[0].args[0] == "search/tv"


@pytest.mark.parametrize("selected, preferred, expected_id", [("tv", "movie", 2), ("movie", "tv", 1)])
def test_explicit_medium_overrides_inferred_preference(selected, preferred, expected_id):
    show = {"id": 2, "name": "Arrival", "original_name": "Arrival", "first_air_date": "2016-01-01"}
    research = research_for([film(1, 0)], [show])
    research.selected_kind = selected
    research.preferred_kind = preferred
    assert research.resolve(FilmInput("Arrival", 2016))["id"] == expected_id
    assert research.get.call_args_list[0].args[0] == f"search/{selected}"


def test_explicit_shows_never_substitutes_same_named_movie():
    research = research_for([film(1, 20000)])
    research.selected_kind = "tv"
    research.wikidata_candidates = Mock(return_value=[])
    with pytest.raises(PipelineError, match="Could not verify"):
        research.resolve(FilmInput("Arrival", 2016))
    assert [call.args[0] for call in research.get.call_args_list] == ["search/tv"]


def test_independent_fallback_also_respects_selected_medium():
    research = research_for([film(1, 20000)])
    research.selected_kind = "tv"
    research.wikidata_candidates = Mock(return_value=[(1, "movie", film(1, 20000))])
    with pytest.raises(PipelineError, match="Could not verify"):
        research.resolve(FilmInput("Arrival", 2016))
    research.wikidata_candidates.assert_called_once()
    assert [call.args[0] for call in research.get.call_args_list] == ["search/tv"]


def test_eligible_preferred_result_skips_unneeded_other_medium_search():
    research = research_for([film(1, 0)])
    research.preferred_kind = "movie"
    original_get = research.get.side_effect

    def get(endpoint, **params):
        if endpoint == "search/tv":
            raise PipelineError("Unneeded search service is unavailable")
        return original_get(endpoint, **params)

    research.get.side_effect = get
    assert research.resolve(FilmInput("Arrival", 2016))["id"] == 1
    assert [call.args[0] for call in research.get.call_args_list] == ["search/movie", "movie/1"]


def test_no_eligible_title_still_requires_verified_identity():
    research = research_for([film(1, 50000, title="A Completely Different Movie")])
    research.wikidata_candidates = Mock(return_value=[])
    with pytest.raises(PipelineError, match="Could not verify"):
        research.resolve(FilmInput("Arrival", 2016))


def coherence_research(release_year=None):
    research = object.__new__(Research)
    row = film(77, 2000, title="Coherence", year="2014")

    def get(endpoint, **params):
        if endpoint == "search/movie":
            return {"results": [row]}
        if endpoint == "search/tv":
            return {"results": []}
        if endpoint == "movie/77/release_dates":
            return {"results": [{"release_dates": [{"release_date": f"{release_year}-09-19T00:00:00Z"}]}]} if release_year else {"results": []}
        if endpoint == "find/tt2866360":
            return {"movie_results": [row]}
        return {**row, "overview": "Friends gather for dinner during an unusual astronomical event.", "videos": {"results": [{"site": "YouTube", "type": "Trailer"}]}, "credits": {}}

    research.get = Mock(side_effect=get)
    return research


def test_festival_release_year_is_valid_even_if_search_uses_theatrical_year():
    research = coherence_research(2013)
    research.wikidata_candidates = Mock(side_effect=AssertionError("Dated release already verifies the input"))
    result = research.resolve(FilmInput("Coherence", 2013))
    assert result["id"] == 77
    assert result["year"] == 2013
    assert result["release_date"].startswith("2014")
    assert "festival" in result["year_evidence"]


def test_nearby_year_alone_never_accepts_the_wrong_film():
    research = coherence_research(2014)
    research.wikidata_candidates = Mock(return_value=[])
    with pytest.raises(PipelineError, match="does not establish that your input is wrong"):
        research.resolve(FilmInput("Coherence", 2013))


def test_wikidata_year_and_imdb_identity_fallback(monkeypatch):
    research = coherence_research()
    claims = {"P577": [{"mainsnak": {"datavalue": {"value": {"time": "+2013-09-19T00:00:00Z", "precision": 11}}}}],
              "P345": [{"mainsnak": {"datavalue": {"value": "tt2866360"}}}]}
    payloads = [{"search": [{"id": "QTEST"}]}, {"entities": {"QTEST": {"id": "QTEST", "labels": {"en": {"value": "Coherence"}}, "claims": claims}}}]
    responses = [Mock(json=Mock(return_value=p)) for p in payloads]
    getter = Mock(side_effect=responses)
    monkeypatch.setattr("movie_pipeline.research.requests.get", getter)
    result = research.resolve(FilmInput("Coherence", 2013))
    assert result["imdb_id"] == "tt2866360"
    assert result["identity_source"] == "https://www.wikidata.org/wiki/QTEST"
    for call in getter.call_args_list:
        assert "Authorization" not in call.kwargs["headers"]


def test_unavailable_independent_source_does_not_invent_identity(monkeypatch):
    import requests
    research = coherence_research()
    monkeypatch.setattr("movie_pipeline.research.requests.get", Mock(side_effect=requests.ConnectionError("unavailable")))
    with pytest.raises(PipelineError, match="Could not verify"):
        research.resolve(FilmInput("Coherence", 2013))
