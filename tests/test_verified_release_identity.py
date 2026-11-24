"""Regression coverage for release identity, API order and research cache migration."""
import json
from unittest.mock import Mock

import pytest

from movie_pipeline.core import FilmInput, PipelineError
from movie_pipeline.research import Research
from movie_pipeline.release_identity import verified_release_years


def _row(identifier, title, year, kind="movie"):
    keys = ("title", "original_title", "release_date") if kind == "movie" else (
        "name", "original_name", "first_air_date")
    return {"id": identifier, keys[0]: title, keys[1]: title, keys[2]: f"{year}-05-13"}


def _research(rows, *, kind="movie", aliases=None, releases=None):
    """Only the supplied same-ID responses can establish identity."""
    research = object.__new__(Research)
    research.token = ""
    research.selected_kind = kind
    aliases, releases = aliases or {}, releases or {}

    def get(endpoint, **params):
        if endpoint == f"search/{kind}":
            return {"results": rows}
        identifier = int(endpoint.split("/")[1])
        if endpoint.endswith("/alternative_titles"):
            payload = aliases.get(identifier, [])
            if isinstance(payload, dict):
                return payload
            return {"id": identifier, "titles" if kind == "movie" else "results":
                    [{"title": name} for name in payload]}
        if endpoint.endswith("/release_dates"):
            payload = releases.get(identifier, [])
            if isinstance(payload, dict):
                return payload
            return {"id": identifier, "results": [{"iso_3166_1": "CA",
                    "release_dates": payload}]}
        assert endpoint == f"{kind}/{identifier}", endpoint
        row = next(row for row in rows if row["id"] == identifier)
        return {**row, "overview": "A lonely young man confronts the consequences of a dangerous wish.",
                "videos": {"results": [{"site": "YouTube", "type": "Trailer",
                                       "key": "example1234", "official": True}]},
                "credits": {"crew": [{"name": "Curry Barker", "job": "Director"}]},
                "images": {"backdrops": [{"file_path": "/right.jpg", "width": 1920}]}}

    research.get = Mock(side_effect=get)
    research.wikidata_candidates = Mock(return_value=[])
    return research


@pytest.mark.parametrize("requested,canonical", [
    ("Money Heist: Berlin", "Berlin"),
    ("Berlin", "Money Heist: Berlin"),
])
def test_berlin_alias_is_symmetric_and_proven_on_same_tmdb_id(requested, canonical):
    row = _row(146176, canonical, 2023, "tv")
    research = _research([row], kind="tv", aliases={146176: [requested]})
    result = research.resolve(FilmInput(requested, 2023))
    assert result["id"] == 146176
    assert result["title"] == canonical
    assert result["kind"] == "tv"
    assert result["year"] == 2023
    assert any(call.args[0] == "tv/146176/alternative_titles"
               for call in research.get.call_args_list)


@pytest.mark.parametrize("canonical", [
    "Money Heist: Berlin Reunion", "Berlin: Behind the Scenes", "Berlin Academy",
])
def test_shared_words_or_prefix_do_not_establish_a_title_alias(canonical):
    row = _row(146176, canonical, 2023, "tv")
    research = _research([row], kind="tv")
    with pytest.raises(PipelineError, match="Could not verify"):
        research.resolve(FilmInput("Berlin", 2023))


def test_alias_from_a_different_tmdb_id_is_not_used():
    row = _row(146176, "Berlin", 2023, "tv")
    wrong = {"id": 999, "results": [{"title": "Money Heist: Berlin"}]}
    research = _research([row], kind="tv", aliases={146176: wrong})
    with pytest.raises(PipelineError, match="Could not verify"):
        research.resolve(FilmInput("Money Heist: Berlin", 2023))


def test_first_eligible_festival_release_beats_later_exact_search_year():
    # Actual diagnostic shape: the popular feature's theatrical date is 2026;
    # a different short film has the exact requested 2025 search year.
    feature = _row(1339713, "Obsession", 2026)
    short = _row(1436161, "Obsession", 2025)
    research = _research([feature, short], releases={1339713: [
        {"release_date": "2025-09-06T00:00:00.000Z", "type": 1, "note": "TIFF"},
        {"release_date": "2026-05-13T00:00:00.000Z", "type": 3, "note": ""},
    ]})
    result = research.resolve(FilmInput("Obsession", 2025))
    assert result["id"] == 1339713
    assert result["directors"] == ["Curry Barker"]
    assert result["backdrops"] == ["/right.jpg"]
    assert verified_release_years(result) == {2025, 2026}
    endpoints = [call.args[0] for call in research.get.call_args_list]
    assert endpoints.index("movie/1339713/release_dates") < endpoints.index("movie/1339713")
    assert "movie/1436161" not in endpoints


def test_unsupported_search_year_cannot_skip_to_a_different_remake():
    research = _research([_row(77, "Obsession", 2018)])
    with pytest.raises(PipelineError, match="Could not verify"):
        research.resolve(FilmInput("Obsession", 2025))


def test_release_dates_payload_cannot_prove_another_tmdb_production():
    research = _research([_row(1339713, "Obsession", 2026)], releases={1339713: {
        "id": 1436161, "results": [{"release_dates": [
            {"release_date": "2025-09-06T00:00:00.000Z", "type": 1}]}]}})
    with pytest.raises(PipelineError, match="Could not verify"):
        research.resolve(FilmInput("Obsession", 2025))


def _film():
    return {"title": "Obsession", "year": 2025, "kind": "movie", "id": 1339713,
            "source": "https://www.themoviedb.org/movie/1339713"}


def _evidence(year, *, identifier=1339713):
    return {"year": year, "source": f"https://api.themoviedb.org/3/movie/{identifier}/release_dates",
            "kind": "festival", "identity": f"movie/{identifier}", "date": f"{year}-09-06"}


def test_only_same_production_evidence_adds_release_years():
    film = {**_film(), "release_years": [1999, 2024, 2025, 2026],
            "release_year_evidence": [_evidence(2026), _evidence(2024, identifier=1436161)]}
    assert verified_release_years(film) == {2025, 2026}


def test_untrusted_year_list_and_free_text_year_evidence_are_not_proof():
    film = {**_film(), "release_years": [2024, 2026], "year_evidence": "A festival was in 2024"}
    assert verified_release_years(film) == {2025}


@pytest.mark.parametrize("bad", [None, "2026", True, {}, {"year": 2026}, []])
def test_malformed_release_year_evidence_is_ignored(bad):
    assert verified_release_years({**_film(), "release_year_evidence": [bad]}) == {2025}


def test_same_id_without_its_canonical_source_is_not_release_proof():
    evidence = {**_evidence(2026), "source": "https://example.com/movie/1339713"}
    assert verified_release_years({**_film(), "release_year_evidence": [evidence]}) == {2025}


def _cache_research(monkeypatch):
    monkeypatch.setenv("WIKIPEDIA_CONTEXT", "false")
    monkeypatch.setenv("AI_BACKEND", "local")
    research = object.__new__(Research)
    research.token = ""
    research.selected_kind = "movie"
    research.get = Mock(side_effect=AssertionError("The cache tests must not access an API"))
    film = {**_film(), "overview": "A lonely young man faces the consequences of a dangerous wish.",
            "release_date": "2026-05-13", "release_year_evidence": [_evidence(2026)],
            "trailers": [], "clips": []}
    research.resolve = Mock(return_value=film)
    return research


def test_current_cache_roundtrip_skips_all_research_calls(monkeypatch, tmp_path):
    research = _cache_research(monkeypatch)
    path = tmp_path / "research.json"
    item = FilmInput("Obsession", 2025)
    first = research.collect([item], path)
    assert first[0]["release_identity_version"] == 1
    research.resolve.reset_mock()
    second = research.collect([item], path)
    assert second == first
    research.resolve.assert_not_called()
    research.get.assert_not_called()


@pytest.mark.parametrize("old", [
    "{broken", "null", "{}", "[null]",
    '[{"title": "Obsession", "year": 2025, "kind": "movie", "id": 1436161}]',
])
def test_old_or_malformed_cache_is_migrated_by_collect(monkeypatch, tmp_path, old):
    research = _cache_research(monkeypatch)
    path = tmp_path / "research.json"
    path.write_text(old, encoding="utf-8")
    films = research.collect([FilmInput("Obsession", 2025)], path)
    research.resolve.assert_called_once()
    assert films[0]["id"] == 1339713
    assert films[0]["release_identity_version"] == 1
    assert json.loads(path.read_text(encoding="utf-8")) == films


def test_current_cache_cannot_be_reused_for_a_different_input_year(monkeypatch, tmp_path):
    research = _cache_research(monkeypatch)
    path = tmp_path / "research.json"
    research.collect([FilmInput("Obsession", 2025)], path)
    research.resolve.reset_mock()
    research.resolve.return_value = {**research.resolve.return_value, "year": 2026}
    result = research.collect([FilmInput("Obsession", 2026)], path)
    research.resolve.assert_called_once_with(FilmInput("Obsession", 2026))
    assert result[0]["year"] == 2026


def _public_film():
    url = "https://en.wikipedia.org/wiki/Obsession_(2025_film)"
    identity = "Obsession is an American supernatural horror film written and directed by Curry Barker."
    quote = "The film had its world premiere at the Toronto International Film Festival on September 6, 2025."
    return {"title": "Obsession", "year": 2026, "kind": "movie", "id": "gemini-verified",
            "source": url, "identity_source": url, "identity_evidence": identity,
            "research_backend": "gemini", "research_method": "public_source_context",
            "research_sources": [{"url": url, "title": "Obsession (2025 film)",
                                  "source_type": "retrieved_public_reference"}],
            "research_evidence": [{"text": identity, "sources": [url]},
                                  {"text": quote, "sources": [url]}],
            "release_year_evidence": [{"year": 2025, "source": url, "identity": url,
                                       "kind": "public_release", "quote": quote}]}


def test_public_release_year_requires_an_exact_quote_on_the_verified_page():
    assert verified_release_years(_public_film()) == {2025, 2026}


@pytest.mark.parametrize("case", ["invented-quote", "different-page", "different-identity"])
def test_public_release_evidence_cannot_cross_page_identity(case):
    film = _public_film()
    evidence = film["release_year_evidence"][0]
    if case == "invented-quote":
        evidence["quote"] = "The film had a release in 2025 that no retrieved source actually documented."
    elif case == "different-page":
        evidence["source"] = "https://en.wikipedia.org/wiki/Obsession_(other_film)"
    else:
        evidence["identity"] = "https://en.wikipedia.org/wiki/Obsession_(other_film)"
    assert verified_release_years(film) == {2026}



def test_public_page_extract_retains_quoted_festival_and_theatrical_years(monkeypatch):
    gemini_research = pytest.importorskip("movie_pipeline.gemini_research")
    lead = "Obsession is a 2025 American supernatural horror film written and directed by Curry Barker."
    festival = "The film had its world premiere at the Toronto International Film Festival on September 6, 2025."
    theatrical = "The film was released theatrically in the United States on May 13, 2026."
    premise = "A lonely young man makes a wish that turns his romantic hopes into a dangerous obsession."
    url = "https://en.wikipedia.org/wiki/Obsession_(2025_film)"
    page = {"title": "Obsession (2025 film)", "url": url, "aliases": ["Obsession"],
            "text": "\n\n".join([lead, premise, festival, theatrical])}
    client = Mock()
    client.generate_json.return_value = {
        "title": "Obsession", "year": 2025, "kind": "movie", "source_index": 0,
        "identity_evidence": lead, "year_evidence": festival,
        "overview_evidence": premise, "research_notes_evidence": [],
        "cast": [], "directors": [{"name": "Curry Barker", "evidence": lead}],
        "creators": [], "genres": [],
    }
    monkeypatch.setattr(gemini_research, "collect_public_sources", Mock(return_value=[page]))
    monkeypatch.setattr(gemini_research, "get_client", Mock(return_value=client))
    film = gemini_research._public_film(FilmInput("Obsession", 2025), "movie")
    assert verified_release_years(film) == {2025, 2026}
    assert client.generate_json.call_count == 1
    assert film["overview"] == premise
    assert film["identity_evidence"] == lead
    quotes = {row["text"] for row in film["research_evidence"]}
    assert festival in quotes and theatrical in quotes
    assert any(row["year"] == 2026 and row["quote"] == theatrical
               for row in film["release_year_evidence"])


@pytest.mark.parametrize("conflict", [None, "director", "imdb"])
def test_public_media_enrichment_stays_with_the_same_production(monkeypatch, tmp_path, conflict):
    pytest.importorskip("movie_pipeline.gemini_research")
    import movie_pipeline.research as module

    monkeypatch.setenv("WIKIPEDIA_CONTEXT", "false")
    monkeypatch.setattr(module, "enabled", lambda: True)
    monkeypatch.setattr(module, "record_event", Mock())
    film = {**_public_film(), "year": 2025,
            "overview": "A lonely young man faces the consequences of a dangerous wish.",
            "cast": ["Michael Johnston"], "directors": ["Curry Barker"],
            "imdb_id": "tt37287335", "backdrops": [], "posters": [], "trailers": []}
    metadata = {**_film(), "cast": ["Michael Johnston"], "directors": ["Curry Barker"],
                "imdb_id": "tt37287335", "backdrops": ["/feature.jpg"],
                "posters": ["/feature-poster.jpg"],
                "trailers": [{"site": "YouTube", "key": "example1234", "type": "Trailer"}],
                "release_year_evidence": [_evidence(2026)]}
    if conflict == "director":
        metadata["directors"] = ["Audrey Collins"]
    elif conflict == "imdb":
        metadata["imdb_id"] = "tt99999999"
    monkeypatch.setattr(module, "research_film", Mock(return_value=film))
    research = object.__new__(Research)
    research.token = "configured-without-network"
    research.selected_kind = "movie"
    research.resolve = Mock(return_value=metadata)
    enriched = research.collect([FilmInput("Obsession", 2025)], tmp_path / "research.json")[0]
    assert enriched["overview"] == film["overview"]
    if conflict:
        assert enriched["backdrops"] == []
        assert enriched["trailers"] == []
    else:
        assert enriched["tmdb_id"] == 1339713
        assert enriched["backdrops"] == ["/feature.jpg"]
        assert enriched["trailers"] == metadata["trailers"]
        assert verified_release_years(enriched) == {2025, 2026}


def test_tmdb_release_year_without_a_dated_event_is_not_proof():
    evidence = _evidence(2026)
    evidence.pop("date")
    assert verified_release_years({**_film(), "release_year_evidence": [evidence]}) == {2025}


def test_tmdb_release_event_date_must_match_its_claimed_year():
    evidence = {**_evidence(2026), "date": "2024-09-06"}
    assert verified_release_years({**_film(), "release_year_evidence": [evidence]}) == {2025}


@pytest.mark.parametrize("quote", [
    "The lead actor was born in 2024 and later starred in the film.",
    "Production began in 2024 with location filming in Canada.",
    "The second season premiered in 2024 with a new cast.",
])
def test_public_incidental_dates_cannot_add_a_release_year(quote):
    film = _public_film()
    url = film["source"]
    film["research_evidence"].append({"text": quote, "sources": [url]})
    film["release_year_evidence"] = [{"year": 2024, "source": url, "identity": url,
                                     "kind": "public_release", "quote": quote}]
    assert verified_release_years(film) == {2026}
