"""Grounded research must verify identities without depending on TMDB."""
import json
from unittest.mock import Mock

import pytest
from movie_pipeline import gemini_research, research as research_module
from movie_pipeline.availability import enrich_availability
from movie_pipeline.core import FilmInput, PipelineError
from movie_pipeline.gemini import GeminiUnavailable
from movie_pipeline.research import Research

ITEM = FilmInput("Arrival", 2016)
IDENTITY = "Arrival (2016) is a science fiction film directed by Denis Villeneuve."
PREMISE = "Linguist Louise Banks works with a team to communicate with mysterious alien visitors."
CAST = "The film stars Amy Adams as Louise Banks."


def grounded(url="https://www.paramount.com/movies/arrival"):
    quotes = [IDENTITY, PREMISE, CAST]
    return {"text": " ".join(quotes),
            "sources": [{"url": url, "uri": url, "title": "Arrival", "index": 0}],
            "grounding": {"groundingChunks": [{"web": {"uri": url, "title": "Arrival"}}],
                          "groundingSupports": [
                              {"segment": {"text": quote}, "groundingChunkIndices": [0]} for quote in quotes]}}


def extracted():
    return {"title": "Arrival", "year": 2016, "kind": "movie",
            "identity_evidence": IDENTITY, "overview_evidence": PREMISE, "research_notes_evidence": [],
            "cast": [{"name": "Amy Adams", "evidence": CAST}],
            "directors": [{"name": "Denis Villeneuve", "evidence": IDENTITY}],
            "creators": [], "genres": [{"name": "science fiction", "evidence": IDENTITY}]}


@pytest.fixture
def provider(monkeypatch):
    client = Mock()
    client.grounded_text.return_value = grounded()
    client.generate_json.return_value = extracted()
    monkeypatch.setattr(gemini_research, "get_client", lambda: client)
    monkeypatch.setattr(research_module, "enabled", lambda: True)
    monkeypatch.setattr(research_module, "record_event", Mock())
    monkeypatch.setenv("WIKIPEDIA_CONTEXT", "false")
    monkeypatch.delenv("TMDB_READ_TOKEN", raising=False)
    return client


def test_grounded_gemini_succeeds_without_tmdb_token(provider, tmp_path):
    research = Research()
    research.resolve = Mock(side_effect=AssertionError("TMDB is not required for grounded success"))
    films = research.collect([ITEM], tmp_path / "research.json")
    film = films[0]
    assert film["title"] == "Arrival" and film["year"] == 2016 and film["kind"] == "movie"
    assert film["overview"] == PREMISE
    assert film["cast"] == ["Amy Adams"] and film["directors"] == ["Denis Villeneuve"]
    assert film["id"].startswith("gemini-") and len(film["id"]) == 27
    assert "tmdb_id" not in film
    assert film["trailers"] == film["clips"] == film["backdrops"] == film["posters"] == []
    assert json.loads((tmp_path / "research.json").read_text())[0] == film
    assert gemini_research.research_film(ITEM, "movie")["id"] == film["id"]
    research.resolve.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("title", "Arrival 2"), ("year", 2017), ("year", True), ("kind", "tv"),
    ("identity_evidence", "An invented film with an unsupported identity."),
    ("overview_evidence", "A made up premise that the evidence never established.")])
def test_mismatched_or_invented_structured_claims_are_rejected(provider, field, value):
    provider.generate_json.return_value[field] = value
    with pytest.raises(GeminiUnavailable):
        gemini_research.research_film(ITEM, "movie")


@pytest.mark.parametrize("case", ["empty", "sources", "supports", "unquoted", "bool-index", "negative-index",
                                  "unknown-index", "different-uri", "wrong-year", "wrong-medium", "sequel"])
def test_grounding_and_explicit_identity_are_required(provider, case):
    result = provider.grounded_text.return_value
    if case == "empty":
        result["text"] = ""
    elif case == "sources":
        result["sources"] = []
    elif case == "supports":
        result["grounding"]["groundingSupports"] = []
    elif case == "unquoted":
        result["grounding"]["groundingSupports"][0]["segment"]["text"] = "Unrelated unquoted content."
    elif case in {"bool-index", "negative-index", "unknown-index"}:
        index = {"bool-index": True, "negative-index": -1, "unknown-index": 3}[case]
        for row in result["grounding"]["groundingSupports"]:
            row["groundingChunkIndices"] = [index]
    elif case == "different-uri":
        result["grounding"]["groundingChunks"][0]["web"]["uri"] = "https://unrelated.example/film"
    else:
        quote = {"wrong-year": IDENTITY.replace("2016", "2017"),
                 "wrong-medium": IDENTITY.replace("film", "television series"),
                 "sequel": IDENTITY.replace("Arrival", "Arrival 2")}[case]
        result["text"] = result["text"].replace(IDENTITY, quote)
        result["grounding"]["groundingSupports"][0]["segment"]["text"] = quote
        provider.generate_json.return_value["identity_evidence"] = quote
    with pytest.raises(GeminiUnavailable):
        gemini_research.research_film(ITEM, "movie")


def test_unsupported_optional_metadata_is_omitted(provider):
    provider.generate_json.return_value["cast"].append({"name": "Invented Actor", "evidence": CAST})
    provider.generate_json.return_value["directors"].append({"name": "Amy Adams", "evidence": CAST})
    film = gemini_research.research_film(ITEM, "movie")
    assert film["cast"] == ["Amy Adams"] and film["directors"] == ["Denis Villeneuve"]


@pytest.mark.parametrize("url,identifier", [
    ("https://www.themoviedb.org/movie/329865-arrival", 329865),
    ("https://themoviedb.org/movie/329865", 329865),
    ("https://www.themoviedb.org/tv/329865", None),
    ("https://themoviedb.org.evil.example/movie/329865", None),
    ("https://www.themoviedb.org/person/329865", None),
    ("https://www.themoviedb.org/movie/0", None)])
def test_tmdb_id_only_comes_from_real_cited_matching_record(provider, url, identifier):
    provider.grounded_text.return_value = grounded(url)
    film = gemini_research.research_film(ITEM, "movie")
    assert film.get("tmdb_id") == identifier
    assert isinstance(film["id"], str) and film["id"].startswith("gemini-")


def test_uncited_tmdb_source_does_not_create_an_id(provider):
    result = provider.grounded_text.return_value
    url = "https://www.themoviedb.org/movie/329865"
    result["sources"].append({"url": url, "title": "Arrival", "index": 1})
    result["grounding"]["groundingChunks"].append({"web": {"uri": url}})
    assert "tmdb_id" not in gemini_research.research_film(ITEM, "movie")


@pytest.mark.parametrize("reason", ["HTTP 429 quota", "HTTP 401 auth"])
def test_provider_failure_uses_original_verified_tmdb_fallback(provider, tmp_path, reason, capsys):
    secret = "fake-private-key"
    provider.grounded_text.side_effect = GeminiUnavailable(reason + " " + secret)
    research = Research()
    fallback = {"title": "Arrival", "year": 2016, "kind": "movie", "id": 329865, "overview": PREMISE}
    research.resolve = Mock(return_value=fallback)
    assert research.collect([ITEM], tmp_path / "research.json") == [fallback]
    research.resolve.assert_called_once_with(ITEM)
    assert secret not in capsys.readouterr().out
    assert secret not in str(research_module.record_event.call_args_list)


def test_failed_gemini_without_tmdb_is_clear_and_never_waives_identity(provider, tmp_path):
    provider.grounded_text.side_effect = GeminiUnavailable("unavailable")
    research = Research()
    research.session.get = Mock(side_effect=AssertionError("No unauthenticated TMDB request"))
    with pytest.raises(PipelineError, match="TMDB fallback requires TMDB_READ_TOKEN"):
        research.collect([ITEM], tmp_path / "research.json")
    research.session.get.assert_not_called()
    assert not (tmp_path / "research.json").exists()


def test_tmdb_auth_error_is_safe_and_actionable(provider, monkeypatch):
    monkeypatch.setenv("TMDB_READ_TOKEN", "fake-private-token")
    research = Research()
    research.session.get = Mock(return_value=Mock(status_code=401))
    with pytest.raises(PipelineError, match="TMDB fallback authentication failed") as error:
        research.get("search/movie", query="Arrival")
    assert "fake-private-token" not in str(error.value)


def test_optional_tmdb_media_failure_does_not_discard_grounded_success(provider, monkeypatch, tmp_path):
    monkeypatch.setenv("TMDB_READ_TOKEN", "configured")
    provider.grounded_text.return_value = grounded("https://www.themoviedb.org/movie/329865")
    research = Research()
    research.resolve = Mock(side_effect=PipelineError("TMDB unavailable"))
    assert research.collect([ITEM], tmp_path / "research.json")[0]["research_backend"] == "gemini"


def test_local_mode_preserves_existing_mandatory_tmdb_contract(provider, monkeypatch):
    monkeypatch.setattr(research_module, "enabled", lambda: False)
    with pytest.raises(PipelineError, match="Set the free TMDB_READ_TOKEN"):
        Research()
    provider.grounded_text.assert_not_called()


def test_gemini_identity_is_not_used_as_tmdb_availability_endpoint(provider, tmp_path, monkeypatch):
    film = gemini_research.research_film(ITEM, "movie")
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    monkeypatch.setattr("movie_pipeline.availability.requests.Session", lambda: session)
    research = Research()
    # Even a cited numeric ID cannot trigger an unauthenticated optional request.
    film["tmdb_id"] = 329865
    result = enrich_availability([film], research, tmp_path / "availability")
    assert result[0]["watch_availability"]["status"] == "unavailable"
    session.get.assert_not_called()


def test_distinctive_research_notes_are_exact_supported_quotations(provider):
    note = "The visitors arrive in twelve spacecraft, leaving the team to decipher their written language."
    result = provider.grounded_text.return_value
    result["text"] += " " + note
    result["grounding"]["groundingSupports"].append(
        {"segment": {"text": note}, "groundingChunkIndices": [0]})
    provider.generate_json.return_value["research_notes_evidence"] = [
        note, "Invented action scene that never appeared in cited evidence."]
    film = gemini_research.research_film(ITEM, "movie")
    assert film["research_notes"] == note
