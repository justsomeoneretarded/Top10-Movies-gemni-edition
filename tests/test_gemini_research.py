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
    monkeypatch.setenv("GEMINI_RESEARCH_MODE", "legacy_google_search")
    monkeypatch.setattr(gemini_research, "collect_public_sources",
                        Mock(side_effect=GeminiUnavailable("test_public_source_unavailable")))
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



def public_page(title="Arrival (2016 film)", identity=IDENTITY, url="https://en.wikipedia.org/wiki/Arrival_(film)"):
    return {"title": title, "url": url, "aliases": [title],
            "text": identity + "\n\n" + PREMISE + "\n\n" + CAST,
            "source_type": "retrieved_public_reference"}


@pytest.fixture
def public_provider(provider, monkeypatch):
    monkeypatch.setenv("GEMINI_RESEARCH_MODE", "public_sources")
    monkeypatch.setattr(gemini_research, "collect_public_sources", Mock(return_value=[public_page()]))
    provider.generate_json.return_value = {**extracted(), "source_index": 0, "year_evidence": IDENTITY}
    return provider


def test_public_context_primary_without_tmdb_or_google_grounding(public_provider, tmp_path):
    public_provider.grounded_text.side_effect = AssertionError("Public context must not call Google Search")
    research = Research()
    research.resolve = Mock(side_effect=AssertionError("No TMDB account is required"))
    film = research.collect([ITEM], tmp_path / "public.json")[0]
    assert film["overview"] == PREMISE
    assert film["research_backend"] == "gemini"
    assert film["research_method"] == "public_source_context"
    assert film["source"] == "https://en.wikipedia.org/wiki/Arrival_(film)"
    assert film["research_sources"][0]["source_type"] == "retrieved_public_reference"
    assert "grounding" not in film
    assert "tmdb_id" not in film
    research.resolve.assert_not_called()
    public_provider.grounded_text.assert_not_called()


@pytest.mark.parametrize("title,year,lead", [
    ("Mutiny", 2026, "Mutiny is a 2026 American action thriller film directed by Jean-Francois Richet."),
    ("War Machine", 2026, "War Machine is a 2026 science fiction action film starring Alan Ritchson."),
    ("Havoc", 2025, "Havoc is a 2025 action thriller film written and directed by Gareth Evans."),
    ("Sisu: Road to Revenge", 2025,
     "Sisu: Road to Revenge is a 2025 Finnish action film and a sequel to Sisu, directed by Jalmari Helander.")])
def test_current_action_film_identity_quotes_need_no_synthetic_sentence(public_provider, monkeypatch, title, year, lead):
    page = public_page(title + f" ({year} film)", lead)
    monkeypatch.setattr(gemini_research, "collect_public_sources", Mock(return_value=[page]))
    public_provider.generate_json.return_value = {
        **extracted(), "title": title, "year": year, "source_index": 0,
        "identity_evidence": lead, "year_evidence": lead,
        "cast": [], "directors": [], "genres": []}
    assert gemini_research.research_film(FilmInput(title, year), "movie")["year"] == year


def test_public_tv_parenthetical_redirect_alias_and_later_premiere_year(public_provider, monkeypatch):
    lead = ("Lioness (also known as Special Ops: Lioness) is an American spy thriller television series "
            "created by Taylor Sheridan. It premiered on July 23, 2023, on Paramount+.")
    page = public_page("Lioness (American TV series)", lead)
    page["aliases"].append("Special Ops: Lioness")
    monkeypatch.setattr(gemini_research, "collect_public_sources", Mock(return_value=[page]))
    public_provider.generate_json.return_value = {
        **extracted(), "title": "Special Ops: Lioness", "year": 2023, "kind": "tv",
        "source_index": 0, "identity_evidence": lead, "year_evidence": lead,
        "cast": [], "directors": [], "creators": [], "genres": []}
    film = gemini_research.research_film(FilmInput("Special Ops: Lioness", 2023), "tv")
    assert film["kind"] == "tv" and film["identity_evidence"] == lead


@pytest.mark.parametrize("case", ["title", "year", "medium", "invented-quote", "production-year", "unverified-alias"])
def test_public_context_rejects_wrong_identity_or_unquoted_claim(public_provider, monkeypatch, case):
    data = public_provider.generate_json.return_value
    if case == "title":
        data["title"] = "Aliens"
    elif case == "year":
        data["year"] = 2017
    elif case == "medium":
        data["kind"] = "tv"
    elif case == "invented-quote":
        data["overview_evidence"] = "This invented plot never appeared in the actual reference page."
    elif case == "unverified-alias":
        page = public_page("Aliens (1986 film)")
        monkeypatch.setattr(gemini_research, "collect_public_sources", Mock(return_value=[page]))
    else:
        page = public_page()
        page["text"] += "\nProduction began in 2017 with filming in Montreal."
        data["year"] = 2017
        data["year_evidence"] = "Production began in 2017 with filming in Montreal."
        monkeypatch.setattr(gemini_research, "collect_public_sources", Mock(return_value=[page]))
    with pytest.raises(GeminiUnavailable):
        gemini_research.research_film(FilmInput("Arrival", 2017 if case == "production-year" else 2016), "movie")


def test_actual_regional_release_statement_can_verify_year(public_provider, monkeypatch):
    lead = "Arrival is a 2016 American science fiction film directed by Denis Villeneuve."
    release = "The film was released in the region on February 3, 2017."
    page = public_page("Arrival (2016 film)", lead)
    page["text"] += "\n" + release
    monkeypatch.setattr(gemini_research, "collect_public_sources", Mock(return_value=[page]))
    public_provider.generate_json.return_value.update(
        year=2017, identity_evidence=lead, year_evidence=release)
    film = gemini_research.research_film(FilmInput("Arrival", 2017), "movie")
    assert film["year_evidence"] == release


def test_public_notes_stay_exact_source_quotations(public_provider, monkeypatch):
    note = "The team must decipher the visitors' written language before tensions erupt between nations."
    page = public_page()
    page["text"] += "\n" + note
    monkeypatch.setattr(gemini_research, "collect_public_sources", Mock(return_value=[page]))
    public_provider.generate_json.return_value["research_notes_evidence"] = [note, "An invented climax."]
    assert gemini_research.research_film(ITEM, "movie")["research_notes"] == note


def test_legacy_google_search_failure_can_use_public_context(public_provider, monkeypatch):
    monkeypatch.setenv("GEMINI_RESEARCH_MODE", "legacy_google_search")
    public_provider.grounded_text.side_effect = GeminiUnavailable("legacy_capability_unavailable")
    assert gemini_research.research_film(ITEM, "movie")["research_method"] == "public_source_context"


def _wiki_response(payload, status=200):
    return Mock(status_code=status, raise_for_status=Mock(), json=Mock(return_value=payload))


def test_public_source_retrieval_uses_returned_url_aliases_and_no_auth(monkeypatch):
    from movie_pipeline import public_research
    search = {"query": {"search": [{"title": "Lioness (American TV series)"}]}}
    lead = ("Lioness (also known as Special Ops: Lioness) is an American television series. "
            "It premiered on July 23, 2023. A CIA officer leads a dangerous undercover operation.")
    page = {"pageid": 1, "ns": 0, "title": "Lioness (American TV series)",
            "fullurl": "https://en.wikipedia.org/wiki/Lioness_(TV_series)"}
    metadata = {"query": {"redirects": [{"from": "Special Ops: Lioness", "to": "Lioness (American TV series)"}],
                          "pages": [page]}}
    extracted = {"query": {"pages": [{**page, "extract": lead}]}}
    get = Mock(side_effect=[_wiki_response(search), _wiki_response(metadata), _wiki_response(extracted)])
    monkeypatch.setattr(public_research.requests, "get", get)
    monkeypatch.setattr(public_research, "record_event", Mock())
    source = public_research.collect_public_sources(FilmInput("Special Ops: Lioness", 2023), "tv")[0]
    assert source["url"] == page["fullurl"]
    assert "Special Ops: Lioness" in source["aliases"]
    assert get.call_count == 3
    for call in get.call_args_list:
        assert "Authorization" not in call.kwargs["headers"]
        assert call.kwargs["allow_redirects"] is False
    assert len(get.call_args_list[1].kwargs["params"]["titles"].split("|")) <= 5
    assert get.call_args_list[-1].kwargs["params"]["pageids"] == "1"


@pytest.mark.parametrize("case", ["html", "missing-url", "wrong-title", "disambiguation"])
def test_public_source_retrieval_rejects_error_or_wrong_pages(monkeypatch, case):
    from movie_pipeline import public_research
    page = {"pageid": 1, "title": "Arrival (2016 film)", "ns": 0, "extract": IDENTITY + " " + PREMISE,
            "fullurl": "https://en.wikipedia.org/wiki/Arrival_(film)"}
    if case == "html":
        page["extract"] = "<html><body>Gateway error. " + IDENTITY + PREMISE + "</body></html>"
    elif case == "missing-url":
        del page["fullurl"]
    elif case == "wrong-title":
        page["title"] = "Aliens (1986 film)"
    else:
        page["pageprops"] = {"disambiguation": ""}
    get = Mock(side_effect=[
        _wiki_response({"query": {"search": []}}),
        _wiki_response({"query": {"pages": [page]}}),
        _wiki_response({"query": {"pages": [page]}})])
    monkeypatch.setattr(public_research.requests, "get", get)
    monkeypatch.setattr(public_research, "record_event", Mock())
    with pytest.raises(GeminiUnavailable):
        public_research.collect_public_sources(ITEM, "movie")


def test_optional_tmdb_artwork_is_retained_after_public_success(public_provider, monkeypatch, tmp_path):
    monkeypatch.setenv("TMDB_READ_TOKEN", "configured")
    research = Research()
    metadata = {"title": "Arrival", "year": 2016, "kind": "movie", "id": 329865,
                "cast": ["Amy Adams"], "directors": ["Denis Villeneuve"],
                "backdrops": ["/actual-art.jpg"], "posters": ["/actual-poster.jpg"],
                "cast_profiles": [{"name": "Amy Adams", "profile_path": "/actual-profile.jpg"}],
                "source": "https://www.themoviedb.org/movie/329865"}
    research.resolve = Mock(return_value=metadata)
    film = research.collect([ITEM], tmp_path / "research.json")[0]
    assert film["tmdb_id"] == 329865 and film["backdrops"] == metadata["backdrops"]
    assert film["cast_profiles"] == metadata["cast_profiles"]
    assert film["overview"] == PREMISE and film["source"].startswith("https://en.wikipedia.org/")
    assert film["id"].startswith("gemini-")


@pytest.mark.parametrize("case", ["cast", "id", "title", "kind", "year"])
def test_optional_tmdb_conflict_does_not_replace_public_facts(public_provider, monkeypatch, tmp_path, case):
    monkeypatch.setenv("TMDB_READ_TOKEN", "configured")
    research = Research()
    metadata = {"title": "Arrival", "year": 2016, "kind": "movie", "id": 329865,
                "cast": ["Amy Adams"], "directors": ["Denis Villeneuve"],
                "backdrops": ["/wrong-film.jpg"], "source": "https://www.themoviedb.org/movie/329865"}
    if case == "cast":
        metadata["cast"] = ["A Different Actor"]
    elif case == "id":
        metadata["id"] = True
    elif case == "title":
        metadata["title"] = "Aliens"
    elif case == "kind":
        metadata["kind"] = "tv"
    else:
        metadata["year"] = 2017
    research.resolve = Mock(return_value=metadata)
    film = research.collect([ITEM], tmp_path / "research.json")[0]
    assert "tmdb_id" not in film and film["backdrops"] == []
    assert film["overview"] == PREMISE


def test_known_tmdb_id_conflict_is_rejected(provider, monkeypatch, tmp_path):
    monkeypatch.setenv("TMDB_READ_TOKEN", "configured")
    provider.grounded_text.return_value = grounded("https://www.themoviedb.org/movie/329865")
    research = Research()
    research.resolve = Mock(return_value={"title": "Arrival", "year": 2016, "kind": "movie", "id": 999,
                                         "cast": ["Amy Adams"], "backdrops": ["/wrong-film.jpg"]})
    film = research.collect([ITEM], tmp_path / "research.json")[0]
    assert film["tmdb_id"] == 329865 and film["backdrops"] == []


def test_aliens_film_cannot_be_accepted_as_a_television_series(public_provider, monkeypatch):
    lead = "Aliens is a 1986 science fiction action film written and directed by James Cameron."
    page = public_page("Aliens (1986 film)", lead)
    monkeypatch.setattr(gemini_research, "collect_public_sources", Mock(return_value=[page]))
    public_provider.generate_json.return_value.update(
        title="Aliens", year=1986, kind="tv", identity_evidence=lead, year_evidence=lead)
    with pytest.raises(GeminiUnavailable, match="medium"):
        gemini_research.research_film(FilmInput("Aliens", 1986), "tv")



@pytest.mark.parametrize("lead,incidental", [
    ("War Machine is a fictional character appearing in American comic books published by Marvel Comics.",
     "War Machine appears in the 2026 film Avengers: Doomsday."),
    ("War Machine is a military concept describing the machinery and organization used in warfare.",
     "War Machine is discussed in a 2026 documentary film."),
    ("War Machine is the stage name of an American professional wrestler.",
     "War Machine appeared in a 2026 film production.")])
def test_canonical_topic_with_incidental_films_is_not_a_movie(public_provider, monkeypatch, lead, incidental):
    page = public_page("War Machine", lead)
    page["text"] += "\n\n" + incidental
    monkeypatch.setattr(gemini_research, "collect_public_sources", Mock(return_value=[page]))
    public_provider.generate_json.return_value.update(
        title="War Machine", year=2026, kind="movie", identity_evidence=incidental, year_evidence=incidental)
    with pytest.raises(GeminiUnavailable, match="not_a_film_or_television_work"):
        gemini_research.research_film(FilmInput("War Machine", 2026), "movie")


@pytest.mark.parametrize("title,page_title,kind,lead", [
    ("War Machine", "War Machine (2026 film)", "movie",
     "War Machine is a 2026 U.S. military science fiction action film starring Alan Ritchson."),
    ("Mr. Robot", "Mr. Robot", "tv",
     "Mr. Robot is an American drama thriller television series created by Sam Esmail. It premiered in 2015."),
    ("Lioness", "Lioness (American TV series)", "tv",
     "Lioness (also known as Special Ops: Lioness) is an American spy thriller television series. It premiered in 2023."),
    ("Birdman or (The Unexpected Virtue of Ignorance)", "Birdman or (The Unexpected Virtue of Ignorance)", "movie",
     "Birdman or (The Unexpected Virtue of Ignorance) is a 2014 American black comedy drama film.")])
def test_work_opening_definition_handles_aliases_and_abbreviations(title, page_title, kind, lead):
    page = public_page(page_title, lead)
    assert gemini_research._work_definition(page, kind)



def test_superhero_and_comic_book_are_allowed_as_actual_film_genres():
    lead = "Iron Man is a 2008 American superhero film based on the Marvel Comics character of the same name."
    assert gemini_research._work_definition(public_page("Iron Man (2008 film)", lead), "movie")
    lead = "Example is a 2026 American comic book film directed by an independent filmmaker."
    assert gemini_research._work_definition(public_page("Example (2026 film)", lead), "movie")



def test_single_page_extract_avoids_real_mediawiki_batch_limit_and_disambiguation(monkeypatch):
    from movie_pipeline import public_research
    old = {"pageid": 1, "ns": 0, "title": "Havoc (2005 film)",
           "fullurl": "https://en.wikipedia.org/wiki/Havoc_(2005_film)"}
    disambiguation = {"pageid": 2, "ns": 0, "title": "Havoc", "pageprops": {"disambiguation": ""},
                      "fullurl": "https://en.wikipedia.org/wiki/Havoc",
                      "extract": "Havoc may refer to various unrelated people, places and films."}
    wanted = {"pageid": 3, "ns": 0, "title": "Havoc (2025 film)",
              "fullurl": "https://en.wikipedia.org/wiki/Havoc_(2025_film)"}
    lead = "Havoc is a 2025 action thriller film written and directed by Gareth Evans."
    premise = "A detective must navigate a dangerous criminal underworld to rescue a politician's estranged son."
    # Real whole-article batching returns a continuation and only the first
    # unsupported/disambiguation extract. Metadata-first retrieval must ignore it
    # and request the exact eligible page ID on its own.
    metadata = {"continue": {"excontinue": 1, "continue": "||info|pageprops"},
                "warnings": {"extracts": {"warnings": "exlimit lowered to 1"}},
                "query": {"pages": [old, disambiguation, wanted]}}
    get = Mock(side_effect=[
        _wiki_response({"query": {"search": [{"title": old["title"]}, {"title": wanted["title"]}]}}),
        _wiki_response(metadata),
        _wiki_response({"query": {"pages": [{**wanted, "extract": lead + "\n\n" + premise}]}})])
    monkeypatch.setattr(public_research.requests, "get", get)
    monkeypatch.setattr(public_research, "record_event", Mock())
    sources = public_research.collect_public_sources(FilmInput("Havoc", 2025), "movie")
    assert sources[0]["title"] == wanted["title"] and premise in sources[0]["text"]
    assert get.call_count == 3
    assert get.call_args_list[1].kwargs["params"]["prop"] == "info|pageprops"
    assert get.call_args_list[2].kwargs["params"]["prop"] == "extracts"
    assert get.call_args_list[2].kwargs["params"]["pageids"] == "3"
    assert "titles" not in get.call_args_list[2].kwargs["params"]


def test_good_public_context_survives_later_optional_extract_timeout(monkeypatch):
    from movie_pipeline import public_research
    import requests
    first = {"pageid": 1, "title": "Arrival", "ns": 0,
             "fullurl": "https://en.wikipedia.org/wiki/Arrival"}
    other = {"pageid": 2, "title": "Arrival (2017 film)", "ns": 0,
             "fullurl": "https://en.wikipedia.org/wiki/Arrival_(2017_film)"}
    text = ("Arrival is an American science fiction film directed by Denis Villeneuve.\n\n"
            + PREMISE + "\n\nThe film premiered in 2016 at an international film festival.")
    get = Mock(side_effect=[
        _wiki_response({"query": {"search": [{"title": other["title"]}]}}),
        _wiki_response({"query": {"pages": [first, other]}}),
        _wiki_response({"query": {"pages": [{**first, "extract": text}]}}),
        requests.Timeout("private request details must not be logged")])
    events = Mock()
    monkeypatch.setattr(public_research.requests, "get", get)
    monkeypatch.setattr(public_research, "record_event", events)
    sources = public_research.collect_public_sources(ITEM, "movie")
    assert len(sources) == 1 and sources[0]["text"] == text
    assert "private request details" not in str(events.call_args_list)
    assert "public_source_timeout" in str(events.call_args_list)


def test_public_http_status_is_diagnostic_without_request_or_response_leaks(monkeypatch):
    from movie_pipeline import public_research
    events = Mock()
    get = Mock(return_value=_wiki_response({"secret": "fake-private-token"}, status=403))
    monkeypatch.setattr(public_research.requests, "get", get)
    monkeypatch.setattr(public_research, "record_event", events)
    with pytest.raises(GeminiUnavailable, match="public_source_http_403") as error:
        public_research.collect_public_sources(ITEM, "movie")
    assert str(error.value) == "public_source_http_403"
    assert "fake-private-token" not in str(events.call_args_list)
    assert events.call_args.kwargs["details"] == {"http_status": 403}


@pytest.mark.parametrize("message,expected", [
    ("public_source_http_403", "public_source_http_403"),
    ("public_source_identity_not_found", "public_source_identity_not_found"),
    ("fake-private-key in https://private.example/request", "unavailable-or-unverified")])
def test_research_fallback_preserves_only_safe_fixed_reason_codes(provider, tmp_path, monkeypatch, message, expected):
    monkeypatch.setattr(research_module, "research_film", Mock(side_effect=GeminiUnavailable(message)))
    research = Research()
    research.resolve = Mock(return_value={"title": "Arrival", "year": 2016, "kind": "movie",
                                         "id": 329865, "overview": PREMISE})
    research.collect([ITEM], tmp_path / "research.json")
    calls = research_module.record_event.call_args_list
    fallback = next(call for call in calls if call.args[:3] == ("research", "gemini", "fallback"))
    assert fallback.kwargs["reason"] == expected
    assert "fake-private-key" not in str(calls)
