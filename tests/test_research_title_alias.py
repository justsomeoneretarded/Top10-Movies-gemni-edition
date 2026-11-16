from unittest.mock import Mock

import pytest

from movie_pipeline.core import FilmInput, PipelineError
from movie_pipeline.research import Research


def record(identifier=250670, title="Maxton Hall - The World Between Us", year=2024, kind="tv"):
    name, original, date = (("title", "original_title", "release_date") if kind == "movie"
                            else ("name", "original_name", "first_air_date"))
    return {"id": identifier, name: title, original: title, date: f"{year}-05-09", "vote_count": 0}


def research_for(rows, *, kind="tv", aliases=None, other_rows=()):
    research = object.__new__(Research)
    research.selected_kind = kind
    aliases = aliases or {}

    def get(endpoint, **params):
        if endpoint == f"search/{kind}":
            return {"results": rows}
        if endpoint.startswith("search/"):
            return {"results": list(other_rows)}
        identifier = int(endpoint.split("/")[1])
        if endpoint.endswith("/alternative_titles"):
            value = aliases.get(identifier, [])
            if isinstance(value, Exception):
                raise value
            if isinstance(value, dict):
                return value
            return {"id": identifier, "titles" if kind == "movie" else "results":
                    [{"title": title} for title in value]}
        assert endpoint == f"{kind}/{identifier}", f"Unexpected endpoint: {endpoint}"
        row = next(row for row in rows if row["id"] == identifier)
        return {**row, "overview": "Two classmates navigate their different worlds.",
                "videos": {"results": [{"site": "YouTube", "type": "Trailer", "key": "example1234"}]},
                "credits": {}}

    research.get = Mock(side_effect=get)
    research.wikidata_candidates = Mock(return_value=[])
    return research


@pytest.mark.parametrize("kind", ["tv", "movie"])
@pytest.mark.parametrize("title,canonical", [
    ("Maxton Hall", "Maxton Hall - The World Between Us"),
    ("Harbor Academy", "Harbor Academy: The Space Between Us"),
    ("Harbor Academy", "Harbor Academy – The Space Between Us"),
])
def test_exact_provider_alias_accepts_subtitle_without_lowering_title_threshold(kind, title, canonical):
    row = record(title=canonical, kind=kind)
    research = research_for([row], kind=kind, aliases={row["id"]: [title]})
    item = FilmInput(title, 2024)
    assert research.title_score(item, row, kind) < 0.88
    result = research.resolve(item)
    assert result["id"] == row["id"]
    assert result["kind"] == kind
    assert result["year"] == 2024
    assert result["source"] == f"https://www.themoviedb.org/{kind}/{row['id']}"
    research.wikidata_candidates.assert_not_called()


@pytest.mark.parametrize("kind", ["tv", "movie"])
def test_verified_alias_preserves_first_eligible_api_result_even_with_zero_votes(kind):
    first = record(kind=kind)
    later = {**record(identifier=2, title="Maxton Hall", kind=kind), "vote_count": 50000}
    research = research_for([first, later], kind=kind, aliases={first["id"]: ["Maxton Hall"]})
    assert research.resolve(FilmInput("Maxton Hall", 2024))["id"] == first["id"]


@pytest.mark.parametrize("kind", ["tv", "movie"])
def test_exact_alias_cannot_accept_wrong_year_remake(kind):
    row = record(year=2023, kind=kind)
    research = research_for([row], kind=kind, aliases={row["id"]: ["Maxton Hall"]})
    with pytest.raises(PipelineError, match="Could not verify"):
        research.resolve(FilmInput("Maxton Hall", 2024))
    assert [call.args[0] for call in research.get.call_args_list] == [f"search/{kind}"]


@pytest.mark.parametrize("title,aliases", [
    ("Maxton Hall - Behind the Scenes", []),
    ("Maxton Hall - Behind the Scenes", ["Maxton Hall Behind the Scenes"]),
    ("Maxton Hall Reunion", ["Maxton Hall"]),
    ("Maxton Hallway - The World Between Us", ["Maxton Hall"]),
])
def test_shared_prefix_without_exact_delimited_provider_alias_is_rejected(title, aliases):
    row = record(title=title)
    research = research_for([row], aliases={row["id"]: aliases})
    with pytest.raises(PipelineError, match="Could not verify"):
        research.resolve(FilmInput("Maxton Hall", 2024))


@pytest.mark.parametrize("payload", [
    {"id": 1, "results": [{"title": "Maxton Hall"}]},
    {"id": True, "results": [{"title": "Maxton Hall"}]},
    {"results": [{"title": "Maxton Hall"}]},
    {"id": 250670, "results": "Maxton Hall"},
    {"id": 250670, "results": [None, {"title": None}]},
])
def test_alias_metadata_must_belong_to_the_same_provider_record(payload):
    row = record()
    research = research_for([row], aliases={row["id"]: payload})
    with pytest.raises(PipelineError, match="Could not verify"):
        research.resolve(FilmInput("Maxton Hall", 2024))


def test_optional_alias_failure_keeps_later_direct_candidate_available():
    first = record()
    later = record(identifier=2, title="Maxton Hall")
    research = research_for([first, later], aliases={first["id"]: PipelineError("Unavailable")})
    assert research.resolve(FilmInput("Maxton Hall", 2024))["id"] == later["id"]


def test_alias_lookup_requests_are_bounded():
    rows = [record(identifier=identifier) for identifier in range(1, 8)]
    research = research_for(rows)
    with pytest.raises(PipelineError, match="Could not verify"):
        research.resolve(FilmInput("Maxton Hall", 2024))
    calls = [call.args[0] for call in research.get.call_args_list]
    assert sum(endpoint.endswith("/alternative_titles") for endpoint in calls) == 5


def test_selected_medium_never_uses_other_medium_alias():
    movie = record(kind="movie")
    research = research_for([], other_rows=[movie], aliases={movie["id"]: ["Maxton Hall"]})
    with pytest.raises(PipelineError, match="Could not verify"):
        research.resolve(FilmInput("Maxton Hall", 2024))
    assert [call.args[0] for call in research.get.call_args_list] == ["search/tv"]


def wikidata_lookup(monkeypatch, *, names=("Maxton Hall", "Maxton Hall - The World Between Us"),
                    year=2024, deprecated=False, group="tv_results", canonical=None):
    research = object.__new__(Research)
    row = record(title=canonical or "Maxton Hall - The World Between Us")
    research.get = Mock(return_value={group: [row]})
    entity = {
        "id": "QTEST", "labels": {"en": {"value": names[0]}},
        "aliases": {"en": [{"value": name} for name in names[1:]]},
        "claims": {
            "P577": [{"mainsnak": {"datavalue": {"value":
                       {"time": f"+{year}-05-09T00:00:00Z", "precision": 11}}}}],
            "P345": [{"rank": "deprecated" if deprecated else "normal",
                       "mainsnak": {"datavalue": {"value": "tt27792190"}}}],
        },
    }
    payloads = [{"search": [{"id": "QTEST"}]}, {"entities": {"QTEST": entity}}]
    getter = Mock(side_effect=[Mock(json=Mock(return_value=payload)) for payload in payloads])
    monkeypatch.setattr("movie_pipeline.research.requests.get", getter)
    return research, getter


def test_wikidata_crosswalk_requires_both_exact_names_on_same_dated_imdb_identity(monkeypatch):
    research, getter = wikidata_lookup(monkeypatch)
    candidates = research.wikidata_candidates(FilmInput("Maxton Hall", 2024))
    assert len(candidates) == 1
    score, kind, row = candidates[0]
    assert score == 1.0
    assert kind == "tv"
    assert row["id"] == 250670
    assert row["_imdb_id"] == "tt27792190"
    assert row["_identity_source"] == "https://www.wikidata.org/wiki/QTEST"
    assert research.get.call_args.args[0] == "find/tt27792190"
    for call in getter.call_args_list:
        assert "Authorization" not in call.kwargs["headers"]


@pytest.mark.parametrize("kwargs", [
    {"names": ("Maxton Hall",)},
    {"names": ("Maxton Hall - The World Between Us",)},
    {"canonical": "Maxton Hall - Behind the Scenes"},
    {"year": 2023},
    {"deprecated": True},
    {"group": "tv_episode_results"},
    {"group": "tv_season_results"},
])
def test_wikidata_alias_crosswalk_rejects_unverified_name_year_or_episode(monkeypatch, kwargs):
    research, _ = wikidata_lookup(monkeypatch, **kwargs)
    assert research.wikidata_candidates(FilmInput("Maxton Hall", 2024)) == []
