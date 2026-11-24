"""Appended TMDB release dates inherit only their verified parent identity."""
from unittest.mock import Mock

import pytest

from movie_pipeline.core import FilmInput, PipelineError
from movie_pipeline.release_identity import tmdb_release_evidence, verified_release_years
from movie_pipeline.research import Research


IDENTIFIER = 1339713


def release_payload(**changes):
    return {
        "results": [{"iso_3166_1": "US", "release_dates": [
            {"release_date": "2025-09-21T00:00:00.000Z", "type": 1, "note": "Fantastic Fest"},
            {"release_date": "2026-05-15T00:00:00.000Z", "type": 3, "note": ""},
        ]}],
        **changes,
    }


def research_for(nested, *, parent_id=IDENTIFIER, standalone=None, later=False):
    research = object.__new__(Research)
    research.selected_kind = "movie"
    first = {"id": IDENTIFIER, "title": "Obsession", "original_title": "Obsession",
             "release_date": "2026-05-13"}
    rows = [first]
    if later:
        rows.append({**first, "id": 1436161, "release_date": "2025-03-28"})

    def get(endpoint, **params):
        if endpoint == "search/movie":
            return {"results": rows}
        if endpoint == f"movie/{IDENTIFIER}/release_dates":
            assert standalone is not None, "An exact-year match uses its appended dates."
            return standalone
        assert endpoint == f"movie/{IDENTIFIER}", f"Unexpected endpoint: {endpoint}"
        return {**first, "id": parent_id, "overview": "A wish takes a sinister turn.",
                "release_dates": nested, "credits": {},
                "videos": {"results": [{"site": "YouTube", "type": "Trailer",
                                       "key": "abcdefghijk"}]}}

    research.get = Mock(side_effect=get)
    research.wikidata_candidates = Mock(side_effect=AssertionError("TMDB already verifies this title."))
    return research


@pytest.mark.parametrize("own_id", [False, True])
def test_exact_year_match_preserves_appended_festival_dates_without_mutating_payload(own_id):
    nested = release_payload(**({"id": IDENTIFIER} if own_id else {}))
    research = research_for(nested)
    result = research.resolve(FilmInput("Obsession", 2026))
    assert verified_release_years(result) == {2025, 2026}
    assert any(row["kind"] == "festival" and row["year"] == 2025
               and row["identity"] == f"movie/{IDENTIFIER}"
               for row in result["release_year_evidence"])
    assert ("id" in nested) is own_id
    assert [call.args[0] for call in research.get.call_args_list] == [
        "search/movie", f"movie/{IDENTIFIER}"]


@pytest.mark.parametrize("nested_id", [None, True, "1339713", 1436161])
def test_explicit_invalid_or_conflicting_nested_identity_is_never_overwritten(nested_id):
    nested = release_payload(id=nested_id)
    result = research_for(nested).resolve(FilmInput("Obsession", 2026))
    assert verified_release_years(result) == {2026}
    assert not any(row["kind"] == "festival" for row in result["release_year_evidence"])
    assert nested["id"] == nested_id


@pytest.mark.parametrize("parent_id", [None, True, "1339713", 1436161])
def test_idless_nested_dates_cannot_bypass_the_parent_identity_check(parent_id):
    with pytest.raises(PipelineError, match="selected production"):
        research_for(release_payload(), parent_id=parent_id).resolve(FilmInput("Obsession", 2026))


def test_standalone_release_payload_still_requires_its_own_matching_identity():
    assert tmdb_release_evidence(release_payload(), "movie", IDENTIFIER) == []
    assert tmdb_release_evidence(release_payload(id=1436161), "movie", IDENTIFIER) == []
    assert {row["year"] for row in tmdb_release_evidence(
        release_payload(id=IDENTIFIER), "movie", IDENTIFIER)} == {2025, 2026}


def test_standalone_festival_proof_preserves_the_earlier_result_over_a_later_exact_year_short():
    research = research_for(release_payload(id=1436161),
                            standalone=release_payload(id=IDENTIFIER), later=True)
    result = research.resolve(FilmInput("Obsession", 2025))
    assert result["id"] == IDENTIFIER
    assert verified_release_years(result) == {2025, 2026}
    assert any(row["kind"] == "festival" and row["year"] == 2025
               and row["identity"] == f"movie/{IDENTIFIER}"
               for row in result["release_year_evidence"])
    assert [call.args[0] for call in research.get.call_args_list] == [
        "search/movie", f"movie/{IDENTIFIER}/release_dates", f"movie/{IDENTIFIER}"]
