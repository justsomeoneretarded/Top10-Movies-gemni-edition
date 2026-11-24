"""Regression coverage for broad, bounded discovery and same-production identity."""
from pathlib import Path

import pytest

from movie_pipeline import editorial_policy, footage
from movie_pipeline.media_discovery import DiscoveryStats, fair_candidates
from movie_pipeline.core import PipelineError
from movie_pipeline.secondary_footage import secondary_candidate_ok


FILM = {
    "id": 42, "tmdb_id": 42, "title": "Night Orchard", "year": 2025, "kind": "movie",
    "cast": ["Ada Morton"], "directors": [], "genres": ["Horror"],
    "homepage": "https://studio.example/films/night-orchard", "imdb_id": "tt12345678",
    "trailers": [],
}
IDS = [f"{letter}" * 11 for letter in "abcdefghijklmno"]


def video(title="Night Orchard Official Trailer", **changes):
    return {"title": title, "duration": 120, "description": "Ada Morton stars.",
            "formats": [{"width": 1920, "height": 1080, "vcodec": "h264"}], **changes}


@pytest.mark.parametrize("primary", [False, True])
def test_movie_queries_include_dated_and_year_free_trailer_and_own_genre_scenes(primary, monkeypatch):
    monkeypatch.setenv("VIDEO_GENRE", "comedy")
    queries = editorial_policy.search_queries(FILM, primary=primary)
    assert len(queries) <= 7
    assert any("2025" in query and "trailer" in query for query in queries)
    assert any("2025" not in query and "trailer" in query for query in queries)
    assert any("2025" not in query and "scene" in query for query in queries)
    assert any("scary scene" in query or "suspense scene" in query for query in queries)
    assert not any("funny" in query or "comedy" in query for query in queries)
    assert all("Ada Morton" not in query for query in queries)


def test_query_genres_are_independent_and_a_director_is_optional_context():
    comedy = {**FILM, "title": "Sunday Errands", "genres": [{"name": "Comedy"}],
              "directors": ["Dana Parks"]}
    comedy_queries = editorial_policy.search_queries(comedy, primary=True)
    horror_queries = editorial_policy.search_queries(FILM, primary=True)
    assert any("funny scene" in query for query in comedy_queries)
    assert not any("scary scene" in query for query in comedy_queries)
    assert not any("funny scene" in query for query in horror_queries)
    assert any("Dana Parks" not in query for query in comedy_queries)
    assert all(query.startswith("Sunday Errands") for query in comedy_queries)


@pytest.mark.parametrize("primary", [False, True])
def test_tv_queries_keep_requested_season_in_year_free_families(primary):
    show = {**FILM, "kind": "tv", "year": 2014, "requested_season": {"season_number": 3}}
    queries = editorial_policy.search_queries(show, primary=primary)
    assert any("2014" not in query and "season 3" in query for query in queries)
    assert all("season 3" in query for query in queries)


@pytest.mark.parametrize("accept", [footage.candidate_ok, secondary_candidate_ok])
@pytest.mark.parametrize("changes", [
    {"description": "Ada Morton stars."},
    {"description": "", "channel": "Orchard Network", "tags": []},
    {"description": "Written by Dana Vale."},
    {"description": "Official production: https://studio.example/films/night-orchard"},
    {"description": "IMDb: https://www.imdb.com/title/tt12345678/"},
])
def test_missing_director_does_not_require_a_director_if_identity_is_corroborated(accept, changes):
    film = {**FILM, "networks": ["Orchard Network"], "creators": ["Dana Vale"]}
    assert accept(video(**changes), film)


@pytest.mark.parametrize("accept", [footage.candidate_ok, secondary_candidate_ok])
@pytest.mark.parametrize("description", [
    "Another cast. https://studio.example/films/different-orchard",
    "Another cast. https://studio.example/",
    "Another cast. https://www.imdb.com/title/tt99999999/",
])
def test_shared_studio_host_or_another_imdb_production_cannot_prove_identity(accept, description):
    assert not accept(video(description=description), FILM)


@pytest.mark.parametrize("accept", [footage.candidate_ok, secondary_candidate_ok])
@pytest.mark.parametrize("title", [
    "Night Orchard 2025 AI-Generated Trailer",
    "Night Orchard 2025 Concept Trailer",
    "Night Orchard 2025 Trailer Reaction",
    "Night Orchard 2025 Full Movie Scene",
    "Different Orchard 2025 Official Trailer",
    "Night Orchard 1991 Official Trailer",
])
def test_wrong_production_and_non_footage_rejections_remain(accept, title):
    assert not accept(video(title), FILM)


@pytest.mark.parametrize("accept", [footage.candidate_ok, secondary_candidate_ok])
def test_tv_requested_season_is_checked_even_for_curated_video(accept):
    show = {**FILM, "kind": "tv", "requested_season": 3}
    assert accept(video("Night Orchard Season 3 Official Trailer"), show)
    assert not accept(video("Night Orchard Season 2 Official Trailer"), show, curated=True)



@pytest.mark.parametrize("accept", [footage.candidate_ok, secondary_candidate_ok])
def test_source_backed_release_years_accept_the_same_production_but_not_a_raw_year_list(accept):
    evidence = {
        "year": 2026, "source": "https://api.themoviedb.org/3/movie/42/release_dates",
        "kind": "theatrical", "identity": "movie/42",
    }
    candidate = video("Night Orchard 2026 Official Trailer")
    verified = {**FILM, "release_year_evidence": [evidence], "release_identity_version": 1}
    assert accept(candidate, verified)
    assert not accept(candidate, {**FILM, "release_years": [2025, 2026]})
    assert not accept(candidate, {**verified, "release_year_evidence": [
        {**evidence, "identity": "movie/999", "source": "https://api.themoviedb.org/3/movie/999/release_dates"}]})


def test_fair_candidates_reserve_first_unique_result_for_later_families_before_seconds():
    url = lambda letter: f"https://www.youtube.com/watch?v={letter * 11}"
    calls = []
    families = {
        "dated": [url("a"), url("b"), url("c"), url("d")],
        "year-free": [url("a"), url("e"), url("f"), url("g")],
        "genre": [url("h"), url("i"), url("j"), url("k")],
    }

    def search(query):
        calls.append(query)
        return families[query]

    results = list(fair_candidates([url("x"), url("y")], list(families), search, limit=6))
    assert results[:4] == [(url("x"), True), (url("a"), False), (url("e"), False), (url("h"), False)]
    assert len(results) == 6
    assert len({candidate for candidate, _ in results}) == 6
    assert calls == list(families)


def test_fair_candidates_excluded_sources_do_not_consume_metadata_allowance():
    urls = [f"https://www.youtube.com/watch?v={identifier}" for identifier in IDS[:4]]
    results = list(fair_candidates([urls[0]], ["scene"], lambda query: urls[:3],
                                  limit=2, excluded=[urls[0]]))
    assert results == [(urls[1], False), (urls[2], False)]


def test_diagnostics_count_rejections_and_exception_types_without_raw_exception_text(capsys):
    stats = DiscoveryStats()
    stats.seen += 2
    stats.reject("wrong_year")
    stats.download_attempts += 1
    stats.error(ValueError("https://user:secret@proxy.invalid/file?token=secret"))
    summary = stats.summary()
    assert "seen=2" in summary
    assert "rejected=1" in summary and "wrong_year" in summary
    assert "download_attempts=1" in summary
    assert "ValueError" in summary
    output = summary + capsys.readouterr().out
    assert "secret" not in output and "proxy.invalid" not in output


def _fake_downloads(monkeypatch, *, search_results=None, failures=True, reject=False):
    """Exercise the public route loop without a network or a real media file."""
    import yt_dlp
    downloads, inspected, searches = [], [], []
    catalog = {identifier: video("Night Orchard 2025 Official Trailer") for identifier in IDS}
    if reject:
        catalog = {identifier: video("Night Orchard 2025 AI-Generated Concept Trailer") for identifier in IDS}

    class FakeYDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, **kwargs):
            if url.startswith("ytsearch"):
                query = url.split(":", 1)[1]
                searches.append(query)
                identifiers = search_results(query) if search_results else IDS[:4]
                return {"entries": [{"id": identifier} for identifier in identifiers]}
            identifier = url[-11:]
            inspected.append(identifier)
            return catalog[identifier]

        def download(self, urls):
            identifier = urls[0][-11:]
            downloads.append(identifier)
            if failures:
                raise yt_dlp.utils.DownloadError(
                    "temporary failure at https://user:secret@proxy.invalid/private?token=secret")
            Path(self.options["outtmpl"].replace("%(ext)s", "mp4")).write_text(identifier)

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(footage, "duration", lambda path: 120)
    monkeypatch.setattr(footage, "probe", lambda path: {
        "streams": [{"codec_type": "video", "width": 1920, "height": 1080}]})
    monkeypatch.setattr(footage, "require_disk", lambda path: None)
    monkeypatch.setattr(footage.download_access, "routes", lambda: ["direct", "fallback"])
    monkeypatch.setattr(footage.download_access, "options", lambda options, route: options)
    monkeypatch.setattr(footage.download_access, "record_route_success", lambda route: None)
    monkeypatch.setenv("VIDEO_RESOLUTION", "2160")
    monkeypatch.setenv("STILLS_FALLBACK", "false")
    return downloads, inspected, searches


def test_failed_downloads_consume_the_title_budget_across_routes_and_hide_sensitive_errors(monkeypatch, tmp_path, capsys):
    downloads, inspected, searches = _fake_downloads(monkeypatch)
    film = {**FILM, "trailers": [{"key": identifier} for identifier in IDS[:4]]}
    with pytest.raises(PipelineError) as raised:
        footage.download(film, tmp_path)
    assert len(downloads) == footage.MAX_QUALITY_DOWNLOADS == 3
    assert len(inspected) <= footage.MAX_CANDIDATES * 2
    diagnostics = str(raised.value) + capsys.readouterr().out
    for sensitive in ("secret", "proxy.invalid", "token=", "user:"):
        assert sensitive not in diagnostics
    assert "bot check" not in str(raised.value).lower()
    assert not list(tmp_path.glob("*candidate*"))


def test_metadata_rejections_report_no_download_attempts_or_invented_bot_failure(monkeypatch, tmp_path, capsys):
    downloads, inspected, searches = _fake_downloads(monkeypatch, reject=True)
    with pytest.raises(PipelineError) as raised:
        footage.download(FILM, tmp_path)
    assert inspected
    assert downloads == []
    diagnostics = str(raised.value) + capsys.readouterr().out
    assert "rejected" in diagnostics.lower()
    assert "download" in diagnostics.lower() and "0" in diagnostics
    assert "bot" not in str(raised.value).lower()
    assert "youtube access restrictions" not in str(raised.value).lower()


def test_primary_metadata_budget_reaches_later_year_free_and_genre_query_families(monkeypatch, tmp_path):
    queries = ["dated trailer", "year-free trailer", "generic scene", "year-free scene",
               "scary scene", "suspense scene", "premise scene"]
    first_ids = dict(zip(queries, IDS[:7]))
    monkeypatch.setattr(editorial_policy, "search_queries", lambda film, **kwargs: queries)
    downloads, inspected, searches = _fake_downloads(
        monkeypatch, reject=True, search_results=lambda query: [first_ids[query], *IDS[7:10]])
    with pytest.raises(PipelineError):
        footage.download(FILM, tmp_path)
    assert downloads == []
    assert set(first_ids.values()).issubset(inspected)
    assert set(queries).issubset(searches)
    assert len(inspected) <= footage.MAX_CANDIDATES * 2
    assert all(inspected.count(identifier) <= 2 for identifier in set(inspected))


@pytest.mark.parametrize("accept", [footage.candidate_ok, secondary_candidate_ok])
@pytest.mark.parametrize("length", [12, 361, float("nan"), float("inf"), True, "120"])
def test_duration_and_live_checks_still_apply_to_curated_footage(accept, length):
    assert not accept(video(duration=length), FILM, curated=True)
    assert not accept(video(is_live=True), FILM, curated=True)
