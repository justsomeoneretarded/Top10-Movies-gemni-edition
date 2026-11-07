from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import requests

from movie_pipeline.availability import availability_line, enrich_availability
from movie_pipeline.core import save_json


def setup(monkeypatch, results=None, failure=None):
    session = MagicMock()
    session.__enter__.return_value = session
    session.get = Mock(side_effect=failure, return_value=Mock(status_code=200, json=Mock(return_value={"results": results or {}})))
    monkeypatch.setattr("movie_pipeline.availability.requests.Session", Mock(return_value=session))
    research = SimpleNamespace(session=SimpleNamespace(headers={"Authorization": "Bearer test-token"}))
    return session, research


def test_us_listings_preserve_offer_types_and_do_not_leak_auth(monkeypatch, tmp_path):
    session, research = setup(monkeypatch, {"US": {
        "flatrate": [{"provider_name": "Netflix"}, {"provider_name": "netflix"}, {"provider_name": "Hulu"}],
        "rent": [{"provider_name": "Amazon Video"}], "buy": [{"provider_name": "Apple TV"}],
        "ads": [{"provider_name": "Tubi"}], "link": "https://www.themoviedb.org/movie/1/watch?locale=US"},
        "GB": {"flatrate": [{"provider_name": "Wrong region"}]}})
    films = [{"id": 1, "kind": "movie", "title": "Example"}]
    result = enrich_availability(films, research, tmp_path)
    record = result[0]["watch_availability"]
    assert record["subscription"] == ["Netflix", "Hulu"]
    assert record["rent"] == ["Amazon Video"]
    assert record["buy"] == ["Apple TV"]
    assert record["ads"] == ["Tubi"]
    assert record["region"] == "US" and record["status"] == "found"
    assert "US" not in availability_line(record)
    assert availability_line(record) == "You can find it on Netflix and Hulu."
    assert "watch_availability" not in films[0]
    assert session.get.call_args.args[0] == "https://api.themoviedb.org/3/movie/1/watch/providers"
    assert session.get.call_args.kwargs["allow_redirects"] is False
    session.headers.update.assert_called_once_with(research.session.headers)
    assert "test-token" not in next(tmp_path.iterdir()).read_text()


def test_show_endpoint_and_empty_region_are_nonfatal_and_cached(monkeypatch, tmp_path):
    session, research = setup(monkeypatch, {"GB": {"flatrate": [{"provider_name": "Netflix"}]}})
    films = [{"id": 2, "kind": "tv", "title": "Show"}]
    first = enrich_availability(films, research, tmp_path)
    assert first[0]["watch_availability"]["status"] == "not-listed"
    assert availability_line(first[0]["watch_availability"]) == ""
    enrich_availability(films, research, tmp_path)
    assert session.get.call_count == 1
    assert "/tv/2/" in session.get.call_args.args[0]


@pytest.mark.parametrize("failure", [requests.Timeout("credential must not appear"), ValueError("bad JSON")])
def test_network_or_parse_failure_never_blocks_or_caches_false_absence(monkeypatch, tmp_path, capsys, failure):
    _session, research = setup(monkeypatch, failure=failure)
    result = enrich_availability([{"id": 1, "kind": "movie", "title": "Example"}], research, tmp_path)
    assert result[0]["watch_availability"]["status"] == "unavailable"
    assert availability_line(result[0]["watch_availability"]) == ""
    assert not list(tmp_path.iterdir())
    assert "credential" not in capsys.readouterr().out


def test_stale_and_corrupt_cache_are_refreshed(monkeypatch, tmp_path):
    session, research = setup(monkeypatch, {"US": {"flatrate": [{"provider_name": "Hulu"}]}})
    path = tmp_path/"movie-1-US-v1.json"
    save_json(path, {"checked_at": (datetime.now(UTC)-timedelta(days=2)).isoformat(), "region": "US", "status": "not-listed", **{k: [] for k in ("subscription", "free", "ads", "rent", "buy")}})
    result = enrich_availability([{"id": 1, "kind": "movie"}], research, tmp_path)
    assert result[0]["watch_availability"]["subscription"] == ["Hulu"]
    path.write_text("not-json")
    enrich_availability([{"id": 1, "kind": "movie"}], research, tmp_path)
    assert session.get.call_count == 2


@pytest.mark.parametrize("link", ["https://evil.example/watch", "http://www.themoviedb.org/watch", "https://user:pass@www.themoviedb.org/watch"])
def test_untrusted_provider_links_are_dropped(monkeypatch, tmp_path, link):
    _, research = setup(monkeypatch, {"US": {"link": link, "free": [{"provider_name": "Tubi"}]}})
    result = enrich_availability([{"id": 1, "kind": "movie"}], research, tmp_path)
    assert result[0]["watch_availability"]["link"] == ""


@pytest.mark.parametrize("record, expected", [
    ({"rent": ["Amazon Video"], "buy": ["Amazon Video"]}, "You can find it on Prime Video."),
    ({"subscription": ["Paramount Plus Premium", "Paramount+ Amazon Channel"], "rent": ["Apple TV"]},
     "You can find it on Paramount Plus and Apple TV."),
    ({"subscription": ["Sony Pictures Core Amazon Channel"]}, "You can find it on Sony Pictures Core."),
    ({"ads": ["Tubi"]}, "You can find it on Tubi."),
    ({"subscription": ["Peacock Premium Plus"], "free": ["Pluto TV"]}, "You can find it on Peacock and Pluto TV."),
    ({}, ""),
])
def test_spoken_lines_name_only_the_services(record, expected):
    assert availability_line(record) == expected


def test_provider_variants_are_deduplicated_in_short_spoken_line():
    assert availability_line({"subscription": ["Amazon Prime Video", "Amazon Prime Video with Ads"]}) == "You can find it on Prime Video."
    assert availability_line({"subscription": ["Netflix", "Netflix Standard with Ads"]}) == "You can find it on Netflix."
