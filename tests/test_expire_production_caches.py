from datetime import UTC, datetime, timedelta
from io import BytesIO
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest
import yaml

from scripts.expire_production_caches import (
    WORKFLOW_PATH,
    GitHubAPI,
    GitHubAPIError,
    expire,
    main,
)

NOW = datetime(2026, 10, 20, 12, tzinfo=UTC)


def stamp(hours_ago):
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def cache(identifier=1, key="production-v2-100-1", age=30):
    return {"id": identifier, "key": key, "created_at": stamp(age)}


def run(age=30, status="completed", attempt=1, **overrides):
    return {"status": status, "run_attempt": attempt, "updated_at": stamp(age),
            "path": WORKFLOW_PATH, **overrides}


def api(caches, runs=None, legacy_runs=None):
    result = Mock(spec=GitHubAPI)
    result.caches.return_value = caches
    result.run.side_effect = lambda run_id: (runs or {100: run()})[run_id]
    result.production_runs.return_value = legacy_runs or [run()]
    result.delete_cache.return_value = True
    return result


def test_only_production_namespaces_are_deleted():
    source = api([
        cache(1), cache(2, "production-v1-101-1"),
        cache(3, "models-cpu-v3-thumbnails-Linux-turbo-qwen3:4b"),
        cache(4, "setup-python-Linux-x64-pip"), cache(5, "pip-dependencies"),
        cache(6, "another-production-v2-100-1"), cache(7, "production-v20-100-1"),
        cache(8, "production-v2-"), cache(9, "production-v2-not-a-run"),
        cache(10, "production-v1-"), cache(11, "production-v1-unsafe/key"),
    ], {100: run(), 101: run()})
    assert expire(source, NOW)["deleted"] == 2
    assert [call.args[0] for call in source.delete_cache.call_args_list] == [1, 2]


@pytest.mark.parametrize("completion_age,expected", [(23 + 59/60, 0), (24, 1), (25, 1)])
def test_expiry_is_24_hours_after_completion(completion_age, expected):
    source = api([cache(age=40)], {100: run(completion_age)})
    assert expire(source, NOW)["deleted"] == expected


def test_recently_saved_cache_kept_even_when_owning_run_is_old():
    source = api([cache(age=1)], {100: run(40)})
    assert expire(source, NOW)["deleted"] == 0
    source.run.assert_not_called()


@pytest.mark.parametrize("status", ["in_progress", "queued", "waiting", "requested", "pending", "unknown"])
def test_active_or_unknown_runs_keep_old_checkpoints(status):
    source = api([cache(age=50)], {100: run(50, status=status)})
    assert expire(source, NOW)["deleted"] == 0


@pytest.mark.parametrize("status", [403, 404, 500, None])
def test_unavailable_run_metadata_is_retained(status):
    source = api([cache()])
    source.run.side_effect = GitHubAPIError(status)
    assert expire(source, NOW)["deleted"] == 0
    source.delete_cache.assert_not_called()


def test_rerun_completion_restarts_retention_for_old_attempt():
    source = api([cache(age=50)], {100: run(1, attempt=2)})
    assert expire(source, NOW)["deleted"] == 0


def test_cache_for_future_unknown_attempt_is_retained():
    source = api([cache(key="production-v2-100-3")], {100: run(40, attempt=2)})
    assert expire(source, NOW)["deleted"] == 0


def test_failed_and_cancelled_completed_runs_can_expire():
    source = api([cache(1), cache(2, "production-v2-101-1")],
                 {100: run(conclusion="failure"), 101: run(conclusion="cancelled")})
    assert expire(source, NOW)["deleted"] == 2


@pytest.mark.parametrize("metadata", [
    run(path=".github/workflows/other-workflow.yml"),
    run(updated_at="not-a-date"), run(updated_at=None),
    run(updated_at="2026-10-18T12:00:00"), run(run_attempt=None),
])
def test_unverifiable_completion_is_retained(metadata):
    source = api([cache()], {100: metadata})
    assert expire(source, NOW)["deleted"] == 0


def test_legacy_keys_wait_for_day_since_latest_completed_video():
    source = api([cache(key="production-v1-legacy-checkpoint", age=80)],
                 legacy_runs=[run(70), run(25)])
    assert expire(source, NOW)["deleted"] == 1
    source.run.assert_not_called()


@pytest.mark.parametrize("legacy_runs", [[run(50), run(1)], [run(50), run(50, "in_progress")],
                                          [run(50), run(50, "queued")], []])
def test_legacy_keys_are_not_removed_before_latest_video_is_safe(legacy_runs):
    source = api([cache(key="production-v1-legacy-checkpoint", age=80)])
    source.production_runs.return_value = legacy_runs
    assert expire(source, NOW)["deleted"] == 0


def test_legacy_status_unavailable_is_retained():
    source = api([cache(key="production-v1-legacy-checkpoint")])
    source.production_runs.side_effect = GitHubAPIError(403)
    assert expire(source, NOW)["deleted"] == 0


def test_duplicate_cache_rows_are_deleted_once_and_run_lookups_are_reused():
    source = api([cache(), cache(), cache(2, "production-v2-100-2")], {100: run(attempt=2)})
    assert expire(source, NOW)["deleted"] == 2
    source.run.assert_called_once_with(100)
    assert [call.args[0] for call in source.delete_cache.call_args_list] == [1, 2]


def test_paginated_cache_listing_is_complete_before_any_deletion():
    source = GitHubAPI("owner/repo", "token")
    first_page = [cache(n + 1) for n in range(100)]
    source.request = Mock(side_effect=[{"actions_caches": first_page}, GitHubAPIError(500)])
    source.delete_cache = Mock()
    with pytest.raises(GitHubAPIError):
        expire(source, NOW)
    source.delete_cache.assert_not_called()
    assert source.request.call_args_list[1].kwargs == {"per_page": 100, "page": 2}


def test_cache_and_run_listings_follow_pagination():
    source = GitHubAPI("owner/repo", "token")
    source.request = Mock(side_effect=[
        {"actions_caches": [cache(n + 1) for n in range(100)]}, {"actions_caches": [cache(101)]},
        {"workflow_runs": [run() for _ in range(100)]}, {"workflow_runs": [run(50)]},
    ])
    assert len(source.caches()) == 101
    assert len(source.production_runs()) == 101
    assert [call.kwargs["page"] for call in source.request.call_args_list] == [1, 2, 1, 2]


def test_404_during_delete_is_a_harmless_race():
    source = GitHubAPI("owner/repo", "token")
    source.request = Mock(side_effect=GitHubAPIError(404))
    assert source.delete_cache(12) is False
    source.request.assert_called_once_with("/actions/caches/12", method="DELETE")


def test_other_delete_failures_remain_visible():
    source = GitHubAPI("owner/repo", "token")
    source.request = Mock(side_effect=GitHubAPIError(403))
    with pytest.raises(GitHubAPIError):
        source.delete_cache(12)


def test_http_api_uses_scoped_endpoint_and_does_not_report_credentials(monkeypatch):
    response = Mock()
    response.__enter__ = Mock(return_value=BytesIO(b'{"actions_caches": []}'))
    response.__exit__ = Mock(return_value=False)
    opener = Mock(return_value=response)
    monkeypatch.setattr("scripts.expire_production_caches.urlopen", opener)
    source = GitHubAPI("owner/repo", "private-token")
    assert source.caches() == []
    request = opener.call_args.args[0]
    assert request.full_url == "https://api.github.com/repos/owner/repo/actions/caches?per_page=100&page=1"
    assert request.headers["Authorization"] == "Bearer private-token"
    opener.side_effect = HTTPError(request.full_url, 403, "private-token", {}, None)
    with pytest.raises(GitHubAPIError) as error:
        source.caches()
    assert "private-token" not in str(error.value)


def test_network_failure_is_sanitized(monkeypatch):
    monkeypatch.setattr("scripts.expire_production_caches.urlopen", Mock(side_effect=URLError("private-token")))
    with pytest.raises(GitHubAPIError) as error:
        GitHubAPI("owner/repo", "private-token").caches()
    assert "private-token" not in str(error.value)


def test_main_requires_workflow_credentials_without_printing_them(monkeypatch, capsys):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert main() == 1
    assert "GH_TOKEN" in capsys.readouterr().err


def test_cleanup_workflow_is_scheduled_and_write_permission_is_isolated():
    from pathlib import Path
    workflow = yaml.load(Path(".github/workflows/cleanup-production-caches.yml").read_text(), Loader=yaml.BaseLoader)
    assert workflow["on"]["schedule"] == [{"cron": "23 * * * *"}]
    assert "workflow_dispatch" in workflow["on"]
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["cleanup"]
    assert job["permissions"] == {"actions": "write", "contents": "read"}
    assert job["timeout-minutes"] == "15"
    script_step = job["steps"][-1]
    assert script_step["env"] == {"GH_TOKEN": "${{ github.token }}"}
    assert script_step["run"] == "python3 -m scripts.expire_production_caches"
    assert "models" not in workflow["concurrency"]["group"]
