"""Expire only video-production checkpoints, at least 24 hours after completion.

Model, pip, setup-python, and all other cache namespaces are deliberately excluded.
This runs briefly on a schedule; it never keeps a runner waiting for the retention period.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import UTC, datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

RETENTION = timedelta(hours=24)
WORKFLOW_PATH = ".github/workflows/generate-video.yml"
RUN_KEY = re.compile(r"production-v[12]-([1-9]\d*)-([1-9]\d*)\Z")
# Earlier production checkpoints can lack an embedded run/attempt ID. This reserved
# namespace is the only legacy namespace we accept; malformed v2 keys stay untouched.
LEGACY_KEY = re.compile(r"production-v1-[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class GitHubAPIError(RuntimeError):
    def __init__(self, status: int | None = None):
        self.status = status
        super().__init__(f"GitHub API request failed (HTTP {status})." if status else "GitHub API request failed.")


class GitHubAPI:
    def __init__(self, repository: str, token: str, base_url: str = "https://api.github.com"):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("GITHUB_REPOSITORY must be owner/repository.")
        self.root = f"{base_url.rstrip('/')}/repos/{repository}"
        self.token = token

    def request(self, path: str, method: str = "GET", **params):
        url = self.root + path
        if params:
            url += "?" + urlencode(params)
        request = Request(url, method=method, headers={
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Top10Movies-production-cache-cleanup",
        })
        try:
            with urlopen(request, timeout=30) as response:
                body = response.read()
            return json.loads(body) if body else None
        except HTTPError as error:
            # Never print response bodies, tokens, or authentication headers.
            raise GitHubAPIError(error.code) from None
        except (URLError, TimeoutError, ValueError):
            raise GitHubAPIError() from None

    def paginated(self, path: str, field: str) -> list[dict]:
        rows = []
        for page in range(1, 1001):
            data = self.request(path, per_page=100, page=page)
            if not isinstance(data, dict) or not isinstance(data.get(field), list):
                raise GitHubAPIError()
            batch = data[field]
            rows.extend(batch)
            if len(batch) < 100:
                return rows
        # Incomplete listings must not be used to conclude that no production is active.
        raise GitHubAPIError()

    def caches(self) -> list[dict]:
        return self.paginated("/actions/caches", "actions_caches")

    def run(self, run_id: int) -> dict:
        return self.request(f"/actions/runs/{run_id}")

    def production_runs(self) -> list[dict]:
        return self.paginated("/actions/workflows/generate-video.yml/runs", "workflow_runs")

    def delete_cache(self, cache_id: int) -> bool:
        try:
            self.request(f"/actions/caches/{cache_id}", method="DELETE")
        except GitHubAPIError as error:
            if error.status == 404:  # Eviction or another cleanup already removed it.
                return False
            raise
        return True


def timestamp(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value)
    except ValueError:
        return None
    return result.astimezone(UTC) if result.tzinfo else None


def completed_at(run: dict) -> datetime | None:
    if not isinstance(run, dict) or run.get("status") != "completed":
        return None
    # A run's updated_at advances on completion and on later reruns. Using the most
    # recent value is conservative: a rerun gets a fresh 24-hour checkpoint window.
    if str(run.get("path", "")).split("@", 1)[0] != WORKFLOW_PATH:
        return None
    return timestamp(run.get("updated_at"))


def legacy_completion(runs: list[dict]) -> datetime | None:
    """Unattributed old keys wait for a full day after the latest video run."""
    if not runs:
        return None
    dates = [completed_at(run) for run in runs]
    if any(date is None for date in dates):
        return None  # Active/queued/waiting/unknown runs make legacy deletion unsafe.
    return max(dates)


def expire(api: GitHubAPI, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    # Finish pagination before any mutation: a list failure never gives partial cleanup.
    caches = api.caches()
    runs = {}
    legacy_checked = False
    legacy_completed = None
    deleted = 0
    skipped = 0
    seen = set()
    for cache in caches:
        cache_key = cache.get("key", "")
        if not isinstance(cache_key, str):
            continue
        match = RUN_KEY.fullmatch(cache_key)
        legacy = LEGACY_KEY.fullmatch(cache_key) if match is None else None
        if match is None and legacy is None:
            continue
        cache_id = cache.get("id")
        if not isinstance(cache_id, int) or isinstance(cache_id, bool) or cache_id <= 0 or cache_id in seen:
            continue
        seen.add(cache_id)
        created = timestamp(cache.get("created_at"))
        if created is None or now - created < RETENTION:
            skipped += 1
            continue
        if match:
            run_id, attempt = map(int, match.groups())
            if run_id not in runs:
                try:
                    runs[run_id] = api.run(run_id)
                except GitHubAPIError:
                    runs[run_id] = None
            run = runs[run_id]
            completed = completed_at(run)
            current_attempt = run.get("run_attempt") if isinstance(run, dict) else None
            if not isinstance(current_attempt, int) or attempt > current_attempt:
                completed = None
        else:
            if not legacy_checked:
                try:
                    legacy_completed = legacy_completion(api.production_runs())
                except GitHubAPIError:
                    legacy_completed = None
                legacy_checked = True
            completed = legacy_completed
        if completed is None or now - max(created, completed) < RETENTION:
            skipped += 1
            continue
        if api.delete_cache(cache_id):
            deleted += 1
            print(f"Deleted expired production checkpoint cache ID {cache_id}.")
    print(f"Production-cache cleanup: deleted {deleted}; retained {skipped} pending expiry or completion verification. Other cache namespaces were untouched.")
    return {"deleted": deleted, "retained": skipped}


def main() -> int:
    token = os.getenv("GH_TOKEN", "")
    repository = os.getenv("GITHUB_REPOSITORY", "")
    if not token or not repository:
        print("Production-cache cleanup needs GH_TOKEN and GITHUB_REPOSITORY.", file=sys.stderr)
        return 1
    try:
        expire(GitHubAPI(repository, token, os.getenv("GITHUB_API_URL", "https://api.github.com")))
    except (GitHubAPIError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
