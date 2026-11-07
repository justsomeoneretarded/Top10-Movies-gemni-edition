"""Bounded keyless public references, distinct from Google Search grounding."""
from __future__ import annotations

import re
import time
from urllib.parse import urlsplit

import requests

from .core import normalized
from .gemini import GeminiUnavailable, record_event

_API = "https://en.wikipedia.org/w/api.php"
_HEADERS = {"User-Agent": "Top10MoviesGemini/1.0 (https://github.com/justsomeoneretarded/Top10-Movies-gemni-edition)"}


def _get(deadline: float, **params) -> dict:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        record_event("public_source", "wikipedia", "unavailable", "public_source_time_budget")
        raise GeminiUnavailable("public_source_time_budget")
    try:
        response = requests.get(_API, params={**params, "format": "json", "formatversion": 2},
                                headers=_HEADERS, timeout=(min(4, remaining), min(12, remaining)),
                                allow_redirects=False)
        status = response.status_code
        if type(status) is int and status != 200:
            code = f"public_source_http_{status}" if 100 <= status <= 599 else "public_source_http_error"
            record_event("public_source", "wikipedia", "request_failed", code,
                         details={"http_status": status})
            raise GeminiUnavailable(code)
        response.raise_for_status()
        data = response.json()
    except requests.Timeout:
        record_event("public_source", "wikipedia", "request_failed", "public_source_timeout")
        raise GeminiUnavailable("public_source_timeout") from None
    except (requests.RequestException, ValueError, TypeError) as error:
        if isinstance(error, GeminiUnavailable):
            raise
        record_event("public_source", "wikipedia", "request_failed", "public_source_unavailable")
        raise GeminiUnavailable("public_source_unavailable") from None
    if not isinstance(data, dict) or data.get("error"):
        record_event("public_source", "wikipedia", "request_failed", "public_source_invalid_response")
        raise GeminiUnavailable("public_source_invalid_response")
    return data


def title_key(title: str) -> str:
    """Remove only a reference page's film/television disambiguation suffix."""
    title = re.sub(r"\s*\([^()]*(?:film|series|miniseries|television show)\)\s*$", "", title, flags=re.I)
    return normalized(title)


def _page_url(value) -> str:
    if not isinstance(value, str) or len(value) > 4096:
        return ""
    try:
        parsed = urlsplit(value)
        if (parsed.scheme == "https" and parsed.hostname == "en.wikipedia.org"
                and not parsed.username and not parsed.password and parsed.port in (None, 443)
                and parsed.path.startswith("/wiki/") and not parsed.query):
            return value
    except ValueError:
        pass
    return ""


def _pages(data: dict) -> tuple[dict, list]:
    query = data.get("query", {})
    if not isinstance(query, dict):
        raise GeminiUnavailable("public_source_invalid_response")
    pages = query.get("pages", [])
    if isinstance(pages, dict):
        pages = list(pages.values())
    if not isinstance(pages, list):
        raise GeminiUnavailable("public_source_invalid_response")
    return query, pages


def _rank(page: dict, year: int, kind: str) -> int | None:
    title = page["title"]
    suffix = re.search(r"\((?:(\d{4})\s+)?([^()]*(?:film|series|miniseries|television show))\)$", title, re.I)
    description = str(page.get("pageprops", {}).get("wikibase-shortdesc", ""))
    label = normalized((suffix[2] if suffix else "") + " " + description)
    television = bool(re.search(r"\b(?:television|tv|miniseries)\b", label))
    movie = bool(re.search(r"\bfilm\b", label))
    if (kind == "movie" and television and not movie) or (kind == "tv" and movie and not television):
        return None
    declared = int(suffix[1]) if suffix and suffix[1] else None
    if declared is not None:
        if declared == year:
            return 0
        return 2 if abs(declared-year) <= 2 else None
    if re.search(r"\b" + str(year) + r"\b", description) and (movie if kind == "movie" else television):
        return 0
    return 1


def _confident(page: dict, text: str, item, kind: str) -> bool:
    if _rank(page, item.year, kind) == 0:
        return True
    lead = normalized(re.split(r"\n\s*\n|^==", text, maxsplit=1, flags=re.M)[0][:1600])
    medium = r"\b(?:film|movie)\b" if kind == "movie" else r"\b(?:television series|tv series|miniseries)\b"
    return bool(re.search(r"\b" + str(item.year) + r"\b.{0,140}" + medium, lead))


def collect_public_sources(item, kind: str) -> list[dict]:
    """Retrieve metadata, then one full extract per eligible page, within 40s."""
    if kind not in {"movie", "tv"}:
        raise GeminiUnavailable("public_source_invalid_medium")
    deadline = time.monotonic() + 40
    medium = "television series" if kind == "tv" else "film"
    search = _get(deadline, action="query", list="search",
                  srsearch=f'"{item.title}" {item.year} {medium}', srlimit=4, srnamespace=0)
    search_query = search.get("query", {})
    if not isinstance(search_query, dict):
        raise GeminiUnavailable("public_source_invalid_response")
    hits = search_query.get("search", [])
    names = [item.title]
    for hit in hits if isinstance(hits, list) else []:
        name = hit.get("title") if isinstance(hit, dict) else None
        if isinstance(name, str) and name.strip() and name not in names:
            names.append(name)
    # Whole-article extracts support only one page per request. Do not ask for
    # batched extracts: the API can return only a disambiguation page and silently
    # omit all wanted film extracts behind a continuation token.
    result = _get(deadline, action="query", prop="info|pageprops", inprop="url",
                  redirects=1, titles="|".join(names[:5]))
    query, pages = _pages(result)
    edges = []
    for field in ("normalized", "redirects"):
        for row in query.get(field, []) if isinstance(query.get(field, []), list) else []:
            if isinstance(row, dict) and isinstance(row.get("from"), str) and isinstance(row.get("to"), str):
                edges.append((row["from"], row["to"]))
    candidates = []
    for page in pages[:5]:
        if not isinstance(page, dict):
            continue
        props = page.get("pageprops", {})
        if (not isinstance(props, dict) or page.get("missing") or page.get("invalid")
                or page.get("ns", 0) != 0 or "disambiguation" in props
                or type(page.get("pageid")) is not int or page["pageid"] <= 0):
            continue
        title = page.get("title")
        url = _page_url(page.get("fullurl") or page.get("canonicalurl"))
        if not isinstance(title, str) or not url:
            continue
        aliases = {title}
        for _ in range(5):
            aliases.update(before for before, after in edges if after in aliases)
        if title_key(item.title) not in {title_key(alias) for alias in aliases}:
            continue
        rank = _rank(page, item.year, kind)
        if rank is not None:
            candidates.append((rank, page, url, sorted(aliases)))
    candidates.sort(key=lambda row: row[0])
    sources, last_error = [], None
    for _rank_value, page, url, aliases in candidates[:5]:
        try:
            extracted = _get(deadline, action="query", prop="extracts", explaintext=1,
                             exsectionformat="wiki", pageids=str(page["pageid"]))
            _query, extracts = _pages(extracted)
        except GeminiUnavailable as error:
            last_error = error
            if sources or str(error) == "public_source_time_budget":
                break
            continue
        text = next((row.get("extract") for row in extracts
                     if isinstance(row, dict) and row.get("pageid") == page["pageid"]), None)
        if (not isinstance(text, str) or len(text.strip()) < 80
                or re.search(r"<(?:!doctype|html|body|script)\b", text, re.I)):
            continue
        sources.append({"url": url, "title": page["title"], "aliases": aliases,
                        "text": text[:24000], "source_type": "retrieved_public_reference"})
        if _confident(page, text, item, kind):
            break
    if not sources:
        if last_error is not None:
            raise last_error
        raise GeminiUnavailable("public_source_identity_not_found")
    record_event("public_source", "wikipedia", "success", details={"source_count": len(sources)})
    return sources
