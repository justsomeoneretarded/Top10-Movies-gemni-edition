"""Bounded, keyless public reference retrieval for Gemini extraction.

This is retrieved page context, not Google Search grounding. URLs come from
MediaWiki's returned canonical page info; redirects supply actual title aliases.
"""
from __future__ import annotations

import re
import time
from urllib.parse import urlsplit

import requests

from .core import normalized
from .gemini import GeminiUnavailable

_API = "https://en.wikipedia.org/w/api.php"
_HEADERS = {"User-Agent": "Top10MoviesGemini/1.0 (https://github.com/justsomeoneretarded/Top10-Movies-gemni-edition)"}


def _get(deadline: float, **params) -> dict:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise GeminiUnavailable("public_source_time_budget")
    try:
        response = requests.get(_API, params={**params, "format": "json", "formatversion": 2},
                                headers=_HEADERS, timeout=(min(4, remaining), min(12, remaining)),
                                allow_redirects=False)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError, TypeError):
        raise GeminiUnavailable("public_source_unavailable") from None
    if not isinstance(data, dict) or data.get("error"):
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


def collect_public_sources(item, kind: str) -> list[dict]:
    """Search at most five pages and fetch plaintext/canonical URLs in one batch."""
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
    names = names[:5]
    result = _get(deadline, action="query", prop="extracts|info|pageprops", inprop="url",
                  explaintext=1, exsectionformat="wiki", exlimit=5, redirects=1, titles="|".join(names))
    query = result.get("query", {})
    if not isinstance(query, dict):
        raise GeminiUnavailable("public_source_invalid_response")
    pages = query.get("pages", [])
    if isinstance(pages, dict):
        pages = list(pages.values())
    if not isinstance(pages, list):
        raise GeminiUnavailable("public_source_invalid_response")
    edges = []
    for field in ("normalized", "redirects"):
        for row in query.get(field, []) if isinstance(query.get(field, []), list) else []:
            if isinstance(row, dict) and isinstance(row.get("from"), str) and isinstance(row.get("to"), str):
                edges.append((row["from"], row["to"]))
    sources = []
    for page in pages[:5]:
        props = page.get("pageprops", {}) if isinstance(page, dict) else {}
        if not isinstance(props, dict):
            continue
        if (not isinstance(page, dict) or page.get("missing") or page.get("invalid")
                or page.get("ns", 0) != 0 or "disambiguation" in props):
            continue
        title, text = page.get("title"), page.get("extract")
        url = _page_url(page.get("fullurl") or page.get("canonicalurl"))
        if (not isinstance(title, str) or not isinstance(text, str) or len(text.strip()) < 80 or not url
                or re.search(r"<(?:!doctype|html|body|script)\b", text, re.I)):
            continue
        aliases = {title}
        for _ in range(5):
            aliases.update(before for before, after in edges if after in aliases)
        # A search hit alone cannot turn a similarly named work into this title.
        if title_key(item.title) not in {title_key(alias) for alias in aliases}:
            continue
        sources.append({"url": url, "title": title, "aliases": sorted(aliases),
                        "text": text[:24000], "source_type": "retrieved_public_reference"})
    if not sources:
        raise GeminiUnavailable("public_source_identity_not_found")
    return sources
