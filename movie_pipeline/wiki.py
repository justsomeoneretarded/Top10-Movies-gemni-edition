"""Keyless Wikipedia context: lead summary, production and reception notes.

Plot/synopsis sections are deliberately skipped so narration stays
spoiler-light. Everything here is optional enrichment; any failure returns an
empty dict and production continues with TMDB facts alone.
"""
from __future__ import annotations

import re

import requests

HEADERS = {"User-Agent": "Top10Movies/0.2 (https://github.com/justsomeoneretarded/Top10-Movies)"}
WIKIDATA = "https://www.wikidata.org/w/api.php"
WIKIPEDIA = "https://en.wikipedia.org/w/api.php"
PRODUCTION = re.compile(r"production|development|filming|casting|pre-production|visual effects|music|design|writing", re.I)
RECEPTION = re.compile(r"reception|critical|accolades|box office|awards|legacy|ratings|viewership", re.I)
SKIP = re.compile(r"plot|synopsis|summary|episodes|cast|characters|references|external|notes|see also|further", re.I)


def _get(url: str, **params) -> dict:
    response = requests.get(url, params={**params, "format": "json"}, headers=HEADERS, timeout=20)
    response.raise_for_status()
    return response.json()


def _title_from_imdb(imdb_id: str) -> str | None:
    if not re.fullmatch(r"tt\d{5,10}", imdb_id or ""):
        return None
    found = _get(WIKIDATA, action="query", list="search", srsearch=f"haswbstatement:P345={imdb_id}", srlimit=1)
    hits = found.get("query", {}).get("search", [])
    if not hits:
        return None
    entity = hits[0]["title"]
    data = _get(WIKIDATA, action="wbgetentities", ids=entity, props="sitelinks", sitefilter="enwiki")
    return data.get("entities", {}).get(entity, {}).get("sitelinks", {}).get("enwiki", {}).get("title")


def _title_from_search(title: str, year, kind: str) -> str | None:
    suffix = "TV series" if kind == "tv" else "film"
    found = _get(WIKIPEDIA, action="query", list="search", srsearch=f'"{title}" {year} {suffix}', srlimit=5)
    wanted = re.sub(r"[^a-z0-9]", "", title.casefold())
    for hit in found.get("query", {}).get("search", []):
        name = hit["title"]
        if re.sub(r"[^a-z0-9]", "", name.split("(")[0].casefold()) == wanted:
            return name
    return None


def sections(text: str) -> tuple[str, dict[str, str]]:
    """Split a plain-text extract into its lead and level-2 sections (with their subsections)."""
    parts = re.split(r"^==\s*([^=].*?)\s*==\s*$", text, flags=re.M)
    lead = parts[0].strip()
    found = {}
    for heading, body in zip(parts[1::2], parts[2::2]):
        # Subsection headings stay inside their parent section, as plain text.
        found[heading.strip()] = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1.", body, flags=re.M).strip()
    return lead, found


def trim(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[:end+1] if end > limit*0.4 else cut.rsplit(" ", 1)[0] + "…"


def wikipedia_context(film: dict) -> dict:
    try:
        page = _title_from_imdb(film.get("imdb_id") or "") or _title_from_search(film["title"], film.get("year"), film.get("kind", "movie"))
        if not page:
            return {}
        data = _get(WIKIPEDIA, action="query", prop="extracts", explaintext=1, exsectionformat="wiki",
                    titles=page, redirects=1)
        pages = list(data.get("query", {}).get("pages", {}).values())
        if not pages or "extract" not in pages[0]:
            return {}
        lead, parts = sections(pages[0]["extract"])
        production = " ".join(body for name, body in parts.items() if PRODUCTION.search(name) and not SKIP.search(name))
        reception = " ".join(body for name, body in parts.items() if RECEPTION.search(name) and not SKIP.search(name))
        context = {"wiki_url": "https://en.wikipedia.org/wiki/" + pages[0]["title"].replace(" ", "_"),
                   "wiki_summary": trim(lead, 900), "wiki_production": trim(production, 1100),
                   "wiki_reception": trim(reception, 900)}
        return {name: value for name, value in context.items() if value}
    except (requests.RequestException, ValueError, KeyError, TypeError, IndexError):
        return {}
