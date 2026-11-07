"""Optional US watch listings from TMDB's JustWatch-powered provider data.

Listings are checked close to production time. Missing data never blocks a video;
subscription, free-with-ads, rental, and purchase offers remain distinct.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter

from .core import PipelineError, save_json

CATEGORIES = {"subscription": "flatrate", "free": "free", "ads": "ads", "rent": "rent", "buy": "buy"}
TTL = timedelta(hours=24)


def _empty(now: datetime, status: str = "unavailable") -> dict:
    return {"region": "US", "checked_at": now.isoformat(), "status": status,
            "source": "TMDB / JustWatch", "link": "", **{name: [] for name in CATEGORIES}}


def _names(offers) -> list[str]:
    if not isinstance(offers, list):
        return []
    names = []
    seen = set()
    for offer in offers:
        if not isinstance(offer, dict):
            continue
        name = offer.get("provider_name")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100 or any(ord(c) < 32 for c in name):
            continue
        name = " ".join(name.split())
        if any(char in name for char in "<>{}") or name.casefold() in seen:
            continue
        seen.add(name.casefold())
        names.append(name)
    return names


def _link(value) -> str:
    if not isinstance(value, str):
        return ""
    try:
        parsed = urlsplit(value)
        if (parsed.scheme == "https" and parsed.hostname in {"www.themoviedb.org", "themoviedb.org"}
                and not parsed.username and not parsed.password and parsed.port in {None, 443}):
            return value
    except ValueError:
        pass
    return ""


def _cached(path: Path, now: datetime) -> dict | None:
    try:
        record = json.loads(path.read_text())
        checked = datetime.fromisoformat(record["checked_at"])
        if (record.get("region") == "US" and record.get("status") in {"found", "not-listed"}
                and checked.tzinfo is not None and timedelta(0) <= now-checked < TTL
                and all(isinstance(record.get(name), list) and all(isinstance(n, str) for n in record[name]) for name in CATEGORIES)):
            return record
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    return None


def enrich_availability(films: list[dict], research, cache: Path) -> list[dict]:
    """Return augmented film dictionaries; cache successful lookups for 24 hours.

    The optional lookup uses a short-lived session with the existing TMDB auth
    headers and no retries, so a provider outage cannot consume the whole run.
    Authentication is sent only to the fixed api.themoviedb.org endpoint.
    """
    now = datetime.now(UTC)
    cache.mkdir(parents=True, exist_ok=True)
    result = []
    with requests.Session() as session:
        session.headers.update(research.session.headers)
        session.mount("https://", HTTPAdapter(max_retries=0))
        for film in films:
            kind, identifier = film.get("kind"), film.get("id")
            if kind not in {"movie", "tv"} or not isinstance(identifier, int) or isinstance(identifier, bool) or identifier <= 0:
                result.append({**film, "watch_availability": _empty(now)})
                continue
            path = cache / f"{kind}-{identifier}-US-v1.json"
            record = _cached(path, now)
            if record is None:
                record = _empty(now)
                try:
                    response = session.get(f"https://api.themoviedb.org/3/{kind}/{identifier}/watch/providers", timeout=(5, 12), allow_redirects=False)
                    if response.status_code != 200:
                        raise PipelineError("Watch listing provider unavailable")
                    payload = response.json()
                    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
                        raise TypeError("Malformed watch listing response")
                    listings = payload["results"]
                    region = listings.get("US", {})
                    if not isinstance(region, dict):
                        raise TypeError("Malformed regional watch listing")
                    for name, api_name in CATEGORIES.items():
                        record[name] = _names(region.get(api_name))
                    record["link"] = _link(region.get("link"))
                    record["status"] = "found" if any(record[name] for name in CATEGORIES) else "not-listed"
                    save_json(path, record)
                except (requests.RequestException, PipelineError, ValueError, TypeError, OSError):
                    # Never log response bodies, auth headers, or exceptions containing URLs.
                    print(f"Watch listings unavailable for {film.get('title', 'this title')}; continuing without availability claims.", flush=True)
            result.append({**film, "watch_availability": record})
    return result


BRAND_SUFFIX = re.compile(
    r"\s+(?:(?:amazon|apple\s+tv|roku\s+premium|youtube)\s+channel|channel|with\s+ads|standard(?:\s+with\s+ads)?|"
    r"basic(?:\s+with\s+ads)?|premium(?:\s+plus)?|essential|ad[- ]free|kids|store)$", re.IGNORECASE)
BRAND_ALIASES = {"amazon prime video": "Prime Video", "amazon video": "Prime Video", "max": "HBO Max", "hbo max": "HBO Max", "hbo": "HBO Max",
                 "fandango at home": "Fandango at Home", "vudu": "Fandango at Home", "google play movies": "Google Play"}


def brand_name(name: str) -> str:
    """The plain service name a viewer recognizes: no tier, ads, add-on channel or store wording."""
    text = " ".join(str(name).split())
    previous = None
    while previous != text:
        previous = text
        text = BRAND_SUFFIX.sub("", text).strip()
    text = text.replace("+", " Plus").replace("  ", " ").strip()
    return BRAND_ALIASES.get(text.casefold(), text)


# Services a viewer can recognise and open. Only these are ever spoken or put on a thumbnail;
# anything else in the listing data (aggregators like "JustWatch TV", cable or TV-provider
# portals, tiny niche channels, unfamiliar regional names) is silently left out.
KNOWN_SERVICES = {name.casefold(): name for name in (
    "Netflix", "Prime Video", "Hulu", "Disney Plus", "HBO Max", "Paramount Plus", "Peacock", "Apple TV",
    "Starz", "MGM Plus", "AMC Plus", "Shudder", "Crunchyroll", "Tubi", "Pluto TV", "The Roku Channel",
    "Plex", "Fandango at Home", "Google Play", "YouTube", "Kanopy", "Hoopla", "MUBI", "Criterion Channel",
    "BritBox", "Acorn TV", "BET Plus", "Sundance Now", "Hallmark Plus", "Sony Pictures Core", "Freevee",
    "Philo", "Fubo", "Sling TV", "Crackle", "Screambox", "ScreenPix", "Cinemax", "Showtime", "Hi-YAH",
)}
SERVICE_ALIASES = {"apple tv plus": "Apple TV", "apple itunes": "Apple TV", "itunes": "Apple TV",
                   "paramount plus with showtime": "Paramount Plus", "disney": "Disney Plus",
                   "roku channel": "The Roku Channel", "the criterion channel": "Criterion Channel",
                   "fubotv": "Fubo", "sling": "Sling TV", "amazon freevee": "Freevee", "plex player": "Plex",
                   "youtube premium": "YouTube", "youtube tv": "YouTube", "amc": "AMC Plus", "mgm": "MGM Plus",
                   "epix": "MGM Plus", "hoopla digital": "Hoopla", "kanopy library": "Kanopy"}


def known_service(name: str) -> str | None:
    """The plain name of a recognised service, or None for anything viewers would not know or could not open."""
    plain = brand_name(name)
    folded = plain.casefold()
    folded = SERVICE_ALIASES.get(folded, KNOWN_SERVICES.get(folded, "")).casefold() or folded
    return KNOWN_SERVICES.get(folded)


def services(record: dict | None, categories=("subscription", "free", "ads", "rent", "buy"), limit: int = 2) -> list[str]:
    """Recognised service names in listing order, deduplicated, best categories first."""
    if not isinstance(record, dict) or record.get("status") not in {"found", None}:
        return []
    names = []
    for category in categories:
        values = record.get(category, [])
        for value in values if isinstance(values, list) else []:
            name = known_service(value) if isinstance(value, str) and value.strip() else None
            if name and name not in names:
                names.append(name)
    return names[:limit]


def availability_line(record: dict | None) -> str:
    """One short spoken line with just the service names: 'You can find it on Paramount Plus and Prime Video.'"""
    names = services(record)
    return f"You can find it on {' and '.join(names)}." if names else ""
