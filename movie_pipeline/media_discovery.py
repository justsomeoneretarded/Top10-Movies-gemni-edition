"""Bounded discovery shared by the independent primary and extra footage pools."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

from .core import normalized
from .release_identity import verified_release_years

REJECT = re.compile(
    r"\b(concept|ai[\s_-]?(?:generated|trailer)|generated\s+(?:by|with)\s+ai|"
    r"reaction|review|explained|deepfake|full (?:movie|episode|film)|"
    r"behind the scenes|making of|interview|breakdown|recap|ending|easter eggs?|"
    r"ranked|ranking|top \d+|compilation|parody|fan theory|edit audio|slowed|reverb)\b",
    re.IGNORECASE,
)
FOOTAGE_WORDS = {"trailer", "teaser", "clip", "clips", "scene", "scenes", "preview", "sneak", "peek", "promo"}
YEARS = re.compile(r"\b(?:18|19|20)\d{2}\b")


def identity_names(film: dict) -> list[str]:
    """Use any known cast, crew, creator or network; no director is mandatory."""
    names = []
    for key, limit in (("cast", 5), ("directors", 2), ("creators", 3), ("networks", 3), ("crew", 8)):
        values = film.get(key) or []
        if isinstance(values, (str, dict)):
            values = [values]
        for value in values[:limit]:
            name = value.get("name", "") if isinstance(value, dict) else value
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    if isinstance(film.get("director"), str) and film["director"].strip():
        names.append(film["director"].strip())
    return list(dict.fromkeys(names))


def _production_url(value: str) -> str:
    """A full production path, never merely a studio's shared homepage host."""
    try:
        parts = urlsplit(value.rstrip(".,;:!?)"))
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            return ""
        host = parts.hostname.lower().removeprefix("www.")
        path = parts.path.rstrip("/")
        if not path or path == "/":
            return ""
        return urlunsplit(("https", host, path, parts.query, ""))
    except (TypeError, ValueError):
        return ""


def identity_ok(info: dict, film: dict) -> bool:
    names = identity_names(film)
    raw = " ".join(str(info.get(key) or "") for key in ("title", "description", "uploader", "channel"))
    tags = info.get("tags") or []
    if isinstance(tags, list):
        raw += " " + " ".join(map(str, tags))
    text = " " + normalized(raw) + " "
    if any(" " + normalized(name) + " " in text for name in names if normalized(name)):
        return True
    expected = {_production_url(str(film.get("homepage") or ""))}
    imdb = str(film.get("imdb_id") or "")
    if re.fullmatch(r"tt\d{5,12}", imdb):
        expected.add("https://imdb.com/title/" + imdb)
    expected.discard("")
    linked = {_production_url(url) for url in re.findall(r"https?://[^\s<>\"']+", raw)}
    if expected & linked:
        return True
    return not names and not expected  # no identity metadata was available to compare


def rejection_reason(info: dict, film: dict, *, curated: bool = False, secondary: bool = False) -> str | None:
    """Return a safe diagnostic enum; searched candidates still require production identity."""
    if not isinstance(info, dict) or not isinstance(info.get("title"), str):
        return "missing_metadata"
    title = info["title"]
    if REJECT.search(title):
        return "excluded_content"
    if info.get("is_live"):
        return "live"
    length = info.get("duration")
    if (isinstance(length, bool) or not isinstance(length, (int, float))
            or not math.isfinite(length) or not (20 if secondary else 25) <= length <= 360):
        return "duration"
    requested = film.get("requested_season") or film.get("season")
    if isinstance(requested, dict):
        requested = requested.get("season_number")
    labelled = re.findall(r"\bseason\s+(\d+)\b|\bs(\d{1,2})e\d+", title, re.I)
    if str(requested).isdigit() and labelled and all(int(a or b) != int(requested) for a, b in labelled):
        return "wrong_season"
    if curated:
        return None
    expected = set(normalized(str(film.get("title", ""))).split())
    found = set(normalized(title).split())
    if not expected or len(expected & found) / len(expected) < .85:
        return "wrong_title"
    if not found & FOOTAGE_WORDS:
        return "not_footage"
    if not identity_ok(info, film):
        return "wrong_identity"
    # Festival and theatrical labels can differ, but only source-backed dates for
    # this exact production expand its year set. A raw list of guesses never does.
    years = set(map(int, YEARS.findall(title))) - set(map(int, YEARS.findall(str(film.get("title", "")))))
    if film.get("kind") != "tv" and years and not years & verified_release_years(film):
        return "wrong_year"
    return None


def fair_candidates(curated, queries, search, *, limit: int, fanout: int = 4,
                    should_stop=None, on_search_error=None, excluded=()):
    """One unique result per family before seconds, with bounded flat searches.

    The caller still owns metadata inspection, downloader options, temp files and
    reservations. Excluded accepted sources never consume a metadata slot.
    """
    families = [list(dict.fromkeys(curated))[:limit]]
    seen = set(excluded)
    emitted = 0

    def take(family):
        while family:
            url = family.pop(0)
            if isinstance(url, str) and url and url not in seen:
                seen.add(url)
                return url
        return None

    for index in range(1 + len(queries)):
        if emitted >= limit or (should_stop and should_stop()):
            return
        if index:
            try:
                family = list(search(queries[index - 1]))[:fanout]
            except Exception as exc:  # callback reports only safe error categories
                if on_search_error is None:
                    raise
                on_search_error(exc)
                family = []
            families.append(family)
        url = take(families[index])
        if url:
            emitted += 1
            yield url, index == 0
    while emitted < limit:
        progressed = False
        for index, family in enumerate(families):
            if emitted >= limit or (should_stop and should_stop()):
                return
            url = take(family)
            if url:
                progressed = True
                emitted += 1
                yield url, index == 0
        if not progressed:
            return


@dataclass
class DiscoveryStats:
    seen: int = 0
    rejected: Counter = field(default_factory=Counter)
    download_attempts: int = 0
    errors: Counter = field(default_factory=Counter)

    def reject(self, reason: str) -> None:
        self.rejected[reason] += 1
        print(f"Footage candidate rejected: {reason}.", flush=True)

    def error(self, exc: Exception) -> None:
        # Exception text can contain authenticated proxies, cookies or signed URLs.
        self.errors[type(exc).__name__] += 1

    def summary(self) -> str:
        rejects = ",".join(f"{key}:{value}" for key, value in sorted(self.rejected.items())) or "none"
        errors = ",".join(f"{key}:{value}" for key, value in sorted(self.errors.items())) or "none"
        return (f"seen={self.seen}, rejected={sum(self.rejected.values())} ({rejects}), "
                f"download_attempts={self.download_attempts}, errors={errors}")
