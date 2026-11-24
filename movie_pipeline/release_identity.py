"""Release years belong to a verified production, never to a title search alone."""
from __future__ import annotations

import re
from urllib.parse import urlsplit

RELEASE_IDENTITY_VERSION = 1


def _year(value):
    return value if type(value) is int and 1870 <= value <= 2200 else None


def tmdb_identity(film: dict) -> str:
    kind = film.get("kind")
    identifier = film.get("tmdb_id", film.get("id"))
    if kind in {"movie", "tv"} and type(identifier) is int and identifier > 0:
        return f"{kind}/{identifier}"
    return ""


def tmdb_release_evidence(data: dict, kind: str, identifier: int) -> list[dict]:
    """Keep actual dated records returned for this exact TMDB production."""
    if (kind != "movie" or type(identifier) is not int or identifier <= 0
            or not isinstance(data, dict) or data.get("id", identifier) != identifier
            or type(data.get("id", identifier)) is not int):
        return []
    evidence = []
    for country in data.get("results", []) if isinstance(data.get("results"), list) else []:
        if not isinstance(country, dict):
            continue
        for row in country.get("release_dates", []) if isinstance(country.get("release_dates"), list) else []:
            if not isinstance(row, dict):
                continue
            date = row.get("release_date")
            if not isinstance(date, str) or not re.match(r"^\d{4}-\d{2}-\d{2}(?:T|$)", date):
                continue
            year = _year(int(date[:4]))
            if year is None:
                continue
            evidence.append({"year": year, "source": f"https://api.themoviedb.org/3/movie/{identifier}/release_dates",
                             "kind": "festival" if row.get("type") == 1 else "theatrical" if row.get("type") in (2, 3) else "release",
                             "identity": f"movie/{identifier}", "date": date,
                             "region": country.get("iso_3166_1", ""), "note": row.get("note", "")})
    return evidence


def _tmdb_source(source: str, identity: str) -> bool:
    try:
        url = urlsplit(source)
        if url.scheme != "https" or url.username or url.password or url.query or url.fragment:
            return False
        if url.hostname in {"www.themoviedb.org", "themoviedb.org"}:
            return bool(re.fullmatch("/" + re.escape(identity) + r"(?:-[^/]*)?/?", url.path))
        return (url.hostname == "api.themoviedb.org"
                and url.path in {f"/3/{identity}", f"/3/{identity}/release_dates"})
    except (TypeError, ValueError):
        return False


def verified_release_years(film: dict) -> set[int]:
    """Current resolved year plus independently dated, same-production evidence.

    A release_years list is a convenient output, not proof. Public-page additions
    require the exact quotation preserved among that verified page's evidence.
    """
    if not isinstance(film, dict):
        return set()
    base = _year(film.get("year"))
    years = {base} if base is not None else set()
    identity = tmdb_identity(film)
    public_url = film.get("identity_source") or film.get("source")
    evidence = film.get("release_year_evidence", [])
    for row in evidence if isinstance(evidence, list) else []:
        if not isinstance(row, dict):
            continue
        year = _year(row.get("year"))
        source = row.get("source")
        if year is None or not isinstance(source, str) or row.get("kind") not in {
                "festival", "theatrical", "regional", "premiere", "release", "search", "first_air", "public_release"}:
            continue
        if identity and row.get("identity") == identity and _tmdb_source(source, identity):
            years.add(year)
            continue
        quote = row.get("quote")
        if (row.get("kind") != "public_release" or not public_url or source != public_url
                or row.get("identity") != public_url or not isinstance(quote, str)
                or not re.search(r"\b" + str(year) + r"\b", quote)
                or not isinstance(film.get("identity_evidence"), str) or not film["identity_evidence"]):
            continue
        proof = film.get("research_evidence", [])
        if any(isinstance(item, dict) and isinstance(item.get("text"), str)
               and quote in item["text"] and source in item.get("sources", [])
               for item in proof if isinstance(proof, list)):
            years.add(year)
    return years
