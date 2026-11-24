"""Release years belong to a verified production, never to a title search alone."""
from __future__ import annotations

import re
from datetime import date as calendar_date
from urllib.parse import urlsplit

RELEASE_IDENTITY_VERSION = 1


def _year(value):
    return value if type(value) is int and 1870 <= value <= 2200 else None


def _dated_year(value):
    if not isinstance(value, str) or not re.match(r"^\d{4}-\d{2}-\d{2}(?:T|$)", value):
        return None
    try:
        return _year(calendar_date.fromisoformat(value[:10]).year)
    except ValueError:
        return None


def _public_release_quote(quote: str, year: int, names=()) -> bool:
    """A same-page date still needs a release event for the actual work."""
    plain = " ".join(quote.split())
    if (not re.search(r"\b" + str(year) + r"\b", plain)
            or re.search(r"\b(?:born|birth|filming|production began|sequel|prequel|remake|"
                         r"follow.up|re.release|rerelease|season|episode|planned|cancelled)\b", plain, re.I)):
        return False
    subjects = [r"It", r"The (?:film|movie|series|show)"]
    subjects += [re.escape(name) for name in names if isinstance(name, str) and name.strip()]
    subject = r"^(?:" + "|".join(subjects) + r")\b"
    if not re.search(subject, plain, re.I):
        return False
    event = (r"\b(?:was (?:\w+ ){0,3}released|premiered|debuted|"
             r"had its (?:world |international |theatrical )?premiere|"
             r"was first (?:broadcast|aired)|first aired)\b")
    return bool(re.search(event + r".{0,220}\b" + str(year) + r"\b", plain, re.I))


def public_release_evidence(source: dict, kind: str) -> list[dict]:
    """Extract exact release quotations after the caller verified this page."""
    if not isinstance(source, dict) or not isinstance(source.get("text"), str):
        return []
    url = source.get("url")
    if not isinstance(url, str) or not url.startswith("https://"):
        return []
    title = re.sub(r"\s+\([^)]*(?:film|series)[^)]*\)\s*$", "", source.get("title", ""), flags=re.I)
    aliases = source.get("aliases", [])
    names = [title, *[name for name in aliases if isinstance(name, str)]] if isinstance(aliases, list) else [title]
    result = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", source["text"]):
        quote = " ".join(sentence.split())
        for value in dict.fromkeys(re.findall(r"\b(?:18|19|20|21)\d{2}\b", quote)):
            year = int(value)
            if _public_release_quote(quote, year, names):
                result.append({"year": year, "source": url, "kind": "public_release",
                               "identity": url, "quote": quote})
    return result


def tmdb_identity(film: dict) -> str:
    kind = film.get("kind")
    identifier = film.get("tmdb_id", film.get("id"))
    if kind in {"movie", "tv"} and type(identifier) is int and identifier > 0:
        return f"{kind}/{identifier}"
    return ""


def tmdb_release_evidence(data: dict, kind: str, identifier: int) -> list[dict]:
    """Keep actual dated records returned for this exact TMDB production."""
    if (kind != "movie" or type(identifier) is not int or identifier <= 0
            or not isinstance(data, dict) or data.get("id") != identifier
            or type(data.get("id")) is not int):
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
            year = _dated_year(date)
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
    aliases = film.get("identity_aliases", [])
    names = [film.get("title", ""), *aliases] if isinstance(aliases, list) else [film.get("title", "")]
    evidence = film.get("release_year_evidence", [])
    for row in evidence if isinstance(evidence, list) else []:
        if not isinstance(row, dict):
            continue
        year = _year(row.get("year"))
        source = row.get("source")
        if year is None or not isinstance(source, str) or row.get("kind") not in {
                "festival", "theatrical", "regional", "premiere", "release", "search", "first_air", "public_release"}:
            continue
        if (identity and row.get("identity") == identity and _tmdb_source(source, identity)
                and _dated_year(row.get("date")) == year):
            years.add(year)
            continue
        quote = row.get("quote")
        if (row.get("kind") != "public_release" or not public_url or source != public_url
                or row.get("identity") != public_url or not isinstance(quote, str)
                or not _public_release_quote(quote, year, names)
                or not isinstance(film.get("identity_evidence"), str) or not film["identity_evidence"]):
            continue
        proof = film.get("research_evidence", [])
        if not isinstance(proof, list):
            continue
        if any(isinstance(item, dict) and isinstance(item.get("text"), str)
               and quote in item["text"] and isinstance(item.get("sources"), list) and source in item["sources"]
               for item in proof):
            years.add(year)
    return years
