"""Bounded, observable web-photo discovery with verified publicity-still fallback.

Only independently downloaded photos or exact-title publicity stills leave this
module. No extracted trailer frame, invented URL, or unverified identity is used.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse, urljoin
from collections import Counter
import hashlib
import html
import ipaddress
import json
import re
import threading
import time

import requests
from PIL import Image

from .editorial_policy import image_queries
from .core import normalized, save_json

MAX_BYTES = 10_000_000
MAX_HTML_BYTES = 1_500_000
MAX_RESULTS = 6
MAX_QUERIES = 3
MAX_PHOTOS = 3
FILM_SECONDS = 70.0
BAD_CONTEXT = re.compile(
    r"\b(ai[ -]?generated|concept art|fan[ -]?(?:art|made)|wallpaper ai|reaction|"
    r"review thumbnail|youtube thumbnail|gameplay|video game|game wallpaper|game cover|"
    r"red carpet|actor headshot|celebrity headshot|stock portrait|stock photo)\b", re.I)
_CHALLENGE = re.compile(
    r"verify (?:that )?you are human|unusual traffic|solve (?:the|this) challenge|"
    r"please complete the captcha|access to this page has been denied", re.I)
_LOCAL = threading.local()
_FACE_LOCK = threading.Lock()


def public_url(url: str) -> bool:
    try:
        p = urlparse(url)
        if (p.scheme not in ("http", "https") or not p.hostname or p.username
                or p.password or p.port not in (None, 80, 443)):
            return False
        host = p.hostname.lower()
        if host in {"localhost", "metadata.google.internal"} or host.endswith((".local", ".internal")):
            return False
        try:
            return ipaddress.ip_address(host).is_global
        except ValueError:
            return "." in host
    except (TypeError, ValueError):
        return False


def _safe_url(url):
    """Diagnostic locations omit queries/fragments which may contain signed tokens."""
    try:
        p = urlparse(str(url))
        return p._replace(query="", fragment="").geturl()[:500]
    except ValueError:
        return ""


def _report():
    return getattr(_LOCAL, "report", None)


def _count(reason):
    report = _report()
    if report is not None:
        report["counts"][reason] = report["counts"].get(reason, 0) + 1


def _event(kind, **values):
    report = _report()
    if report is not None and len(report[kind]) < 36:
        report[kind].append(values)


def corroborates(film: dict, context: str, page: str) -> bool:
    """Require production context, rejecting known foreign canonical records."""
    if BAD_CONTEXT.search(context + " " + page):
        return False
    p = urlparse(page)
    host, path = (p.hostname or "").lower(), p.path.lower()
    tmdb = re.match(r"/(movie|tv)/(\d+)(?:[-/]|$)", path)
    imdb = re.match(r"/title/(tt\d+)(?:/|$)", path)
    tmdb_host = host in {"www.themoviedb.org", "themoviedb.org"}
    imdb_host = host == "imdb.com" or host.endswith(".imdb.com")
    if tmdb_host and tmdb and film.get("id"):
        if str(film["id"]) != tmdb[2] or film.get("kind", tmdb[1]) != tmdb[1]:
            return False
    known_imdb = film.get("imdb_id")
    if imdb_host and imdb and known_imdb and imdb[1] != known_imdb:
        return False
    stop = {"the", "a", "an", "of", "and", "to", "in", "on"}
    words = set(normalized(film["title"]).split()) - stop
    found = set(normalized(context + " " + page).split())
    if not words or len(words & found) / len(words) < .8:
        return False
    if film.get("kind") != "tv" and film.get("year") and str(film["year"]) not in context + " " + page:
        return False
    season = film.get("requested_season") or film.get("season")
    if isinstance(season, dict):
        season = season.get("season_number")
    mentioned = re.findall(r"\bseason\s+(\d+)\b", context + " " + page, re.I)
    if season and mentioned and str(season) not in mentioned:
        return False
    people = [*film.get("cast", [])[:5], *film.get("directors", [])[:2], *film.get("creators", [])[:2]]
    for name in people:
        parts = normalized(str(name)).split()
        if parts and len(parts[-1]) >= 4 and parts[-1] in found:
            return True
    return bool(tmdb_host and tmdb or imdb_host and imdb)


class ImageMetadataParser(HTMLParser):
    """Bing image metadata may use either quote style and HTML entities."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.items = []
        self.malformed = 0
        self.seen = 0

    def handle_starttag(self, tag, attrs):
        if self.seen >= 200:
            return
        values = dict(attrs)
        raw = values.get("m") or values.get("data-m")
        if raw is None:
            return
        self.seen += 1
        if len(raw) > 12000:
            self.malformed += 1
            return
        try:
            item = json.loads(html.unescape(raw))
            if not isinstance(item, dict):
                raise ValueError
        except (ValueError, TypeError):
            self.malformed += 1
            return
        if "murl" in item or "purl" in item:
            self.items.append(item)


def parse_metadata(text: str):
    parser = ImageMetadataParser()
    try:
        parser.feed(text[:MAX_HTML_BYTES])
        parser.close()
    except (ValueError, TypeError):
        parser.malformed += 1
    return parser.items, {"metadata_seen": parser.seen, "malformed_metadata": parser.malformed}


def _search_html(response):
    """Bound both bytes and time; support simple offline response fixtures."""
    if not hasattr(response, "iter_content"):
        return str(response.text)[:MAX_HTML_BYTES]
    content = bytearray()
    deadline = time.monotonic() + 12
    for block in response.iter_content(65536):
        if time.monotonic() > deadline:
            _count("search_time_limit")
            break
        if not block:
            continue
        content.extend(block)
        if len(content) > MAX_HTML_BYTES:
            _count("search_size_limit")
            del content[MAX_HTML_BYTES:]
            break
    return content.decode(getattr(response, "encoding", None) or "utf-8", errors="replace")


def search(film: dict) -> list[dict]:
    rows, seen = [], set()
    for query in image_queries(film, limit=MAX_QUERIES):
        report = _report()
        if report is not None and time.monotonic() > report["_deadline"]:
            _count("film_time_limit")
            break
        query_report = {"query": query[:300], "status": "unavailable", "results": 0,
                        "accepted": 0, "rejections": {}}
        rejected = Counter()
        response = None
        try:
            response = requests.get(
                "https://www.bing.com/images/search", params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"}, timeout=(4, 8), stream=True)
            query_report["http_status"] = int(response.status_code)
            if response.status_code in (403, 429):
                query_report["status"] = "blocked_or_rate_limited"
                _count(query_report["status"])
                continue
            response.raise_for_status()
            body = _search_html(response)
            if _CHALLENGE.search(body):
                query_report["status"] = "challenge"
                _count("search_challenge")
                continue
            items, parsed = parse_metadata(body)
            query_report.update(parsed, results=len(items))
            query_report["status"] = "results" if items else (
                "malformed_metadata" if parsed["malformed_metadata"] else "no_image_metadata")
            for item in items:
                url, page = item.get("murl", ""), item.get("purl", "")
                if not isinstance(url, str) or not isinstance(page, str):
                    rejected["invalid_location"] += 1
                    continue
                context = " ".join(str(item.get(k, "")) for k in ("t", "desc", "purl"))[:1200]
                if not public_url(url) or not public_url(page):
                    rejected["nonpublic_location"] += 1
                elif BAD_CONTEXT.search(context):
                    rejected["unrelated_or_artificial_context"] += 1
                elif any(host in url + " " + page for host in ("ytimg.com", "youtube.com", "pinterest.com")):
                    rejected["creator_thumbnail_or_aggregator"] += 1
                elif not corroborates(film, context, page):
                    rejected["identity_not_corroborated"] += 1
                elif url in seen:
                    rejected["duplicate_url"] += 1
                else:
                    seen.add(url)
                    rows.append({"url": url, "page": page, "query": query, "context": context[:300]})
                    query_report["accepted"] += 1
                if len(rows) >= MAX_RESULTS:
                    break
            if rows and query_report["accepted"]:
                query_report["status"] = "accepted_candidates"
        except requests.RequestException as exc:
            query_report["status"] = "request_failed"
            query_report["error_class"] = type(exc).__name__
            _count("search_request_failed")
        finally:
            if response is not None:
                response.close()
            query_report["rejections"] = dict(rejected)
            _event("queries", **query_report)
        if len(rows) >= MAX_RESULTS:
            break
    return rows


def _valid_image(image):
    return (image.width >= 800 and image.height >= 450
            and image.width * image.height <= 40_000_000)


def read_image(url: str, cache: Path):
    original_url = url
    if not public_url(url):
        _count("download_nonpublic")
        return None
    target = cache / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".jpg")
    if target.exists():
        try:
            with Image.open(target) as cached:
                if not _valid_image(cached):
                    raise ValueError("cached dimensions")
                image = cached.convert("RGB")
            _count("download_cache_hit")
            return image
        except (OSError, ValueError, Image.DecompressionBombError):
            _count("invalid_cached_image")
            target.unlink(missing_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 16
    failure = "redirect_limit"
    try:
        for _ in range(4):
            if not public_url(url):
                failure = "redirect_nonpublic"
                return None
            if time.monotonic() > deadline:
                failure = "download_time_limit"
                return None
            response = requests.get(url, timeout=(4, 8), stream=True, allow_redirects=False,
                                    headers={"User-Agent": "Mozilla/5.0"})
            try:
                if response.status_code in (301, 302, 303, 307, 308):
                    url = urljoin(url, response.headers.get("Location", ""))
                    continue
                response.raise_for_status()
                if not response.headers.get("Content-Type", "").lower().startswith("image/"):
                    failure = "not_an_image"
                    return None
                data = bytearray()
                for block in response.iter_content(65536):
                    if time.monotonic() > deadline:
                        failure = "download_time_limit"
                        return None
                    data.extend(block)
                    if len(data) > MAX_BYTES:
                        failure = "image_byte_limit"
                        return None
            finally:
                response.close()
            with Image.open(BytesIO(data)) as original:
                if not _valid_image(original):
                    failure = "image_dimensions"
                    return None
                image = original.convert("RGB")
            image.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
            image.save(target, quality=94)
            failure = "accepted"
            _count("image_downloaded")
            return image
    except requests.RequestException as exc:
        failure = "download_" + type(exc).__name__
    except (OSError, ValueError, Image.DecompressionBombError):
        failure = "invalid_image_bytes"
    finally:
        if failure != "accepted":
            _count(failure)
        _event("downloads", url=_safe_url(original_url), result=failure)
    return None


def _duplicates(image, other, value, other_value):
    """A hash collision in dark or flat images is not proof of a duplicate."""
    if (value ^ other_value).bit_count() > 5:
        return False
    import numpy as np
    a = np.asarray(image.convert("RGB").resize((32, 18)), dtype=float)
    b = np.asarray(other.convert("RGB").resize((32, 18)), dtype=float)
    return float(np.abs(a - b).mean()) < 9.0


def gather_photos(films: list[dict], work: Path) -> tuple[list, list[dict]]:
    from .thumbnail_studio import Candidate, face_quality, frame_sharpness, has_burned_text, _dhash
    from .stills_trailer import fetch_original

    def one(pair):
        index, film = pair
        report = {"film": film["title"], "film_index": index, "queries": [], "downloads": [],
                  "counts": {}, "accepted_web": 0, "accepted_publicity": 0,
                  "_deadline": time.monotonic() + FILM_SECONDS}
        _LOCAL.report = report
        accepted = []

        def accept(image, source, notes):
            if image is None:
                _count("image_unavailable")
                return
            if not _valid_image(image):
                _count("quality_dimensions")
                return
            if frame_sharpness(image) < 20:
                _count("quality_blur")
                return
            if has_burned_text(image):
                _count("quality_text_or_logo")
                return
            image = image.convert("RGB")
            image.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
            value = _dhash(image)
            if any(_duplicates(image, old.image, value, old.notes["hash"]) for old in accepted):
                _count("duplicate_image")
                return
            # YuNet has mutable input size; downloads stay parallel, face inference does not.
            with _FACE_LOCK:
                face, quality, good = face_quality(image)
            candidate = Candidate(image, index, source, face=face, score=quality)
            candidate.notes = {"good_face": good, "face_quality": quality, "hash": value, **notes}
            accepted.append(candidate)
            report["accepted_web" if source == "web-image" else "accepted_publicity"] += 1

        try:
            rows = search(film)
            for row in rows[:MAX_RESULTS]:
                if time.monotonic() > report["_deadline"]:
                    _count("film_time_limit")
                    break
                accept(read_image(row["url"], work / "web"), "web-image",
                       {"url": _safe_url(row["url"]), "source_page": _safe_url(row["page"]),
                        "query": row["query"], "identity_evidence": row["context"],
                        "source_kind": "web-image"})
                if len(accepted) >= MAX_PHOTOS:
                    break
        except (OSError, ValueError, KeyError, TypeError, requests.RequestException) as exc:
            _count("discovery_" + type(exc).__name__)

        # Supplement a sparse web pool, not only an entirely empty one. These
        # remain explicitly labelled exact-record publicity stills.
        for path in film.get("backdrops", [])[:6]:
            if len(accepted) >= MAX_PHOTOS:
                break
            if time.monotonic() > report["_deadline"]:
                _count("film_time_limit")
                break
            try:
                accept(fetch_original(path, work / "publicity"), "publicity-fallback",
                       {"source_kind": "publicity-fallback",
                        "url": "https://image.tmdb.org/t/p/original" + path,
                        "fallback_reason": ("No usable independently searched photo" if not report["accepted_web"]
                                            else "Supplemented a sparse verified web-photo pool"),
                        "identity_evidence": "TMDB image attached to this exact title ID"})
            except (OSError, ValueError, requests.RequestException) as exc:
                _count("publicity_" + type(exc).__name__)
        report["status"] = ("web_and_publicity" if report["accepted_web"] and report["accepted_publicity"]
                            else "web" if report["accepted_web"] else
                            "publicity_fallback" if accepted else "no_verified_photo")
        report.pop("_deadline", None)
        _LOCAL.report = None
        return accepted, report

    with ThreadPoolExecutor(max_workers=3) as pool:
        grouped = list(pool.map(one, enumerate(films)))
    result, provenance, diagnostics = [], [], []
    for group, report in grouped:
        diagnostics.append(report)
        for candidate in group:
            value = candidate.notes["hash"]
            if any(_duplicates(candidate.image, old.image, value, old.notes["hash"]) for old in result):
                report["counts"]["global_duplicate"] = report["counts"].get("global_duplicate", 0) + 1
                continue
            result.append(candidate)
            provenance.append({"film": films[candidate.film]["title"], **candidate.notes})
    save_json(work / "image-discovery.json", provenance)
    save_json(work / "image-discovery-report.json", {
        "schema_version": 2, "provider": "bing-html", "limits": {
            "queries_per_title": MAX_QUERIES, "candidates_per_title": MAX_RESULTS,
            "accepted_photos_per_title": MAX_PHOTOS, "workers": 3,
            "seconds_per_title": FILM_SECONDS}, "titles": diagnostics,
        "accepted": len(result), "independently_searched": sum(c.source == "web-image" for c in result),
        "publicity_fallback": sum(c.source == "publicity-fallback" for c in result)})
    return result, provenance
