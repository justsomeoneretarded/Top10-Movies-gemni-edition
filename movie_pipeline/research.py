from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .core import fold_accents, FilmInput, PipelineError, normalized, save_json
from .wiki import wikipedia_context
from .release_identity import (RELEASE_IDENTITY_VERSION, tmdb_identity,
                               tmdb_release_evidence, verified_release_years)
from .gemini import GeminiUnavailable, enabled, record_event
from .gemini_research import research_film
from .public_research import title_key


class Research:
    def __init__(self):
        self.token = os.getenv("TMDB_READ_TOKEN", "")
        if not self.token and not enabled():
            raise PipelineError("Set the free TMDB_READ_TOKEN Actions secret (TMDB API Read Access Token).")
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))

    def get(self, endpoint: str, **params):
        if not self.token:
            raise PipelineError("TMDB fallback requires TMDB_READ_TOKEN when Gemini source research is unavailable or cannot verify this title.")
        response = self.session.get("https://api.themoviedb.org/3/" + endpoint, params=params, timeout=40)
        # Avoid including request URLs or token-bearing headers in exception text.
        if response.status_code in (401, 403):
            raise PipelineError(f"TMDB fallback authentication failed (HTTP {response.status_code}). Check TMDB_READ_TOKEN; no unverified title was accepted.")
        if response.status_code != 200:
            raise PipelineError(f"TMDB {endpoint} returned HTTP {response.status_code}. Check the token or retry later.")
        return response.json()

    @staticmethod
    def title_score(item: FilmInput, found: dict, kind: str) -> float:
        names = (found.get("title" if kind == "movie" else "name", ""),
                 found.get("original_title" if kind == "movie" else "original_name", ""))
        return max(SequenceMatcher(None, normalized(item.title), normalized(name or "")).ratio() for name in names)

    def title_alias_matches(self, item: FilmInput, found: dict, kind: str) -> bool:
        """Verify an exact alternative title on this search result's own TMDB ID."""
        identifier = found.get("id")
        if type(identifier) is not int or identifier <= 0:
            return False
        try:
            data = self.get(f"{kind}/{identifier}/alternative_titles")
        except (PipelineError, requests.RequestException, ValueError, TypeError):
            # Optional corroboration may fail; never turn a prefix into an identity.
            print("TMDB alternative-title lookup unavailable; no unverified alias accepted.", flush=True)
            return False
        if not isinstance(data, dict) or type(data.get("id")) is not int or data["id"] != identifier:
            return False
        aliases = data.get("titles" if kind == "movie" else "results", [])
        if not isinstance(aliases, list):
            return False
        return any(isinstance(alias, dict) and isinstance(alias.get("title"), str)
                   and normalized(alias["title"]) == normalized(item.title) for alias in aliases)

    def wikidata_candidates(self, item: FilmInput) -> list:
        """Independent identity fallback; never send the TMDB bearer token to Wikimedia."""
        endpoint = "https://www.wikidata.org/w/api.php"
        headers = {"User-Agent": "Top10Movies/0.1 (https://github.com/justsomeoneretarded/Top10-Movies)"}
        try:
            response = requests.get(endpoint, params={"action": "wbsearchentities", "search": item.title, "language": "en", "limit": 5, "format": "json"}, headers=headers, timeout=20)
            response.raise_for_status()
            ids = [x["id"] for x in response.json().get("search", [])]
            if not ids:
                return []
            response = requests.get(endpoint, params={"action": "wbgetentities", "ids": "|".join(ids), "props": "labels|aliases|claims", "languages": "en", "format": "json"}, headers=headers, timeout=20)
            response.raise_for_status()
            candidates = []
            for entity in response.json().get("entities", {}).values():
                names = [entity.get("labels", {}).get("en", {}).get("value", "")]
                names += [a.get("value", "") for a in entity.get("aliases", {}).get("en", [])]
                verified_names = {normalized(name) for name in names if isinstance(name, str) and normalized(name)}
                if normalized(item.title) not in verified_names:
                    continue
                claims = entity.get("claims", {})
                dates = [c.get("mainsnak", {}).get("datavalue", {}).get("value", {}) for c in claims.get("P577", []) if c.get("rank") != "deprecated"]
                if not any(isinstance(d, dict) and d.get("time", "")[1:5] == str(item.year) and d.get("precision", 0) >= 9 for d in dates):
                    continue
                for claim in claims.get("P345", []):
                    imdb = claim.get("mainsnak", {}).get("datavalue", {}).get("value", "")
                    if claim.get("rank") == "deprecated" or not isinstance(imdb, str) or not re.fullmatch(r"tt\d{7,10}", imdb):
                        continue
                    found = self.get(f"find/{imdb}", external_source="imdb_id", language="en-US")
                    for kind, group in (("movie", "movie_results"), ("tv", "tv_results")):
                        for row in found.get(group, []):
                            score = self.title_score(item, row, kind)
                            # Both names must occur on the same dated, IMDb-linked entity.
                            # A shared prefix alone never establishes an alias.
                            canonical = (row.get("title" if kind == "movie" else "name", ""),
                                         row.get("original_title" if kind == "movie" else "original_name", ""))
                            if any(isinstance(name, str) and normalized(name) in verified_names for name in canonical):
                                score = 1.0
                            date = row.get("release_date" if kind == "movie" else "first_air_date", "")
                            if score >= 0.88:
                                row = {**row, "_year_evidence": f"Wikidata {entity['id']} publication date", "_imdb_id": imdb, "_identity_source": f"https://www.wikidata.org/wiki/{entity['id']}"}
                                candidates.append((score, kind, row))
            return candidates
        except (requests.RequestException, ValueError, KeyError):
            print("Independent Wikidata lookup unavailable; no unverified year match accepted.", flush=True)
            return []

    def resolve(self, item: FilmInput) -> dict:
        candidates = []
        selected_kind = getattr(self, "selected_kind", None)
        kinds = ((selected_kind,) if selected_kind in ("movie", "tv") else
                 ("tv", "movie") if getattr(self, "preferred_kind", "movie") == "tv" else ("movie", "tv"))
        for kind in kinds:
            alias_checks = 0
            data = self.get(f"search/{kind}", query=item.title, language="en-US", include_adult="false")
            results = list(data.get("results", []))
            if fold_accents(item.title) != item.title:
                plain = self.get(f"search/{kind}", query=fold_accents(item.title), language="en-US", include_adult="false")
                known = {row.get("id") for row in results}
                results += [row for row in plain.get("results", []) if row.get("id") not in known]
            for found in results:
                if (not isinstance(found, dict) or type(found.get("id")) is not int or found["id"] <= 0):
                    continue
                date = found.get("release_date" if kind == "movie" else "first_air_date", "")
                date = date if isinstance(date, str) else ""
                similarity = self.title_score(item, found, kind)
                # Any exact provider alias can prove the identity in either
                # direction. A shared prefix or stripped brand name cannot.
                if similarity < 0.88 and alias_checks < 5 and (date[:4] == str(item.year) or kind == "movie"):
                    alias_checks += 1
                    if self.title_alias_matches(item, found, kind):
                        similarity = 1.0
                        found = {**found, "_verified_alias": item.title}
                if similarity < 0.88:
                    continue
                eligible = date[:4] == str(item.year)
                if not eligible and kind == "movie":
                    # Check this earlier result's actual festival/regional dates
                    # before considering a later same-title exact-year record.
                    try:
                        releases = self.get(f"movie/{found['id']}/release_dates")
                        evidence = tmdb_release_evidence(releases, kind, found["id"])
                    except (PipelineError, requests.RequestException, ValueError, TypeError):
                        evidence = []
                    if any(row["year"] == item.year for row in evidence):
                        eligible = True
                        found = {**found, "_year_evidence": "TMDB dated festival/regional release record",
                                 "_release_year_evidence": evidence}
                if eligible:
                    candidates.append((similarity, kind, found))
                    break
            if candidates:
                break
        if not candidates:
            candidates = self.wikidata_candidates(item)
            if selected_kind:
                candidates = [candidate for candidate in candidates if candidate[1] == selected_kind]
        if not candidates:
            raise PipelineError(f"Could not verify {item.title} ({item.year}) using search dates, dated release records, or the independent Wikidata/IMDb-ID fallback. This does not establish that your input is wrong.")
        # Owner policy: always keep TMDB's first eligible result. Preserve the API
        # order instead of reranking by votes, exactness, or a second confidence gate.
        # Title/year checks still exclude unrelated titles and unsupported years.
        _, kind, match = candidates[0]
        print(f"Selected first matching TMDB result: {item.title} ({item.year}) -> {kind}/{match['id']}.", flush=True)
        if match.get("_year_evidence"):
            print(f"Verified {item.title} ({item.year}) via {match['_year_evidence']} (search release date: {match.get('release_date', '')}).", flush=True)
        details = self.get(f"{kind}/{match['id']}", append_to_response="videos,credits,external_ids,keywords,images,release_dates", include_image_language="en,null", language="en-US")
        if (not isinstance(details, dict) or type(details.get("id")) is not int
                or details["id"] != match["id"]):
            raise PipelineError("TMDB details do not belong to the selected production.")
        videos = details.get("videos", {}).get("results", [])
        # Original-language trailers may exist when English-localized metadata has none.
        if not videos:
            videos = self.get(f"{kind}/{match['id']}/videos").get("results", [])
        # Vimeo copies are kept as a fallback for runners that YouTube refuses.
        # Official clips (single scenes the studio posted) give extra, different footage for the
        # second and third sources; trailers and teasers stay the primary choice.
        clips = [v for v in videos if v.get("site") == "YouTube" and v.get("type") == "Clip" and v.get("official", True)][:6]
        videos = [v for v in videos if v.get("site") in ("YouTube", "Vimeo") and v.get("type") in ("Trailer", "Teaser")]
        videos.sort(key=lambda v: (v.get("official", False), v.get("type") == "Trailer", v.get("site") == "YouTube"), reverse=True)
        overview = details.get("overview", "").strip()
        if not overview:
            raise PipelineError(f"No factual synopsis available for {item.title}; refusing to invent one.")
        crew = details.get("credits", {}).get("crew", [])
        release_evidence = list(match.get("_release_year_evidence", []))
        # Appended dates inherit the already-verified parent identity. Copy only
        # ID-less nested payloads; explicit conflicting IDs remain rejected.
        release_dates = details.get("release_dates")
        if isinstance(release_dates, dict) and "id" not in release_dates:
            release_dates = {**release_dates, "id": details["id"]}
        release_evidence += tmdb_release_evidence(release_dates, kind, match["id"])
        canonical_source = f"https://www.themoviedb.org/{kind}/{match['id']}"
        for date in (match.get("release_date" if kind == "movie" else "first_air_date"),
                     details.get("release_date" if kind == "movie" else "first_air_date")):
            if isinstance(date, str) and re.match(r"^\d{4}-\d{2}-\d{2}(?:T|$)", date):
                release_evidence.append({"year": int(date[:4]), "source": canonical_source,
                                         "kind": "search" if kind == "movie" else "first_air",
                                         "identity": f"{kind}/{match['id']}", "date": date})
        return {
            "title": details.get("title", details.get("name")), "year": item.year,
            "kind": kind, "id": match["id"], "tmdb_id": match["id"], "overview": overview,
            "requested_title": item.title, "input_year": item.year,
            "release_identity_version": RELEASE_IDENTITY_VERSION,
            "release_year_evidence": release_evidence,
            "identity_aliases": list(dict.fromkeys([item.title, details.get("title", details.get("name", item.title))])),
            "homepage": details.get("homepage"),
            "release_date": details.get("release_date", details.get("first_air_date")),
            "genres": [g["name"] for g in details.get("genres", [])],
            "directors": [c["name"] for c in crew if c.get("job") == "Director"],
            "cast": [c["name"] for c in details.get("credits", {}).get("cast", [])[:5]],
            "cast_profiles": [{"name": c["name"], "profile_path": c.get("profile_path", ""),
                               "character": c.get("character", "")} for c in details.get("credits", {}).get("cast", [])[:5]],
            "status": details.get("status"), "trailers": videos, "clips": clips,
            "year_evidence": match.get("_year_evidence", "TMDB search release date"),
            "imdb_id": match.get("_imdb_id") or details.get("imdb_id") or details.get("external_ids", {}).get("imdb_id"),
            "identity_source": match.get("_identity_source"),
            "source": f"https://www.themoviedb.org/{kind}/{match['id']}",
            **extra_facts(details, kind),
        }

    def collect(self, items: list[FilmInput], path: Path) -> list[dict]:
        path = Path(path)
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            cached = None
        if research_cache_valid(cached, items, getattr(self, "selected_kind", None)):
            return cached
        films = []
        for item in items:
            print(f"Research: {item.title} ({item.year})", flush=True)
            film = None
            if enabled():
                try:
                    kind = getattr(self, "selected_kind", None) or getattr(self, "preferred_kind", "movie")
                    film = research_film(item, kind)
                    record_event("research", "gemini", "success", details={"sources": len(film["research_sources"])})
                except (GeminiUnavailable, requests.RequestException, ValueError, TypeError, KeyError) as error:
                    # Provider exceptions can contain credential-bearing request URLs.
                    # Log only a fixed reason, never the exception or response body.
                    reason = "unavailable-or-unverified"
                    if isinstance(error, GeminiUnavailable) and re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+){1,15}", str(error)):
                        reason = str(error)
                    record_event("research", "gemini", "fallback", reason=reason)
                    print("Gemini source research unavailable or unverified; trying verified TMDB research.", flush=True)
            if film is None:
                film = self.resolve(item)
                if enabled():
                    record_event("research", "tmdb", "success", reason="verified-fallback")
            elif self.token:
                # Optional media enrichment still passes the original TMDB identity
                # resolver. An outage or conflicting ID cannot replace Gemini facts.
                try:
                    metadata = self.resolve(item)
                    aliases = {title_key(film["title"])}
                    aliases.update(title_key(alias) for alias in film.get("identity_aliases", []) if isinstance(alias, str))
                    identifier = metadata.get("id")
                    compatible = (type(identifier) is int and identifier > 0
                                  and metadata.get("kind") == film["kind"]
                                  and metadata.get("year") == film["year"]
                                  and metadata.get("source") == f"https://www.themoviedb.org/{film['kind']}/{identifier}"
                                  and title_key(str(metadata.get("title", ""))) in aliases
                                  and (film.get("tmdb_id") is None or film["tmdb_id"] == identifier))
                    if (film.get("imdb_id") and metadata.get("imdb_id")
                            and film["imdb_id"] != metadata["imdb_id"]):
                        compatible = False
                    for role in ("cast", "directors"):
                        known = {normalized(name) for name in film.get(role, []) if isinstance(name, str)}
                        found = {normalized(name) for name in metadata.get(role, []) if isinstance(name, str)}
                        if known and found and not known.intersection(found):
                            compatible = False
                    if compatible:
                        film["tmdb_id"] = identifier
                        film["release_year_evidence"] = [
                            *film.get("release_year_evidence", []),
                            *metadata.get("release_year_evidence", [])]
                        for name in ("trailers", "clips", "backdrops", "posters", "cast_profiles", "networks",
                                     "companies", "release_date", "status", "imdb_id", "runtime_minutes"):
                            if metadata.get(name):
                                film[name] = metadata[name]
                        for name in ("cast", "directors", "creators", "genres"):
                            if not film.get(name) and metadata.get(name):
                                film[name] = metadata[name]
                        film["tmdb_source"] = metadata["source"]
                    else:
                        record_event("research", "tmdb", "unavailable", reason="optional-media-identity-conflict")
                except (PipelineError, requests.RequestException, ValueError, TypeError, KeyError):
                    record_event("research", "tmdb", "unavailable", reason="optional-media-enrichment")
            # Optional, keyless background notes; never required for production.
            if os.getenv("WIKIPEDIA_CONTEXT", "true").lower() != "false":
                notes = wikipedia_context(film)
                film.update(notes)
                print(f"Wikipedia notes for {film['title']}: {', '.join(k for k in notes if k != 'wiki_url') or 'none found'}.", flush=True)
            film["requested_title"] = item.title
            film["input_year"] = item.year
            film["release_identity_version"] = RELEASE_IDENTITY_VERSION
            if not isinstance(film.get("release_year_evidence"), list):
                film["release_year_evidence"] = []
            film["release_years"] = sorted(verified_release_years(film))
            films.append(film)
        save_json(path, films)
        return films


def research_cache_valid(films, items, selected_kind=None) -> bool:
    """Only current identity-verified records may bypass research."""
    if not isinstance(films, list) or len(films) != len(items):
        return False
    for film, item in zip(films, items):
        if (not isinstance(film, dict) or type(film.get("release_identity_version")) is not int
                or film["release_identity_version"] != RELEASE_IDENTITY_VERSION
                or normalized(str(film.get("requested_title", ""))) != normalized(item.title)
                or type(film.get("input_year")) is not int or film["input_year"] != item.year
                or type(film.get("year")) is not int or film["year"] != item.year
                or film.get("kind") not in {"movie", "tv"}
                or selected_kind in {"movie", "tv"} and film["kind"] != selected_kind
                or not isinstance(film.get("title"), str) or not film["title"].strip()
                or not isinstance(film.get("overview"), str) or not film["overview"].strip()
                or not isinstance(film.get("release_year_evidence"), list)):
            return False
        if film.get("research_backend") == "gemini":
            if (not isinstance(film.get("identity_evidence"), str) or not film["identity_evidence"]
                    or not isinstance(film.get("research_evidence"), list) or not film["research_evidence"]):
                return False
        elif (not tmdb_identity(film)
              or film.get("source") != f"https://www.themoviedb.org/{tmdb_identity(film)}"):
            return False
    return True


def extra_facts(details: dict, kind: str) -> dict:
    """Additional TMDB details that make narration specific; all optional."""
    keywords = details.get("keywords", {})
    keywords = keywords.get("keywords", keywords.get("results", [])) if isinstance(keywords, dict) else []
    runtime = details.get("runtime") or next(iter(details.get("episode_run_time") or []), None)
    facts = {
        "tagline": (details.get("tagline") or "").strip(),
        "runtime_minutes": runtime if isinstance(runtime, int) and runtime > 0 else None,
        "keywords": [k["name"] for k in keywords[:10] if isinstance(k, dict) and k.get("name")],
        "original_language": details.get("original_language"),
    }
    votes = details.get("vote_count") or 0
    if isinstance(votes, int) and votes >= 300 and isinstance(details.get("vote_average"), (int, float)):
        facts["tmdb_rating"] = round(float(details["vote_average"]), 1)
        facts["tmdb_votes"] = votes
    if kind == "tv":
        facts.update(seasons=details.get("number_of_seasons"), episodes=details.get("number_of_episodes"),
                     networks=[n["name"] for n in details.get("networks", [])[:3] if n.get("name")],
                     creators=[c["name"] for c in details.get("created_by", [])[:3] if c.get("name")])
    images = details.get("images", {}) if isinstance(details.get("images"), dict) else {}
    backdrops = [b for b in images.get("backdrops", []) if isinstance(b, dict) and b.get("file_path") and (b.get("width") or 0) >= 1280]
    # Text-free stills first (no language), then the most-voted.
    backdrops.sort(key=lambda b: (b.get("iso_639_1") is None, b.get("vote_average") or 0, b.get("width") or 0), reverse=True)
    facts["backdrops"] = [b["file_path"] for b in backdrops[:14]]  # thumbnails use the first few; stills fallback uses more
    if not facts["backdrops"] and details.get("backdrop_path"):
        facts["backdrops"] = [details["backdrop_path"]]
    # Text-free posters (character posters, key art): one actor, big, sharp - ideal thumbnail faces.
    posters = [p for p in images.get("posters", []) if isinstance(p, dict) and p.get("file_path")
               and p.get("iso_639_1") is None and (p.get("height") or 0) >= 1200]
    posters.sort(key=lambda p: (p.get("vote_average") or 0, p.get("height") or 0), reverse=True)
    facts["posters"] = [p["file_path"] for p in posters[:8]]
    return {name: value for name, value in facts.items() if value not in (None, "", [])}
