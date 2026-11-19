from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from . import download_access
from .core import PipelineError, duration, normalized, probe, save_json
from .video_settings import export_size, require_disk

REJECT = re.compile(r"\b(concept|ai.?generated|reaction|review|explained|deepfake|full (?:movie|episode|film))\b", re.IGNORECASE)
MAX_TRAILER_BYTES = 800_000_000
MAX_CANDIDATES = 8
MAX_QUALITY_DOWNLOADS = 3


def identity_names(film: dict) -> list[str]:
    """People and networks that a genuine trailer for this title usually names (from TMDB)."""
    names = [*film.get("cast", [])[:5], *film.get("directors", [])[:2], *film.get("creators", [])[:3],
             *film.get("networks", [])[:3]]
    return [str(name) for name in names if isinstance(name, str) and name.strip()]


def identity_ok(info: dict, film: dict) -> bool:
    """A searched video must name someone or something from the right title (cast, director,
    creator or network), so a remake or another country's version with the same name is rejected."""
    names = identity_names(film)
    if not names:
        return True  # nothing to check against
    text = " " + normalized(" ".join(str(info.get(field) or "") for field in ("title", "description", "uploader", "channel"))
                            + " " + " ".join(map(str, info.get("tags") or []))) + " "
    for name in names:
        words = normalized(name).split()
        if not words:
            continue
        if f" {' '.join(words)} " in text or (len(words) > 1 and len(words[-1]) >= 4 and f" {words[-1]} " in text):
            return True
    return False


def search_query(film: dict, extra: str = "official trailer") -> str:
    """YouTube search text; network names are metadata, not mandatory query terms."""
    from .editorial_policy import season_hint
    return " ".join(str(part) for part in (film["title"], film["year"], season_hint(film), extra) if part)


def candidate_ok(info: dict, film: dict, curated: bool = False) -> bool:
    title = info.get("title", "")
    length = info.get("duration") or 0
    if REJECT.search(title) or info.get("is_live") or not 25 <= length <= 360:
        return False
    if curated:
        return True
    years = set(re.findall(r"\b(?:18|19|20)\d{2}\b", title)) - set(re.findall(r"\b(?:18|19|20)\d{2}\b", film["title"]))
    if film.get("kind") != "tv" and film.get("year") and years and str(film["year"]) not in years:
        return False
    words = set(normalized(film["title"]).split())
    candidate_words = set(normalized(title).split())
    overlap = len(words & candidate_words) / max(1, len(words))
    return (overlap >= 0.85 and bool(candidate_words & {"trailer", "teaser", "scene", "clip", "preview"})
            and identity_ok(info, film))


def effective_height(width, height) -> float:
    """Pixels available for a landscape 16:9 crop, rather than a height-only label."""
    if (isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, (int, float))
            or not isinstance(height, (int, float)) or not math.isfinite(width) or not math.isfinite(height)
            or width <= 0 or height <= 0):
        return 0
    return min(float(height), float(width)*9/16)


def advertised_quality(info: dict, cap: int) -> float:
    """Prefer a source's best usable in-cap format without downloading every trailer."""
    formats = info.get("formats")
    rows = formats if isinstance(formats, list) else [info]
    choices = []
    for row in rows:
        if not isinstance(row, dict) or row.get("vcodec") == "none":
            continue
        width, height = row.get("width"), row.get("height")
        quality = effective_height(width, height)
        if not quality or height > cap:
            continue
        estimated = row.get("filesize") or row.get("filesize_approx")
        if isinstance(estimated, (int, float)) and estimated > MAX_TRAILER_BYTES:
            continue
        choices.append(quality)
    return max(choices, default=0)


def _bot_check(exc: Exception) -> bool:
    return "confirm you're not a bot" in str(exc).casefold().replace("’", "'")


def _blocked():
    return PipelineError(
        "YouTube blocked this runner with a sign-in/bot check. "
        "Trying more video IDs will not resolve a runner-wide block. "
        "Optionally configure the private YOUTUBE_COOKIES Actions secret with a valid "
        "Netscape cookies.txt export and start a new run from main. "
        "Cookies may expire or still be rejected on GitHub-hosted runners; "
        "use a self-hosted runner or a permitted alternative footage source if access remains blocked. "
        "See README: YouTube sign-in/bot errors."
    )


class _RouteBlocked(Exception):
    """The current network route met a sign-in/bot check before any trailer succeeded."""


def _is_youtube(url: str) -> bool:
    return "youtube.com/" in url or "youtu.be/" in url


def trailer_url(trailer: dict) -> str | None:
    """Watch URL for a TMDB video record; YouTube first, Vimeo as a non-YouTube fallback."""
    key = str(trailer.get("key", ""))
    site = trailer.get("site", "YouTube")
    if site == "YouTube" and re.fullmatch(r"[\w-]{11}", key):
        return f"https://www.youtube.com/watch?v={key}"
    if site == "Vimeo" and re.fullmatch(r"\d{5,12}", key):
        return f"https://vimeo.com/{key}"
    return None


def _clean_candidate(path: Path):
    for partial in path.parent.glob(path.stem + ".*"):
        if partial.is_file():
            partial.unlink(missing_ok=True)


def _valid_cached(path: Path) -> bool:
    try:
        record = json.loads(path.with_suffix(".json").read_text())
        return (path.exists() and record.get("quality_selection_version") == 2
                and effective_height(record.get("source_width"), record.get("source_height")) > 0
                and 25 <= duration(path) <= 360 and path.stat().st_size <= MAX_TRAILER_BYTES)
    except (OSError, ValueError, TypeError, AttributeError, PipelineError):
        return False


_LOW_QUALITY_REFRESHES: set[tuple[str, str, str]] = set()


def _cached_record(path: Path) -> dict:
    try:
        value = json.loads(path.with_suffix(".json").read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _record_quality_attempt(path: Path, status: str) -> None:
    record = _cached_record(path)
    if not record:
        return
    run_id = os.getenv("GITHUB_RUN_ID", "").strip()
    if run_id:
        record["quality_refresh_run"] = run_id
        record["quality_refresh_attempt"] = os.getenv("GITHUB_RUN_ATTEMPT", "1").strip() or "1"
    record["quality_refresh_status"] = status
    try:
        save_json(path.with_suffix(".json"), record)
    except OSError:
        pass


def _invalidate_source_analyses(path: Path) -> None:
    """New canonical bytes need new cuts, dialogue and vision—not stale sidecars."""
    protected = {path, path.with_suffix(".json")}
    for derived in path.parent.glob(path.stem + ".*"):
        if derived not in protected and derived.is_file():
            try:
                derived.unlink(missing_ok=True)
            except OSError:
                # Fingerprinted analysis caches also reject these stale bytes.
                pass


def _install_candidate(attempt: Path, path: Path, metadata: dict) -> None:
    """Install a complete media/metadata pair with rollback of an earlier copy."""
    candidate_record = attempt.with_suffix(".json")
    save_json(candidate_record, metadata)
    previous = path.with_name(path.stem + ".quality-previous.mp4")
    previous_record = previous.with_suffix(".json")
    _clean_candidate(previous)
    had_media, had_record = path.exists(), path.with_suffix(".json").exists()
    media_backed = record_backed = new_media = new_record = False
    try:
        if had_media:
            path.replace(previous)
            media_backed = True
        if had_record:
            path.with_suffix(".json").replace(previous_record)
            record_backed = True
        attempt.replace(path)
        new_media = True
        candidate_record.replace(path.with_suffix(".json"))
        new_record = True
    except OSError:
        # Never remove an untouched incumbent when an earlier rename failed.
        if new_media:
            path.unlink(missing_ok=True)
        if new_record:
            path.with_suffix(".json").unlink(missing_ok=True)
        if media_backed:
            previous.replace(path)
        if record_backed:
            previous_record.replace(path.with_suffix(".json"))
        raise
    else:
        try:
            _clean_candidate(previous)
        except OSError:
            pass
        _invalidate_source_analyses(path)


def download(film: dict, root: Path) -> Path:
    import yt_dlp
    root.mkdir(parents=True, exist_ok=True)
    height = export_size()[1]
    path = root / f"{film['kind']}-{film['id']}-{height}p.mp4"
    cached = _valid_cached(path)
    incumbent = _cached_record(path) if cached else {}
    stills_cached = bool(incumbent.get("stills_fallback"))
    cached_quality = effective_height(incumbent.get("source_width"), incumbent.get("source_height"))
    # Verify the bytes, not a stale sidecar's advertised resolution.
    if cached and not stills_cached:
        try:
            stream = next((item for item in probe(path).get("streams", [])
                           if item.get("codec_type") == "video"), {})
            actual_quality = effective_height(stream.get("width"), stream.get("height"))
            if actual_quality:
                cached_quality = actual_quality
            else:
                cached_quality = 0
        except (PipelineError, OSError, ValueError, TypeError, AttributeError):
            # A failed optional quality probe does not destroy a readable incumbent.
            cached_quality = 0
    run_id = os.getenv("GITHUB_RUN_ID", "").strip()
    run_attempt = os.getenv("GITHUB_RUN_ATTEMPT", "1").strip() or "1"
    refresh_key = (str(path.resolve()), run_id, run_attempt)
    if cached:
        if not stills_cached and cached_quality >= min(1080, height):
            return path
        if (refresh_key in _LOW_QUALITY_REFRESHES
                or run_id and incumbent.get("quality_refresh_run") == run_id
                and incumbent.get("quality_refresh_attempt", "1") == run_attempt):
            return path
        _LOW_QUALITY_REFRESHES.add(refresh_key)
        # Persist the attempt marker before network work, so a warm-up's limited
        # result is not downloaded again by the producer in the same Actions run.
        _record_quality_attempt(path, "attempting_hd_refresh")
    try:
        require_disk(root)
    except PipelineError:
        if cached:
            _record_quality_attempt(path, "retained_disk_reserve")
            return path
        raise
    target = path.with_name(path.stem + ".quality-refresh.mp4") if cached else path
    if cached:
        _clean_candidate(target)
    base = {"quiet": True, "no_warnings": True, "noplaylist": True, "socket_timeout": 30,
            "retries": 3, "fragment_retries": 3, "js_runtimes": {"node": {}}}
    state = {"attempts": 0, "best_quality": 0.0, "winning_route": None}
    last_error: Exception | None = None
    for route in download_access.routes():
        if state["attempts"] >= MAX_QUALITY_DOWNLOADS:
            break
        state["route"] = route
        try:
            _download_route(film, root, target, height, download_access.options(base, route), yt_dlp,
                            state=state)
        except _RouteBlocked:
            if route == "direct" and download_access.fallback_proxy():
                print(f"YouTube bot check on the direct route for {film['title']}; retrying through the fallback route.", flush=True)
            continue
        except PipelineError as exc:
            last_error = exc
            continue
        # A readable SD copy does not end route discovery while the same shared,
        # bounded three-download allowance can still find an actual HD copy.
        if state["best_quality"] >= min(1080, height):
            break
    selected = target.exists() and state["best_quality"] > 0
    if selected and state.get("winning_route"):
        download_access.record_route_success(state["winning_route"])
    if cached:
        try:
            if selected and (stills_cached or state["best_quality"] > cached_quality + 1):
                metadata = _cached_record(target)
                _install_candidate(target, path, metadata)
                _record_quality_attempt(path, "upgraded")
                print(f"Upgraded cached footage for {film['title']} from {cached_quality:.0f}p to {state['best_quality']:.0f}p.", flush=True)
            else:
                _record_quality_attempt(path, "retained_no_better_source")
        except (OSError, PipelineError, ValueError):
            if not _valid_cached(path):
                raise
            _record_quality_attempt(path, "retained_upgrade_install_failed")
            print(f"Optional HD refresh could not be installed for {film['title']}; retaining its verified source.", flush=True)
        finally:
            _clean_candidate(target)
        return path
    if selected:
        if state["best_quality"] < min(1080, height):
            _LOW_QUALITY_REFRESHES.add(refresh_key)
            _record_quality_attempt(path, "limited_source_available")
        return path
    return _stills_or_raise(film, root, path, stills_cached, last_error or _blocked())

def _stills_or_raise(film: dict, root: Path, path: Path, stills_cached: bool, error: Exception) -> Path:
    """When no trailer can be downloaded, use the film's TMDB images instead of stopping the video."""
    if stills_cached and path.exists():
        return path
    if os.getenv("STILLS_FALLBACK", "true").strip().lower() == "false":
        raise error
    from .core import notice
    from .stills_trailer import build
    try:
        built = build(film, path, root)
    except Exception as exc:  # noqa: BLE001 -- report the original download failure
        print(f"Stills fallback unavailable for {film.get('title')} ({type(exc).__name__}: {exc}).", flush=True)
        raise error from exc
    notice("No trailer: using stills", f"{film.get('title')} ({film.get('year')}): no trailer could be downloaded "
           f"({type(error).__name__}); its section uses official TMDB images. Consider re-running or swapping this film.", "warning")
    return built


def _download_route(film: dict, root: Path, path: Path, height: int, common: dict, yt_dlp, *, state: dict | None = None) -> Path:
    state = state if state is not None else {"attempts": 0, "best_quality": 0.0, "winning_route": None}
    candidates = []
    for trailer in film["trailers"][:MAX_CANDIDATES]:
        url = trailer_url(trailer)
        if url and (url, True) not in candidates:
            candidates.append((url, True))
    # YouTube sources first; Vimeo remains available when YouTube refuses the runner.
    candidates.sort(key=lambda entry: not _is_youtube(entry[0]))
    errors = []
    ranked = []
    blocked = False
    seen = set()

    def inspect(entries):
        nonlocal blocked
        for url, curated in entries:
            if url in seen or len(seen) >= MAX_CANDIDATES or (blocked and _is_youtube(url)):
                continue
            seen.add(url)
            try:
                with yt_dlp.YoutubeDL(common) as ydl:
                    info = ydl.extract_info(url, download=False)
                if info and candidate_ok(info, film, curated):
                    ranked.append((advertised_quality(info, height), len(ranked), url, curated, info))
            except (yt_dlp.utils.DownloadError, PipelineError, OSError, ValueError) as exc:
                errors.append(type(exc).__name__)
                if _bot_check(exc):
                    blocked = True

    inspect(candidates)
    # If curated choices are missing or low quality, spend remaining metadata slots
    # on verified official search matches. Search results never bypass title checks.
    from .editorial_policy import search_queries
    for query in search_queries(film, primary=True):
        if blocked or len(seen) >= MAX_CANDIDATES or max((r[0] for r in ranked), default=0) >= height:
            break
        try:
            with yt_dlp.YoutubeDL({**common, "extract_flat": True}) as ydl:
                result = ydl.extract_info(f"ytsearch4:{query}", download=False)
            discovered = [(f"https://www.youtube.com/watch?v={entry['id']}", False)
                          for entry in (result or {}).get("entries", []) if entry and re.fullmatch(r"[\w-]{11}", entry.get("id", ""))]
            inspect(discovered)
        except (yt_dlp.utils.DownloadError, PipelineError, OSError, ValueError) as exc:
            errors.append(type(exc).__name__)
            blocked = blocked or _bot_check(exc)
    if blocked and not ranked:
        raise _RouteBlocked()
    ranked.sort(key=lambda row: (-row[0], row[1]))
    best_quality = state["best_quality"]
    selected = path.exists() and best_quality > 0
    for advertised, order, url, curated, info in ranked:
        if state["attempts"] >= MAX_QUALITY_DOWNLOADS:
            break
        if selected and (best_quality >= height or (advertised and advertised <= best_quality)):
            continue
        if blocked and _is_youtube(url):
            continue
        attempt = root / f"{path.stem}.candidate-{order:02}.mp4"
        _clean_candidate(attempt)
        try:
            options = {**common, "format": f"bv[height<={height}]+ba/b[height<={height}]",
                       "format_sort": ["res", "vcodec:h264"], "merge_output_format": "mp4",
                       "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
                       "outtmpl": str(attempt.with_suffix(".%(ext)s")), "overwrites": True,
                       "max_filesize": MAX_TRAILER_BYTES}
            require_disk(root)
            state["attempts"] += 1
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])
            if not attempt.exists() or not 25 <= duration(attempt) <= 360:
                raise PipelineError("Downloaded trailer failed file/duration checks.")
            if attempt.stat().st_size > MAX_TRAILER_BYTES:
                raise PipelineError("Trailer exceeds the 800 MB per-film storage budget.")
            metadata = probe(attempt)
            video = next((stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "video"), {})
            actual = effective_height(video.get("width"), video.get("height"))
            if not actual:
                raise PipelineError("Downloaded trailer has no valid video dimensions.")
            if actual > best_quality:
                _install_candidate(attempt, path, {"source": url, "title": info.get("title"), "channel": info.get("channel"),
                          "curated": curated, "requested_height": height, "source_width": video.get("width"),
                          "source_height": video.get("height"), "quality_selection_version": 2,
                          "film_kind": film.get("kind"), "film_id": film.get("id"),
                          "download_route": state.get("route")})
                best_quality = state["best_quality"] = actual
                state["winning_route"] = state.get("route")
                selected = True
        except (yt_dlp.utils.DownloadError, PipelineError, OSError, ValueError) as exc:
            errors.append(type(exc).__name__)
            if _bot_check(exc):
                blocked = True
                if selected:
                    break
        finally:
            _clean_candidate(attempt)
    if selected:
        print(f"Selected trailer provides about {best_quality:.0f}p of landscape detail; final export quality will be chosen after all trailers are checked.", flush=True)
        return path
    path.unlink(missing_ok=True)
    path.with_suffix(".json").unlink(missing_ok=True)
    if blocked:
        raise _RouteBlocked()
    raise PipelineError(f"No usable trailer for {film['title']} ({film['year']}). Tried {len(seen)} candidates. YouTube access restrictions, unavailable trailers or format changes may be responsible. No movie was substituted. Failures: {', '.join(errors)}")
