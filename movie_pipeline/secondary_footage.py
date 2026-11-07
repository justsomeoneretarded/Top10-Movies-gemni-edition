"""Best-effort extra trailers: different imagery, same title, and no quality downgrade."""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from .core import PipelineError, duration, normalized, probe, save_json
from . import download_access
from .footage import MAX_TRAILER_BYTES, _bot_check, _clean_candidate
from .video_settings import export_size, require_disk

REJECT = re.compile(r"\b(concept|ai[ -]?generated|reaction|review|explained|deepfake|full (?:movie|episode|film)|"
                    r"behind the scenes|making of|interview|breakdown|recap|ending|easter eggs?|ranked|ranking|top \d+|"
                    r"compilation|parody|fan theory|edit audio|slowed|reverb)\b", re.IGNORECASE)
# Words that mark official footage: trailers and teasers, and single scenes or clips.
FOOTAGE_WORDS = {"trailer", "teaser", "clip", "clips", "scene", "scenes", "preview", "sneak", "peek", "promo"}
# What a scene search asks for, per genre (an action list wants chases and fights, a comedy funny scenes).
SCENE_SEARCH = {"action": "action scene", "adventure": "action scene", "thriller": "scene", "scifi": "scene",
                "fantasy": "battle scene", "horror": "scary scene", "comedy": "funny scene", "romance": "scene",
                "family": "scene", "drama": "scene", "documentary": "clip"}
FAN = re.compile(r"\b(?:fan[ -]?(?:made|edit)|unofficial)\b", re.IGNORECASE)
MAX_METADATA = 12
MAX_DOWNLOADS = 3


def secondary_candidate_ok(info: dict, film: dict, curated: bool = False) -> bool:
    title = info.get("title", "")
    if not isinstance(title, str) or REJECT.search(title) or info.get("is_live"):
        return False
    length = info.get("duration") or 0
    # Clips are often 20-60 s; anything shorter is a logo, promo or fragment. Over 6 minutes is not a clip.
    if not isinstance(length, (int, float)) or not 20 <= length <= 360:
        return False
    years = set(re.findall(r"\b(?:18|19|20)\d{2}\b", title))
    years -= set(re.findall(r"\b(?:18|19|20)\d{2}\b", film["title"]))
    if film.get("kind") != "tv" and years and str(film["year"]) not in years:
        return False
    requested = film.get("requested_season")
    if isinstance(requested, dict):
        requested=requested.get("season_number")
    labelled=re.findall(r"\bseason\s+(\d+)\b|\bs(\d{1,2})e\d+", title, re.I)
    if str(requested).isdigit() and labelled and all(int(a or b) != int(requested) for a,b in labelled):
        return False
    if curated:
        return True
    expected = set(normalized(film["title"]).split())
    found = set(normalized(title).split())
    overlap = len(expected & found)/max(1, len(expected))
    if overlap < .85 or not found & FOOTAGE_WORDS:  # a scene clip need not say 'trailer'
        return False
    from .footage import identity_ok
    if not identity_ok(info, film):
        return False  # same name, different title (a remake or another country's version)
    return True  # A genuine matching scene need not come from a verified uploader.


def _hashes(path: Path) -> list[int]:
    """Sample useful frames and compute tiny perceptual difference fingerprints."""
    length = duration(path)
    capture = cv2.VideoCapture(str(path))
    hashes = []
    try:
        for seconds in np.linspace(length*.10, length*.85, 18):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(seconds)*1000)
            ok, image = capture.read()
            if not ok:
                continue
            gray = cv2.cvtColor(cv2.resize(image, (320,180)), cv2.COLOR_BGR2GRAY)
            if float(gray.std()) < 8 or float(gray.mean()) < 12:
                continue
            from .shots import content_core  # letterboxed and full-frame copies of a shot must match
            small = cv2.resize(content_core(gray), (9,8), interpolation=cv2.INTER_AREA)
            bits = (small[:,1:] > small[:,:-1]).reshape(-1)
            value = sum(int(bit) << index for index, bit in enumerate(bits))
            # Count a held/repeated scene once, rather than treating every frame as new.
            if not any((value ^ old).bit_count() <= 4 for old in hashes):
                hashes.append(value)
    finally:
        capture.release()
    return hashes


def novelty_from_hashes(primary: list[int], candidate: list[int]) -> dict | None:
    if len(primary) < 4 or len(candidate) < 3:
        return None
    novel = sum(not any((value ^ old).bit_count() <= 8 for old in primary)
                for value in candidate)
    fraction = novel/len(candidate)
    # A single-scene clip has few distinct frames (3 is enough); it counts as new only if every one is new.
    # At least half of an extra trailer must be new footage; a recut of the same shots adds only repeats.
    distinct = (novel >= 3 and fraction >= .5) if len(candidate) >= 4 else novel == len(candidate)
    return {"novel_frames": novel, "sampled_frames": len(candidate), "novel_fraction": fraction, "distinct": distinct}


def scene_novelty(primary: Path, candidate: Path) -> dict | None:
    """A visual diversity heuristic; this does not verify actor or movie identity."""
    return novelty_from_hashes(_hashes(primary), _hashes(candidate))


def _quality(info: dict, minimum_size: tuple[int,int], cap: int) -> float | None:
    width, height = minimum_size
    formats = info.get("formats")
    rows = formats if isinstance(formats, list) else [info]
    known = False
    choices = []
    for row in rows:
        if not isinstance(row, dict) or row.get("vcodec") == "none":
            continue
        w, h = row.get("width"), row.get("height")
        if not isinstance(w, (int,float)) or not isinstance(h, (int,float)) or not math.isfinite(w) or not math.isfinite(h):
            continue
        known = True
        size = row.get("filesize") or row.get("filesize_approx")
        if w >= width and height <= h <= cap and not (isinstance(size,(int,float)) and size > MAX_TRAILER_BYTES):
            choices.append(min(h,w*9/16))
    return max(choices) if choices else (None if known else 0)


def _primary_id(source: str) -> str:
    parsed = urlsplit(source)
    if parsed.hostname in {"youtube.com", "www.youtube.com"}:
        return parse_qs(parsed.query).get("v", [""])[0]
    if parsed.hostname == "youtu.be":
        return parsed.path.strip("/")
    return ""


def _valid_cache(path: Path, primary_source: str, minimum_size: tuple[int,int]) -> bool:
    try:
        record = json.loads(path.with_suffix('.json').read_text())
        return (record.get('secondary_version') == 1 and record.get('primary_source') == primary_source
                and record.get('novelty',{}).get('distinct') is True
                and record.get('source_width',0) >= minimum_size[0] and record.get('source_height',0) >= minimum_size[1]
                and path.exists() and 0 < path.stat().st_size <= MAX_TRAILER_BYTES and 20 <= duration(path) <= 360)
    except (OSError, ValueError, TypeError, AttributeError, PipelineError):
        return False


def download_secondary(film: dict, root: Path, primary: Path, minimum_size: tuple[int,int],
                       *, existing: list[Path] | None = None, slot: int = 2) -> Path | None:
    """Fetch an optional slot from two to five, excluding accepted sources."""
    try:
        return _download_secondary(film, root, primary, minimum_size, existing=existing, slot=slot)
    except Exception:  # noqa: BLE001 -- optional enrichment must never discard the primary/video.
        print(f"No additional usable trailer for {film.get('title', 'this title')}; keeping the accepted trailers.", flush=True)
        return None


from .footage import search_query  # noqa: E402


def _download_secondary(film: dict, root: Path, primary: Path, minimum_size: tuple[int,int],
                        *, existing: list[Path] | None = None, slot: int = 2) -> Path | None:
    import yt_dlp
    from .resource_budget import reserve_source, release_source
    if slot not in (2, 3, 4, 5):
        return None
    existing = [Path(p) for p in (existing or []) if Path(p) != primary]
    accepted_sources = [json.loads(p.with_suffix(".json").read_text()).get("source", "") for p in existing]
    record = json.loads(primary.with_suffix('.json').read_text())
    if record.get('stills_fallback'):
        return None  # no trailer was reachable for this film; do not search again
    source = record.get('source')
    if not isinstance(source,str) or not source:
        return None
    root.mkdir(parents=True,exist_ok=True)
    suffix = "extra" if slot == 2 else f"extra-{slot}"
    target = root / f'{primary.stem}-{suffix}.mp4'
    if _valid_cache(target,source,minimum_size):
        cached = json.loads(target.with_suffix(".json").read_text())
        if cached.get("existing_sources", []) == accepted_sources and cached.get("source") not in accepted_sources:
            if reserve_source(root,cached.get("source",""),target):
                return target
            return None
    _clean_candidate(target)
    require_disk(root)
    cap = export_size()[1]
    # Optional extra footage uses whichever route already worked for primary trailers.
    common = download_access.options({'quiet':True,'no_warnings':True,'noplaylist':True,'socket_timeout':12,'retries':1,
                                      'fragment_retries':1,'js_runtimes':{'node':{}}}, download_access.routes()[0])
    seen = {_primary_id(source), *(_primary_id(url) for url in accepted_sources)}
    excluded_count = len(seen)
    ranked = []
    blocked = False

    def inspect(entries):
        nonlocal blocked
        for identifier,curated in entries:
            if identifier in seen or len(seen)-excluded_count >= MAX_METADATA or not re.fullmatch(r'[\w-]{11}', identifier):
                continue
            seen.add(identifier)
            url = f'https://www.youtube.com/watch?v={identifier}'
            try:
                with yt_dlp.YoutubeDL(common) as ydl:
                    info = ydl.extract_info(url,download=False)
                if info and secondary_candidate_ok(info,film,curated):
                    quality = _quality(info,minimum_size,cap)
                    if quality is not None:
                        ranked.append((quality,len(ranked),url,curated,info))
            except (yt_dlp.utils.DownloadError,PipelineError,OSError,ValueError) as exc:
                # Primary footage already exists. A secondary bot check is never fatal.
                if _bot_check(exc):
                    blocked = True
                    break
                continue

    # Official TMDB trailers, teasers and clips (trusted: the studio's own uploads) come before any search.
    # Clips first: a single studio scene is far more likely to add new footage than another trailer cut.
    curated = [*film.get('clips', []), *film.get('trailers', [])]
    inspect([(video.get('key',''),True) for video in curated[:min(4,MAX_METADATA)] if video.get('site','YouTube') == 'YouTube'])
    # Reserve actual metadata capacity for fan edits when curated discovery has few options.
    # Then searches: scenes that fit the list's genre (chases and fights for action, funny scenes for
    # comedy), then other trailers. Searched results need not use verified uploaders but must name the
    # right title, year and cast (secondary_candidate_ok).
    from .editorial_policy import search_queries
    for query in search_queries(film):
        if blocked or len(seen)-excluded_count >= MAX_METADATA:
            break
        try:
            with yt_dlp.YoutubeDL({**common,'extract_flat':True}) as ydl:
                result = ydl.extract_info(f"ytsearch4:{query}", download=False)
            inspect([(entry.get('id',''),False) for entry in (result or {}).get('entries',[]) if entry][:2])
        except (yt_dlp.utils.DownloadError,PipelineError,OSError,ValueError):
            blocked = True
    # Prefer a useful single scene, then quality; uploader status is not a gate.
    ranked.sort(key=lambda row:(not (set(normalized(str(row[4].get('title',''))).split()) & {"clip","clips","scene","scenes"}),
                                -row[0], row[1]))
    for _quality_value,order,url,curated,info in ranked[:MAX_DOWNLOADS]:
        attempt = root / f'{primary.stem}-{suffix}-candidate-{order:02}.mp4'
        _clean_candidate(attempt)
        if not reserve_source(root,url,target):
            continue
        retained=False
        try:
            options = {**common,'format':f'bv[height<={cap}]+ba/b[height<={cap}]',
                       'format_sort':['res','vcodec:h264'],'merge_output_format':'mp4',
                       'postprocessors':[{'key':'FFmpegVideoRemuxer','preferedformat':'mp4'}],
                       'outtmpl':str(attempt.with_suffix('.%(ext)s')),'overwrites':True,'max_filesize':MAX_TRAILER_BYTES}
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])
            if not attempt.exists() or not 20 <= duration(attempt) <= 360 or attempt.stat().st_size > MAX_TRAILER_BYTES:
                continue
            metadata = probe(attempt)
            video = next((stream for stream in metadata.get('streams',[]) if stream.get('codec_type') == 'video'),{})
            width,height = video.get('width',0),video.get('height',0)
            if width < minimum_size[0] or height < minimum_size[1]:
                continue
            if existing:
                # The third trailer must add scenes beyond the union of both
                # accepted trailers, not merely differ from each one separately.
                hashes = [value for path in [primary, *existing] for value in _hashes(path)]
                novelty = novelty_from_hashes(hashes, _hashes(attempt))
            else:
                novelty = scene_novelty(primary,attempt)
            if not novelty or not novelty['distinct']:
                continue
            save_json(attempt.with_suffix('.json'), {'source':url,'primary_source':source,'title':info.get('title'),
                      'channel':info.get('channel'),'curated':curated,'fan_edit':bool(FAN.search(info.get('title',''))),
                      'source_width':width,'source_height':height,'secondary_version':1,'novelty':novelty,'existing_sources':accepted_sources,'slot':slot})
            attempt.replace(target)
            attempt.with_suffix('.json').replace(target.with_suffix('.json'))
            retained=True
            print(f"Added trailer {slot} for {film['title']} with different sampled imagery.",flush=True)
            return target
        except (yt_dlp.utils.DownloadError,PipelineError,OSError,ValueError) as exc:
            if _bot_check(exc):
                break
        finally:
            if not retained:
                release_source(root,url,target)
            _clean_candidate(attempt)
    if blocked or ranked:
        print(f"No distinct trailer {slot} accepted for {film['title']}; keeping the accepted trailers.",flush=True)
    return None
