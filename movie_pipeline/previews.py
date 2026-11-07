"""Continuous visual previews before narration, with conservative trailer audio.

CLIP relevance and motion are preferences only. A weak match never blocks a
usable preview; dialogue is muted whenever a clean audio window is uncertain.
"""
from __future__ import annotations

import math
from pathlib import Path

from .core import PipelineError
from .shots import (
    duration,
    frame_text,
    get_frame,
    matches_title_card,
    moving_limit,
)

FPS = 30


def _number(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _seconds(value: float) -> float:
    value = _number(value)
    if value <= 0:
        raise ValueError("Preview duration must be positive.")
    return max(1, round(value * FPS)) / FPS


def _relevance(selector, source: Path, text: str, count: int) -> list[float]:
    try:
        values = selector.relevance(source, text)
        return [max(-1.0, min(1.0, _number(values[i]))) for i in range(count)]
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
        return [0.0] * count


def _dialogue(source: Path, dialogue: dict[str, list] | None) -> list[tuple[float, float]] | None:
    if dialogue is None or str(source) not in dialogue:
        return None
    cues = dialogue[str(source)]
    if not isinstance(cues, list):
        return None
    result = []
    for cue in cues:
        if not isinstance(cue, dict):
            return None
        start = _number(cue.get("start"), -1)
        end = _number(cue.get("end"), -1)
        if start < 0 or end <= start:
            return None
        result.append((start, end))
    return result


def _overlap(start: float, seconds: float, cues: list[tuple[float, float]]) -> bool:
    # A little room around recognized speech avoids cutting into its first consonant.
    return any(start < end + .12 and start + seconds > beginning - .12 for beginning, end in cues)


def _freshness(selector, source: Path, start: float, seconds: float) -> float:
    try:
        value = float(selector.freshness(source,start,seconds))
    except (AttributeError,RuntimeError,TypeError,ValueError):
        value = 1.0
    try:
        visual = float(selector.visual_freshness(source,start,seconds))
        if math.isfinite(visual):
            value = min(value,visual)
    except (AttributeError,RuntimeError,TypeError,ValueError):
        pass
    return max(0.0,min(1.0,value)) if math.isfinite(value) else 1.0


def _unused(selector, source: Path, start: float, end: float) -> list[tuple[float,float]]:
    try:
        values = selector.unused_windows(source,start,end)
        return [(float(low),float(high)) for low,high in values] if isinstance(values,list) else []
    except (AttributeError,RuntimeError,TypeError,ValueError):
        return []


def _reserve(selector, source, start: float, seconds: float, kind: str):
    method = getattr(selector,"reserve",None)
    if callable(method):
        method(Path(source),start,seconds,kind)


def _card(selector, source: Path, index: int) -> bool:
    try:
        return selector.card_kind(source,index) in {"title","promo"}
    except (AttributeError,RuntimeError,TypeError,ValueError):
        return False


def _best_preview(selector, sources, shot_lookup, text, seconds, action, dialogue):
    if not sources:
        raise PipelineError("No footage source is available for a preview.")
    query = (("A gripping action highlight: a fight, chase, stunt or explosion. " if action else
              "A striking cinematic character or environment that establishes the mood. ") + text)
    candidates = []
    for source_index, value in enumerate(sources):
        source = Path(value)
        shots = shot_lookup.get(str(source), [])
        scores = _relevance(selector, source, query, len(shots))
        valid = []
        blocked = [(_number(shot.get("start")),_number(shot.get("end"))) for i,shot in enumerate(shots)
                   if isinstance(shot,dict) and _card(selector,source,i)]
        allowed = [i for i in range(len(shots)) if not _card(selector,source,i)] or list(range(len(shots)))
        for index, shot in enumerate(shots):
            if index not in allowed:
                continue
            if not isinstance(shot, dict):
                continue
            start = max(0.0, _number(shot.get("start"), -1))
            end = _number(shot.get("end"), -1)
            if end <= start:
                continue
            valid.append((index, shot, start, end))
        maximum_end = max((end for _, _, _, end in valid), default=seconds)
        speech = _dialogue(source, dialogue)
        for index, shot, start, end in valid:
            available = end - start
            full_window = available >= seconds
            margin = min(.08, max(0.0, (available - seconds) / 2))
            initial = start + margin
            # If every source shot is short, use one continuous trailer window rather
            # than synthesizing several cuts or freezing the preview. Keep it inside
            # the sampled source extent whenever that extent is long enough.
            if not full_window:
                initial = min(initial, max(0.0, maximum_end - seconds))
            starts = [initial]
            starts += [low for low,high in _unused(selector,source,start,end) if high-low >= seconds]
            if full_window and speech:
                starts += [last + .14 for _, last in speech if start <= last + .14 <= end - seconds]
                starts += [first - .14 for first, last in speech
                           if start <= first-.14 <= end-seconds and last+.14 <= first-.14+seconds]
            for candidate_start in dict.fromkeys(starts):
                still = any(candidate_start < stop and candidate_start+seconds > beginning for beginning,stop in blocked)
                if still:
                    # A continuous short-shot fallback must never run through a
                    # detected title/logo. Hold one clean scene frame instead.
                    candidate_start = (start+end)/2
                muted = speech is None or any(
                    (beginning-.12 < candidate_start < stop+.12)
                    or (beginning-.12 < candidate_start+seconds < stop+.12)
                    for beginning, stop in speech or [])
                muted = muted or still
                motion = max(0.0, min(60.0, _number(shot.get("motion")))) / 60
                score = scores[index] * 1.25
                # Scare/impact beats (sudden sound jumps with motion) make the best reveals and cold opens.
                weight = getattr(selector, "peak_weight", 0.6)
                score += _number(shot.get("peak")) * (weight if isinstance(weight, (int, float)) else 0.6)
                score += motion * .8 if action else -.18 * motion
                score += min(available, 8.0) * .025
                score += .16 if not muted else 0
                freshness = _freshness(selector,source,candidate_start,seconds)
                whole = _freshness(selector,source,start,available)
                tier = 3 if freshness >= .999 and whole >= .999 else (2 if freshness >= .999 else (1 if freshness > .2 else 0))
                try:
                    score -= float(selector.reuse_penalty(source,candidate_start,seconds))
                except (AttributeError,RuntimeError,TypeError,ValueError):
                    pass
                try:
                    score -= float(selector.visual_reuse_penalty(source,candidate_start,seconds))
                except (AttributeError,RuntimeError,TypeError,ValueError):
                    pass
                result = {"source":str(source),"start":candidate_start,"duration":seconds,"muted":muted}
                if still:
                    result["still"] = True
                candidates.append((tier,full_window and not still, score, -source_index, -candidate_start,result))
    if not candidates:
        # Analysis may be incomplete even though the source itself is usable.
        # A quiet continuous opening remains preferable to aborting production.
        return 0.0, {"source": str(sources[0]), "start": 0.0,
                     "duration": seconds, "muted": True}
    tier,full,score,_source,_start,result = max(candidates,key=lambda value:value[:5])
    return score + (.5 if full else 0) + tier*4, result


def choose_preview(selector, sources: list[Path], shot_lookup: dict[str, list], text: str,
                   seconds: float, action: bool, dialogue: dict[str, list] | None = None) -> dict:
    """Choose one uninterrupted source interval before a movie's narration.

    Known speech-free intervals can retain trailer sound. Recognized speech,
    malformed transcripts and untranscribed extra trailers are muted; the final
    music bed fills those intervals without beginning halfway through dialogue.
    """
    result = _best_preview(selector,sources,shot_lookup,text,_seconds(seconds),action,dialogue)[1]
    _reserve(selector,result["source"],result["start"],1/FPS if result.get("still") else result["duration"],"review-preview")
    return result


def choose_title_card(selector, sources: list[Path], shot_lookup: dict[str, list], title: str,
                      seconds: float = 1.5) -> dict | None:
    """Find the actual title lettering for the end of this title's own review.

    Studio/platform logos never qualify. OCR is optional and limited to likely
    card images; an unavailable tool or an unreadable title simply skips this
    embellishment. Holding the verified frame avoids crossing a trailer cut.
    """
    seconds = _seconds(seconds)
    for value in sources:
        source = Path(value)
        cards = getattr(selector,"cards",{})
        texts = getattr(selector,"card_texts",{})
        kinds = cards.get(str(source),[]) if isinstance(cards,dict) else []
        known_texts = texts.get(str(source),[]) if isinstance(texts,dict) else []
        for index,shot in enumerate(shot_lookup.get(str(source),[])):
            if index >= len(kinds) or kinds[index] != "title":
                continue
            text = known_texts[index] if index < len(known_texts) else ""
            point = (_number(shot.get("start"))+_number(shot.get("end")))/2
            if not text:
                try:
                    text = frame_text(get_frame(source,point))
                except (PipelineError,OSError,RuntimeError,ValueError):
                    continue
            if matches_title_card(text,title):
                _reserve(selector,source,_number(shot.get("start")),
                         _number(shot.get("end"))-_number(shot.get("start")),"review-title-card")
                return {"source":str(source),"start":point,"duration":seconds,"muted":True,"still":True}

    # Normal shot indexing intentionally skips end slates. Sample at most four
    # frames in the final fifteen seconds; do not add these promos to shot pools.
    for value in sources:
        source = Path(value)
        try:
            length = duration(source)
        except (PipelineError,OSError,RuntimeError,ValueError):
            continue
        if not math.isfinite(length) or length <= 0:
            continue
        low,high = max(0,length-min(15,length*.15)),max(0,length-.3)
        points = [low+(high-low)*fraction for fraction in (.1,.35,.6,.85)]
        for point in points:
            try:
                image = get_frame(source,point)
                gray = image.convert("L").resize((160,90))
                histogram = gray.histogram()
                pixels = sum(histogram)
                # A dark slate with sparse bright lettering is much safer than
                # running OCR over actor subtitles or end-credit sequences.
                if sum(histogram[:60])/pixels <= .55 or not .01 < sum(histogram[141:])/pixels < .4:
                    continue
                if not matches_title_card(frame_text(image),title):
                    continue
            except (PipelineError,OSError,RuntimeError,TypeError,ValueError):
                continue
            _reserve(selector,source,max(0,point-seconds/2),seconds,"review-title-card")
            return {"source":str(source),"start":point,"duration":seconds,"muted":True,"still":True}
    return None


def cold_open_plan(selector, pools: list[list[Path]], shot_lookup: dict[str, list], text: str,
                   action: bool, dialogue: dict[str, list] | None = None, seconds: float = 5) -> list[dict]:
    """Open on two strong, different titles in five seconds before any voiceover.

    Shorter requests use one continuous image instead of creating tiny flashes.
    Frame-based division keeps the returned duration exactly on the export grid.
    """
    total = _seconds(seconds)
    count = min(2, sum(bool(pool) for pool in pools)) if total >= 3 else 1
    if not any(pools):
        raise PipelineError("No footage is available for the opening preview.")
    frames = round(total * FPS)
    first_frames = math.ceil(frames / count)
    # All films are compared at the same duration so rankings remain comparable.
    choices = []
    for film_index, pool in enumerate(pools):
        if pool:
            score, shot = _best_preview(selector, pool, shot_lookup, text, first_frames/FPS, action, dialogue)
            choices.append((score, -film_index, film_index, shot))
    choices.sort(key=lambda value: value[:2], reverse=True)
    result = []
    remaining = frames
    for _, _, film_index, shot in choices[:count]:
        length = min(first_frames, remaining)
        result.append({**shot, "duration": length/FPS, "film_index": film_index})
        _reserve(selector,shot["source"],shot["start"],1/FPS if shot.get("still") else length/FPS,"cold-open")
        remaining -= length
    return result


SIZZLE_GENRES = {"action", "adventure", "scifi", "fantasy", "thriller"}


def sizzle_plan(selector, pools: list[list[Path]], shot_lookup: dict[str, list], seconds: float = 25.0,
                shot_seconds: float = 2.6) -> list[dict]:
    """An opening reel of the list's biggest moments (about 25 seconds) before any voiceover.

    For action-type lists: the strongest moment of every title first, then each title's next
    best, in about 2.6-second hits with the trailers' own sound under the music. Text cards
    are skipped and every window is reserved so the reviews don't replay it.
    """
    # Rhythm: about 3 s per hit, quickening slightly (never under 2.2 s); the last hit absorbs the rest.
    total_frames = round(seconds * FPS)
    lengths, used = [], 0
    while used < total_frames:
        progress = used / max(1, total_frames)
        frames = round(FPS * (shot_seconds + 0.4 - 0.6 * progress))  # steady, slightly quickening
        frames = max(round(2.2 * FPS), frames)
        if total_frames - used - frames < round(2.2 * FPS):
            frames = total_frames - used  # the final hit absorbs the remainder
        lengths.append(frames)
        used += frames
    frames_each = max(lengths)
    count = len(lengths)
    ranked = []
    for film_index, pool in enumerate(pools):
        if not pool:
            continue
        path = pool[0]
        options = []
        for index, shot in enumerate(shot_lookup.get(str(path), [])):
            kind = getattr(selector, "card_kind", None)
            if callable(kind) and kind(path, index) is not None:
                continue
            start, end = float(shot["start"]), float(shot["end"])
            if end - start < shot_seconds * 0.8:
                continue
            centre = float(shot.get("punch_at", (start + end) / 2))
            begin = max(start, min(centre - shot_seconds * 0.4, end - shot_seconds))
            options.append((float(shot.get("peak", 0)) * 2 + min(float(shot.get("motion", 0)), 60) / 60, begin))
        options.sort(reverse=True)
        if options:
            ranked.append((options[0][0], film_index, path, options))
    if not ranked:
        return []
    ranked.sort(key=lambda row: -row[0])  # the title with the single best moment opens the reel
    plan, depth = [], 0
    while len(plan) < count and any(depth < len(row[3]) for row in ranked):
        for _, film_index, path, options in ranked:
            if depth < len(options) and len(plan) < count:
                begin = options[depth][1]
                length = lengths[len(plan)] / FPS
                # Muted: the music carries the reel. Each clip's own sound, levelled separately and
                # cut every few seconds, made the intro jump up and down harshly.
                plan.append({"source": str(path), "start": begin, "duration": length, "muted": True,
                             "still": False, "film_index": film_index})
                _reserve(selector, str(path), begin, length, "cold-open")
        depth += 1
    return plan


def outro_recap_plan(selector, pools: list[list[Path]], shot_lookup: dict[str, list],
                     paragraph_seconds: list[float], text: str, action: bool = False) -> list[list[dict]]:
    """Revisit at most two titles in calm beats no longer than twelve seconds.

    Paragraph boundaries advance within the current shot rather than restart
    it. Longer outros use more fresh windows, not extended motion or a regular
    play/freeze pattern. The final three-second ending hold is added separately.

    ``action`` is accepted for call-site consistency, but the outro deliberately
    favors stable, lower-motion imagery even for an action countdown.
    """
    paragraph_frames = [max(0, math.ceil(_number(seconds) * FPS)) for seconds in paragraph_seconds]
    total_frames = sum(paragraph_frames)
    if not total_frames:
        return [[] for _ in paragraph_frames]
    available = [(index, pool) for index, pool in enumerate(pools) if pool]
    if not available:
        raise PipelineError("No footage is available for the closing recap.")
    count = 2 if len(available) >= 2 and total_frames >= 14 * FPS else 1
    # The recap is a montage across titles: it keeps shorter (12 s) excerpts than the reviews.
    beat_count = max(1, math.ceil(total_frames / (min(12.0, moving_limit(selector)) * FPS)))
    first_frames = math.ceil(total_frames / beat_count)
    sample_seconds = first_frames / FPS
    choices = []
    for film_index, pool in available:
        score, preview = _best_preview(selector, pool, shot_lookup, text, sample_seconds, False, None)
        choices.append((score, -film_index, film_index, preview))
    choices.sort(key=lambda value: value[:2], reverse=True)
    beats = []
    remaining = total_frames
    active = choices[:count]
    while remaining:
        _, _, film_index, _ = active[len(beats) % len(active)]
        frames = min(first_frames, remaining)
        pool = next(pool for index, pool in available if index == film_index)
        _, preview = _best_preview(selector, pool, shot_lookup, text, frames/FPS, False, None)
        if preview.get("still"):
            frames = min(frames, 4 * FPS)
        live_frames = 1 if preview.get("still") else frames
        beats.append({"source": preview["source"], "start": preview["start"],
                      "film_index": film_index, "frames": frames, "live_frames": live_frames})
        _reserve(selector,preview["source"],preview["start"],live_frames/FPS,"outro-recap")
        remaining -= frames
    plans = []
    beat_index = 0
    used = 0
    for frames in paragraph_frames:
        plan = []
        remaining = frames
        while remaining:
            beat = beats[beat_index]
            take = min(remaining, beat["frames"] - used)
            live = max(0, min(take, beat["live_frames"] - used))
            start_frame = min(used, beat["live_frames"] - 1)
            plan.append({"source": beat["source"], "start": beat["start"] + start_frame / FPS,
                         "duration": take / FPS, "freeze": (take - live) / FPS if live else 0,
                         "still": live == 0, "film_index": beat["film_index"],
                         "visual_reason": "calm-outro-recap"})
            remaining -= take
            used += take
            if used == beat["frames"]:
                beat_index += 1
                used = 0
        plans.append(plan)
    return plans
