"""Plan complete native-audio scene beats before rendering the narration.

Semantic scores propose candidates. Transcript boundaries and original scene windows
constrain them; missing analysis yields fewer beats, never a failed episode.
"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
import numpy as np
from .core import ffmpeg, key, save_json, probe
from . import gemini_editing

def _overlap(a,b):
    return a["start"] < b["start"]+b["duration"] and a["start"]+a["duration"] > b["start"]

def complete_window(candidate: dict, cues: list[dict], extent: tuple[float,float]) -> dict | None:
    """Grow to full recognized utterances instead of truncating their words."""
    start,end=candidate["start"],candidate["start"]+candidate["duration"]
    low,high=extent
    ordered=sorted(cues,key=lambda c:c.get("start",0))
    indices=[i for i,c in enumerate(ordered) if c.get("end",0)>start-.12 and c.get("start",high)<end+.12]
    relevant=[]
    terminal=lambda c: bool(re.search(r"[.!?][\"'’”)]?$",str(c.get("text","")).strip()))
    if indices:
        first,last=indices[0],indices[-1]
        # Whisper captions are word-aligned display chunks, not necessarily
        # whole sentences. Walk back/forward to a real utterance boundary.
        while first>0 and not terminal(ordered[first-1]) and ordered[first]["start"]-ordered[first-1]["end"]<.8:
            first-=1
        while last+1<len(ordered) and not terminal(ordered[last]) and ordered[last+1]["start"]-ordered[last]["end"]<.8:
            last+=1
        relevant=ordered[first:last+1]
        start=min(start,relevant[0]["start"]-.14)
        end=max(end,relevant[-1]["end"]+.20)
        if start<low+.04 or end>high-.04:
            return None
        if end-start>22 or not terminal(relevant[-1]):
            return None
    if not 3 <= end-start <= 22:
        return None
    return {**candidate,"start":max(low,start),"duration":end-max(low,start),
            "muted":False,"native_verified":True,
            "cues":[dict(c) for c in relevant]}

def _scores(selector,source,query,count):
    try:
        values=np.asarray(selector.identity_relevance(source,query),dtype=float)
        return values if len(values)==count else np.full(count,.25)
    except (AttributeError,RuntimeError,ValueError,KeyError,TypeError):
        return np.full(count,.25)

def candidates(selector,pool,shot_lookup,film):
    result=[]
    for source in pool:
        shots=shot_lookup.get(str(source),[])
        if not shots:
            continue
        visual_advice=gemini_editing.analyse_source(Path(source))
        meaningful=_scores(selector,source,"A complete cinematic moment: a confrontation, physical comedy, chase, fight, crash, danger, fire or an emotional exchange.",len(shots))
        energetic=_scores(selector,source,"People visibly fighting, running from danger, shooting, a vehicle crash or an explosion.",len(shots))
        quiet=_scores(selector,source,"A static quiet view, an empty room or a person standing still.",len(shots))
        relevant=_scores(selector,source,str(film.get("overview") or film["title"]),len(shots))
        features=getattr(selector,"features",{}).get(str(source))
        if features is not None:
            try:
                features=features.detach().cpu().numpy()
            except AttributeError:
                features=np.asarray(features)
        for i,shot in enumerate(shots):
            start=float(shot["start"])
            for last in range(i,min(i+3,len(shots))):
                indices=list(range(i,last+1))
                if any(getattr(selector,"card_kind",lambda *_:None)(source,j) is not None for j in indices):
                    break
                if last>i:
                    gap=float(shots[last]["start"])-float(shots[last-1]["end"])
                    # Joined shots must be genuinely adjacent with consistent
                    # visual context, not a jump to a new trailer location.
                    if gap>.55 or features is None or float(np.dot(features[last-1],features[last]))<.78:
                        break
                end=float(shots[last]["end"])
                length=end-start
                if length>22:
                    break
                if length<3:
                    continue
                fresh=getattr(selector,"freshness",lambda *_:1)(source,start,length)
                visual=getattr(selector,"visual_freshness",lambda *_:1)(source,start,length)
                if fresh<.98 or visual<.8:
                    continue
                strength=float(np.mean(energetic[indices]-quiet[indices]))
                score=float(np.mean(meaningful[indices]+relevant[indices]))+.10*fresh+.06*min(length/10,1)
                # URL analysis is approximate: only rank already coherent, fresh,
                # non-card local candidates. Native boundaries still pass through
                # _enrich/complete_window and _usable after selection.
                adjustment,notes=gemini_editing.soft_rank(visual_advice,start,end)
                advice=({"visual_analysis":{"provider":visual_advice["provider"],
                         "cache":visual_advice["cache"],"score_adjustment":adjustment,
                         "notes":notes}} if visual_advice else {})
                result.append({**advice,"source":str(source),"start":start,"duration":length,
                               "score":score+adjustment,"action_strength":strength,
                               "motion":float(np.mean([shots[j].get("motion",0) for j in indices])),
                               "film_index":None,"scene_index":i,"scene_end":end,
                               "extent":max(float(x["end"]) for x in shots),
                               "muted":True,"cues":[]})
    return sorted(result,key=lambda c:-c["score"])

def _enrich(candidate, dialogue, recognizer, work):
    source=Path(candidate["source"])
    try:
        if json.loads(source.with_suffix(".json").read_text()).get("stills_fallback"):
            return None  # a synthetic stills reel has no worthwhile original scene sound
    except (OSError,ValueError,TypeError):
        pass
    if not any(stream.get("codec_type") == "audio" for stream in probe(source).get("streams", [])):
        return None
    if str(source) in dialogue:
        cues=dialogue[str(source)]
        extent=(0.0,candidate["extent"])
    else:
        if recognizer is None:
            return None
        low=max(0,candidate["start"]-2)
        high=min(candidate["extent"], candidate["start"]+candidate["duration"]+2)
        token=key(source.name,source.stat().st_size,round(low,3),round(high,3),"native-v1")
        cache=work/f"{token}.json"
        if cache.exists():
            cues=json.loads(cache.read_text())
        else:
            from .voice import captions
            audio=work/f"{token}.wav"
            try:
                ffmpeg("-ss",low,"-i",source,"-t",high-low,"-vn","-ar",16000,"-ac",1,audio)
                local=captions(audio,recognizer)
                cues=[{**c,"start":c["start"]+low,"end":c["end"]+low} for c in local]
                save_json(cache,cues)
            finally:
                audio.unlink(missing_ok=True)
        extent=(low,high)
    return complete_window(candidate,cues,extent)

def _usable(selector,clip,shot_lookup):
    source=Path(clip["source"])
    if getattr(selector,"freshness",lambda *_:1)(source,clip["start"],clip["duration"])<.98:
        return False
    if getattr(selector,"visual_freshness",lambda *_:1)(source,clip["start"],clip["duration"])<.8:
        return False
    for i,shot in enumerate(shot_lookup.get(str(source),[])):
        if shot["start"]<clip["start"]+clip["duration"] and shot["end"]>clip["start"]:
            if getattr(selector,"card_kind",lambda *_:None)(source,i) is not None:
                return False
    return True

def _reserve(selector,clip):
    getattr(selector,"reserve",lambda *_:None)(Path(clip["source"]),clip["start"],clip["duration"],"native-scene")

def plan_episode(selector,pools,shot_lookup,films,sections,speech,dialogue,recognizer,work):
    """Return opening, natural rank previews, and optional review insertions.

    Image matching is required for this optional planner. The established preview
    path remains available to minimal or degraded selectors.
    """
    if not callable(getattr(selector,"identity_relevance",None)):
        return None
    from .core import duration
    work.mkdir(parents=True,exist_ok=True)
    options={i:candidates(selector,pool,shot_lookup,films[i]) for i,pool in enumerate(pools)}
    used=[]
    opening=[]
    # Extend the opening only when actual semantic action evidence exists.
    energetic=sorted([{**clip,"film_index":i} for i,rows in options.items() for clip in rows
                      if clip["action_strength"]>.025 and clip["motion"]>4],
                     key=lambda c:-(c["action_strength"]+c["score"]))
    for candidate in energetic:
        if len(opening)>=2 or sum(c["duration"] for c in opening)>=12:
            break
        if any(c["film_index"]==candidate["film_index"] for c in opening):
            continue
        try:
            clip=_enrich(candidate,dialogue,recognizer,work)
        except (OSError,ValueError,RuntimeError):
            clip=None
        if clip and _usable(selector,clip,shot_lookup) and sum(c["duration"] for c in opening)+clip["duration"]<=18:
            opening.append(clip);used.append(clip);_reserve(selector,clip)
    if not opening:
        all_clips=[{**clip,"film_index":i} for i,rows in options.items() for clip in rows]
        if all_clips:
            best=max(all_clips,key=lambda c:c["score"])
            # Quiet lists get a brief silent picture, not chopped movie dialogue.
            clip={**best,"duration":min(3.2,best["duration"]),"muted":True,"cues":[]}
            opening=[clip];used.append(clip);_reserve(selector,clip)
    previews,inserts={},{}
    for section_index,section in enumerate(sections):
        if section.get("kind")!="film":
            continue
        index=section["film"]
        # The title announcement precedes the preview; native inserts belong
        # only after complete review paragraphs, never between a name and its scene.
        body_positions=list(range(1 if section.get("rank_intro_version") else 0, len(section["paragraphs"])))
        paragraph_seconds=[duration(speech[(section_index,p)]) for p in body_positions]
        wanted=min(5,1+int(sum(paragraph_seconds)//30),len(paragraph_seconds)+1)
        chosen=[]
        examined=0
        for row in options.get(index,[]):
            if len(chosen)>=wanted or examined>=max(6,wanted*2):
                break
            candidate={**row,"film_index":index}
            if getattr(selector,"visual_freshness",lambda *_:1)(Path(candidate["source"]),candidate["start"],candidate["duration"]) < .8:
                continue
            if any(c["source"]==candidate["source"] and _overlap(c,candidate) for c in used):
                continue
            examined+=1
            try:
                clip=_enrich(candidate,dialogue,recognizer,work)
            except (OSError,ValueError,RuntimeError):
                clip=None
            if clip is None or not _usable(selector,clip,shot_lookup):
                continue
            if any(c["source"]==clip["source"] and _overlap(c,clip) for c in used):
                continue
            chosen.append(clip);used.append(clip);_reserve(selector,clip)
        if chosen:
            previews[section_index]=chosen[0]
            remaining=chosen[1:len(paragraph_seconds)+1]
            if remaining:
                # Spread pauses across complete paragraphs, including a useful scene
                # after the final review sentence when appropriate.
                positions=np.linspace(0,len(body_positions)-1,len(remaining),dtype=int)
                inserts[section_index]={body_positions[int(p)]:clip for p,clip in zip(positions,remaining)}
    return {"opening":opening,"previews":previews,"inserts":inserts,
            "report":{"opening_seconds":sum(c["duration"] for c in opening),
                      "visual_hint_sources":len({c["source"] for c in used if c.get("visual_analysis")}),
                      "visual_hint_timing_authority":"local_shots_and_complete_transcript",
                      "native_moments":sum(c.get("native_verified",False) and not c.get("muted",True) for c in used),"review_counts":{str(i):1+len(inserts.get(i,{})) for i in previews}}}
