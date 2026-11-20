"""Bounded production-quality orchestration; technical protections stay downstream."""
from __future__ import annotations
from copy import deepcopy
import json
import math
from pathlib import Path
from .core import PipelineError, save_json


def _section_identity(script):
    sections = script.get("sections", []) if isinstance(script, dict) else []
    if not isinstance(sections, list) or not sections:
        return None
    result = []
    for section in sections:
        if not isinstance(section, dict) or not isinstance(section.get("paragraphs"), list):
            return None
        if not section["paragraphs"] or not all(isinstance(p, str) and p.strip() for p in section["paragraphs"]):
            return None
        result.append((section.get("kind"), section.get("film")))
    return result


def repair_measured_depth(writer, title, brief, films, script, spec, estimated_seconds, trailer_seconds):
    """One caller-bounded targeted repair; no sentence is inserted solely for a clock."""
    receipt = {"phase": "measured", "estimated_seconds": round(estimated_seconds, 3),
               "target_seconds": spec.target, "status": "not_needed"}
    if not math.isfinite(estimated_seconds) or estimated_seconds >= spec.minimum:
        return script, False, receipt
    method = getattr(writer, "repair_depth", None)
    if not callable(method):
        receipt.update(status="unavailable", reason="writer_has_no_depth_repair")
        return script, False, receipt
    try:
        candidate = method(title, brief, films, deepcopy(script),
                           target_seconds=spec.target, estimated_seconds=estimated_seconds,
                           trailer_seconds=trailer_seconds, automatic=spec.automatic,
                           max_repairs=2)
        if _section_identity(candidate) != _section_identity(script) or _section_identity(candidate) is None:
            receipt.update(status="retained", reason="repair_changed_section_identity")
            return script, False, receipt
        for old, new in zip(script["sections"], candidate["sections"]):
            if old.get("kind") != "film" and old["paragraphs"] != new["paragraphs"]:
                receipt.update(status="retained", reason="repair_changed_non_review_speech")
                return script, False, receipt
            if (old.get("kind") == "film" and old["paragraphs"][0].startswith("Number ")
                    and old["paragraphs"][0] != new["paragraphs"][0]):
                receipt.update(status="retained", reason="repair_changed_rank_announcement")
                return script, False, receipt
        before = [section["paragraphs"] for section in script["sections"]]
        after = [section["paragraphs"] for section in candidate["sections"]]
        changed = before != after
        receipt.update(status="repaired" if changed else "retained",
                       reason="useful_depth_repair" if changed else "no_supported_expansion",
                       changed_film_indices=sorted({
                           old["film"] for old, new in zip(script["sections"], candidate["sections"])
                           if old.get("kind") == "film" and old["paragraphs"] != new["paragraphs"]}))
        return candidate if changed else script, changed, receipt
    except Exception as exc:
        receipt.update(status="unavailable", reason=type(exc).__name__)
        return script, False, receipt


def write_quality_receipts(work: Path, output: Path, script: dict, *, measured_repairs=None, visual_repairs=None):
    """Publish current-job evidence, never stale reports from restored job caches."""
    work, output = Path(work), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "production-quality-report.json", {
        "schema_version": 1, "script_depth": script.get("package", {}).get("depth_review", {}),
        "measured_repairs": measured_repairs or [], "visual_repairs": visual_repairs or {}})
    for name in ("quality-report.json", "duration-report.json", "audio-review.json",
                 "soundtrack-report.json", "native-scenes.json", "visual-repair-report.json",
                 "script-quality-report.json"):
        try:
            data = json.loads((work / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        save_json(output / name, data)
    paragraphs = [p for section in script.get("sections", []) for p in section.get("paragraphs", [])]
    (output / "final-script.txt").write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")


def fallback_pool_inputs(pools, trailers, shot_sets, shot_lookup, original_primary_paths, failed_sources,
                         *, affected_films=None):
    """One bounded render fallback, retaining title identity and original speech.

    A multi-source FFmpeg error does not identify every bad input. Prefer the
    already accepted original primary when optional inputs are implicated, and
    never move a source from another film into this film's pool.
    """
    failed = {str(path) for path in failed_sources}
    originals = {str(path) for path in original_primary_paths}
    implicated_extras = failed - originals
    excluded = implicated_extras or failed
    affected = {index for index, pool in enumerate(pools)
                if any(str(path) in failed for path in pool)}
    for index in affected_films or ():
        if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(pools):
            affected.add(index)
    changes = []
    for index in sorted(affected):
        pool = pools[index]
        remaining = [path for path in pool if str(path) not in excluded]
        if not remaining:
            raise PipelineError("No alternate verified source remains for the affected title.")
        original = original_primary_paths[index]
        chosen = original if original in remaining else remaining[0]
        if str(chosen) not in shot_lookup:
            raise PipelineError("Alternate source has no matching local shot analysis.")
        changes.append((index, chosen, [chosen, *[p for p in remaining if p != chosen]]))
    for index, chosen, remaining in changes:
        pools[index][:] = remaining
        trailers[index] = chosen
        shot_sets[index] = shot_lookup[str(chosen)]
    return bool(changes)


def choose_narrated_draft(previous, distance, script, speech, *, prefer_quality=False):
    """Adopt a complete substantive repair even when its clock is no closer.

    Missing takes still enter the existing same-speaker recovery for an initial
    draft, but must never overwrite an already complete narration.
    """
    expected = {(i, p) for i, section in enumerate(script["sections"])
                for p in range(len(section["paragraphs"]))}
    complete = set(speech) == expected and all(path is not None for path in speech.values())
    if previous is None:
        return (distance, script, speech), True
    previous_speech = previous[2]
    previous_complete = bool(previous_speech) and all(path is not None for path in previous_speech.values())
    if (complete and prefer_quality) or (
            (complete or not previous_complete) and distance < previous[0]):
        return (distance, script, speech), True
    return previous, False


def refresh_stale_plan(selector, chosen, pools, trailers, shot_sets, source_films,
                       text, action, default_film):
    """Replan only removed inputs; preserve current clips and cross-title identity."""
    from .source_pool import primary_fallback_plan

    result = []
    for shot in chosen:
        if shot.get("image"):
            result.append(shot)
            continue
        source = str(shot.get("source") or trailers[default_film])
        film_index = source_films.get(source, shot.get("film_index", default_film))
        if (not isinstance(film_index, int) or isinstance(film_index, bool)
                or not 0 <= film_index < len(pools)):
            raise PipelineError("A planned source has no verified title identity.")
        if source in {str(path) for path in pools[film_index]}:
            result.append(shot)
            continue
        result.extend(primary_fallback_plan(selector, [{**shot, "film_index": film_index}],
            trailers, shot_sets, text, action, film_index))
    return result
