"""Conservative, bounded repairs of empty visual gaps; original audio is immutable.

These helpers never shorten a part or change the timeline. Dark photographed
scenes, lettering, edge fades and uncertain contexts remain review advisories.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
from PIL import Image

from .core import PipelineError, ffmpeg, probe, save_json
from .shots import get_frame

_ALLOWED = {"narration", "review-preview", "trailer-audio", "cold-open"}
_MAX_GAP = 1.5
_MIN_GAP = .4


def _finite(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _time_limit():
    return max(10.0, min(180.0, _finite(os.getenv("VISUAL_REPAIR_SECONDS", "120"), 120)))


def _stream(metadata, kind):
    return next((row for row in metadata.get("streams", []) if row.get("codec_type") == kind), {})


def _video_info(path):
    metadata = probe(path)
    video = _stream(metadata, "video")
    if not video:
        raise ValueError("video_missing")
    frames = int(video.get("nb_frames") or 0)
    numerator, denominator = str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1").split("/")
    fps = float(numerator) / float(denominator)
    seconds = _finite(video.get("duration"), _finite(metadata.get("format", {}).get("duration")))
    if frames <= 0 or not 1 <= fps <= 120 or seconds <= 0:
        raise ValueError("stable_frame_clock_unavailable")
    return {"seconds": seconds, "frames": frames, "fps": fps,
            "width": int(video["width"]), "height": int(video["height"]),
            "has_audio": bool(_stream(metadata, "audio"))}


def _black_intervals(path, timeout):
    # Return only decoder-confirmed, almost entirely black image runs.
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-threads", "2", "-filter_threads", "1", "-i", str(path), "-vf",
         "scale=320:-2,blackdetect=d=0.4:pix_th=0.08", "-an", "-f", "null", "-"],
        capture_output=True, text=True, timeout=max(1, timeout), check=False)
    if result.returncode:
        raise ValueError("visual_scan_unavailable")
    return [(float(a), float(b)) for a, b in
            re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", result.stderr)]


def _picture_stats(image):
    gray = np.asarray(image.convert("L").resize((320, 180)), dtype=np.float32)
    # Ignore only the small logo/tracker corners; central lettering remains.
    useful = gray.copy()
    useful[:28, :86] = 0
    useful[:28, -86:] = 0
    useful[-26:, :86] = 0
    useful[-26:, -86:] = 0
    mask = np.ones(gray.shape, dtype=bool)
    mask[:28, :86] = mask[:28, -86:] = False
    mask[-26:, :86] = mask[-26:, -86:] = False
    values = useful[mask]
    return {"mean": float(values.mean()), "std": float(values.std()),
            "bright_fraction": float(np.mean(values > 28))}


def _context_allowed(event):
    if event.get("type") not in _ALLOWED or event.get("rank_announcement"):
        return False
    if event.get("section_kind") in {"intro", "outro", "subscribe"}:
        return False
    if len(event.get("films", [])) > 1:
        return False
    return bool(event.get("film"))


def _classify(path, start, end, bounds, event):
    """A repair needs a truly empty interior and clean context on BOTH sides."""
    low, high = bounds
    if not _context_allowed(event):
        return {"status": "advisory", "reason": "intentional_or_mixed_graphic_context"}
    if not _MIN_GAP - .035 <= end - start <= _MAX_GAP:
        return {"status": "advisory", "reason": "gap_outside_safe_repair_length"}
    if start - low < .3 or high - end < .3:
        return {"status": "advisory", "reason": "edge_fade_or_adjacent_part"}
    samples = [get_frame(path, start + (end - start) * fraction) for fraction in (.2, .5, .8)]
    for image in samples:
        stats = _picture_stats(image)
        if stats["std"] > 1.8 or stats["bright_fraction"] > .0005 or stats["mean"] > 4:
            return {"status": "advisory", "reason": "dark_scene_or_visible_text"}
    before = get_frame(path, max(low, start - .2))
    after = get_frame(path, min(high - .02, end + .2))
    if any((stats := _picture_stats(image))["mean"] < 12 or stats["std"] < 9
           for image in (before, after)):
        return {"status": "advisory", "reason": "uncertain_neighbouring_picture"}
    # Original dialogue can deliberately bridge a fade for dramatic effect.
    # Without stronger evidence, retain native-audio scene pauses as advisories.
    if event.get("type") in {"trailer-audio", "review-preview", "cold-open"}:
        return {"status": "advisory", "reason": "native_scene_pause_requires_review"}
    return {"status": "repairable", "reason": "empty_gap_under_continuous_narration",
            "hold": before}


def _audio_digest(path):
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-threads", "2", "-filter_threads", "1", "-xerror", "-err_detect", "explode", "-i", str(path),
         "-map", "0:a:0", "-vn", "-f", "hash", "-hash", "sha256", "-"],
        capture_output=True, text=True, timeout=90, check=False)
    if result.returncode or not re.search(r"SHA256=[0-9a-f]{64}", result.stdout):
        raise ValueError("original_audio_cannot_be_verified")
    return result.stdout.strip()


def _replace_empty_gap(path, start, end, hold, directory):
    info = _video_info(path)
    if not info["has_audio"]:
        raise ValueError("audio_missing")
    original_stat = path.stat()
    digest = _audio_digest(path)
    directory.mkdir(parents=True, exist_ok=True)
    image = directory / "clean-neighbour.png"
    candidate = directory / "candidate.mp4"
    hold.convert("RGB").save(image)
    first = max(0, math.ceil(start * info["fps"]))
    last = min(info["frames"] - 1, math.ceil(end * info["fps"]) - 1)
    if last < first:
        raise ValueError("empty_frame_range")
    try:
        ffmpeg("-i", path, "-loop", 1, "-framerate", info["fps"], "-i", image,
               "-filter_complex", f"[0:v][1:v]freezeframes=first={first}:last={last}:replace=0[v]",
               "-map", "[v]", "-map", "0:a:0", "-c:v", "libx264", "-preset", "fast",
               "-crf", 14, "-pix_fmt", "yuv420p", "-r", info["fps"],
               "-frames:v", info["frames"], "-c:a", "copy", "-movflags", "+faststart", candidate)
        checked = _video_info(candidate)
        if (checked["frames"] != info["frames"] or checked["width"] != info["width"]
                or checked["height"] != info["height"]
                or abs(checked["seconds"] - info["seconds"]) > .002
                or _audio_digest(candidate) != digest):
            raise ValueError("repair_changed_audio_or_frame_clock")
        ffmpeg("-xerror", "-err_detect", "explode", "-i", candidate, "-f", "null", "-")
        new_stat = path.stat()
        if (new_stat.st_size, new_stat.st_mtime_ns) != (original_stat.st_size, original_stat.st_mtime_ns):
            raise ValueError("part_changed_during_repair")
        candidate.replace(path)
    finally:
        candidate.unlink(missing_ok=True)
    return {"first_frame": first, "last_frame": last, "audio": "decoded_sha256_identical",
            "frames": info["frames"], "duration_seconds": info["seconds"]}


def repair_visual_parts(parts, timeline, work, *, max_repairs=3):
    """Repair at most three safe blank gaps; never discard or alter source audio."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "basis": "rendered_parts_before_mix",
              "repaired": [], "advisories": [], "unresolved": [], "inspected_parts": 0}
    limit = max(0, min(3, int(max_repairs)))
    if len(parts) != len(timeline):
        report["unresolved"].append({"reason": "part_timeline_identity_mismatch"})
        save_json(work / "report.json", report)
        return report
    deadline = time.monotonic() + _time_limit()
    for index, (value, event) in enumerate(zip(parts, timeline)):
        if time.monotonic() >= deadline or len(report["repaired"]) >= limit:
            report["uninspected_parts"] = len(parts) - index
            break
        if not _context_allowed(event):
            continue
        path = Path(value)
        try:
            info = _video_info(path)
            intervals = _black_intervals(path, min(45, deadline - time.monotonic()))
            report["inspected_parts"] += 1
            for start, end in intervals[:6]:
                row = {"part": index, "film": event.get("film"), "type": event.get("type"),
                       "start": start, "end": end}
                verdict = _classify(path, start, end, (0, info["seconds"]), event)
                row.update({key: value for key, value in verdict.items() if key != "hold"})
                if verdict["status"] != "repairable":
                    report["advisories"].append(row)
                    continue
                if len(report["repaired"]) >= limit or time.monotonic() >= deadline:
                    report["unresolved"].append({**row, "reason": "bounded_repair_allowance"})
                    continue
                try:
                    proof = _replace_empty_gap(path, start, end, verdict["hold"],
                                               work / f"part-{index:03}-{len(report['repaired']):02}")
                    report["repaired"].append({**row, **proof, "status": "repaired"})
                except (PipelineError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
                    report["unresolved"].append({**row, "status": "retained", "reason": type(error).__name__})
        except (PipelineError, OSError, RuntimeError, ValueError, TypeError, subprocess.SubprocessError) as error:
            report["advisories"].append({"part": index, "reason": "scan_unavailable",
                                         "error": type(error).__name__})
    report["status"] = "repaired" if report["repaired"] else "reviewed"
    save_json(work / "report.json", report)
    return report


def inspect_visual_export(video, timeline, report):
    """Classify remaining final-export dark intervals without changing any media."""
    result = {"basis": "actual_finished_export", "repairable_parts": [], "advisories": []}
    try:
        info = _video_info(Path(video))
        speed = _finite(report.get("export_speed"), 0)
        if speed <= 0:
            # The caller should supply finish()'s exact tempo. Inference is useful
            # for advisory inspection only, never for an automatic mapped repair.
            result["status"] = "exact_tempo_unavailable"
            result["advisories"] = [{"reason": "export_speed_required_for_repair_mapping"}]
            return result
        intervals = _black_intervals(video, 120)
        for start, end in intervals[:30]:
            owners = [(i, event) for i, event in enumerate(timeline)
                      if event.get("start", 0) / speed <= start + .02
                      and (event.get("start", 0) + event.get("duration", 0)) / speed >= end - .02]
            if len(owners) != 1:
                result["advisories"].append({"start": start, "end": end, "reason": "part_boundary_or_ending"})
                continue
            index, event = owners[0]
            low = event["start"] / speed
            high = (event["start"] + event["duration"]) / speed
            verdict = _classify(Path(video), start, end, (low, high), event)
            row = {"part": index, "film": event.get("film"), "start": start, "end": end,
                   **{k: v for k, v in verdict.items() if k != "hold"}}
            if verdict["status"] == "repairable":
                result["repairable_parts"].append(index)
            else:
                result["advisories"].append(row)
        result["repairable_parts"] = sorted(set(result["repairable_parts"]))
        result["status"] = "review_needed" if result["repairable_parts"] else "checked"
    except (PipelineError, OSError, RuntimeError, ValueError, TypeError, subprocess.SubprocessError) as error:
        result.update(status="inspection_unavailable", error=type(error).__name__)
    return result
