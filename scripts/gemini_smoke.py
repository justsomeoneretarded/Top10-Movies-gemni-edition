"""Small real-API contract check; no paid image generation or voice substitution."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
from movie_pipeline import gemini
from movie_pipeline.core import FilmInput
from movie_pipeline.research import Research

def main():
    root = Path("gemini-validation")
    root.mkdir(exist_ok=True)
    report = {"configuration": gemini.identity(), "checks": {}, "billing_verified": False}
    errors = []
    if not gemini.enabled():
        print("Gemini validation requires the configured GEMINI_API repository secret.", flush=True)
        return 1
    client = gemini.get_client()
    schema = {"type": "object", "properties": {"status": {"type": "string", "enum": ["ok"]}},
              "required": ["status"], "additionalProperties": False}
    try:
        response = client.generate_json('Return exactly {"status":"ok"}.', schema, max_output_tokens=2048)
        if response != {"status": "ok"}:
            raise ValueError("Unexpected text contract")
        report["checks"]["structured_text"] = {"status": "passed", "model": client.text_model}
    except Exception as exc:
        report["checks"]["structured_text"] = {"status": "failed", "reason": type(exc).__name__ + ": " + gemini._redact(str(exc))[:180]}
        errors.append("structured_text")
    try:
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (512, 288), "black")
        ImageDraw.Draw(image).rectangle((130, 60, 380, 230), fill="red")
        vision_schema = {"type": "object", "properties": {"description": {"type": "string"}},
                         "required": ["description"], "additionalProperties": False}
        observed = client.generate_json("Describe the visible shape and colors in one short sentence.",
                                        vision_schema, images=(image,), max_output_tokens=2048)
        if not observed.get("description", "").strip():
            raise ValueError("Empty image description")
        report["checks"]["image_understanding"] = {"status": "passed", "model": client.vision_model,
                                                   "description": observed["description"][:300]}
    except Exception as exc:
        report["checks"]["image_understanding"] = {"status": "failed", "reason": type(exc).__name__ + ": " + gemini._redact(str(exc))[:180]}
        errors.append("image_understanding")
    try:
        saved = os.environ.pop("TMDB_READ_TOKEN", None)
        previous = os.environ.get("WIKIPEDIA_CONTEXT")
        os.environ["WIKIPEDIA_CONTEXT"] = "false"
        try:
            research = Research()
            research.selected_kind = research.preferred_kind = "movie"
            films = research.collect([FilmInput("Havoc", 2025)], root / "research.json")
        finally:
            if saved is not None:
                os.environ["TMDB_READ_TOKEN"] = saved
            if previous is None:
                os.environ.pop("WIKIPEDIA_CONTEXT", None)
            else:
                os.environ["WIKIPEDIA_CONTEXT"] = previous
        if len(films) != 1 or films[0].get("research_backend") != "gemini" or not films[0].get("research_sources"):
            raise ValueError("Verified source research not demonstrated")
        report["checks"]["verified_source_research_without_tmdb"] = {
            "status": "passed", "title": films[0]["title"], "year": films[0]["year"],
            "source_count": len(films[0]["research_sources"]),
            "method": films[0].get("research_method", "unknown")}
    except Exception as exc:
        report["checks"]["verified_source_research_without_tmdb"] = {
            "status": "failed", "reason": type(exc).__name__ + ": " + gemini._redact(str(exc))[:180]}
        errors.append("verified_source_research_without_tmdb")
    report["checks"]["custom_voice"] = {
        "status": "not_generated",
        "reason": "A real actor reference, same-speaker consent and registered voice ID are required. Chatterbox preserves the existing reference identity."}
    # Video understanding, exactly as production asks for it: a public YouTube link, then an uploaded file.
    from movie_pipeline import gemini_editing
    video_schema = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}
    for name, kwargs, schema, prompt in (
            ("youtube_simple", {}, video_schema, "Summarize this video in one sentence."),
            ("youtube_scene_map", {"model": gemini_editing.video_models(client)[0]}, gemini_editing.VIDEO_SCHEMA,
             gemini_editing.scene_prompt(60))):
        try:
            value = client.generate_json(prompt, schema, youtube_url="https://www.youtube.com/watch?v=9hE5-98ZeCg",
                                         video_duration=60, max_output_tokens=4096, thinking="low", **kwargs)
            report["checks"][name] = {"status": "passed", "keys": sorted(value)[:5]}
        except Exception as exc:
            report["checks"][name] = {"status": "failed", "reason": type(exc).__name__ + ": " + gemini._redact(str(exc))[:240]}
            errors.append(name)
    try:
        import shutil
        import subprocess
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not installed on this runner")
        clip = root / "smoke.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=24:d=6",
                        "-f", "lavfi", "-i", "sine=f=440:d=6", "-shortest", "-c:v", "libx264", "-c:a", "aac", str(clip)], check=True)
        uploaded = client.upload_video(clip, timeout=120)
        value = client.generate_json("Describe this short test video in one sentence.", video_schema,
                                     video_file=uploaded, video_duration=6, max_output_tokens=1024, thinking="low")
        report["checks"]["uploaded_video"] = {"status": "passed", "keys": sorted(value)[:5]}
    except Exception as exc:
        report["checks"]["uploaded_video"] = {"status": "failed", "reason": type(exc).__name__ + ": " + gemini._redact(str(exc))[:240]}
        errors.append("uploaded_video")
    for name in ("youtube_simple", "youtube_scene_map", "uploaded_video"):
        check = report["checks"][name]
        print(f"::{'notice' if check['status'] == 'passed' else 'warning'} title=Gemini video check: {name}::{check['status']} {check.get('reason', '')}", flush=True)
    usage = Path(os.getenv("GEMINI_USAGE_REPORT", "work/gemini-usage.jsonl"))
    if usage.exists():
        events = []
        for line in usage.read_text().splitlines()[-40:]:
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
        report["provider_events"] = events
    report["passed"] = not errors
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Gemini live report: " + json.dumps(report, sort_keys=True))
    print(json.dumps({"passed": report["passed"], "failed_checks": errors,
                      "models": {"text": client.text_model, "vision": client.vision_model}}, sort_keys=True))
    return int(bool(errors))

if __name__ == "__main__":
    sys.exit(main())
