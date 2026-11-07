"""Renderer-only sample; synthetic footage and espeak are QA placeholders.

This does not select the production narrator or touch any production request.
"""
from __future__ import annotations
import argparse
import base64
import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from movie_pipeline.core import duration, ffmpeg, probe
from movie_pipeline.rank_intro import prepare_rank_intros
from movie_pipeline.render import (
    LOSSLESS_AUDIO, concat_file, label_image, speech_video, trailer_video,
    validate_audio_video_duration,
)
from movie_pipeline.video_settings import export_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("intro-preview"))
    root = parser.parse_args().output_dir
    work = root/"rank-work"
    work.mkdir(parents=True, exist_ok=True)
    script = prepare_rank_intros({"sections": [{"kind": "film", "film": 0,
        "paragraphs": ["Number 1: Sample Scene. A quiet moment lets the story begin."]}]},
        [{"title": "Sample Scene"}])
    announcement, body = script["sections"][0]["paragraphs"]
    tracks = {}
    for name, text in (("announcement", announcement), ("body", body),
                       ("scene", "We have to go now. Stay close. We can make it.")):
        path = work/f"{name}.wav"
        subprocess.run(["espeak-ng", "-s", "155", "-w", str(path), text], check=True)
        tracks[name] = path
    width, height = export_size()
    source = work/"source.mp4"
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=s={width}x{height}:r=30:d=8",
           "-i", tracks["scene"], "-map", "0:v", "-map", "1:a",
           "-af", "apad", "-t", 8, "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-threads", 2, *LOSSLESS_AUDIO, source)
    label = work/"rank.png"
    label_image(label, "Sample Scene", 1, 2026)
    reveal = root/"rank-announcement.mp4"
    shot = {"source": str(source), "start": 0, "duration": duration(tracks["announcement"]),
            "still": True, "still_motion": "zoom_in"}
    speech_video(source, tracks["announcement"], [shot], reveal, label,
                 work/"reveal", label_seconds=duration(tracks["announcement"]))
    native = root/"native-preview.mp4"
    trailer_video(source, 0, 8, native, muted=False)
    review = root/"review-body.mp4"
    body_shot = {"source": str(source), "start": 0, "duration": duration(tracks["body"]), "still": True}
    speech_video(source, tracks["body"], [body_shot], review, None, work/"body")
    parts = [root/"movie-ranks-ident.mp4", reveal, native, review]
    listing = work/"parts.txt"
    concat_file(parts, listing)
    output = root/"intro-and-rank-sequence.mp4"
    ffmpeg("-f", "concat", "-safe", 0, "-i", listing, "-map", "0:v", "-map", "0:a",
           "-vf", "setpts=N/(30*TB)", "-af", "asetpts=N/SR/TB",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", 18, "-threads", 2,
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", output)
    seconds = validate_audio_video_duration(output)
    expected = sum(duration(part) for part in parts)
    assert math.isclose(seconds, expected, abs_tol=.12), (seconds, expected)
    starts = [0]
    for part in parts[:-1]:
        starts.append(starts[-1]+duration(part))
    report = {"qa_placeholders_only": True, "sequence": [
        {"kind": kind, "start": start, "duration": duration(part)}
        for kind, start, part in zip(("brand", "announcement", "native-preview", "review"), starts, parts)],
        "announcement": announcement, "review": body, "total": seconds,
        "audio": [s for s in probe(output)["streams"] if s["codec_type"] == "audio"]}
    (root/"sequence-report.json").write_text(json.dumps(report, indent=2))
    contact = Image.new("RGB", (960,216), "#101014")
    draw = ImageDraw.Draw(contact)
    for i, (at, title) in enumerate(((starts[1]+.9, "Announcement + animated card"),
                                   (starts[2]+1, "Full preview - no narrator"),
                                   (starts[3]+.5, "Review resumes"))):
        frame = work/f"contact-{i}.jpg"
        ffmpeg("-ss", at, "-i", output, "-frames:v", 1, "-vf", "scale=320:180", frame)
        with Image.open(frame) as image:
            contact.paste(image.convert("RGB"), (i*320, 0))
        draw.text((i*320+10,190), title, fill="white")
    image_path = root/"rank-contact-sheet.jpg"
    contact.save(image_path, quality=90)
    print("SEQUENCE_REPORT", json.dumps(report))
    print("RANK_IMAGE_BEGIN")
    print(base64.b64encode(image_path.read_bytes()).decode())
    print("RANK_IMAGE_END")


if __name__ == "__main__":
    main()
