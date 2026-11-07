"""Thumbnail Lab: build only the five thumbnails for a list, in about twenty minutes.

Research, one trailer per title (top five), shot analysis, optional AI scene descriptions,
then the thumbnail studio. Writes lab-output/ with the five thumbnails, a side-by-side
sheet and the studio's report, so designs can be judged without a full video run.

Usage: python -m movie_pipeline.thumbnail_lab   (reads VIDEO_TITLE, VIDEO_BRIEF, MOVIE_LIST)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw

from .core import notice


def sheet(output: Path):
    """All thumbnails on one image, labelled, for quick review."""
    files = sorted(output.glob("thumbnail-0*.jpg"))
    if not files:
        return
    tile_w, tile_h = 640, 360
    canvas = Image.new("RGB", (tile_w * 2 + 30, (tile_h + 40) * ((len(files) + 1) // 2) + 10), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(files):
        x, y = 10 + (index % 2) * (tile_w + 10), 10 + (index // 2) * (tile_h + 40)
        with Image.open(path) as image:
            canvas.paste(image.convert("RGB").resize((tile_w, tile_h)), (x, y))
        draw.text((x, y + tile_h + 8), path.name, fill=(230, 230, 230))
    canvas.save(output / "sheet.jpg", quality=88)


def main():
    from .cli import setup_run
    from .footage import download
    from .render import thumbnails
    from .shots import analyse
    started = time.time()
    title, movies = os.getenv("VIDEO_TITLE", ""), os.getenv("MOVIE_LIST", "")
    brief = os.getenv("VIDEO_BRIEF", "").strip() or f"{title}."  # the lab's brief is optional
    output = Path("lab-output")
    output.mkdir(parents=True, exist_ok=True)
    ctx = setup_run(title, brief, movies, Path("work/jobs"), output, started)
    films = ctx.films[:5]
    trailers, shot_sets = [], []
    from . import scenes
    for film in films:
        trailer = download(film, Path("work/footage"))
        shots = analyse(trailer, trailer.with_suffix(".shots.json"))
        if scenes.enabled():
            scenes.describe_trailer(trailer, shots)
        trailers.append(trailer)
        shot_sets.append(shots)
    scenes.unload()
    phrases = [p for p in os.getenv("LAB_PHRASES", "").split(";") if p.strip()]
    thumbnails(trailers, shot_sets, phrases, output, films=films, title=title, brief=brief,
               work=Path("work/thumbnail-work"), use_api=False, use_cutouts=True)
    report = Path("work/thumbnail-work/thumbnail-report.json")
    if report.exists():
        (output / "thumbnail-report.json").write_text(report.read_text())
        data = json.loads(report.read_text())
        if not isinstance(data, dict):  # the fallback designs write a plain list
            data = {"genre": "fallback designs", "thumbnails": [{"file": t.get("file", "?"), "style": t.get("style", "?"),
                    "films": [str(t.get("film", ""))]} for t in data if isinstance(t, dict)]}
        notice("Thumbnail Lab", f"{data.get('genre')}: " + "; ".join(f"{t['file']} {t['style']} ({', '.join(t['films'])})"
                                                               for t in data.get("thumbnails", []))
               + f". Good faces: {data.get('good_faces')} of {data.get('candidates')}. {(time.time()-started)/60:.0f} min.")
    sheet(output)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as exc:  # report the cause on the run page (logs are not always readable)
        notice("Thumbnail Lab failed", f"{type(exc).__name__}: {str(exc)[-600:]}\n{traceback.format_exc()[-1500:]}", "error")
        raise
