"""Self-contained branding QA: real video plus a small visual contact sheet.

Run python scripts/branding_preview.py --output-dir /tmp/movie-ranks-branding.
No models, movie downloads or API keys are needed.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw

from movie_pipeline.branding import IDENT_SECONDS, ident_frame, render_ident
from movie_pipeline.core import probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=Path("branding-preview"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("VIDEO_RESOLUTION","1280x720")
    video = args.output_dir/"movie-ranks-ident.mp4"
    render_ident(video,args.output_dir/"work")
    contact = Image.new("RGB",(960,432),"#101014")
    draw = ImageDraw.Draw(contact)
    for i,t in enumerate((.10,.35,.60,.85,1.2,1.85)):
        x,y=(i%3)*320,(i//3)*216
        frame=ident_frame(t,640,360).resize((320,180),Image.Resampling.LANCZOS)
        contact.paste(frame,(x,y))
        draw.text((x+12,y+191),f"{t:.2f}s",fill="white")
    picture=args.output_dir/"contact-sheet.jpg"
    contact.save(picture,quality=90)
    print("IDENT_PROBE",probe(video))
    print("BRANDING_IMAGE_BEGIN")
    print(base64.b64encode(picture.read_bytes()).decode())
    print("BRANDING_IMAGE_END")


if __name__ == "__main__":
    main()
