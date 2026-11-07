"""Generate short real-model script samples and clearly labelled reaction QA."""
from __future__ import annotations
import argparse
import base64
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

from movie_pipeline.script import Writer, film_facts, OPENING, CLOSING
from movie_pipeline.cta import REACTIONS, _asset

SAMPLES = [
    {"title": "The Hangover", "year": 2009, "kind": "movie", "genres": ["Comedy"],
     "overview": "Three friends wake up in Las Vegas after a bachelor party with no memory of the night before. Their friend, the groom, is missing. They must retrace the night and find him before his wedding.",
     "brief": "Funny movie recommendations. Tell the actual situation clearly; let its absurdity carry the humor."},
    {"title": "The Conjuring", "year": 2013, "kind": "movie", "genres": ["Horror"],
     "overview": "The Perron family moves into an old farmhouse in Rhode Island. Strange events escalate, and paranormal investigators Ed and Lorraine Warren come to help the family understand what is happening.",
     "brief": "Horror recommendations. Explain the family, the house and the threat in plain conversational language."},
    {"title": "The Bear", "year": 2022, "kind": "tv", "genres": ["Drama", "Comedy"],
     "overview": "Carmy, a chef from fine dining, returns to Chicago to run his late brother's sandwich shop. He faces debt, a chaotic kitchen and a staff used to doing things their own way while trying to keep the business alive.",
     "brief": "Great shows to watch. Make the actual work, people and conflict easy to picture; do not label the whole video with one genre."},
]


def reaction_preview(root):
    contact = Image.new("RGB", (960, 250*((len(REACTIONS)+2)//3)), "#101014")
    draw = ImageDraw.Draw(contact)
    for i, reaction in enumerate(REACTIONS):
        path = _asset(reaction, root/"decoded-reactions")
        with Image.open(path) as image:
            image.seek(getattr(image, "n_frames", 1)//2)
            picture = ImageOps.contain(image.convert("RGB"), (308, 210))
        x,y=(i%3)*320,(i//3)*250
        contact.paste(picture, (x+(320-picture.width)//2,y+(210-picture.height)//2))
        draw.text((x+10,y+224), reaction["theme"], fill="white")
    path=root/"reaction-contact-sheet.jpg"
    contact.save(path,quality=90)
    print("REACTION_IMAGE_BEGIN")
    print(base64.b64encode(path.read_bytes()).decode())
    print("REACTION_IMAGE_END")


def writing_preview(root, revise=False):
    writer = Writer(cache_dir=root/"writing-cache")
    rows = []
    saved = json.loads(Path("tests/fixtures/editorial_raw_samples.json").read_text()) if revise else []
    try:
        for index, film in enumerate(SAMPLES):
            facts = {**film_facts(film), "rank": index+1}
            draft = (saved[index]["draft"] if revise and index > 0 else
                     writer.narration("Movie and show recommendations", film["brief"],
                                      f"rank {index+1}: {film['title']}", facts, None))
            edited, delivery = writer.edit("Movie and show recommendations", film["brief"],
                f"the review of {film['title']}", facts, draft, rank=index+1)
            row={"title":film["title"],"draft":draft,"edited":edited,"delivery":delivery}
            rows.append(row)
            (root/"script-samples.json").write_text(json.dumps(rows,indent=2,ensure_ascii=False))
            print("SCRIPT_SAMPLE",json.dumps(row,ensure_ascii=False),flush=True)
    finally:
        writer.unload()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--writing",action="store_true")
    parser.add_argument("--reactions",action="store_true")
    parser.add_argument("--revise",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=Path("natural-review"))
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.reactions: reaction_preview(args.output_dir)
    if args.writing: writing_preview(args.output_dir, args.revise)


if __name__=="__main__":
    main()
