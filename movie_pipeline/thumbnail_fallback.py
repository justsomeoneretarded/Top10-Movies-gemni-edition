"""Complete a degraded thumbnail package without inventing a movie image."""
from pathlib import Path
import os
from PIL import Image, ImageDraw, ImageOps
from .core import save_json

def complete(output: Path, films, work: Path):
    missing=[i for i in range(1,11) if not (output/f"thumbnail-{i:02}.jpg").exists()]
    if not missing:
        return
    from .render import brand_font
    from .image_discovery import gather_photos
    photos, provenance=[],[]
    try:
        photos,provenance=gather_photos(films or [],work/"fallback-photos")
    except Exception as exc:
        print(f"Optional fallback photo discovery unavailable ({type(exc).__name__}).",flush=True)
    report=[]
    for i in missing:
        if photos:
            candidate=photos[(i-1)%len(photos)]
            image=ImageOps.fit(candidate.image,(1280,720),centering=(.35 if i%2 else .65,.4))
            source=candidate.notes
        else:
            image=Image.new("RGB",(1280,720),("#10131c","#19121a","#111c1c")[(i-1)%3])
            draw=ImageDraw.Draw(image)
            draw.text((640,280),"MOVIE RANKS",anchor="mm",font=brand_font("display",84),fill="white")
            title=" ".join(os.getenv("VIDEO_TITLE","Movie recommendations").split()[:8])
            draw.text((640,410),title,anchor="mm",font=brand_font("body-medium",28),fill="#dddddd")
            source={"source_kind":"typographic-fallback","reason":"No verified still image available"}
        ImageDraw.Draw(image).rectangle((2,2,1277,717),outline="#d92935",width=5)
        image.convert("RGB").save(output/f"thumbnail-{i:02}.jpg",quality=94)
        report.append({"file":f"thumbnail-{i:02}.jpg",**source})
    save_json(output/"thumbnail-fallback.json",{"designs":report,"provenance":provenance})
