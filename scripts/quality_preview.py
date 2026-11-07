"""Render observable brand and ten real-photo thumbnails on the validation runner."""
from pathlib import Path
import os,json,base64
import requests
from PIL import Image,ImageDraw
from movie_pipeline.branding import render_ident, logo_image
from movie_pipeline.thumbnail_studio import build
from movie_pipeline.core import save_json

root=Path("quality-preview");root.mkdir(exist_ok=True)
logo_image(root/"brand.png")
render_ident(root/"ident.mp4",root/"ident")
token=os.getenv("TMDB_READ_TOKEN","")
films=[]
if token:
    session=requests.Session()
    session.headers["Authorization"]="Bearer "+token
    for name in ("Project Hail Mary","The Odyssey","Hoppers"):
        response=session.get("https://api.themoviedb.org/3/search/movie",params={"query":name,"year":2026},timeout=20)
        response.raise_for_status()
        results=response.json().get("results",[])
        if not results:
            continue
        movie=results[0];identifier=movie["id"]
        detail=session.get(f"https://api.themoviedb.org/3/movie/{identifier}",params={"append_to_response":"credits"},timeout=20)
        detail.raise_for_status();details=detail.json()
        response=session.get(f"https://api.themoviedb.org/3/movie/{identifier}/images",timeout=20)
        response.raise_for_status();images=response.json()
        films.append({"id":identifier,"title":movie["title"],"year":2026,"kind":"movie","genres":[],
                      "cast":[p["name"] for p in details.get("credits",{}).get("cast",[])[:5]],
                      "directors":[p["name"] for p in details.get("credits",{}).get("crew",[]) if p.get("job")=="Director"],
                      "backdrops":[r["file_path"] for r in images.get("backdrops",[])[:8]],
                      "posters":[r["file_path"] for r in images.get("posters",[])[:4]]})
if films:
    report=build(films,[],[],["Worth watching"],root,root/"work",
                 title="8 Movies of 2026 You Should Watch Right Now",brief="A broad movie recommendation list.")
    save_json(root/"design-report.json",report)
    print("THUMBNAIL_DIMENSIONS",json.dumps({str(i):Image.open(root/f"thumbnail-{i:02}.jpg").size for i in range(1,11)}))
    sheet=Image.new("RGB",(1280,800),"#121216")
    draw=ImageDraw.Draw(sheet)
    for i in range(1,11):
        image=Image.open(root/f"thumbnail-{i:02}.jpg").convert("RGB")
        image.thumbnail((250,141))
        x=((i-1)%5)*256;y=((i-1)//5)*180
        sheet.paste(image,(x,y))
        draw.text((x+5,y+145),f"Option {i}",fill="white")
    logo=Image.open(root/"brand.png").convert("RGB");logo.thumbnail((400,225))
    sheet.paste(logo,(20,400))
    draw.text((450,440),"MOVIE RANKS: real-photo designs, no invented faces",fill="white")
    sheet.save(root/"contact-sheet.jpg",quality=88)
    print("QA_IMAGE_BEGIN")
    print(base64.b64encode((root/"contact-sheet.jpg").read_bytes()).decode())
    print("QA_IMAGE_END")
else:
    print("Real-photo QA skipped: TMDB token unavailable or titles unresolved; brand render completed.")
