"""Small observable review assets from the actual finished export, never a substitute for watching it."""
from pathlib import Path
import base64,json
from PIL import Image,ImageDraw
from movie_pipeline.core import duration,save_json
from movie_pipeline.shots import get_frame

def build(output=Path("output"),jobs=Path("work/jobs")):
    video=output/"final-video.mp4"
    manifests=sorted(jobs.glob("*/timeline.json"),key=lambda p:p.stat().st_mtime,reverse=True)
    if not video.exists() or not manifests:
        return
    data=json.loads(manifests[0].read_text())
    speed=float(data.get("speed",1))
    rows=[]
    events=data.get("events",[])
    for kind in ("brand-ident","cold-open","review-preview","narration","trailer-audio","subscribe","end-screen"):
        selected=[e for e in events if e.get("type")==kind]
        for event in selected[:3]:
            tick=(event["start"]+min(event["duration"]*.5,3))/speed
            rows.append((tick,kind,event.get("film","")))
    rows=sorted(rows)[:20]
    sheet=Image.new("RGB",(1280,228*((len(rows)+3)//4)),"#121216")
    draw=ImageDraw.Draw(sheet)
    for i,(tick,kind,film) in enumerate(rows):
        image=get_frame(video,tick).convert("RGB")
        image.thumbnail((316,178))
        x=(i%4)*320;y=(i//4)*228
        sheet.paste(image,(x,y))
        draw.text((x+4,y+182),f"{tick:.1f}s {kind}",fill="white")
        draw.text((x+4,y+199),film[:42],fill="#cccccc")
    sheet.save(output/"quality-contact-sheet.jpg",quality=88)
    report={"review_basis":"Actual finished export; sampled frames are advisory, not a quality guarantee",
            "duration_seconds":duration(video),"sampled_frames":len(rows),"native_moments":[
                {"film":e.get("film"),"seconds":e["duration"]/speed,"start":e["start"]/speed}
                for e in events if e.get("type")=="trailer-audio"],
            "chapters":data.get("chapters",[])}
    save_json(output/"quality-summary.json",report)
    print("QUALITY_EXPORT_IMAGE_BEGIN")
    print(base64.b64encode((output/"quality-contact-sheet.jpg").read_bytes()).decode())
    print("QUALITY_EXPORT_IMAGE_END")
    print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":
    build()
