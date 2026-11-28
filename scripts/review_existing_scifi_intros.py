"""Inspect existing output only: no model, source download or production calls."""
import base64, json, math, re, shutil, subprocess, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from PIL import Image, ImageDraw, ImageFont
root=Path("payload").resolve()
out=Path("review-output"); out.mkdir(exist_ok=True)
for archive in list(root.rglob("*.zip"))[:4]:
    with zipfile.ZipFile(archive) as z:
        entries=z.infolist()
        if len(entries)>1500 or sum(i.file_size for i in entries)>1_000_000_000:
            raise RuntimeError("Unexpected archive size")
        dest=root/"unpacked"/archive.stem
        for item in entries:
            p=PurePosixPath(item.filename)
            if p.is_absolute() or ".." in p.parts or ((item.external_attr>>16)&0o170000)==0o120000:
                raise RuntimeError("Unsafe archive path")
            target=dest.joinpath(*p.parts)
            if not target.resolve().is_relative_to(root): raise RuntimeError("Unsafe destination")
            if item.is_dir(): target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(item) as src,target.open("wb") as dst: shutil.copyfileobj(src,dst)
files=sorted(p for p in root.rglob("*") if p.is_file())
print("REVIEW_FILE_LIST "+json.dumps([{"path":str(p.relative_to(root)),"bytes":p.stat().st_size} for p in files][:400]),flush=True)
def scrub(v):
    if isinstance(v,dict):
        return {k:("[redacted]" if re.search(r"(?:secret|cookie|authorization|api.?key|access.?token)",k,re.I) else scrub(x)) for k,x in v.items()}
    if isinstance(v,list): return [scrub(x) for x in v]
    if isinstance(v,str):
        v=re.sub(r"(?i)([?&](?:key|token|api_key|access_token)=)[^&\s]+",r"\1[redacted]",v)
        return re.sub(r"AIza[A-Za-z0-9_-]{20,}","[redacted]",v)
    return v
budget=1_800_000
for p in files:
    if p.suffix.lower()==".json":
        try:
            clean=scrub(json.loads(p.read_text()))
            text=json.dumps(clean,ensure_ascii=True,separators=(",",":"))
            if len(text)<=260000 and len(text)<=budget:
                print("REVIEW_JSON "+json.dumps({"path":str(p.relative_to(root)),"data":clean},ensure_ascii=True),flush=True)
                budget-=len(text)
            else:
                print("REVIEW_LARGE_JSON "+json.dumps({"path":str(p.relative_to(root)),"bytes":len(text),"keys":list(clean)[:50] if isinstance(clean,dict) else "list"}),flush=True)
        except (ValueError,UnicodeError,OSError): print("REVIEW_UNREADABLE_JSON "+str(p.relative_to(root)),flush=True)
    elif p.suffix.lower() in {".srt",".vtt"} or (p.suffix.lower() in {".txt",".md"} and any(w in p.name.lower() for w in ("script","description","chapter","title"))):
        try: print("REVIEW_TEXT "+json.dumps({"path":str(p.relative_to(root)),"text":scrub(p.read_text()[:50000])},ensure_ascii=True),flush=True)
        except (UnicodeError,OSError): pass
videos=[p for p in files if p.name=="final-video.mp4"]
if not videos: videos=sorted((p for p in files if p.suffix.lower()==".mp4"),key=lambda p:p.stat().st_size,reverse=True)
if not videos: raise RuntimeError("No finished video")
video=videos[0]
probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_format","-show_streams","-of","json",str(video)]))
print("REVIEW_PROBE "+json.dumps(scrub(probe)),flush=True)
dur=float(probe["format"]["duration"])
ranges=[("main-intro",0,17.4),("rank-5-coherence",17.4,14),("rank-4-predestination",69.366667,14),("rank-3-prestige",116.033334,14),("rank-2-arrival",166.200001,14),("rank-1-inception",217.333335,14)]
font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",16)
for name,start,length in ranges:
    length=min(length,max(0,dur-start))
    folder=out/name; folder.mkdir(exist_ok=True)
    times=[start+i for i in range(int(math.ceil(length)))]+[start+max(0,length-.12)]
    def frame(task):
        i,t=task; dest=folder/f"{i:03}.jpg"
        subprocess.run(["ffmpeg","-v","error","-nostdin","-threads","1","-ss",f"{t:.6f}","-i",str(video),"-frames:v","1","-vf","scale=320:180","-q:v","3","-y",str(dest)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        return dest
    with ThreadPoolExecutor(max_workers=2) as pool: paths=list(pool.map(frame,enumerate(times)))
    cols=4; sheet=Image.new("RGB",(cols*320,math.ceil(len(paths)/cols)*208),(22,22,22)); draw=ImageDraw.Draw(sheet)
    for i,(p,t) in enumerate(zip(paths,times)):
        x=(i%cols)*320;y=(i//cols)*208
        with Image.open(p) as im: sheet.paste(im.convert("RGB"),(x,y))
        mm,ss=divmod(t,60)
        draw.text((x+6,y+184),f"{name}  {int(mm):02}:{ss:05.2f}",font=font,fill="white")
    jpeg=out/f"{name}.jpg";sheet.save(jpeg,quality=84,optimize=True)
    print("REVIEW_IMAGE "+json.dumps({"name":name,"start":start,"length":length,"mime":"image/jpeg","base64":base64.b64encode(jpeg.read_bytes()).decode()}),flush=True)
    mp3=out/f"{name}.mp3"
    subprocess.run(["ffmpeg","-v","error","-nostdin","-threads","1","-ss",str(start),"-i",str(video),"-t",str(length),"-vn","-ac","1","-ar","24000","-c:a","libmp3lame","-b:a","64k","-y",str(mp3)],check=True)
    print("REVIEW_AUDIO "+json.dumps({"name":name,"start":start,"length":length,"mime":"audio/mpeg","base64":base64.b64encode(mp3.read_bytes()).decode()}),flush=True)
    proc=subprocess.run(["ffmpeg","-hide_banner","-nostdin","-ss",str(start),"-i",str(video),"-t",str(length),"-vn","-af","astats=metadata=0:reset=0","-f","null","-"],text=True,capture_output=True)
    stats=[line for line in proc.stderr.splitlines() if any(w in line for w in ("RMS level dB","Peak level dB","Noise floor dB","Number of NaNs","Number of Infs"))]
    print("REVIEW_AUDIO_STATS "+json.dumps({"name":name,"statistics":stats[-15:]}),flush=True)
print("REVIEW_EXTRACTION_COMPLETE",flush=True)
