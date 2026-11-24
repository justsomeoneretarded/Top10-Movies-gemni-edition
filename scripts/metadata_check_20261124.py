"""Read-only metadata evidence; no model calls, media downloads or production."""
import json, os, urllib.request, urllib.parse, urllib.error
base="https://api.themoviedb.org/3/"
token=os.environ.get("TMDB_READ_TOKEN","").strip()
if not token:
    raise SystemExit("TMDB secret unavailable")
def get(path, **params):
    url=base+path+"?"+urllib.parse.urlencode(params)
    req=urllib.request.Request(url,headers={"Authorization":"Bearer "+token,"Accept":"application/json","User-Agent":"MovieRanksMetadataValidation/1.0"})
    try:
        with urllib.request.urlopen(req,timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        print("TMDB_STATUS "+path+" "+str(e.code))
    except Exception as e:
        print("TMDB_UNAVAILABLE "+path+" "+type(e).__name__)
    return {}
def minimal(row,kind):
    return {k:row.get(k) for k in ("id","name","original_name","title","original_title","release_date","first_air_date","vote_count")}
selected=[]
for kind,query in [("tv","Money Heist: Berlin"),("tv","Berlin"),("movie","Obsession")]:
    rows=get("search/"+kind,query=query,language="en-US",include_adult="false").get("results",[])
    print("SEARCH "+json.dumps({"kind":kind,"query":query,"results":[minimal(r,kind) for r in rows[:10]]}))
    for row in rows[:10]:
        name=str(row.get("name") or row.get("title") or "").casefold()
        date=str(row.get("release_date") or row.get("first_air_date") or "")
        if ((kind=="tv" and name=="berlin" and date.startswith("2023"))
            or (kind=="movie" and name=="obsession" and date[:4] in {"2025","2026"})):
            pair=(kind,row.get("id"))
            if pair not in selected:
                selected.append(pair)
for kind,ident in selected[:4]:
    d=get(kind+"/"+str(ident),append_to_response="external_ids,credits,videos,images,release_dates,alternative_titles",language="en-US",include_image_language="en,null")
    releases=d.get("release_dates",{}).get("results",[])
    record={"kind":kind,"id":ident,"name":d.get("name") or d.get("title"),"date":d.get("release_date") or d.get("first_air_date"),
        "imdb_id":d.get("external_ids",{}).get("imdb_id") or d.get("imdb_id"),"homepage":d.get("homepage"),
        "overview":d.get("overview"),"directors":[x.get("name") for x in d.get("credits",{}).get("crew",[]) if x.get("job")=="Director"],
        "cast":[{"name":x.get("name"),"character":x.get("character")} for x in d.get("credits",{}).get("cast",[])[:5]],
        "release_records":[{"country":r.get("iso_3166_1"),"dates":[{k:x.get(k) for k in ("release_date","type","note")} for x in r.get("release_dates",[])]} for r in releases],
        "aliases":d.get("alternative_titles",{}),"videos":[{k:v.get(k) for k in ("site","key","name","official","type","published_at")} for v in d.get("videos",{}).get("results",[])[:10]],
        "images":{"backdrops":len(d.get("images",{}).get("backdrops",[])),"posters":len(d.get("images",{}).get("posters",[]))}}
    print("DETAIL "+json.dumps(record,ensure_ascii=True))
