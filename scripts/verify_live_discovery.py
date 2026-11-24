"""Read-only checks of live metadata; no AI inference or media downloads."""
import copy
import json
import os
from movie_pipeline.core import FilmInput
from movie_pipeline.research import Research
from movie_pipeline.release_identity import verified_release_years
from movie_pipeline.editorial_policy import search_queries
from movie_pipeline.footage import candidate_ok

if os.environ.get("AI_BACKEND") != "local":
    raise SystemExit("Live metadata check requires inference disabled.")
research=Research()
research.selected_kind=research.preferred_kind="tv"
berlin=research.resolve(FilmInput("Money Heist: Berlin",2023))
assert berlin.get("id")==146176 and berlin.get("kind")=="tv", "Wrong series for branded alias"
print("LIVE_ALIAS "+json.dumps({"title":berlin.get("title"),"id":berlin.get("id"),"year":berlin.get("year")}))
research.selected_kind=research.preferred_kind="movie"
film=research.resolve(FilmInput("Obsession",2025))
assert film.get("id")==1339713 and film.get("kind")=="movie", "Wrong production for festival-year title"
years=verified_release_years(film)
assert {2025,2026}.issubset(years), "Lost verified festival/theatrical dates"
print("LIVE_RELEASE "+json.dumps({"title":film.get("title"),"id":film.get("id"),"year":film.get("year"),"verified_years":sorted(years),"backdrops":len(film.get("backdrops",[]))}))
cinema=research.resolve(FilmInput("Obsession",2026))
assert cinema.get("id")==1339713 and {2025,2026}.issubset(verified_release_years(cinema)), "Lost dates from nested release metadata"
print("LIVE_NESTED_RELEASE verified2026 record retains2025 festival evidence")

queries=search_queries(film,primary=True)
year_free=[q for q in queries if not any(str(y) in q for y in years)]
assert year_free and any("scene" in q.lower() or "trailer" in q.lower() for q in year_free)
print("LIVE_QUERIES "+json.dumps(queries))
without_director=copy.deepcopy(film)
for field in ("directors","director","creators","creator","crew"):
    without_director.pop(field,None)
cast=without_director.get("cast") or []
person=cast[0].get("name","") if cast and isinstance(cast[0],dict) else cast[0] if cast else ""
info={"title":"Obsession (2026) Official Trailer","duration":100,
      "description":"Featuring "+str(person),"is_live":False}
assert candidate_ok(info,without_director), "Verified footage incorrectly requires director or identical premiere-year label"
assert not candidate_ok({**info,"title":"Obsession (1976) Official Trailer"},without_director), "Wrong remake year accepted"
print("LIVE_CANDIDATE verified2026 accepted without director;1976 rejected")
print("LIVE_METADATA_CHECK_PASSED")
