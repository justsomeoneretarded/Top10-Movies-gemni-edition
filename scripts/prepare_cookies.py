"""Read optional Actions secret without exposing it to shell interpolation or logs."""
import os
from pathlib import Path

from movie_pipeline.auth import prepare_cookies

path = prepare_cookies(os.getenv("YOUTUBE_COOKIES", ""), Path(os.environ["RUNNER_TEMP"]))
if path:
    with Path(os.environ["GITHUB_ENV"]).open("a") as stream:
        stream.write(f"YOUTUBE_COOKIE_FILE={path}\n")
    print("Optional YouTube session cookies configured.")
