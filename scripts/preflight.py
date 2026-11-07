"""Validate input and selected narrator before large model downloads."""
import os
from pathlib import Path

from movie_pipeline.core import (
    PipelineError,
    parse_movies,
    selected_content_type,
    selected_narrator,
    voice_reference,
)
from movie_pipeline.timing import workflow_duration
from movie_pipeline.video_settings import download_ceiling

title, brief = os.getenv("VIDEO_TITLE", ""), os.getenv("VIDEO_BRIEF", "")
if not title.strip() or not brief.strip():
    raise PipelineError("Title and description/creative brief are required.")
films = parse_movies(os.getenv("MOVIE_LIST", ""))
if not os.getenv("TMDB_READ_TOKEN"):
    raise PipelineError("Configure the free TMDB_READ_TOKEN repository secret before running.")
spec = workflow_duration(title, len(films))
height = download_ceiling()
name = selected_narrator(brief)
kind = selected_content_type()
voice_reference(name, Path("assets/voices"))
mode = "Auto" if spec.automatic else "selected"
print(f"Input valid: {len(films)} titles; content type {kind or 'auto'}; narrator {name}; {mode} length about {spec.target/60:.1f} minutes "
      f"(soft range {spec.minimum/60:.1f}–{spec.maximum/60:.1f}); up to {height}p; source-based export chosen after downloads.")
