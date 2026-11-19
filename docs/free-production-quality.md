# Free production quality improvements

Integration work remains isolated on feat/free-production-quality until full CI and review pass.
Main baseline: 8414b58f0a2b1f9fff75a627b0201ef1475fe474.
Other repositories, paid experiment code, voices, model/Python caches and strict speech/source/AV checks remain outside this change.

Ownership / durable component branches:
- work/free-quality-depth: script.py, timing.py; Writer.repair_depth with bounded, substantive repair and soft runtime guidance.
- work/free-quality-hd: footage.py, secondary_footage.py, source_pool.py, video_settings.py; promote_source_pool, discovery_size, choose_export_from_pools, source_quality_report.
- work/free-quality-coverage: gemini.py, gemini_editing.py; begin_accepted_analysis, primary-first reserved coverage, bounded retries/backoff.
- work/free-quality-thumbnails: image_discovery.py, thumbnail_studio.py; robust bounded discovery, honest diagnostics and diverse composition repair.
- work/free-quality-scenes: shots.py, previews.py, moments.py, review.py, quality_repair.py; coherent moments and audio-preserving visual repair.
- Root integration: cli.py, production_quality.py, integration tests, final reports, CI and documentation.

Component work-in-progress commits use [skip ci]; the final combined commit must run the full suite.
Production is not restarted until validated code is merged and a correctly aligned new request is created.
Creative targets and optional enrichments remain nonfatal; no paid models, paid keys, model probes or YouTube publication.
