# MOVIE RANKS — Gemini edition

Independent Gemini-first copy of the normal [Top10-Movies](https://github.com/justsomeoneretarded/Top10-Movies) pipeline. Source baseline: main commit `5daddc9c891dc39664e47084405223daccb53673`, with the pre-Graphify production code from `c6f82a62`. The normal and 2.0-ish repositories are untouched. Original asset hashes and modes are recorded in [BASELINE.json](BASELINE.json).

Run **[Generate Video — Gemini](https://github.com/justsomeoneretarded/Top10-Movies-gemni-edition/actions/workflows/generate-video.yml)** on main. Enter the same title, brief, title/year list, narrator, duration and 4K preference used in the normal repository. The additional backend selector defaults to **Gemini with local fallback**; **Local** keeps the original route.

## What Gemini does

| Stage | Primary route | Existing fallback |
|---|---|---|
| Research | Google Search grounding with Gemini 2.5 Flash; cited title/year/medium and premise validation | Verified TMDB research plus optional Wikipedia |
| Script and packaging | Gemini 3.8 Flash for episode planning, narration, editorial passes, title/description and thumbnail concepts | Existing Gemma 4 12B/Ollama writer, fetched only when needed |
| Frame understanding | Gemini 3.8 Flash describes the actual trailer frames | Existing local Qwen vision and image matching |
| Native scene selection | Bounded public-YouTube analysis gives approximate visual interest windows | Existing local shot/scene analysis and transcript boundaries |
| Thumbnail shot selection | Gemini judges actual candidate frames for expression, clarity, genre fit and logos | Existing image/face heuristics and local composition |
| Narration | Gemini 3.8 Flash TTS **only for an explicitly registered matching custom voice** | Original Chatterbox using the committed narrator reference |

Gemini advice does not bypass correct-title checks, source rejection, finite video bounds, scene continuity, complete dialogue, caption alignment or final AV/speech verification. It cannot guarantee a subjective quality improvement or click-through rate.

The default text/vision model is a current documented free-tier model. If that model returns a genuine unavailable-model response or a final read timeout, the provider may use the documented free Gemini 2.5 Flash alternative. Actual successful models and fallback events are reported separately from configured model/cache identity. Gemini requests run in one final production process after research, paced by default at one start every twelve seconds. Downloads and music remain parallel. Local mode keeps the original parallel planning/script/voice warm-ups.

## Repository secrets

Set these under **Settings → Secrets and variables → Actions**:

- **GEMINI_API**: your Gemini Developer API key. **GEMINI_API_KEY** is also accepted.
- **TMDB_READ_TOKEN**: optional verified-metadata fallback. Use TMDB's API Read Access Token, not its shorter API Key. Successful grounded Gemini research does not require it.
- **YOUTUBE_COOKIES**: optional Netscape cookie export for yt-dlp when YouTube requires authentication.
- **TELEGRAM_BOT_TOKEN**, **TELEGRAM_CHAT_ID**: optional completion notifications.

Use an **unbilled Free Tier Google project**. Google does not provide a per-request "free only" switch, and model metadata cannot establish a key's billing status. This edition restricts calls to documented free-tier models, excludes paid native image generation and bounds requests/retries. Supplying a key from a billed project can still incur that project's normal API charges. Quotas and availability depend on the Google project.

### Why yt-dlp and cookies still exist

Gemini can understand a public YouTube URL. Its response does not contain a downloadable video file. yt-dlp still discovers/acquires source footage; FFmpeg decodes and renders it. Gemini cannot remove YouTube access restrictions or make cookies unnecessary on every runner.

### Warguy and Kit Laser

The original **Warguy, Kit Laser and Female** OGG reference files are copied byte-for-byte. Existing dropdown choices and Warguy default are preserved.

Current Gemini voice replication requires a real clean recording plus a separate real recording of the same adult reading Google's required consent statement. An existing character/synthetic reference alone is insufficient. This repository never invents consent or silently substitutes a stock Gemini speaker.

After legitimate enrollment with Google's Voices API, save the returned custom ID/key as the matching optional Actions secret:

- GEMINI_VOICE_WARGUY
- GEMINI_VOICE_KIT_LASER
- GEMINI_VOICE_FEMALE

Registered IDs start with voice_ or voicekey_. Repository variables with these names are supported for compatibility, but secrets are recommended. Preset names are deliberately not accepted as a substitute for a selected cloned narrator.

Without a registered matching voice, the reference uses Original Chatterbox. A backend is locked for the episode: initial Gemini unavailability may choose local before any Gemini take; once a Gemini take exists, a later failure asks for a retry with the same identity. Parallel voice warm-up is disabled for registered Gemini voices to prevent mixed-speaker archives. Normal local voice warm-up remains available.

## Output and diagnostics

The normal downloadable package is retained: final video, subtitles, title/description, chapters, ten thumbnails, credits and technical reports. An optional comparison export keeps its existing control.

Additional reports:

- **ai-production-report.json** records configured models, actual backend events and research provenance.
- Per-job **gemini-usage-*.jsonl** and **gemini-preflight-*.json** reports distinguish authenticated access, API results and fallback. They contain no API key, prompt bodies or raw voice keys.
- Grounded research sources appear in **sources-and-credits.txt**.
- Gemini scene hints are advisory caches alongside accepted footage; local windows and recognized speech remain timing authority.

The default thumbnail path remains actual images/frames composed locally. Gemini native image-generation APIs have no documented free API tier, so they are not called.

## Controls and validation

Useful optional repository variables include GEMINI_TEXT_MODEL, GEMINI_VISION_MODEL, GEMINI_RESEARCH_MODEL, GEMINI_TTS_MODEL, GEMINI_MAX_REQUESTS, GEMINI_MAX_VIDEO_SECONDS, GEMINI_MIN_INTERVAL_SECONDS, GEMINI_MAX_RETRIES and GEMINI_TIMEOUT_SECONDS. The application rejects models outside its free-tier allowlist.

Defaults bound requests per process/job, pace calls and stop repeated quota/auth failures. Model quota failures do not disable unrelated models. Google quotas are project-wide, so parallel jobs still share those external limits. Optional video analysis is capped and cannot multiply indefinitely with extra sources.

**Tests** runs existing regressions plus mocked provider, research, voice, visual, workflow and integration contracts. **Gemini Live Validation** makes a small bounded real-key check of structured output, frame understanding and grounded research without a TMDB token. It does not claim voice-clone enrollment, YouTube download success or subjective video quality.

Model/Python caches, production cleanup and two-day final artifact retention are preserved. New repositories begin with cold caches. Gemini/model/voice fingerprints isolate checkpoints from normal/local runs.

## API documentation

- [Gemini API models](https://ai.google.dev/gemini-api/docs/models)
- [Pricing and free-tier eligibility](https://ai.google.dev/gemini-api/docs/pricing)
- [Search grounding](https://ai.google.dev/gemini-api/docs/generate-content/google-search)
- [Video understanding and public YouTube input](https://ai.google.dev/gemini-api/docs/generate-content/video-understanding)
- [Speech generation](https://ai.google.dev/gemini-api/docs/speech-generation)
- [Voice replication and consent](https://ai.google.dev/gemini-api/docs/voice-replication)
