# MOVIE RANKS — Gemini edition

Independent copy of the normal [Top10-Movies](https://github.com/justsomeoneretarded/Top10-Movies) pipeline at `5daddc9c891dc39664e47084405223daccb53673` (historical production code `c6f82a62`). All bundled voices, music, fonts and meme assets are retained. Gemini integration is being built on an isolated feature branch. See [baseline provenance](BASELINE.json).

# Top10 Movies

An unattended, CPU-based GitHub Actions pipeline: enter a video title, creative brief, and movie/show names with years, then choose Movies or Shows and a narrator from dropdowns. Receive a countdown with automatically chosen source-matched resolution, up to 4K, with an Auto or selected video length, uploadable subtitles, description, recommended title, and three thumbnails.

**No paid inference APIs.** Scriptwriting uses Gemma 4 12B through local Ollama; narration uses Chatterbox; shot matching uses CLIP; captions use faster-whisper with the exact script text. GitHub account usage/storage limits still apply. This is an initial implementation, not a guarantee of editorial quality or successful downloads for every title. End-to-end production must be verified on your runner with your configured secrets.

## One-time setup

1. Obtain an **API Read Access Token** from [TMDB API settings](https://www.themoviedb.org/settings/api). Add it as repository Actions secret **`TMDB_READ_TOKEN`** in **Settings → Secrets and variables → Actions**. Use the long bearer/read token, not the short API key. Source and attribution notes are included in `sources-and-credits.txt`; the requested disclaimer is omitted from the video description. TMDB terms still apply, and commercial use may require separate terms from TMDB.
2. The three narrator reference clips are already included in `assets/voices/`. **No voice secrets or audio uploads are needed.**
3. Enable Actions if GitHub asks. No paid LLM/TTS/image accounts are needed.

| Narrator | Included reference | Narrator dropdown |
|---|---|---|
| Warguy | `assets/voices/warguy.ogg` | Warguy (default) |
| Kit Laser | `assets/voices/kit-laser.ogg` | Kit Laser |
| Female | `assets/voices/female.ogg` | Female |

These are public 12-second reference excerpts starting five seconds into each of the three recordings supplied by the owner, published with their approval. The original full-length recordings are not required for production. The clips are mono Opus files, normalized for cloning. Voice similarity has not been subjectively auditioned here.

To replace a reference, use `python scripts/prepare_voice.py NAME /path/to/audio.mp3 --start 20 --seconds 12`, then commit only the resulting `.ogg` file. Choose clear speech by one permitted speaker without music. Do not commit `.secrets.env` files or API tokens. Optional private base64 references remain supported when no local file exists; they are not needed for this repository.

## Generate a video

Open **[Actions → Generate Video](https://github.com/justsomeoneretarded/Top10-Movies-gemni-edition/actions/workflows/generate-video.yml) → Run workflow**.

After a code update, start a **new run** with branch **main** selected. GitHub's **Re-run jobs** button repeats the original run's commit; it does not use the latest fixes from main.

There are eight inputs: title, description / creative brief, movies, a **Movies or Shows** dropdown, a **Narrator** dropdown, a **Speech model** dropdown, a **Video length in minutes** selector, and a **Try 4K** checkbox. Defaults are Movies, Warguy, original Chatterbox, Auto and Try 4K checked. The script fact-check checkbox and factual/spoiler gates have been removed. A general writing editor still improves clarity and removes filler; it does not enforce source-claim or spoiler checks. Movie identity, audio completeness and timing checks remain separate.

```text
Title:
Top 5 Mind-Bending Science Fiction Movies

Description / creative brief:
Make a cinematic countdown with clear story setups and a useful opening hook.
Build tension, feature striking visuals, and let brief
trailer dialogue play between narration sections. Make the thumbnail
about mystery and impossible worlds. Avoid exaggerated claims.

Movies:
Arrival (2016); Inception (2010); Coherence (2013); Interstellar (2014); Predestination (2014)

Movies or Shows:
Movies (default), or Shows for a list of TV / streaming series

Narrator:
Kit Laser (Warguy is the default; Female is also available)

Speech model:
original (default)

Video length in minutes:
Auto (default), or choose a number such as 8

Try 4K:
Checked (default)
```

List titles **best first** (#1 first); the rendered countdown plays from #5 down to #1. Use semicolons or newlines. Choose **Shows** for an all-show list and provide each show's first-premiere year. Choose **Movies** for an all-movie list. This choice controls research even if the video title uses a different word. Supports 2–15 titles. A “Top N” title must match the list length. Upcoming titles work when TMDB has their release year, synopsis, and a usable trailer; the writer is instructed to avoid presenting them as watched/released.

The **Narrator** dropdown defaults to **Warguy** and overrides voice instructions in the brief. Select **Kit Laser** or **Female** when desired; Female is never selected automatically. Local/API use without an explicit selector retains the earlier brief-based narrator selection, defaulting to Warguy.

When the run succeeds, download **Artifacts → movie-video-N**. The ZIP contains:

```text
final-video.mp4
subtitles.srt
description.txt
recommended-title.txt
title-options.txt        top three titles, for YouTube Studio's title test
tags.txt                 comma-separated search tags for the Tags field
pinned-comment.txt       a comment question to post and pin
thumbnail-01.jpg         number one's face over a dark scene
thumbnail-02.jpg         diagonal panels
thumbnail-03.jpg         strongest genre frame
thumbnail-04.jpg         streaming-service mark
thumbnail-05.jpg         short headline
review.html              one-page quality report: contact sheet, checks and thumbnail scores
sources-and-credits.txt
```

With the optional comparison export enabled (repository variable `COMPARISON_EXPORT=true`), the ZIP also contains `final-video-augmented.mp4`, `comparison-readme.txt` and `comparison-settings.json`.

Upload the SRT separately in YouTube Studio. Captions cover narration and retained intelligible trailer dialogue; captions and chapter times are adjusted if final audio tempo changes. Narration captions use the exact script words (so titles and names are spelled as written) with speech-recognition timing; trailer dialogue captions are speech-recognized and can contain transcription errors. Subtitles are not burned into the picture. New artifacts expire after **48 hours (two days)** from upload. The description includes the first two spoken paragraphs, a plain-language topic summary, 3–5 relevant hashtags, chapters, and a featured movies/shows list. Search-friendly metadata does not guarantee ranking. Chapters are also embedded in the MP4. To get chapters on YouTube, paste the timestamped description into your upload; the workflow does not upload or update YouTube videos.

## Production stages

1. Validate inputs, the video-length selection, the Try 4K setting, and the selected voice before downloading models.
2. Resolve title/year against the selected Movies or Shows medium, then add TMDB tagline, runtime, keywords, ratings, stills and keyless Wikipedia summary/production/reception notes (plot sections are not collected), always selecting the first eligible result in TMDB's returned order. Do not rerank by vote counts or reject matching namesakes as ambiguous. If no search-year match exists, check dated festival/regional release records, then use the Wikidata/IMDb-ID fallback within the chosen medium. Never accept a nearby year without a supporting date. Local use without a type selection retains automatic movie/TV discovery.
3. Discover TMDB-linked YouTube (and Vimeo, as a fallback) trailers/teasers and bounded YouTube search matches. Prefer an official primary trailer. An extra source may be a fan edit of the same movie/show when its title/year checks match; concept, AI-generated, fake, reaction, and review videos are excluded.
4. Fetch primary footage in a first round: download one usable primary trailer for every title before looking for second trailers. Choose the highest shared export resolution up to the checkbox ceiling. In separate second and third rounds, try to add trailers with different scenes for each title at that resolution. Finish each round for all titles before starting the next; parallel footage warm-ups fetch primary trailers only. Discovery and retries are bounded; keep at most three trailers per title temporarily in `work/footage/`. One usable trailer is enough, and missing or rejected extra trailers never fail production. No user-supplied URLs or footage are required. Only accessible non-DRM footage is supported.
5. Look up US streaming providers using TMDB's JustWatch-powered watch listings, with a 24-hour cache. Keep subscription, free, ads, rental, and purchase options distinct. Missing or unavailable listings are skipped without failing production.
6. Plan the whole episode with `gemma4:12b` through local Ollama, then write plot-led reviews with varied openings and a general spoken-language editing pass. Story context is separated from ratings, taglines and database readouts. There is no adjective-replacement dictionary, mandatory opening rotation, retained-length percentage, or rank-based narration quota. Editing may shorten a section substantially. Auto supplies no review word target; selected minutes remain soft ±25% guidance.
7. Clone the selected reference with **original Chatterbox on CPU**, using one model and the same selected voice reference throughout the episode. Preserve the existing audio processing chain. Generate bounded complete thoughts, account for existing silence between phrases, and choose pitch-preserving pacing from the measured take rather than blindly stacking speed multipliers. Short title announcements keep their articulation. Retry incomplete or invalid speech with the same voice.
8. Transcribe narration and primary-trailer dialogue with CPU/int8 faster-whisper. Verify overall narration and its opening/closing words, retrying incomplete speech. Place one funny like/subscribe break at a review boundary around 30–50% of measured runtime; shorten its audio to keep it below 10 seconds after final tempo adjustment.
9. Match frames from the accepted source pool loosely to the current narration using CLIP. Use up to three trailers when useful. Shot length follows each line's delivery: hype lines cut at up to about 4.5 seconds (3.8 for action lists), normal lines about 7, calm 9 and dramatic 10, with spaced face holds and a multi-title intro. Weak visual matches never cause a relevance-validation failure.
10. Open with a 2.3-second animated MOVIE RANKS ident: colored light ribbons resolve into the glossy red M with an original two-hit sting. The footage-led cold open stays brief unless genuinely compelling action justifies more time. Every ranked entry first speaks “Number N: Title” over a matching trailer still and animated rank/title/year card. A complete narration-free preview follows, then the review begins without repeating the announcement. Native scene audio takes priority over the music bed during suitable previews. Chapters begin at the spoken reveal, and subtitles use each separately verified speech take. Close with the existing end screen and final hold.
11. Export H.264/AAC at 30fps, using the highest source-matched resolution shared by all selected trailers, up to 4K when checked or 1080p when unchecked. Apply modest pitch-preserving tempo adjustment only when needed. Master the finished audio to −14 LUFS integrated with a −1 dBTP ceiling (YouTube's playback loudness). Embed chapters and write synchronized SRT and description chapter timestamps.
12. Build five thumbnails that read as the video's genre at a glance. The genre comes from the title, brief and the films' TMDB genres (horror, thriller, action, sci-fi, comedy, drama). Frames are found by searching the AI description of every trailer shot for the genre's signature moments (for horror: terrified faces, lurking figures, masks, creatures, blood, darkness), plus TMDB stills; the local vision model rates how instantly each frame reads as the genre and flags logos or text, and frames with captions, title cards or a corner logo are rejected. Every design uses the titles' own official stills and trailer frames, uncut (no generated backgrounds and no pasted-on actors). Five designs: number one's most genre-defining face, big and lit; diagonal panels of the most genre-defining frame from three titles; the list's single strongest genre frame, full bleed; number one with a large streaming-service mark (its real service when listed, otherwise Netflix, HBO Max or Prime Video at random); and a countdown design: the title's number huge in yellow, two bold words and a red year tag on the left, the list's biggest moment on the right. Each genre has its own grade and glow colour (horror: crushed blacks, cold desaturated tones with reds kept, heavy vignette, grain, red rim), checked at phone size. A service badge appears only when number one is listed on that service. The previous headline designs remain as a fallback: a face-aware hero still of number one with a large two-tone headline, a cutout of a person from number one's still (BiRefNet, falling back to U2Net) over the next picks, and the Netflix-style option. The generated-scene designs remain the fallback, or the default with `THUMBNAIL_SOURCE=ai`. Rank 6–8 model-proposed titles for length, count, topic words and no bait, and write tags and a pinned comment: an either/or question about the top two picks ("Which 2026 show are you watching first: A or B?"), which the narrator also asks at the end, followed by the plan's own question.

**Retention.** Right after the intro, a three-second flash of number one (unnamed, "#1 — stay for number one") gives viewers a reason to stay. Each entry opens with a different reveal (punch-in, whip-pan, pull-back or accent flash, each with a matching sound); number one always gets the punch-in. A small countdown tracker sits lower left, where the rank and title box appears: it comes in once that box has gone and leaves before the next one arrives. In the last seconds of every entry an "UP NEXT #4 Title" pill takes its place (number one stays secret), so viewers who aren't interested skip ahead with the chapters instead of leaving. Turn these off with the `TEASE_NUMBER_ONE=false` or `UP_NEXT=false` environment variables.
13. Validate output files, usable media streams and duration, thumbnail dimensions/size, and a full FFmpeg decode. A healthy export is not rejected solely for missing the length goal. Upload the package for two days, remove downloaded trailers and render intermediates, and save bounded speech/script checkpoints. An independent hourly maintenance workflow expires production caches after the run has been completed for at least 24 hours.

No approval or publication steps occur. Successful runs deliver downloadable files; this repository does not upload to YouTube.

## How a run is split across machines

One **Run workflow** press starts several jobs on separate free runners:

1. **prepare** — research only (TMDB, Wikipedia, watch listings) and the video's genre; a few minutes.
2. **plan** — the episode plan: the video's promise, hooks, comparison anchors and standout scenes.
3. **footage** — one job per title, starting right after research: primary and second trailer download, end-slate and text-card scan, sound-peak and shot analysis, AI scene descriptions.
4. **music** — an original instrumental score in the video's genre with ACE-Step 1.5 (MIT); if it fails, the bundled licensed track is used. Turn off with the repository variable `GENRE_MUSIC=false`.
5. **script** — one job per title plus intro and outro, after the plan: each writes and edits its section and records its narration.
6. **produce** — the full pipeline. It reuses every warm-up result and continues with the joining pass, remaining narration, editing, mixing, thumbnails, the review report and packaging, then uploads the single **movie-video-N** artifact.

Warm-up jobs are optional: if one fails, produce does that part itself, so a failed warm-up only costs time.

## Optional repository variables

Set these in **Settings → Secrets and variables → Actions → Variables**. None are required.

| Variable | Default | Effect |
|---|---|---|
| Speech model input / local `TTS_MODEL` | `original` | Select original Chatterbox, `turbo`, or `nano` for the whole run. |
| `THUMBNAIL_SOURCE` | `stills` | `ai` uses the generated-scene thumbnail designs instead of TMDB stills. |
| `THUMBNAIL_TEXT` | `none` | `minimal` adds a 2–3 word label along the bottom of the panels thumbnail. |
| `COMPARISON_EXPORT` | `false` | `true` also exports the augmented comparison video (roughly doubles final encode time). |
| `SCENE_DESCRIPTIONS` | `true` | `false` skips the AI scene descriptions (Qwen3-VL) used for matching footage to narration. |
| `DIALOGUE_SEPARATION` | `true` | `false` plays trailer dialogue moments with the trailer's own music instead of separating voices (Demucs). |

## Trailer download fallbacks

GitHub-hosted runners are often refused by YouTube with a sign-in/bot check. Each run layers free, optional fixes, and keeps using whichever route works:

1. The private `YOUTUBE_COOKIES` secret, when configured.
2. A local proof-of-origin token server (`bgutil-ytdlp-pot-provider`, Docker) that the yt-dlp plugin uses automatically.
3. A Cloudflare WARP route (free; `wgcf` + `wireproxy` as a local SOCKS proxy) used only after a bot check on the direct route.
4. Vimeo copies of trailers listed by TMDB.

These are best-effort workarounds, not guarantees; the clear bot-check failure message remains when every route is refused.

## Benchmark Free Models workflow

**Actions → Benchmark Free Models** measures candidate free models on the same CPU runner and uploads samples to listen to: script models through Ollama, Chatterbox Turbo and the original Chatterbox, Qwen3-TTS voice cloning, and ACE-Step 1.5 instrumental music. Timings appear as annotations on the run page. Choose `all` or one family (`llm`, `voice`, `music`).

## CPU performance and reliability

Original Chatterbox is the default for future runs. The **Speech model** input applies to all warm-up and final jobs; it does not read the old repository `TTS_MODEL` variable. Explicit Turbo/Nano choices remain available. Local CLI runs use `TTS_MODEL=original` by default. All deliveries use one selected model and reference, without a separate emotional model or darker recording. Original's expression changes gently by delivery; local `VOICE_EMOTION=false` keeps its neutral settings without changing models. Generation stays bounded to about 45 words per original-model request to leave room below its speech-token limit, preserving the full text across requests. Original can take longer on CPU; incomplete speech retains same-model retries and transcript verification. Its own weights are downloaded by the existing pinned `chatterbox-tts` dependency and cached under the selected model. Speech caches include model, reference, delivery settings and phrase policy, so Turbo recordings cannot be returned for original Chatterbox. Existing model/Python caches are preserved.

Public and private GitHub-hosted runners have different resource allowances. The workflow has a 350-minute timeout; first runs download several GB of models and dependencies. Model and job caches accelerate later runs, but GitHub may evict them. Free storage/minute quotas are finite; inspect your account's billing settings. Standard GitHub runners do not provide a free GPU. A failed/timed-out run delivers no partial package. Re-running a failed job can restore checkpoints saved by the failure handler; runner cancellation/timeouts may prevent cache saving.

Some upstream model/tool dependencies are pinned, while yt-dlp remains updateable to track changing sites. Ollama's installer and model tag track their upstream distribution. For strict reproducibility, pin their versions/model digests after a verified production run. Do not expect the first unbenchmarked run to have a guaranteed completion time.

YouTube can block data-center downloads or require authentication. Neither yt-dlp nor this workflow guarantees access. It does not bypass DRM or bot challenges. The pipeline stops clearly when required footage is missing instead of substituting a different film. A self-hosted runner is an alternative if hosted-runner access is unreliable. Publicly visible trailers are copyrighted; downloading and using them must comply with source terms and applicable rights. Short clips and commentary do not guarantee clearance or prevent Content ID claims.

Automated shot matching, trailer dialogue selection, factual accuracy, and narration quality remain imperfect. CLIP matching is a heuristic; the system does not understand every scene or guarantee spoiler avoidance. The three thumbnails are options, not tested CTR winners. The implementation uses clean cuts and rank graphics; it does not claim to reproduce a professional human editor's judgment.

## Resolution, disk space, and downloaded footage

Model setup now frees only unused SDKs on GitHub-hosted runners, preserves Python/model caches, and checks working-disk headroom before large installs. Ollama uses a pinned, checksum-verified archive with CPU runtime extraction and bounded retries. If optional voice pre-generation would leave too little space for the script model, the main job records that narration later with the same voice and quality. See [runner setup and disk safeguards](docs/RUNNER-SETUP.md). Start a new run from **main** to use updates; re-running a historical run uses its historical code.

The **Try 4K** checkbox is checked by default. Checked means “choose the best shared quality up to **3840×2160**”; unchecked caps the search and export at **1920×1080**. The brief no longer controls resolution, and no `VIDEO_RESOLUTION` repository variable is needed. Thumbnails remain 1280×720 JPEGs.

The pipeline downloads the best primary trailer for each title, checks their actual dimensions, then chooses the highest export resolution all primary sources support. All-4K sources can produce 4K; a mix of 4K and 1080p produces 1080p; 1440p, 720p, or lower sources lower the shared export accordingly. A weaker source is not stretched to force a 4K-sized file. If a low-resolution candidate would lower the result, the downloader first checks other candidate trailers for that movie and prefers a better source when available. It retains at most three trailers per title, with an 800 MB cap for each file.

After the first-trailer round has completed for every movie/show, the workflow starts separate second- and third-trailer rounds to enrich each title, without requiring another input. Each extra source must meet the already-selected export resolution, so optional extra footage cannot lower the result or force upscaling. Sampled-frame perceptual hashes estimate whether it adds different imagery instead of repeating a reupload. This is a scene-diversity heuristic, not proof that every scene is new or that a fan edit contains only footage from the intended title. Metadata checks still tie candidates to the same movie/show. Missing, inaccessible, too-similar, low-resolution, or unusable extra trailers are skipped; production continues with the primary source. A working-disk reserve also takes priority over keeping extra footage.

Missing 4K footage does not cause a resolution error: production continues at the available shared quality. Logs and the cached job's quality-report JSON record accepted source dimensions, including optional extra trailers, and the selected export resolution. All accepted trailer source URLs are included in the description. Website access and native 4K availability are not guaranteed.

4K is supported but expensive on a standard CPU runner: it has four times as many pixels as 1080p, can take substantially longer to encode, and may reach the 350-minute job timeout. Standard runners have finite temporary disk, and models, dependencies, trailers, rendered parts, concatenated input, and final output can coexist during production. Conservative free-space checks stop with an explanation when the working reserve is insufficient. 1080p is the practical choice for more dependable free runs; no 4K completion-time guarantee is made.

New artifacts expire after **48 hours (two days)** from upload. That reduces how long finished packages occupy artifact storage; it does not increase disk space while a job is running. This setting applies to newly uploaded artifacts; previously uploaded packages keep their original expiration. Production caches have a separate 24-hour cleanup policy below. Model and Python dependency caches are preserved by this cleanup, but GitHub's own quotas and eviction rules still apply.

During a run, trailers and their analysis files live under `work/footage/`; rendered intermediate clips live under `work/jobs/`. The cleanup step runs after success or failure, deletes primary and optional extra trailers plus rendered MP4s, and preserves `output/`, model weights, and the bounded speech/script checkpoints. Trailers must therefore be downloaded again on a new run. A cancelled job or runner shutdown can prevent cleanup or checkpoint saving, but GitHub-hosted runner disks are temporary and are discarded with the runner. These directories are never committed to the repository by the workflow.

## Streaming availability and descriptions

Watch listings use **US availability** from TMDB's JustWatch-powered endpoint. Successful lookups are cached for 24 hours and refreshed as needed. After a review, narration can say where the title is listed, distinguishing a subscription from a free-with-ads, rental, or purchase offer. The spoken line omits the country as requested. The description labels the listings as US and records the check date; availability can still change or be incomplete. Missing data or a provider outage omits the claim and does not fail the video.

`description.txt` begins with the first two spoken script paragraphs, followed by a simple topic summary and 3–5 relevant hashtags. It also includes timed chapters and a heading that matches the content: Featured movies, Featured shows / series, or a mixed heading. The former TMDB disclaimer sentence is omitted from this file as requested; source and attribution notes are kept separately in `sources-and-credits.txt`.

## Local development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q

# Full production also needs FFmpeg, fonts, Ollama, and CPU speech dependencies.
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[voice,thumbnails]'
ollama serve
# In another terminal:
ollama pull gemma4:12b
export TMDB_READ_TOKEN='your-read-token'
# Narrator reference clips are already included in assets/voices/.
movie-pipeline --title 'Top 2 Sci-Fi Movies' \
  --brief 'Plot-led and conversational. Use Warguy.' \
  --movies 'Arrival (2016); Interstellar (2014)'
```

Never put API tokens or voice base64 in workflow inputs. Inputs are visible in run metadata. The tests exercise input parsing, narrator defaults/negation, subtitles, trailer filtering, workflow safety, and actual FFmpeg rendering/thumbnail generation without downloading AI models or copyrighted media.

## YouTube sign-in/bot errors

If research succeeds but downloads say “Sign in to confirm you’re not a bot”, YouTube is rejecting the runner/session. It is not a TMDB-token or movie-list error. Switching video IDs or downloader forks does not reliably resolve it.

Optional authenticated access:

1. Export your YouTube session cookies as a **Netscape-format `cookies.txt`**, following [yt-dlp's current export guidance](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies). Use a trusted local export method; treat the export as account credentials. Do not share it in chat, issues, or public repository files.
2. Add a repository Actions **secret** named `YOUTUBE_COOKIES`. Paste the file's complete contents, not a filename, JSON, or base64. The secret must fit GitHub's 48 KB limit; export YouTube cookies only.
3. Start a **new Run workflow on main** so it uses the updated workflow. Re-running an older run uses that run's old workflow revision.

The workflow writes cookies to a permission-restricted file under `RUNNER_TEMP`, outside footage/model caches and output artifacts, and removes the file afterward. Cookies remain private even though narrator reference clips are public. Use authenticated downloads only where permitted by the source's terms and applicable rights.

**This is an authentication option, not a guaranteed fix or CAPTCHA bypass.** Sessions expire, and GitHub-hosted IP addresses may remain blocked even with valid cookies. In that case a self-hosted runner on a network where normal permitted access works, or another authorized footage source, is required. There is no dependable promise of unattended, free YouTube downloading forever from GitHub-hosted runners.

### Release-year differences and independent identity lookup

The resolver follows a **first matching result** policy: after filtering by title and supported release year, it always uses the first result returned by TMDB within the Movies or Shows selection. Vote counts, low popularity, zero votes, or additional matching records do not cause an ambiguity failure. This preserves TMDB's search ranking rather than attempting a second ranking. TMDB can occasionally put a different version first, so this policy favors unattended completion over additional identity checks. Logs show the selected TMDB kind and ID. Independent lookup is used only when no eligible TMDB/release-date result exists, not as an extra gate for a valid first result.

A title may premiere at a festival in one year and reach theaters in the next. The resolver preserves the year you entered and checks supporting release dates before rejecting it. If needed, a keyless Wikidata lookup checks the English label/aliases, publication-year claims, and IMDb identifier. TMDB's IMDb-ID lookup then retrieves the same film's metadata and trailer references. This is an independent **identity/year check**, not scraping IMDb reviews or plots. Synopses and trailer references still come from TMDB. Source outages and incomplete claims can still prevent verification; logs report that without assuming the input is wrong. No additional account or API secret is required.

## Choose a video length

Use the workflow's **Video length in minutes** selector. The creative brief no longer controls runtime.

- **Auto (default):** plan an appropriate amount of commentary for the list, then let complete reviews determine the result. A top-five shows video is usually planned around 7–9 minutes; a top-ten movies video can reasonably be around 13–15 minutes. Auto does not turn a small list into an arbitrary 100-minute video or stretch useful narration to fill a fixed quota.
- **A selected number:** every minute option uses a **±25% soft target**. Selecting **8** means **6–10 minutes**; selecting **5** means **3:45–6:15**; selecting **15** means **11:15–18:45**. The selector offers every whole minute from 1 to 30.

The initial Auto estimate is only a resource-planning fallback. Auto reviews have no word target, and a useful short script is kept without length-padding retries. For selected minutes, an unusually long draft may receive a bounded shortening pass; shorter useful drafts are not expanded to reach a quota. The finished video remains at 1× speed. Requested minutes still allow ±25%, but a healthy outlier is delivered rather than failing solely on length. Chapters and subtitles use the actual measured narration and rendered timeline.

This does not remove real resource or media failures: longer videos require more generation/render time and storage, and GitHub's 350-minute job timeout and account quotas still apply. Missing required footage, broken media, or failed narration can still stop production. Automated tests check timing selection, script-budget scaling, and media export; full live model output still needs verification on the runner.

## Steadier editing, hooks, and calls to action

Moving excerpts follow the delivery of the line they cover (about 4.5 seconds for hype lines up to about 10 for dramatic ones), never above a twelve-second ceiling even when footage is exhausted. The planner shortens that ceiling when a final playback slowdown is expected. Small tails are balanced into readable beats instead of extending a shot or adding a freeze at every paragraph boundary. The intro uses roughly three-second images from several titles, covering at least half the list when its duration allows that without rapid flashes. The first visual can come from anywhere in a trailer, not just its opening.

Occasional 4–7-second holds show a useful face or a calm, loosely relevant detail. A shared cooldown requires at least two moving excerpts and about fourteen seconds between deliberate holds, including across paragraph and review boundaries. They use a slow zoom of at most 3.5%, sometimes a gentle pan, with varied directions. A freeze at the end of moving footage begins its zoom from the existing framing rather than jumping in. Small color/contrast adjustments stay consistent across footage; reaction GIFs remain untouched and trailer title cards remain static for readability. The clean export adds no shaking, flashes, flipped footage, decorative borders, or audio distortion. The separate comparison export applies the documented augmentation. Limited source footage can still require compromises in shot choice.

Scene choices share one viewing history across the cold open, previews, reviews, and recap. Unused source intervals take priority over replaying a previously attractive scene; longer shots continue from an unused point rather than restarting. Sampled perceptual hashes also discourage replaying the same imagery from a different trailer. Once the available footage is exhausted, the editor spaces reuse apart instead of repeatedly looping the same opening or increasing the cut rate. These visual heuristics cannot identify every duplicate shot.

Each ranked entry follows **spoken rank/title with its animated card → complete native preview with no narrator → review body**. Announcement and review are generated as separate complete speech takes; inserting a scene never requires cutting through a word. A subtle moving still from the selected trailer holds the card, then the preview plays from its natural beginning without a second rank reveal. Optional native moments remain between review paragraphs. A preview can be shorter or longer when its natural boundaries justify it; missing optional scenes do not fail production. Chapters start at the announcement.

Detected trailer title/promo cards are excluded from the middle of reviews. When a title card can be matched to the current movie/show using local OCR, its frame is held briefly at the end of that review. Missing or unreadable title cards simply skip this optional beat; the opening rank/title/year label still identifies the entry. Tesseract runs only on a bounded set of candidate frames and needs no API key.

CLIP matches the current thought in the script loosely to available frames across the accepted trailers; relevance is a preference, not a pass/fail condition. The intro can choose the stronger source for each movie/show, and review paragraphs can draw on both while keeping the same steady pacing. It is not a guarantee that every shot illustrates a sentence precisely. When narration discusses a performer or character and a suitable face frame is detected, the edit can pause on it for a few seconds. Face detection is not actor-identity verification. Short dialogue inserts use the primary trailer and keep their selected audio/video context, with narration silent during the insert. Second-trailer footage is muted under the original commentary.

Hooks use the owner's conversational, curiosity-driven examples as a style direction, adapted to the actual title count and genre. They avoid inventing viewing history, statistics, awards, or availability. Slang can appear naturally without being forced into every line. A rare optional “Absolute Cinema” meme image may be used only for a matching hook and approximately one in eight eligible jobs. The image is fetched from Imgflip; missing images or failed downloads simply skip the meme. It is never a required production dependency.

The spoken intro aims for two useful sentences and roughly eight seconds, after the five-second visual cold open. It gives the topic and a concrete reason to watch instead of explaining how a countdown works or stacking empty teasers. Script checks repair numeric formatting debris, obvious filler, and repeated bait. Older cached scripts receive this editorial refresh while valid review text and its speech cache remain reusable. If the recorded intro runs long, one bounded rewrite can shorten the complete thought; eight seconds remains guidance, never a hard rejection or an audio trim.

A funny like-and-subscribe message appears once between reviews, using the measured timeline to target 30–50% of the video. It never interrupts a review. The writing model can create a fresh short line paired with one of nine locally bundled reaction assets. There are also 54 matching backup jokes, with reproducible selection across the full library instead of one fixed genre-to-meme routing. Checksums, image decoding and complete-speech verification remain. Missing reactions fall back to a held film image without failing production. The existing like/subscribe/bell animation and synchronized sounds stay. The end-screen plate now lasts **10 seconds**, followed by the existing **3-second final hold**.

Speech has a small leading/trailing silence margin. Verification checks the beginning and ending as well as overall transcript coverage, so a paragraph cannot pass merely because most words survived. Every intermediate uses the same lossless 24-bit ALAC, 48 kHz stereo configuration. Older or incompatible parts have only their audio normalized before assembly, preserving their video bytes. This prevents mixing 16-bit narration packets with a 24-bit preview header, which previously corrupted the assembled voice. The final MP4 uses AAC once, with timestamps reset at the final mix.

Assembly strictly decodes the joined audio before the final render and verifies that both streams cover the complete edit. Timestamp-declared gaps are preserved during mixing so older source audio does not shrink against the picture. The finished export receives another audio decode/duration check and bounded checks of its first, middle, and last spoken paragraphs against their verified recordings. Uncertain acoustic comparisons can request independent recognition; an inconclusive secondary check is reported rather than treated as proof of damaged speech. Definite missing/corrupted narration prevents delivery. Failure diagnostics include `audio-assembly.json` and `audio-review.json`. Real FFmpeg regression tests cover mixed intermediate formats, narration boundaries, final tempo changes, and the ending hold; actual model pronunciation and delivery still vary.

The final encoder reads video parts sequentially and rebuilds their display clock instead of copying every part into one intermediate MP4. Independently encoded H.264 clips can have different B-frame reorder delays, which made that old copy step fail with “Non-monotonic DTS.” Audio is joined separately without loss and checked before mixing. Precise stream durations determine the boundaries, avoiding accumulated container-rounding error across many parts, while declared video/audio timing gaps remain intact. This keeps the existing final video encode and avoids creating another full-size video intermediate. Regression checks include a 63-part mixed-delay sequence, every output frame at normal speed, scene/audio boundaries, and both supported tempo extremes.

Chapters include Intro, each movie/show with its timeline position, and Outro. They are adjusted after final tempo changes, written into `description.txt`, and embedded in the MP4. YouTube chapter display requires pasting the timestamped description into the video upload and remains subject to YouTube's chapter eligibility rules.

Editing is for pacing and original commentary. Cuts, shuffling, frame holds, overlays, memes, or music do not by themselves establish fair use, prevent Content ID claims, or guarantee YouTube monetization. Rights and YouTube's reused-content assessment are separate from the edit effects.

### Voice speed and failed-run recovery

Speaking-rate measurements are now logged as diagnostics. A fast or slow delivery alone does not reject a segment: the transcription stage checks whether the words were actually spoken, and final duration control handles overall pacing. Voice synthesis still rejects empty, silent, non-finite, or extremely long/runaway output after bounded retries.

All spoken sections share the same recovery process. Recognition first checks the recording normally, then retries the unchanged recording without voice-activity trimming and with a wider decoding search. Only uncertain clips request an additional CPU/int8 `small.en` recognition model; it has a separate reusable `models-cpu-speech-verifier-*` cache and leaves existing model/Python caches in place. Recognition never receives the expected script as a prompt. Harmless spelling differences such as `high-five` versus `highfive` are normalized, while missing opening/closing phrases, substantial repetition, and reordered clauses remain failures. Correct recordings are retained without changing their voice or speed.

Genuinely incomplete speech gets fresh synthesis attempts in the selected voice, using separate candidate files so rejected audio cannot hide behind a cache hit or overwrite an accepted take. Difficult review paragraphs can be regenerated as complete thoughts, retaining the original words and verified caption timing. The subscribe break is checked before full narration generation. It can try another short joke paired with the same reaction image, rather than forcing a long take to play unnaturally fast. Its recording stays within eight seconds, leaving room for the visible animation while remaining below ten seconds.

If speech remains unusable after bounded recovery, the workflow preserves the quality checks and explains what it expected and recognized. A failure-only `speech-diagnostics-N` artifact contains the verification report and at most one small rejected recording, retained for 48 hours. Successful runs do not upload this extra artifact. Models and Python dependency caches keep their existing retention behavior; saved usable narration checkpoints remain eligible for reuse.

A new workflow run can restore earlier research, script, and completed narration checkpoints while they are retained. Downloaded footage and rendered MP4 intermediates are excluded. End-of-job cleanup keeps at most the newest two job folders, bounded to 2 GB total before saving; models use a separate cache. Each job's paths include its inputs, Movies/Shows choice, and selected voice, so different selections cannot reuse the wrong research, script, or narration. Neither cookies nor voice secrets are included in the checkpoint cache.

The **Expire Production Caches** workflow runs hourly and can also be started manually from Actions. It uses GitHub's automatic workflow token; no new secret is needed. It deletes only `production-v1-*` and `production-v2-RUN_ID-ATTEMPT` checkpoint caches, by cache ID, once both the cache and the completed production run are at least **24 hours old**. Active or unverifiable runs are skipped. A rerun restarts the completion window. Unattributed older `production-v1` keys wait until all video runs have ended and the latest completion is at least a day old. Failed/cancelled runs also receive a day to retry before their verified completed-run caches expire. Legacy caches may therefore be kept longer while new production continues.

The schedule normally removes eligible checkpoints within **24–25 hours after completion**, and can run later if GitHub delays scheduled jobs. It does not hold a production runner open for a day. Cleanup has its own short job and narrowly scoped Actions write permission; the video-production job retains read-only repository permissions.

| Stored item | Retention / cleanup | Effect |
|---|---|---|
| Production checkpoints (`production-v1-*`, `production-v2-*`) | Eligible after 24 hours from verified completion; checked hourly | Reruns can reuse scripts/speech before expiry; afterward they may need to regenerate them. |
| Finished downloadable package | New uploads retained for 48 hours | Available to download independently of production-cache cleanup. Downloaded copies remain yours. |
| Model caches (`models-cpu-*`) | Untouched by this cleanup | Existing model downloads remain reusable while GitHub retains the caches. |
| setup-python / pip and all other cache keys | Untouched by this cleanup | Dependency caching keeps its existing behavior. |
| Downloaded trailers and rendered intermediates | Removed at the end of production | Temporary runner files are not kept in checkpoint caches. |

Deleting checkpoints does not change video quality, remove an existing artifact, or require downloading cached models again. The cost is reduced recovery/reuse: rerunning the same input after expiry can regenerate research, scripts, speech, captions, and thumbnail work and take longer. A completely new video's work generally must be generated anyway. Model/dependency caches remain subject to GitHub's own storage limits and eviction; this workflow does not guarantee permanent retention. The two-day artifact setting applies to new uploads and does not extend older artifacts' original expiry.

## Telegram completion reminder

After the video is generated, validated, and successfully uploaded to GitHub Artifacts, the workflow can send one formatted Telegram reminder with the recommended title and GitHub run link, then pin that same message. Five blank lines separate the heading, title, and download sections. It does not upload the video or other files to Telegram. Failed video runs do not send a success reminder. Missing credentials or a Telegram delivery error do not fail an otherwise successful video job.

One-time setup:

1. In Telegram, open the official **@BotFather** and send `/newbot`. Choose a name and a unique bot username. Copy the bot token privately.
2. Open your new bot's chat and press **Start**, or send `/start`. Bots cannot initiate a private conversation until you do this.
3. Get your chat ID: in a private browser session open `https://api.telegram.org/botYOUR_TOKEN/getUpdates`, replacing `YOUR_TOKEN` locally with your token. Find `message` → `chat` → `id` for your own `/start` message. If the result is empty, send another message to your bot and refresh. Never share the token-bearing URL or response; it may contain private chat details. Close the private browser afterward.
4. In repository **Settings → Secrets and variables → Actions**, add:
   - `TELEGRAM_BOT_TOKEN`: the token from BotFather.
   - `TELEGRAM_CHAT_ID`: your numeric chat ID.
5. Start a new Generate Video run on **main**. A reminder is sent only after its finished package uploads successfully.

If getUpdates reports a webhook conflict, use a dedicated new bot for this workflow instead of changing another application's bot. If a token is exposed, revoke it through BotFather and replace the GitHub secret. The linked artifacts retain the repository's GitHub access requirements; new uploads expire after 48 hours. Delivery is best effort: a network failure can prevent confirmation, and re-running a successful production job can send another reminder.

### Channel pinning and spacing

Completion reminders have bold labels and five empty lines between their three main sections. The bot pins the message it just sent; it does not delete posts or unpin other messages. For a channel, make the bot an administrator with **Post Messages** and **Edit Messages** rights (the Bot API requires edit rights to pin in channels). For a group/supergroup, enable **Pin Messages**. Pinning failure leaves the sent reminder intact and does not fail video production. Telegram clients control exact text spacing and notification behavior; pinned-message alerts in channels may be silent.

## Redesigned thumbnail generation

By default thumbnails are now built from official TMDB stills (see production stage 12). With `THUMBNAIL_SOURCE=ai`, or when stills are unavailable, the generated-scene pipeline below is used. It requests complete cinematic scenes, including subjects, lighting, and environment together. The script model proposes six concepts from the title, brief, premises, and commentary. A lightweight composition-cue selector picks one concept per style. Each selected concept gets up to two image seeds (six image requests maximum); phone-size sharpness, contrast, exposure, and duplicate checks pick a candidate. These are visual heuristics, not a vision model, anatomy checker, semantic verifier, or CTR predictor.

The reference direction is expressive close-ups, intense genre-appropriate scenes, coherent dramatic lighting, strong depth, and minimal clutter. Horror gets dedicated dread/silhouette concepts; action gets chase/impact concepts. Generated scenes are illustrative artwork, not verified movie stills or guaranteed actor likenesses.

- **Option 1:** minimal-text cinematic hero.
- **Option 2:** bold headline, at most five words, with the generated subject on the right and readable text on the left.
- **Option 3:** cinematic alternative with the requested red Netflix-style N every time. The badge does not verify Netflix availability or endorsement.
- **Provider:** retains anonymous Pollinations Classic, with no keys or paid fallback. No unverified free FLUX service is claimed. Requests are sequential, at least 45 seconds apart, requested at 1024×576 and resized without cropping away provider branding. The additional candidates can add several minutes. Cached image requests are reused.
- **Fallback:** provider failure, blank images, or duplicate winners use local artwork and trailer subject cutouts. CPU rembg/U2Net and OpenCV extraction are now fallback tools only; successful complete scenes never get pasted trailer subjects over them.
- **Outputs:** three 1280×720 JPEGs below 2 MB each. Candidate scores, selected seeds, fallback methods, and concepts are recorded in `thumbnail-work/thumbnail-report.json`, outside the downloadable package. Updated concept/image cache versions prevent old background-only plans from being reused.

Audio uses original Chatterbox by default; Pocket TTS remains excluded. Only generated visual prompts are sent to the public image provider, not voice recordings, tokens, or the whole script. Free service availability and quality are not guaranteed. Live provider quality must be checked on the GitHub runner; offline tests cover selection, composition, cache behavior, fallback, and file constraints.


### Clean and augmented comparison exports

The comparison export is **optional** (repository variable `COMPARISON_EXPORT=true`). When enabled, production delivers **two videos**. `final-video.mp4` is the clean edit; `final-video-augmented.mp4` uses the same scenes, narration, music, subtitles and chapters, then applies a fixed subpixel warp, faint Perlin noise, approximately 1% local disjoint pixel swaps, and small audio variations only outside protected narration intervals. Narration and a safety margin around it are not time/formant augmented. Frame counts and audio lengths are checked. The comparison uses streaming frames and overlapping audio windows rather than a huge raw-video file. Settings are included in `comparison-settings.json`. Augmentation can still affect perceived quality, does not anonymize recognizable people or voices, and provides no protection from copyright claims. The second encode takes additional CPU time and approximately doubles final video storage; artifacts still expire after 48 hours.

The old continuous sine-tone/pulse bed was audible in the finished video's music-only opening (normalized waveform correlation 0.998 against the generated source). Both procedural music generators have now been removed; bundled licensed recordings provide the bed. One theme is chosen for the whole video from the title, then brief: action, mystery, romance or drama. No static is added. Trailer sound stays out of narrated sections, appearing only in deliberately selected previews and up to two breaks per longer review. Complete speech windows can play in previews; boundary-crossing speech remains muted.

Narration pacing uses genre and delivery as preferences, then measures the actual words and duration of each take. Comfortable readings remain unchanged; slow or fast readings receive a bounded pitch-preserving adjustment. Extra phrase pauses account for silence already present, rather than using a fixed punctuation table. The finished footage is never globally sped up or slowed down to fit a runtime target.

Character mentions may freeze a face/character frame from the same film’s indexed trailers at the corresponding caption time. No external headshots are downloaded. CLIP cast/character matching is a soft editorial hint, not verified identity; uncertain matches are skipped without failing production. Shot detection now requires an appearance change as well as movement, so fast action alone does not force an edit.

The midroll and spoken outro retain the visible like/subscribe/bell interaction and short synchronized UI sounds. Provider details remain in the written description. Script editing rewrites weak ideas as whole sentences instead of substituting words. Script factual and spoiler gates are removed; research identity, speech-completeness, media-decode and timing safeguards remain.

A separate `audio-inspection` artifact contains bounded background-only and edit-without-music samples plus audio checks. It expires after 48 hours. Temporary audio samples are removed before saving production caches; model and Python caches are unchanged.

The CTA mixer renders its full audio separately from the animated image sequence. Both mixing inputs cover the complete section and are drained with `amix duration=longest`, avoiding FFmpeg 6's early EOF with unequal input packet sizes. The mux does not hide missing samples with padding; strict audio/video duration checks run before replacement. The final music mix also drains both inputs before the existing explicit output boundary.


### Voice clarity and trailer-only character freezes

The procedural sine-instrument bed has been removed entirely. Three bundled, hash-checked excerpts from Scott Buckley's Titan, Celestial and Aurora supply quiet, genre-based recorded music, with smooth crossfades and ducking under speech. CC BY 4.0 attribution and edit disclosure are added automatically to the public description and credits. No music download or paid API is needed at runtime. Keep the music attribution when uploading.

Narration uses a gentle 65 Hz rumble filter, low-level downward expansion, a small presence adjustment, and pitch-preserving `atempo` pacing without formant reconstruction. The final mix uses a fixed gain and peak limiter instead of a second dynamic loudness normalization pass. Music receives a small reduction in the voice range. Speech is protected from time/formant augmentation in the comparison, including at overlapping processing-window boundaries. Isolated before/after pacing samples join the bounded inspection artifact; final videos and diagnostics still expire after 48 hours. These changes do not repair an intrinsically poor TTS take; a reported problem in an expired export needs an uploaded audio excerpt for exact diagnosis.

Character freezes use actual source timestamps in that film's trailers, with restrained motion. External actor portraits have been removed. A weak or ambiguous face/character match is skipped instead of substituting a headshot or stopping the job.
