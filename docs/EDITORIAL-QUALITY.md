# Movie Ranks editorial pipeline

## Episode intent

A genre in the title, or an explicit genre request in the brief, chooses the episode theme.
A general or mixed list uses neutral instrumental music and a general opening. A request
for an action scene in one review does not turn the entire episode into an action list.
Each title's verified metadata separately guides footage and image discovery.

## Discovery and edit

All primary footage is obtained before extras. Four optional rounds can bring each title
to five accepted sources; the default concurrency is two titles per round (FOOTAGE_WORKERS,
bounded to 1–3). Independent downloads, file paths and a shared disk reservation prevent
parallel workers from spending the same free space. Extra sources must add imagery beyond
the combined accepted pool. Fewer sources are normal and do not fail production.

Queries include original-year trailer, official trailer and best-scene families, with a
year-free best-scene query for every show and additional per-title genre concepts. Network
names and verified uploader status are not discovery gates. Title identity, remake/season
checks, fake-footage rejection, decode checks, resolution and byte limits remain necessary.
Metadata checks are useful evidence, not infallible visual identity verification.

Native scene candidates use whole analyzed windows and up to three adjacent shots only
when their visual context agrees. Dialogue windows grow to word-aligned utterance boundaries.
Incomplete or uncertain clips are skipped rather than arbitrarily cut. Original sound is
retained, and music steps back smoothly. The scene planner searches every accepted source.
It aims for up to five useful native moments in longer reviews, limited by paragraph
boundaries; one good scene is preferable to filling a quota. These moments add real time.

The opening includes a 2.3-second original M ident and an original two-note sting. Strong
visual action evidence can justify a longer opening (up to 18 seconds of selected scenes);
quieter footage gets a short visual opening. These are conservative editorial bounds, not
a claim that motion scores understand every joke, action payoff or emotional scene.
The former mandatory genre sizzle and number-one flash are off.

Scene-description work samples six frames per source within a shared 420-second budget.
Image matching remains available when optional descriptions fail. Thumbnail vision review
has its own short bound. Increasing source count does not enable unlimited model calls.

## Script and voice

Provider information stays in the description, not speech. Formulaic comparison bridges
and recurring number-one teasers are not compulsory. A delete-only episode pass can remove
repeated or empty sentences across reviews while preserving announcements, premises and
existing facts. It does not replace removed filler to reach a word count. Runtime targets
remain soft; captions and chapters follow the actual measured edit.

Warguy remains the default voice. Turbo stays consistent across delivery styles unless
the user deliberately enables the optional expressive model. Coherent TTS groups contain
up to three complete sentences and about 45 words. Whole-episode TTS is avoided because
long calls increase omissions, repetitions, voice drift, CPU use and retry cost.
A clean 15–25-second recording in the desired style can help the clone, but existing
5–30-second references remain supported. A model change needs a controlled voice comparison.

Dry speech, processed/paced speech, music and finished-mix samples remain diagnostic assets.
The speech chain uses gentler de-essing and compression. Routine transition effects are
off (TRANSITION_FX=true opts in); the original ident and synchronized CTA sounds remain.
Automated transcript/decode checks cannot establish that a voice sounds natural to a human.

## Ten thumbnails

Options 1–2 preserve their hero/split composition logic, adding only a thin frame.
Options 3–5 use image-led hero/duo compositions without compulsory service branding,
large countdown numbers or spectacle templates.
Options 6–10 use independently searched photos with source pages and query provenance;
verified title-specific publicity stills are the fallback. They never silently use extracted
video frames. A severe all-photo outage produces explicitly reported typographic fallback
designs, not invented actors. Red is the safe frame default; explicit genre can choose an accent.

Search uses a bounded Bing image-result plan across the whole list. URLs are actual result
sources; private/local targets, unreadable images and obvious fake/watermarked contexts
are rejected. Text context and image heuristics cannot guarantee actor identity, so source
provenance and final human visual review matter. No score guarantees click-through rate.

## Retained safeguards and review

Technical protections remain: bounded downloads/retries, identity checks, disk headroom,
audio decode and speech verification, A/V timing, subtitles, chapters and source credits.
The existing model and Python caches and cleanup/retention policies remain. Dry temporary
takes are removed after bounded inspection samples are saved.

Editorial Quality Review renders real-photo thumbnails and the ident on GitHub's runner.
The normal tests cover media/voice integrity and orchestration. Finished production also
exports quality-contact-sheet.jpg and quality-summary.json from the real video, alongside
the existing advisory review report. Sampled frames and passing tests do not replace
watching/listening to the final export.

## Starting production from a request file

Generate Video still supports its existing manual controls. It is also reusable.
A push changing requests/current.json on the production-requests branch starts the narrow
Movie Ranks Production Request launcher, which validates fields and invokes Generate Video.
It uses the same secrets, selectors, retention and notifications. JSON values are passed as
typed workflow inputs, never interpolated into shell code. This path exists for connected
clients that can commit files but do not expose workflow_dispatch.

## Remaining creative defaults

The following defaults are centralized rather than silently eliminated: rank weighting gives
later recommendations more narration; review pace profiles limit moving excerpts; freeze
holds have a cooldown; scene planners use bounded lengths, visual-similarity thresholds
and transcript punctuation as heuristics; the ident and end-screen have fixed short
durations; quiet TTS padding protects consonants; title pronunciations support the user's
pronunciations.json. The stock-word replacement table still simplifies familiar clichés
and can reduce language variety; editorial generation and delete-only episode review are
preferred over adding more compulsory catchphrases. Font, contrast, crop and accent
defaults ensure a usable composition when optional models fail. None of these creative
preferences is a reason to reject an otherwise usable finished video.
