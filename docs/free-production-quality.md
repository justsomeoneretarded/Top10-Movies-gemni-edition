# Free production quality

The Gemini edition retains the existing free/local routes, selected narrator and Original Chatterbox. No paid experiment code is imported and neither other repository changes. Creative targets remain guidance; source identity, complete speech verification, AV timing, resource reserves, model/Python caches and output retention remain enforced.

## Writing and duration

Auto supplies soft per-review planning. A bounded editorial pass identifies thin, generic or repetitive reviews and can rewrite up to three using unused details quoted from the supplied research. A short substantive review can remain short. No word-count padding or generic filler is inserted.

After speech generation, an episode below its soft range may receive one more targeted pass (at most two reviews). Unchanged text reuses its speech cache. A complete substantive repair can be adopted even if its duration is no closer to the target; incomplete replacement speech retains the earlier complete draft. Reports say whether the measured repair was adopted. Timing targets never bypass speech verification.

## Source quality

All primaries are fetched before optional rounds, up to five accepted sources per title. Extra discovery uses each primary's native detail, capped at 1080p for discovery, instead of demanding the provisional final canvas. A cached low-resolution primary can receive one bounded refresh per run/attempt; unsuccessful or inferior candidates never replace the accepted copy.

Accepted pools promote a sharper same-title source. Export size is chosen after every round and settled again after indexing fallbacks, before rendering parts. Try 4K respects the existing majority/detail and maximum-upscale policy; it does not promise native 4K. Explicit 1080p remains available. Failed promoted sources fall back once within the same title. Precomputed scenes referencing removed sources are reselected without transferring another source's dialogue timestamps.

Quality reports distinguish accepted sources from files actually rendered and record upscale conditions. Source-derived caches use actual file-byte identity so refreshing a file cannot reuse stale shots or frames.

## Gemini analysis

Discovery has a limited share of the analysis allowance. Accepted-source analysis starts after downloads, gives every primary a chance before extras, and precedes optional frame descriptions. Valid caches are reused before budget checks.

Request, attempted-video and retry limits are not reset. Failed attempts count. Default video retries are zero; service failures use bounded model backoff instead of repeatedly trying URL/upload variants. Actual fallback models and missing coverage are reported. Local editing remains available.

## Scenes and visual repair

Verified connected analysis windows can supply complete native scenes beyond short shot clusters. Dialogue boundaries stay inside the coherent scene and preserve complete words. Short preview fallbacks hold a clean same-source frame instead of blindly crossing unknown cuts.

At most three visual repair attempts are allowed across initial parts and one possible final-export follow-up. Only short, nearly uniform internal blank gaps under continuous narration qualify. Replacement is a clean neighboring same-part frame hold. Dark textured scenes, text cards, native pauses and deliberate intro/outro transitions remain advisory. Replacement is atomic and must retain decoded audio, frame count, canvas and duration, and pass decoding. Final export is encoded again only after a new successful repair. Full downstream speech and AV checks still run.

## Ten thumbnails

Options 1–2 retain their composition logic. Free image discovery handles more search-result formats and reports blocked/challenged/empty searches and candidate rejections. Searches span the full list; verified publicity stills supplement sparse results. Options 6–10 remain still-image-only, never silently replaced by extracted video frames.

Options 3–10 use bounded local crop, phone-readability and diversity checks with conservative reselection. Provenance and fallback reasons are recorded. These checks cannot predict click-through rates or guarantee good design. No paid image generation is introduced.

## Output evidence and validation

Existing files remain. The package also includes production-quality-report.json, current-job quality/duration/audio/native-scene/visual/script reports when available, and final-script.txt. Thumbnail discovery/composition diagnostics remain in thumbnail-sources.json. Restored old-job reports are not copied as current evidence.

Generated-media regressions cover blank repair versus dark scenes/text cards, exact decoded-audio preservation, frame clocks, source-byte cache invalidation and coherent scene boundaries. Integration regressions cover source-quality policy, warm advice, same-title fallback and script-repair adoption. Full GitHub CI is required before deployment. Passing tests alone does not establish subjective video quality.
