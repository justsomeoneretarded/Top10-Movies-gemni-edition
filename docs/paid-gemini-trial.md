# Isolated paid Gemini test

This opt-in experiment is isolated from the normal Generate Video — Gemini workflow and from the original and 2.0-ish repositories. Only a validated request pushed to paid-trial-requests:requests/paid-trial.json starts it. There is one API producer job.

- Gemini 4 Carbon (low thinking) writes the episode plan, narration, editorial pass and packaging.
- Gemini 3.8 Flash reviews footage/frame candidates and extracts source-verified public-page facts. This is not native Google Search grounding.
- Gemini 3 Pro Image edits real-source compositions for thumbnails 3–5. Independent Flash identity review retains originals when uncertain. Options 1–2 and 6–10 retain their current composition logic.
- Warguy, Original Chatterbox, speed 1.2, existing music and encoding stay selected. No stock Gemini voice or cloning enrollment.
- Existing yt-dlp routes, identity/speech/AV checks, systemd resource guard, model/Python caches, cleanup and two-day output retention remain.

The dedicated GEMINI_API_KEY_PRO secret is the only key. A real paid-only Carbon preflight shares the same durable ledger as production. Metadata alone cannot prove paid access.

## Estimated spending

The application bounds this one test to a conservative estimated $5, 100 generation requests, input/output token ceilings and three image generations. Each request first counts its exact input, reserves 5%+32 tokens of input headroom plus maximum output/thinking, and then generates once. Only internally consistent usage releases text reservations. Unknown outcomes and image costs retain reservations. No prompts, URLs or keys are stored in the ledger.

This is an application estimate, not a Google billing hard cap. Provider accounting, price changes and spending elsewhere in the project are outside this ledger. Prices are conservatively pinned and expire on 2026-12-31.

Paid transport retries, key/model rotation and video URL-to-upload retries are disabled. Optional media analysis and thumbnail failures record fallback. Paid script failure stops rather than claiming a local script is a paid result. GitHub reruns are blocked because a new runner would reset the ledger. An additional paid attempt needs separate authorization.

CI tests the provider, durable budget, images, workflow, preflight and existing regression suite without billed calls. Production records actual models, fallbacks and usage. The final package includes budget/preflight receipts; separate audit artifacts retain thumbnail references and generated candidates. Paid models do not guarantee better writing, editing or click-through performance.
