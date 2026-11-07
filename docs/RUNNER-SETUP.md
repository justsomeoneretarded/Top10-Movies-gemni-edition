# Runner model setup and disk space

The hosted Ubuntu runner's working disk is separate from the storage used by
uploaded video artifacts. Expiring an artifact does not free space inside a
currently running machine.

Before the planning, script, footage and production jobs restore caches or install
large dependencies, scripts/runner_disk.py removes only these unused preinstalled
SDKs on recognized GitHub-hosted Linux runners:

- /usr/share/dotnet
- /usr/local/lib/android
- /opt/ghc
- /opt/hostedtoolcache/CodeQL

It skips self-hosted/unrecognized runners and symlinks. Workspace files, output,
model weights, Ollama models, Python installations, pip caches and all Actions cache
keys are preserved. The helper measures actual relevant filesystems before and
after cleanup, logs free bytes/inodes, and checks headroom before large operations.
A low-space failure is reported before the runner fills up and loses its own logs.

Script jobs leave room for the approximately 8 GB Gemma 4 12B model. If there is not
enough additional room for optional voice pre-generation after restoring the model
cache, VOICE_WARM is disabled for that job. Script preparation still continues, and
the final job generates narration with the same selected voice/model and quality.

Ollama uses a pinned official archive and SHA256, bounded transfer retries and a
task-owned installation. CPU runtime files are retained; unnecessary bundled CUDA
and Vulkan backends are excluded during extraction. There is no separate official
CPU archive: this is a reduced extraction of the official Linux archive. Temporary
archives are removed before pulling model weights. No systemd service or GPU driver
installation is needed for the job's local server.

The same bootstrap handles script-model setup and --serve-only scene-description
setup. It reuses a healthy local server, preserves existing model downloads, and
retries interrupted model pulls with finite time limits.

Runner Setup Checks performs both simulated failure-path tests and a real hosted
runner check: it restores the current model cache, installs CPU speech dependencies,
sets up Gemma 4 12B, performs CPU inference, repeats setup to check reuse, and verifies
remaining disk headroom. It does not render or publish a video or send Telegram
messages.

The existing 24-hour production-checkpoint cleanup and two-day final-artifact
retention remain unchanged. Normal new runs from main use the fix; rerunning a
historical run continues to use that historical workflow revision.
