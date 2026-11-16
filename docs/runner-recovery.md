# Production runner recovery

The production process and its Ollama server run in the same cgroup. The guard leaves at least 2 GiB and 20% of effective physical RAM for the host, reserves CPU capacity where available, limits workload swap and bounds FFmpeg filter threads. The requested models, voices, resolution, encoding quality and verification checks are unchanged. Reduced parallelism can increase production time.

A required-control probe runs before production. The guard never silently retries an application outside its limits. Resource measurements are written to work/runner-diagnostics/resources.jsonl; low host memory or less than 1 GiB free disk produces a controlled failure rather than continuing toward host exhaustion.

During production, bounded checkpoints are uploaded every 15 minutes and before/after the long command. Each snapshot contains allowlisted script/research state, hash-verified generated speech and scrubbed diagnostics. It excludes downloaded footage, rendered video, original voice references, credentials and model/Python caches. At most two recent snapshots are retained when artifact deletion succeeds; checkpoint artifacts expire after one day. The finished package still expires after two days.

After a new version has been deployed, start a new workflow run to use that version. Rerunning an older failed run uses its original commit. For a rerun of a run already using this recovery system, the helper attempts to restore the latest checkpoint from an earlier attempt of that same run, then the ordinary identity, speech and AV checks remain authoritative.

A sudden external host or network failure cannot be prevented by repository code. Only checkpoints whose upload completed before host loss can be recovered. Optional checkpoint-service failures are reported but never discard an otherwise valid finished video.

Validation covers both unit tests and a real Actions runner: successful and failing commands, cancellation cleanup, controlled memory-limit failure without losing the runner, live short-title identity verification and an uploaded checkpoint restored into an isolated directory.
