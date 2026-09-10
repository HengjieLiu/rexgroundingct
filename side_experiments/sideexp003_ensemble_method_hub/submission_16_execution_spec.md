---
created: 2026-09-09
updated: 2026-09-09
status: active
---

# Sixteen-worker continuation for s001

The user authorized increasing d1/d2/d3 case concurrency from four to sixteen
after both retained smoke cases finish. Keep the frozen s001 job, inputs,
numerical functions, postprocessing, validation and destination identities.
This separate execution policy supersedes only the four-worker scheduling
setting. The new container has a 16-core CPU allowance; numerical libraries
remain single-threaded in each case process. Waves 2–5 remain held.

Pin and stop only the old supervisor with pidfd SIGSTOP while its two smoke
workers continue. The old supervisor keeps the job lock. Wait for both complete
smoke journals and verify all six output hashes. After release tests pass,
queue SIGTERM before SIGCONT to the pinned supervisor and wait for the old
container and all its workers to exit. Archive old state and launch evidence;
never remove prediction files, worker journals or source caches. A missing
or failed smoke, source drift or ownership mismatch blocks the handoff.

The continuation validates the original frozen contract and bound cache
manifests, acquires the same job lock, verifies/reuses completed transactions,
and streams sixteen pending cases using the original process_case function.
It uses original package/ZIP functions and the same full acceptance criteria.
Unpublished temporary outputs remain this job's scratch. A separate execution
manifest binds the new source files and original job hash without altering
or relabeling the original job or completed outputs.

Require 300 GiB available RAM before launch. Defer new case dispatch below
64 GiB available RAM, allowing already-active workers to finish. Record memory,
phase timings, completed cases and elapsed time every minute. Estimate remaining
case work after ten newly processed cases, excluding retained/reused cases.
Final completion records include original and continuation provenance and
reproduction commands. No GPU access, label access, network or Docker socket
inside the worker container. The host handoff controls only the identified old
submission supervisor/container and the newly named continuation container.
