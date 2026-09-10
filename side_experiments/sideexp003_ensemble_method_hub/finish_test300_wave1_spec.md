# Finish test300 Wave 1 and hold ranks 5–20

Authorized 2026-09-09: let ranks 1–4 complete inference, publication and strict
validation, then stop before ranks 5–20. Preserve all resulting caches and the
completed val200 search. Investigate the unexpectedly slow test export before
authorizing further waves.

The frozen running coordinator has no wave-boundary stop control. The separate
`finish_test300_wave1.py` guard pins its exact host PID with a pidfd, records
identity and intent, and sends SIGSTOP only to that coordinator. Existing Docker
workers run independently and retain their frozen export/publication checks.
The stopped coordinator retains launch.lock and cannot launch Wave 2. The guard
monitors the four workers every 30 seconds and changes no CPU search process.

Completion requires all four worker processes to exit successfully and all four
300-case / 582-prompt manifests and validation records to pass the existing
completed-cache predicate. Then queue SIGTERM to the stopped coordinator and
send SIGCONT so its existing handler exits without advancing to the next wave.
After its exit, acquire launch.lock, retain the original exit state, collect
the four completed caches into an inventory, sync the catalog, and write
`stopped_after_wave1`. The guard removes no arrays or containers. A failed guard
leaves the scheduler stopped, preserving the block on later waves.

Runtime evidence: `control/finish_wave1_request.json`, `finish_wave1_state.json`,
`logs/finish_wave1.log`, and `wave1_hold_report.md`. The control request persists
after completion; a subsequent launch requires an explicit new decision.
Do not send SIGCONT to the scheduler manually: that could launch Wave 2.

After Wave 1, compare matched test/val execution paths and separately time GPU
prediction, native restoration, contiguous-array conversion, checks/hashing,
local writes, and shared publication. Compare memory strides and float dtype
at each conversion. Measure on representative small and large cases without
changing the existing caches or claiming bitwise equivalence without checks.
The earlier stack sample implicates native-layout copying for one large case;
it does not establish a full-run time breakdown. Do not restart ranks 5–20
until the bottleneck is measured and an improved path is reviewed.
