---
created: 2026-09-09
updated: 2026-09-09
status: active
---

# Leaderboard evidence

The first extraction is [V++_b3](snapshots/20260909T222726Z/Vpp_b3.report.md),
retrieved on 2026-09-09 and recorded in `results.json`. It exactly matches the
user-provided public result. Its association with the local b3 artifact remains
unconfirmed because no upload event or archive hash has been supplied. Existing challenge-wide archives remain in
`challenge_info/`; do not duplicate or rewrite them here.

Future result records should include result_id, source_url, extracted_at_utc,
original_displayed_name, external_id, leaderboard/cohort identity, reported
metrics with original precision, and the evidence snapshot path and SHA-256.
A confirmed match records submission_id and submission_event_id referencing
an actual registry submission-history event. Keep ambiguous results unmatched
with null IDs; do not infer an upload from a prepared prediction folder or
match similar names automatically. Multiple dated observations are history,
not replacements for an earlier score.

Public small source snapshots go in `snapshots/<UTC timestamp>/`. Withheld
scores, credentials and private account data are not collected by these tools.
Manual future entry of a result must identify its evidence and actual cohort.
Local val200 measurements are not official leaderboard results.

See [Dice aggregation verification](dice_aggregation.md) for the case-wise
versus finding-wise distinction and correction to the initial snapshot note.
