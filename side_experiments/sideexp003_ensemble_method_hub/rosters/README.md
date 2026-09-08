---
created: 2026-09-07
updated: 2026-09-07
status: active
---

# Rosters

The first authorized roster freezes the current top 20 checkpoints by overall
val200 Dice. It is fresh-only: prior SideExp002 arrays remain cataloged but are
not inputs or fallbacks for its export job.

Name files `roster_<slug>.json`. Freeze the catalog SHA, fixed dataset/fold SHA,
ordered catalog candidate IDs, checkpoint/config/cache hashes, deduplicated `N`,
and reviewed eligibility exceptions. Once referenced by a run, a roster is
immutable; changes require a new roster file and run ID.
