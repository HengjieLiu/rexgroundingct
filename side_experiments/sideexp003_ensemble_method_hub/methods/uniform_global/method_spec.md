---
created: 2026-09-08
updated: 2026-09-08
status: active
---

# Method: Uniform Global

Average equal-weight post-sigmoid probabilities from every frozen roster member
and evaluate at probability threshold `>= 0.5`. Report mean Dice and hits over
aligned findings plus official-category breakdowns.

Diagnostic runs may stream and discard ensemble chunks. They must set
`metrics_only=true`, identify selection/evaluation reuse of val200, and must not
publish a formal recipe or derived logits.
