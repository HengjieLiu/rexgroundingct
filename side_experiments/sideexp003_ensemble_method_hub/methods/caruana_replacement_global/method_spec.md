---
created: 2026-09-08
updated: 2026-09-08
status: active
---

# Method: Caruana Replacement Global

Start with the frozen rank-1 candidate. At every subsequent basket position,
evaluate adding every candidate, including candidates already in the basket.
Select by full-scope mean finding Dice, then hits, then candidate ID. Selection
frequency defines the final discrete weight.

Predictions use post-sigmoid probability averaging and threshold `>= 0.5`.
Full-val diagnostic runs are optimistic same-set oracles, not OOF estimates or
formal recipes. They may not materialize derived logits without a later user
decision.
