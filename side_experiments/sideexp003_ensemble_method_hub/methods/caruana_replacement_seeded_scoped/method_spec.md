---
created: 2026-09-08
updated: 2026-09-08
status: active
---

# Method: Caruana Replacement Seeded Scoped

This method is a metrics-only, optimistic full-val200 diagnostic over the
frozen fresh top-16 roster. It runs independently for `all`, `2a`, `2b`, `2c`,
and `2d` while sharing every source-logit read across scopes.

Each scope begins with four fixed, distinct leaderboard candidates, each with
multiplicity one. At basket sizes 5 through 16, every top-16 candidate remains
eligible, including candidates already in the basket. The selected addition
maximizes scope mean finding Dice, then scope hits, then ascending candidate
ID. Selection continues through K=16 even when the best marginal Dice is
negative.

Predictions use post-sigmoid probability averaging and threshold `>= 0.5`.
Hits use finding Dice `>= 0.1`. Because selection and evaluation use the same
val200 findings, the curves are same-set oracles and cannot be promoted to a
formal recipe. This method never materializes derived logits.
