---
created: 2026-09-07
updated: 2026-09-07
status: specified_not_run
---

# Method: Greedy Category

## Prediction contract

Use the same five frozen case folds, equal post-sigmoid probability averaging,
fixed `0.5` threshold, and deterministic tie-break as `greedy_global`.

Run independent OOF greedy selection for official categories `2a`, `2b`, `2c`,
and `2d`; each category gets a separately user-confirmed final `M`. Categories
with fewer than 20 val200 findings use the already confirmed global recipe.
Category `2f` is explicitly `val200 unavailable` and also uses the global
fallback. Produce full-val per-category oracle curves only as diagnostics; they
are not formal recipes.

At materialization time, route each known finding by its official category and
still write one evaluator-layout `(F,X,Y,Z)` array per case. Unknown/missing
category metadata is an error, never a guessed route.

## Required outputs

For each independently searched category and every `M=1..Mmax`, save OOF
Dice/hits, fold member sequences, selection frequency, marginal Dice, full-val
oracle, and explicit fallback decisions. Report the same three `M` candidates as
the global method, then wait for the user to choose `M` per category.

## Status

Specification only. No roster, run, inference, greedy search, or final logits
exist yet.
