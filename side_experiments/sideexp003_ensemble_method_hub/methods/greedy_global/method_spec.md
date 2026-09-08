---
created: 2026-09-07
updated: 2026-09-07
status: specified_not_run
---

# Method: Greedy Global

## Prediction contract

- five deterministic case folds, frozen by hash in every run;
- select members on four folds and evaluate only on the held-out fold;
- sigmoid each cached logit, equal-weight average probabilities, threshold at
  exactly `0.5`;
- no weight, logit-domain, or threshold search;
- without-replacement forward selection using a running probability sum.

At each step, maximize training-fold mean finding Dice. Resolve exact ties by
higher hit count, then lexicographically smaller catalog candidate ID. The run
must reject any finding overlap between train and held-out case folds.

## Required outputs

For every `M=1..Mmax`, save OOF Dice/hits, each fold's selected member sequence,
selection frequency, marginal Dice, and the full-val refit curve. Report, but do
not automatically select:

- Dice peak;
- smallest ensemble within `0.001` Dice of the peak;
- highest-hit ensemble within `0.002` Dice of the peak.

The user confirms final `M`. Only then refit members on all val200 and materialize
one final derived-logit recipe.

## Status

Specification only. No roster, run, inference, greedy search, or final logits
exist yet.
