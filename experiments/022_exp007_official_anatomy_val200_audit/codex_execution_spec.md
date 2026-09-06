---
created: 2026-09-05
status: approved
---

# Exp022 execution spec

## Objective and prior evidence

Implement the user-approved category-by-category audit of all 381 val200
findings. Exp007 cont e050/abs e150 baseline is 0.3460234857111323 Dice and
296 hits. Exp010 showed that a small global gain can hide individual clipping
harms. Exp020 verifies official CT-RATE source identity and geometry but remains
pending human visual review. This new authorization covers diagnostic audit
only, not integration or a change in Exp020 review status.

## Inputs and preprocessing

Canonical config: `configs/experiments/022_exp007_official_anatomy_val200_audit.json`.
Runtime: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/022_exp007_official_anatomy_val200_audit`.
Use hash-pinned fixed val200 and saved Exp007 evaluator, native binary predictions,
released finding-first GT (positive instance IDs become foreground with >0)
and Exp020 masks/CT headers. Preprocessing unchanged:
no normalization, resampling, new cache, inference windows, or model execution.
Use the native CT index grid; verify source mask affine, spatial orthogonality,
prediction/GT finding-first shape and established export/header conventions.
Reproduce every saved Dice to 1e-7 and every saved hit before comparisons.

## Frozen semantic review

Read every prompt, create one manually interpreted row per (scan, finding), and
seal before measuring new outcomes. Record extent, target/landmark distinction,
scope, risk, ambiguity and rationale. AI-authored semantic review is not clinical
or human review. Explicit bilateral/general distribution overrides dominant or
largest-location clauses. Ambiguous lobe assignments widen to side/whole lung;
unsupported target compartments bypass all prompt-routed constraints.
No exact bronchus, pleura, fissure or aberrant vessel mask exists in total;
do not substitute trachea, lung or aorta as an exact lesion compartment.

## Comparisons and measurement

Regions: whole lung for all findings as a deliberately indiscriminate diagnostic;
prompt-eligible whole lung; prompt-supported side; finest semantically supported
lobe union (falling back to broader scope). Each has exact mask and axis-aligned
bbox, margins 0/5/10/20 mm. Mask expansion uses Euclidean voxel-center distance
with CT spacing; bbox uses voxel centers within inclusive bounds plus physical
margin. Empty/missing region means recorded unavailable and no-op for routed
policies, not forced empty. No GT-driven region, margin, or policy selection.

Measure GT containment, TP/FP removal, Dice, precision/recall, hits, empty output,
and all-label GT overlap. Aggregate full and selected populations, category,
case, locality and risk strata. Paired CT-cluster bootstrap: 2,000 draws,
seed 20260905. Category draws resample its contributing CTs; sparse groups are
descriptive. All policy comparisons reported; exploratory intervals are not
multiple-comparison adjusted or evidence of held-out improvement.

## Execution and gates

CPU only, at most four workers, single numerical thread per worker. Source
mounts read-only; only Exp022 runtime writable. Pin config, code, review, sources
and environment. Resume only hash-identical completed case records; atomic
writes. First run unit tests and a one-case smoke, then all 200 cases. Stop on
input, geometry, identity, or baseline mismatch rather than dropping findings.

Commands inside the documented Docker mount layout:

```bash
python scripts/rexgroundingct/test_audit_022_official_anatomy.py
python scripts/rexgroundingct/audit_022_official_anatomy.py freeze --review /review/exp022_semantic_review.tsv
python scripts/rexgroundingct/audit_022_official_anatomy.py measure --limit 1 --workers 1
python scripts/rexgroundingct/audit_022_official_anatomy.py measure --workers 4
python scripts/rexgroundingct/report_022_official_anatomy.py
python scripts/rexgroundingct/visualize_022_official_anatomy.py --workers 2
python scripts/rexgroundingct/extend_022_visual_review.py
python scripts/rexgroundingct/visualize_022_official_anatomy.py --review-notes /mnt/shengdata1/hengjie/experiments/rexgroundingct/022_exp007_official_anatomy_val200_audit/reports/visual_notes.tsv
python scripts/rexgroundingct/report_022_official_anatomy.py
python scripts/rexgroundingct/check_022_official_anatomy.py --require-visual-review
```

## Completion and closeout

External Markdown report + all-381 appendix, CSV/JSON, per-case evidence,
frozen interpretations, and visual manifest. Generate and actually inspect
category gain/harm extrema, every lost-hit/fully-excluded target finding, and
Exp020 fragmentation flag; record uncertainty and distinguish AI from human QC.
Sync only aggregate redacted report/provenance, update current status, run
focused and repository checks. Stop for user review without adopting a method.

## Recorded implementation correction

The initial one-case smoke stopped before writing measurements because GT
contains positive instance IDs (observed 0..7), not only 0/1. The evaluator's
existing >0 foreground convention is retained; binary predictions are still
required. `revise-implementation` preserved the original seal and wrote a linked
measurement seal before any completed measurements. Prompt interpretations,
config, source files, policy definitions, and margins were not changed.

Runtime image: `rexgroundingct-totalsegmentator:2.16.0-cu126`, image SHA256
`047cfe9e32a9001de0f55d04bffdb1ad81bf59b8f9f7de049103cf5d15d0acb7`.
This is the dependency environment only; anatomical inputs remain the official
CT-RATE outputs with the producer-bound 2.7.0 label mapping. No GPU is exposed.

## Completed diagnostic extension

The initial required set contained 37 panels. Three additional panels inspect
all remaining regressions under the highest-observed fixed routed comparison.
This is explicitly post-outcome diagnostic selection, not a policy revision.
All 40 panels were inspected via full-resolution contact sheets, with the
largest dense-consolidation failure also opened individually. The original
Exp020 fragmentation panel was opened separately. Structured notes and image
hashes are external; no human or clinical QC sign-off is asserted.

Post-measurement label-overlap diagnostics summarize prompt/GT/anatomy
discordance, including a descriptive >5% GT reporting threshold. They do not
filter cases, change the frozen 381 interpretations, or alter any policy.
