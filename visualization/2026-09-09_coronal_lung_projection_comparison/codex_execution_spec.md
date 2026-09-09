# Execution specification

## Full-val200 extension (approved 2026-09-09)

Completed on 2026-09-09: [762 figures and 28 category galleries](results/README.md).
All acceptance checks passed, including 15 focused tests and visual inspection
of 21 PNGs covering every nonempty category.

Render all 200 fixed validation cases / 381 findings as 762 PNGs, retaining one
3×5 block per finding per CT background. Package under this folder's `results/`
with two embedded Markdown galleries per official category (`full_body.md`,
`lung_only.md`), an overview, finding metrics, figure index, and compact
provenance. Category 2f has two empty pages and no images. PNGs alone receive
a scoped Git LFS rule and an exception to the visualization PNG ignore rule.
The user will stage, commit, and push later.

Use the frozen Exp024/025 model selections and prompt routes. Correct exactly
15 findings matching the case-insensitive phrases `upper lobes of both lungs`,
`both upper lobes`, or `bilateral upper lobes`: labels 10/12, fine eligibility
enabled, physical 20 mm dilation. Seven of these routes were previously fine-
ineligible. Preserve all other fields/routes and source artifacts. PP2 follows
frozen whole-lung eligibility; ineligible policies leave masks unchanged.

Hash shared checkpoints once per invocation; use four CPU workers, one CT load
per case, and per-case support reuse. Case completion records are written only
after both backgrounds, all findings, scientific checks, and artifact hashes
pass. Resume requires matching input hashes, routing, code/settings, and output
hashes. Derived masks and detailed records stay in the external visualization
runtime. Repository output is an explicit exception for this curated package.

Acceptance: 381 PNGs per background; 28 category pages; exact category/finding
census and valid links; at most ten images per collapsible group; nine policy
summary rows; native geometry and source composition; saved-policy equivalence;
original aggregate/per-finding Dice; pilot corrected PP3 references; identical
overlays/Dice across backgrounds; empty cells and valid PNGs; targeted correction
tests; LFS attribute/ignore checks. Visually inspect a deterministic set covering
every nonempty category, multiple findings, corrected/ineligible routes, and
the original pilot. Preserve `PASS_PENDING_MANUAL_VISUAL_REVIEW` for anatomy.

## Completed single-case pilot

Approved task: generate four 3×5 coronal figures for fixed-val200 index 189,
`train_3026_a_2.nii.gz`, covering findings 0/1 independently. Compare the July
whole-volume mean CT background with a mean restricted to exact official lung
voxels. Preserve full GT/prediction overlays, voxelwise TP-priority coloring,
native geometry, lung window, and full-volume 3D Dice.

Reuse the frozen Exp024/025 overall/category/iso07 model selection, raw masks,
and available PP2 masks. Reconstruct Exp007 PP2. Derive corrected PP3 from raw
for all rows using native-physical-space 20 mm dilation of labels 10 and 12.
This is the user's explicitly selected correction to the existing bilateral
whole-lung routing, not a change to any frozen test method or source file.
Preprocessing/model inference is unchanged; only CPU visualization and derived
binary masks are produced. No checkpoints are loaded into a model.

Implementation lives in this sibling visualization folder and imports existing
July renderer helpers and Exp024 physical dilation. Keep CT, anatomy, GT, and
source predictions read-only. Read source provenance from Exp020/022/024/025;
do not modify their records. Serialize the complete input/output paths and
hashes, source model provenance, grid checks, original/corrected routes, support
labels, fixed window, metrics, and verification outcomes externally.

Gates: read-only header/path validation; focused synthetic tests; source
checkpoint/mask hashes and native-grid checks; category source-channel equality;
saved-versus-derived PP2 equality; original PP3 equivalence; published baseline
metrics; corrected Exp007 PP3 Dice 0.37675890074900004 / 0.24743654767197254;
four successful PNGs with four blank cells each; visual inspection of all PNGs.
Stop on any unexplained mismatch. Record the official mask status as
`PASS_PENDING_MANUAL_VISUAL_REVIEW` throughout.

Use the existing VoxTell Docker image on CPU because host Python lacks nibabel.
Mount repository and input data read-only and grant output access only to the
shared visualization tree. Run as the host UID/GID. Keep generated images,
derived NIfTIs, metrics and case provenance outside Git. Update the visualization
index and this folder's README with successful execution results.
