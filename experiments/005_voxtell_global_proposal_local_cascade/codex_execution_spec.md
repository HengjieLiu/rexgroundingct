# Experiment 005: Global Proposal to Local Segmentation

## Hypothesis

A dedicated full-field proposal model can improve localization by selecting a
small set of candidate regions before the expensive native-resolution VoxTell
segmentation stage. Stage 1 is judged primarily by whether its candidate
`192^3` regions hit and fully include each finding, not by coarse-mask Dice.

## Difference From Existing Coarse-to-Fine Code

The external workspace contains a useful prompt-conditioned low-resolution
U-Net, but its current coarse-to-fine path downsamples the same local patch and
fuses the result back into that patch. Experiment 005 adapts that architecture
to a separate global proposal problem:

- Stage 1 sees the complete CT at 4 mm isotropic resolution.
- Stage 1 predicts valid local-crop centers for each text prompt.
- Stage 2 receives only proposed native-resolution crops.
- The two stages are evaluated both separately and end to end.

## Stage 1 Geometry

The audited exp004 2 mm cache is the source. Each image is downsampled to 4 mm
with trilinear interpolation and center-padded to `192^3`, giving a 768 mm
field of view in every axis. The dataset audit shows that all train and
validation cropped CTs fit this grid.

For each finding, native-space target bounds define the set of starts for which
a native `192^3` crop fully contains the finding. Those valid starts are mapped
to the 4 mm grid and form the proposal target. If a finding itself exceeds 192
voxels in any native axis, a crop-hit target is used and explicitly recorded.

## Variants

| Variant | GPU | Supervision |
| --- | ---: | --- |
| `strict_inclusion` | 0 | Exact full-inclusion candidate-center region |
| `inclusion_margin16mm` | 1 | Inclusion region expanded by four 4 mm voxels |

Both variants consume the same deterministic 5,000-event prompt schedule,
train for 50 standardized epochs, and use batch 1 without DDP.

The hit-oriented loss is the negative log fraction of predicted probability
mass assigned to valid proposal centers. This aligns optimization with proposal
ranking while avoiding the extreme-value bias of a maximum-over-`192^3` loss.

## Candidate Evaluation

Stage 1 uses 3D nonmaximum suppression and reports top-1, top-3, and top-5:

- Finding hit rate: at least one target voxel lies in the proposed crops.
- Full-inclusion rate: one proposed crop contains every target voxel.
- Target coverage: fraction of target voxels covered by the union of crops.
- Results are also broken down by challenge category and native mask size.

Fixed val20 proposal reports are written every 500 updates. These reports are
available while training continues and are the primary early-termination
signal.

## Stage 2

The local segmenter is exp003 `v123_opt_poscrop_emptyloss` epoch 100. For each
finding, it runs on the top three proposed native `192^3` crops. Probabilities
are restored into the native cropped volume with voxelwise maximum merging,
then returned to evaluator orientation.

End-to-end val20 runs occur at proposal updates 1,000, 2,500, and 5,000.
At update 5,000, both variants receive proposal and cascade val200 evaluation,
including saved probability maps.

## Runtime And Stop Control

The detached experiment has a hard 24-hour timeout. Intermediate checkpoints
and reports are immutable. To stop cleanly after the current update, create:

- `control/STOP_ALL`
- `control/STOP_strict_inclusion`
- `control/STOP_inclusion_margin16mm`

Cascade jobs defer when their assigned GPU has more than 15 GiB allocated or
an exp004 checkpoint is awaiting evaluation.
