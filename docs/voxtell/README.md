---
created: 2026-07-22
updated: 2026-07-28
status: active
---

# VoxTell ReXGroundingCT Experiments

## Folder Map

- `README.md`: experiment workflow overview, launch commands, and current
  VoxTell status.
- `preprocessing.md`: direct inference loading, preprocessing, orientation, and
  export details.
- `preprocessing_variants.md`: standard preprocessing/cache variants and the
  required contract for future spacing, normalization, and cache experiments.
- `normalization.md`: normalization findings and planned ablations.
- `ct_hu_normalization_analysis.md`: validation-wide HU audit, z-score outlier
  analysis, target-intensity distributions, InstanceNorm caveat, and the
  experiment 011 normalization interpretation.
- `ddp_and_effective_batch_size.md`: DDP, gradient accumulation, global batch
  size, and comparison rules for VoxTell fine-tuning.
- `training_dynamics.md`: canonical per-experiment loss plots, segmented-metric
  merging, cross-experiment gallery organization, and live Exp011 refresh
  rules.
- `training_sampling_strategy_audit_bilingual_2026-08-22.md`: bilingual audit
  of the current positive/negative prompt sampling strategy, multi-finding
  behavior, risks, and prioritized improvements.
- `asymmetric_proposal_refinement.md`: dual-branch high-recall proposal,
  soft-guided refinement, asymmetric losses, fusion alternatives, literature,
  and the experiment 008 design.
- `exp008_variant_reference.md`: exact side-by-side experiment 008 model/data
  flow charts, losses, gradient paths, training contract, and interim evidence.
- `bronchopulmonary_segments_and_pseudo_segment_priors.md`: ReX segment-language
  evidence, S1-S10 anatomy, TotalSegmentator capability gap, geometric and
  airway-guided segment priors, and their validation safeguards.
- `prompts/`: bilingual prompt-reference docs generated from MICCAI metadata.

## Paths

- ReXGroundingCT metadata and masks: `/data/hengjie/datasets/rexgroundingct`
- CT-RATE CT subset: `/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct`
- Experiment root: `/mnt/shengdata1/hengjie/experiments/rexgroundingct`
- Repo-local experiment index: `/home/hengjie/code_sync/rexgroundingct/experiments`

The CT files are stored separately because the ReXGroundingCT download is about
`3.4 GB`, while the CT-RATE subset is about `336.78 GiB` for `train + val` and
`369.15 GiB` for `train + val + test`.

## Experiment File Policy

Experiment configs are canonical in the repo under `configs/experiments/`.
Runtime config files under `/mnt/shengdata1` are immutable snapshots recorded
with SHA256 hashes in each experiment manifest. Edit the repo config first, then
launch a new run or intentionally overwrite the runtime snapshot.

The repo-local `experiments/` directory is a lightweight index for easier
inspection:

- `experiments/README.md` and `experiments/registry.yaml` summarize experiment
  status and important paths.
- `experiments/<id>/report.md` and `metrics_summary.json` are small synced
  snapshots of runtime reports and metrics.
- `experiments/<id>/runtime` is an ignored symlink to the runtime directory.
- Logs, predictions, checkpoints, NIfTI files, and raw evaluator outputs stay on
  `/mnt/shengdata1` and are ignored by git.

After a run writes new reports or metrics, sync the repo index:

```bash
python /home/hengjie/code_sync/rexgroundingct/scripts/rexgroundingct/sync_experiment_index.py
```

Before committing or launching from an existing runtime directory, check for
drift:

```bash
python /home/hengjie/code_sync/rexgroundingct/scripts/rexgroundingct/check_experiment_consistency.py
python /home/hengjie/code_sync/rexgroundingct/scripts/rexgroundingct/check_experiment_consistency.py --strict-runtime
```

## Data Readiness

Inside the Docker container:

```bash
python /workspace/scripts/rexgroundingct/poll_ct_subset.py --splits val
python /workspace/scripts/rexgroundingct/poll_ct_subset.py --splits train val
```

Exit code `0` means the requested split set is complete and no `.incomplete`
files were found.

## Experiment 001

Run after `val` readiness passes:

```bash
bash /workspace/scripts/rexgroundingct/run_001_voxtell_val_eval.sh
```

The current standardized Experiment 001 result is the corrected-orientation
200-case MICCAI validation quick/global evaluation. The challenge ranking metric
is **Mean global Dice per finding**, not mean Dice per case.

Current canonical quick/global result:

- Mean global Dice per finding: `0.225228`
- Mean global Dice per case: `0.234287`
- Hit rate: `0.535433`
- Findings/prompts: `381`
- 4-GPU inference wall time: about `20:04`
- Quick/global evaluation time: about `18-20s`

This standardized result does not include corrected instance precision/recall/F1;
the previous canonical full official evaluation had an orientation export bug and
was removed from the standard Experiment 001 outputs.

From the host, use the watcher to poll until validation CTs are complete and
then start experiment 001 in Docker:

```bash
IMAGE=rexgroundingct-voxtell:cu126 \
INTERVAL=600 \
bash /home/hengjie/code_sync/rexgroundingct/scripts/rexgroundingct/watch_and_run_001.sh
```

Do not start this watcher until the Docker image has been built.

To run the full gated pipeline from the host, use:

```bash
IMAGE=rexgroundingct-voxtell:cu126 \
INTERVAL_VAL=600 \
INTERVAL_TRAIN_VAL=600 \
RUN_FULL_FT=1 \
bash /home/hengjie/code_sync/rexgroundingct/scripts/rexgroundingct/watch_and_run_pipeline.sh
```

This waits for validation CTs, runs experiment 001, and only if that exits
successfully waits for `train + val` CTs before starting experiment 002.

### Orientation Diagnostic

The first full validation pass exposed an orientation/export issue. Before any
full rerun, verify the fix on the selected single validation case inside
experiment 001:

```bash
bash /workspace/scripts/rexgroundingct/run_001_orientation_case_val_idx001.sh
```

This writes only to
`/mnt/shengdata1/hengjie/experiments/rexgroundingct/001_voxtell_v1_1_miccai200_val_eval/diagnostics/orientation_case_val_idx001`.
The corrected single-case diagnostic is kept as provenance for the orientation
fix.

## Experiment 002

The text-conditioned fine-tuning path is intentionally separate from public
`voxtell-finetune`. Public `voxtell-finetune` is an nnU-Net encoder-transfer
workflow and is not the primary challenge-valid free-text grounding experiment.

First precompute prompt embeddings:

```bash
bash /workspace/scripts/rexgroundingct/run_002_precompute_text_embeddings.sh
```

Run the 4-GPU smoke loop after `train + val` readiness passes:

```bash
bash /workspace/scripts/rexgroundingct/run_002_voxtell_text_ft_smoke.sh
```

Then start the longer 4-GPU run:

```bash
MAX_ITERATIONS=10000 CHECKPOINT_EVERY=500 \
bash /workspace/scripts/rexgroundingct/run_002_voxtell_text_ft_full.sh
```

The training script is explicitly gated with `--allow-experimental-train` inside
the launcher because this is a challenge-specific text-conditioned path, not the
public `voxtell-finetune` encoder-transfer path.
