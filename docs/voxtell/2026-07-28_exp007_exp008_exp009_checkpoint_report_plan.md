---
created: 2026-07-28
updated: 2026-07-28
status: active
report: "2026-07-28_exp007_exp008_exp009_checkpoint_report.md"
---

# Exp007/Exp008/Exp009 Checkpoint Report Generation Plan

This file records how
`2026-07-28_exp007_exp008_exp009_checkpoint_report.md` was generated so the
report can be reproduced or refreshed later.

## Purpose

The report is a major VoxTell model-selection checkpoint comparing:

- checkpoint A: exp006 `v123_cached_e5_d4` epoch 100;
- exp009 plain continuation from checkpoint A;
- four exp008 dual-head continuations from checkpoint A;
- three exp009 S3 attention continuations from checkpoint A;
- exp007 DDP4 epoch 50 and epoch 100 as val200 reference rows.

## Source Runtime Paths

```text
EXP006=/mnt/shengdata1/hengjie/experiments/rexgroundingct/006_voxtell_cached_native_v123_lr_ablation/runs/exp006_cached_native_lr_20260725T050001Z/v123_cached_e5_d4
EXP007=/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/runs/exp007_full_20260725T231624Z/ddp_bs4
EXP008=/mnt/shengdata1/hengjie/experiments/rexgroundingct/008_voxtell_dual_branch_proposal_refinement_ablation/runs/exp008_dual_branch_20260726T182647Z
EXP009=/mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation/runs/exp009_s3_attention_20260727T081747Z
```

Primary metric files are:

```text
<run-or-arm>/eval_epoch{epoch:03d}_val20/reports/val_quick_global_eval_summary.json
<run-or-arm>/eval_epoch{epoch:03d}_val200/reports/val_quick_global_eval_summary.json
```

## Metric Schema

Each table cell uses:

```text
Dice / hit rate (hits/total)
```

Read fields from each JSON:

```text
Dice      = mean_global_dice_per_finding
hit rate  = hit_rate
hits      = total_hits
total     = total_findings
```

Round Dice and hit rate to four decimals in the report.

## Included Models

Nine primary report rows:

1. exp006 `v123_cached_e5_d4` checkpoint A.
2. exp009 `baseline_cont100`.
3. exp008 `v1_sharedfusion_softguide`.
4. exp008 `v1_dualfusion_softguide`.
5. exp008 `v2_dualfusion_precision`.
6. exp008 `v3_dualfusion_softguide_joint`.
7. exp009 `s3v1_fixedrho_suppress_half_quarter`.
8. exp009 `s3v2_balanced_feature_half_quarter`.
9. exp009 `s3v3_logit_residual_half_quarter`.

Val200 reference rows:

1. exp007 DDP4 epoch 50.
2. exp007 DDP4 epoch 100.

DDP4 references appear only in the val200 comparison tables.

## Table Definitions

The report contains:

1. Model lineage table:
   checkpoint A, exp009 plain, four exp008 arms, three exp009 S3 arms, and
   two DDP4 reference rows.
2. Final val200 9-primary-model summary.
3. Dual-head val200 comparison:
   checkpoint A, exp009 plain, four exp008 arms, DDP4 epoch 50, DDP4 epoch 100.
4. S3 val200 comparison:
   checkpoint A, exp009 plain, three exp009 S3 arms, DDP4 epoch 50, DDP4
   epoch 100.
5. Dual-head val20 evolution:
   exp006 trajectory, exp009 plain, four exp008 arms.
6. S3 val20 evolution:
   exp006 trajectory, exp009 plain, three exp009 S3 arms.

Val20 epochs are:

```text
0, 5, 20, 40, 60, 80, 100
```

Missing-data policy:

- Exp008 E5 val20 was not evaluated; show `--`.
- Exp006 val20 E0 is the fixed val20 PT reference from exp006 final summary:
  `0.2986 / 0.5806 (18/31)`.
- Exp006 val20 E5/E20/E40/E60/E80/E100 are read from the exp006
  `v123_cached_e5_d4` runtime summaries.

## Regeneration Sketch

Use this Python pattern from the repository root:

```python
import json
from pathlib import Path

def read_metric(path):
    data = json.loads(Path(path).read_text())
    return {
        "dice": float(data["mean_global_dice_per_finding"]),
        "hit_rate": float(data["hit_rate"]),
        "hits": int(data["total_hits"]),
        "findings": int(data["total_findings"]),
    }

def fmt(metric):
    if metric is None:
        return "--"
    return (
        f"{metric['dice']:.4f} / {metric['hit_rate']:.4f} "
        f"({metric['hits']}/{metric['findings']})"
    )
```

Then read val20/val200 summaries from the source runtime paths above, preserve
the row ordering in this plan, and paste regenerated Markdown tables into the
main report.

## Verification

After updating the report, run:

```bash
git diff --check -- \
  docs/voxtell/2026-07-28_exp007_exp008_exp009_checkpoint_report.md \
  docs/voxtell/2026-07-28_exp007_exp008_exp009_checkpoint_report_plan.md
```

No code tests are required for this documentation-only report.

## Interpretation Defaults

- Treat exp009 `baseline_cont100` as the shared plain continuation baseline.
- State explicitly that exp008 lacks its own concurrent plain-control arm.
- Treat final val200 as the primary model-selection signal.
- Treat val20 as a small trajectory/debugging probe: 20 cases and 31 findings.
