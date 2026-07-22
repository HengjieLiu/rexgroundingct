---
created: 2026-07-22
updated: 2026-07-22
status: active
---

# Hardware And Storage Assumptions

## Known Hardware

- Primary experiment configs assume 4 GPUs.
- Experiment 002 config records NVIDIA RTX 6000 Ada Generation GPUs with 48 GiB
  VRAM each.

## Storage

- Large CT volumes and runtime outputs live on `/mnt/shengdata1`.
- Gated challenge metadata and masks live under `/data/hengjie`.
- The Git repo should hold code, configs, small summaries, and documentation.

## Practical Notes

- Full train plus validation CT storage is hundreds of GiB.
- Validation inference and fine-tuning launchers should check CT readiness before
  starting expensive runs.
- Keep checkpoint and prediction paths outside tracked source directories.
