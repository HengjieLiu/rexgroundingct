---
created: 2026-07-22
updated: 2026-09-08
status: active
---

# Documentation Map

Start here when looking for project context.

## Folder Map

- `ai_workflow.md`: AI-assisted research workflow, execution spec rules,
  closeout loop, agent command/reflection split, and canonical checks.
- `current_status.md`: current decisions, active work, blockers, and next
  actions.
- `submission.md`: challenge submission packaging and final pre-submit checks.
- `gpu8_transfer_runbook.md`: GPU9-to-GPU8 transfer and verification for code,
  project data, Docker, and shared `/mnt` storage.
- `val200_leaderboards/`: dated fixed-val200 model leaderboard snapshots and
  refresh notes.
- `templates/`: reusable execution spec and experiment closeout templates.
- `brainstorm/`: dated research notes covering model capability gaps,
  counterfactual supervision, and candidate redesigns.
- `voxtell/`: VoxTell experiment workflow, preprocessing notes, normalization
  notes, and prompt-analysis references.
- `../dataset/README.md`: gated ReXGroundingCT and CT-RATE download workflow.
- `../experiments/README.md`: repo-local experiment index and drift policy.
- `../challenge_info/README.md`: source-faithful challenge archive and latest
  official-source snapshot.
- `../literature/`: locally stored papers and PDFs used as background reading.

## Agent Notes

Agents should read `../AGENTS.md` first, then the relevant folder `README.md`
files for the area being changed. For substantial experiment, submission, or
workflow work, also read `ai_workflow.md` and `current_status.md`.
