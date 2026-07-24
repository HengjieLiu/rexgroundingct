---
created: 2026-07-23
updated: 2026-07-23
status: active
purpose: Reusable prompt for adapting prior AI-assisted research workflows.
---

# AI Workflow Upgrade Prompt

Use this prompt when adapting prior project workflow lessons into a lightweight
research repository.

```text
Inspect the current repo and the available prior AI workflow summaries. Identify
which patterns are useful for this project without causing a codebase refactor.
Rank the recommendations from most important to least important. Favor durable
but lightweight project memory: execution specs before long runs, small
repo-local summaries, smoke-test gates, decision/todo updates, prompt archives,
artifact hygiene, and canonical checks.

Do not move large data, predictions, checkpoints, logs, or model weights into
Git. Do not reorganize scripts into a package unless the current project already
requires it. Turn accepted recommendations into concrete docs, templates,
checks, and small hygiene fixes. After implementation, run the canonical local
checks and report any failures honestly.
```

## Expected Artifacts

- AI workflow docs page.
- Current status or decision log.
- Execution spec template.
- Experiment closeout template.
- Prompt archive README.
- Canonical check command.
- Any small hygiene fixes found during repo inspection.
