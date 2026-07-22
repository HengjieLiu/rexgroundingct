---
created: 2026-07-22
updated: 2026-07-22
status: active
---

# Workflow Rules

## Defaults

- Read `AGENTS.md`, all files in `AGENTS/`, and relevant folder READMEs before
  changing files.
- Prefer small topic-focused commits with explicit path staging.
- Do not use broad `git add .`; stage the exact files intended for each commit.
- Preserve user or runtime changes that are unrelated to the current task.
- Use `rg` for repo search and `git mv` for tracked file moves.

## Checks

Run these before committing experiment or script changes:

```bash
python -m py_compile scripts/rexgroundingct/*.py
python scripts/rexgroundingct/check_experiment_consistency.py
git diff --check
git diff --cached --check
```

For documentation-only changes, still run the Git whitespace checks.

## Commit Hygiene

- Keep generated or synced artifacts in their intended commit topic.
- Keep heavyweight runtime artifacts out of Git.
- Mention any skipped checks in the final handoff.
