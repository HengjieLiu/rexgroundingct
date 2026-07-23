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

Run the canonical workflow check before committing experiment or script changes:

```bash
python scripts/rexgroundingct/check_repo_workflow.py
```

Expanded manual sequence:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest challenge_info.tests.test_update_challenge_info
python scripts/rexgroundingct/check_experiment_consistency.py
python -m py_compile scripts/rexgroundingct/*.py challenge_info/update_challenge_info.py dataset/download_ct_rate_challenge_subset.py
git diff --check
git diff --cached --check
git ls-files '*__pycache__*' '*.pyc'
```

For documentation-only changes, still run the Git whitespace checks.

## Experiment Specs

- Write `experiments/<experiment-id>/codex_execution_spec.md` before substantial
  implementation or long experiment launches.
- Use `docs/templates/codex_execution_spec.md` as the starting point.
- After meaningful results, sync the experiment index and update
  `docs/current_status.md` if the next decision changes.

## Commit Hygiene

- Keep generated or synced artifacts in their intended commit topic.
- Keep heavyweight runtime artifacts out of Git.
- Mention any skipped checks in the final handoff.
