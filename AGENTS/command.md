---
created: 2026-07-24
updated: 2026-09-07
status: active
---

# Agent Commands

This file stores reusable quick commands for common user requests. Commands are
short task patterns an agent can apply directly after reading the required repo
context.

Use `workflow.md` for durable repo workflow rules. Use `workflow_reflect/` for
workflow reflection, scaffold review, upgrade prompts, and historical workflow
evidence.

## Command Index

| Command | How To Call It | Use When |
| --- | --- | --- |
| Topic-Clustered Commit Preparation | `prepare commits by topic, wait for approval` | The user wants a reviewable commit plan for a dirty tree before any staging or commits. |
| Add ExpXXX to SideExp003 Leaderboards | `add ExpXXX to SideExp003 leaderboards` | A new single-checkpoint fixed-val200 evaluation should enter the deduplicated ensemble catalog. |
| Watch SideExp003 Fresh Top-20 Cache | `watch SideExp003 top20 logits` | Inspect the guarded fresh-only val200 export without launching greedy search. |

## Watch SideExp003 Fresh Top-20 Cache

Read the SideExp003 README, immutable roster, and cache job spec. Then run the
one-shot status command first:

```bash
python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  cache watch --job j001_top20_val200_fresh --once
```

Use `--interval 60` only when the user asks to keep monitoring. Report current
wave, case counts, failures, GPU safety state, and ETA. Never substitute a
legacy cache, start greedy, or select `M` as part of this command.

## Add ExpXXX to SideExp003 Leaderboards

Use this command when the user asks to add or refresh an experiment in the
SideExp003 checkpoint and official-category leaderboards.

Recommended shorthand:

```text
add Exp027 to SideExp003 leaderboards
```

1. Read the SideExp003 README and execution spec. Do not modify the canonical
   experiment registry or configs as part of this command.
2. Run:

   ```bash
   python side_experiments/sideexp003_ensemble_method_hub/hub.py \
     catalog add --experiment 027 --dry-run
   ```

3. Verify exactly `200 cases / 381 findings`, the fixed dataset SHA, mask
   threshold provenance, unambiguous checkpoint mapping, and checkpoint-SHA
   deduplication. Category metrics must be recomposed by `(case name, finding
   index)`.
4. If duplicate evaluators disagree, stop and return the generated override
   stub. Never choose a canonical evaluator without user direction. After the
   user decides, merge the selected path and rationale into SideExp003
   `catalog_overrides.json`, then repeat the dry-run.
5. Apply only after the dry run passes:

   ```bash
   python side_experiments/sideexp003_ensemble_method_hub/hub.py \
     catalog add --experiment 027 --apply
   python side_experiments/sideexp003_ensemble_method_hub/hub.py check
   python -m unittest discover \
     -s side_experiments/sideexp003_ensemble_method_hub -p 'test_*.py'
   python scripts/rexgroundingct/check_repo_workflow.py
   git diff --check
   ```

Both Markdown leaderboards are generated atomically from
`checkpoint_catalog.json`; never maintain ranking rows manually. Records with
missing threshold or checkpoint provenance remain cataloged as `needs_review`
and are roster-ineligible by default.

## Topic-Clustered Commit Preparation

Use this command when the user asks to prepare commits, group dirty-tree changes
by topic, or draft a commit plan for review.

Recommended shorthand:

```text
prepare commits by topic, wait for approval
```

```text
Prepare a series of topic-clustered commits for this repo.

First ground yourself in the repo:
- read AGENTS.md, every Markdown file under AGENTS/, and relevant folder README.md files;
- inspect git status, ignored files, submodule state, and the size/type of untracked artifacts;
- treat existing dirty-tree changes as user work unless clearly produced by this task.

Group changes by topic, ownership boundary, and reviewability. For each proposed
commit, provide:
- a concise subject line;
- a comprehensive but compact message body;
- exact paths to include;
- paths to keep out of Git or add to .gitignore;
- verification to run before and after staging.

Use explicit path staging only. Do not use git add .

Do not commit runtime outputs, logs, predictions, checkpoints, model weights,
medical image volumes, Python caches, notebook checkpoints, large generated
figures, credentials, or unrelated dirty-tree changes.

For repo-local synced experiment summaries, refresh them only at the correct
point in the sequence so repo_commit and provenance fields describe the intended
committed code/config state.

Stop after presenting the commit plan and wait for user approval before staging
or committing.
```

After approval, stage one commit at a time with explicit paths, inspect
`git diff --cached --name-status` before every commit, use the approved subject
and detailed body, run the agreed checks, and report final commit hashes plus
any remaining ignored or untracked artifacts.
