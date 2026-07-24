---
created: 2026-07-24
updated: 2026-07-24
status: active
---

# Agent Commands

This file stores reusable quick commands for common user requests. Commands are
short task patterns an agent can apply directly after reading the required repo
context.

Use `workflow.md` for durable repo workflow rules. Use `workflow_reflect/` for
workflow reflection, scaffold review, upgrade prompts, and historical workflow
evidence.

## Topic-Clustered Commit Preparation

Use this command when the user asks to prepare commits, group dirty-tree changes
by topic, or draft a commit plan for review.

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
