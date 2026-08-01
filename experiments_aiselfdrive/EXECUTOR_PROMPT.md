# Poll for the first eligible experiment

Paste the following as one message into a separate Codex chat window:

```text
Work in /home/hengjie/code_sync/rexgroundingct.

Monitor /home/hengjie/code_sync/rexgroundingct/experiments_aiselfdrive every
60 seconds. Continue polling while the portfolio is valid but temporarily has
no eligible experiment. Do not mistake the technical report alone for a
completed plan, and do not edit or invent missing plan files while waiting.

The portfolio is runnable only when all of these are true:
1. experiments_aiselfdrive/portfolio.yaml exists.
2. experiments_aiselfdrive/tools/experimentctl.py exists.
3. `python experiments_aiselfdrive/tools/experimentctl.py validate --all`
   exits successfully.
4. `python experiments_aiselfdrive/tools/experimentctl.py next --json`
   returns `"available": true`, id `asd_000_evidence_lock_error_atlas`, and a
   claimable status such as `ready`.

If `next --json` returns `no_eligible_experiment`, inspect `STATUS.md` and the
ASD-000 state. Keep polling when the reason is a temporary plan migration,
lease, retry wait, or declared dependency wait. If validation fails, retain
the structured error output: keep polling when errors are explicitly caused by
an in-progress plan/state migration or missing control-plane lock and relevant
file modification times are still changing. Stop and report when the same
non-migration validation error remains for three polls without file changes,
the state contains a terminal scientific/lineage blocker requiring user input,
the experiment is finished, or the user interrupts you. Never repair those
errors from this waiting session. Send a short status update at least every
five polling cycles.

Before acting, obey the repository AGENTS.md instructions: read every Markdown
file under AGENTS/ and every applicable README.md.

When the runnable-selection conditions pass:
1. Claim only `asd_000_evidence_lock_error_atlas` using `experimentctl.py
   claim --id asd_000_evidence_lock_error_atlas --agent-id
   <your-stable-agent-id>`.
2. Execute its stages in DAG order, including `execution_mode: agent`
   implementation stages, exactly within each stage's allowed write paths.
   Pass the same `--agent-id` to every `run-stage` and `complete-stage` call.
3. During an agent stage, call `experimentctl.py heartbeat --id
   asd_000_evidence_lock_error_atlas --agent-id <your-stable-agent-id>` every
   60 seconds until completion evidence is recorded.
4. Re-run validation after implementation stages.
5. Continue until this experiment is `finished`, `blocked`, or otherwise
   terminal.
6. Never start asd_005 or another experiment in this session.
7. Preserve unrelated worktree changes and do not modify the immutable
   technical report or a claimed experiment plan.
8. Use only declared local inputs. Treat `/common/...`, H100/H200, and Slurm
   references as historical.
9. Put new code/config/state under experiments_aiselfdrive and heavy artifacts
   only in the declared external runtime directory.
10. Use Docker for GPU access and enforce the declared GPU-memory preflight.
11. If blocked or invalid, record the exact blocker and evidence in state; do
    not guess or silently weaken a gate.

Give me a short update whenever state changes and a final report containing
status, outcome, completed stages, metrics, artifact paths, and the exact
blocker or next decision.
```

The polling agent must not claim an experiment merely because its directory
exists. A passing global validator and an available `next --json` result are
the readiness barrier.
