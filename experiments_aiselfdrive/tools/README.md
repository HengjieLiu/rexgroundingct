# Self-drive tools

`experimentctl.py` is the only supported writer for execution state, event
history, leases, and generated dashboards. Scientific implementation scripts
belong in their owning experiment directory.

Validate and select from the repository root:

```bash
python experiments_aiselfdrive/tools/experimentctl.py validate --all
python experiments_aiselfdrive/tools/experimentctl.py next --json
```

Every execution mutation is owner-bound. Use the same stable `--agent-id` for
`claim`, `run-stage`, `heartbeat`, and `complete-stage`; a different agent may
not complete or refresh that lease. Command stages are supervised by the
controller and heartbeat automatically. Agent stages return a bounded work
order and require explicit completion evidence at the path declared by the
plan.

Use `heartbeat --id <id> --agent-id <owner>` every 60 seconds during an agent
stage. Heartbeats update only the canonical live lease; `events.jsonl` remains
reserved for material state transitions.

Intentional plan changes use an audited revision rather than editing live
state by hand:

```bash
python experiments_aiselfdrive/tools/experimentctl.py replan \
  --id <id> --agent-id <owner> --reason <reason> \
  --previous-plan-source git:<commit>:<repo-relative-plan-path> \
  --adopt-current
```

The controller verifies the recovered predecessor, archives the old state
under ignored `.runtime/replans/`, appends a hash-chained `plan_revised`
event, resets stages, and preserves revision lineage. After a coordinated
registry migration, pin all non-draft claim/artifact obligations and executable
control-plane sources with:

```bash
python experiments_aiselfdrive/tools/experimentctl.py lock-control-plane \
  --agent-id <owner>
```

Do not repin merely to hide drift; validation must explain any changed source.
After every candidate reaches a terminal T3 decision (including an audited
early scientific no-go), issue the deterministic, maximum-two T4 decision with:

```bash
python experiments_aiselfdrive/tools/experimentctl.py promote \
  --agent-id <owner>
```

Promotion records are controller-produced, hash-verified, and required before
the selected T4 stage can run. Non-selected candidates proceed to closeout.
Each candidate's T4 stage declares a treatment arm and a distinct disabled-
module control arm. Their checkpoint and metric artifacts must be separate,
and completion evidence must prove equal parent, data, schedule, and update
count hashes before either arm can support promotion claims.

For command stages that launch Docker directly, the controller injects the
lease environment and labels. A script that launches a nested container must
propagate:

```text
REXGROUNDINGCT_AISELFDRIVE_EXPERIMENT_ID
REXGROUNDINGCT_AISELFDRIVE_LEASE_ID
REXGROUNDINGCT_AISELFDRIVE_DOCKER_LABEL
rexgroundingct.aiselfdrive.experiment=<experiment-id>
rexgroundingct.aiselfdrive.lease=<lease-id>
```

These labels let stale-lease recovery verify the owned container rather than
reclaiming work from elapsed time alone.
