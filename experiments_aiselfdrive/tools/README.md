# Self-drive tools

`experimentctl.py` is the only supported writer for execution state, event
history, leases, and generated dashboards. Scientific implementation scripts
belong in their owning experiment directory.

Use `heartbeat --id <id> --agent-id <owner>` every 60 seconds during an agent
stage. Heartbeats update only the canonical live lease; `events.jsonl` remains
reserved for material state transitions.
