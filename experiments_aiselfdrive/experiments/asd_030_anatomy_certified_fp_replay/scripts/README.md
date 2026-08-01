# ASD-030 script contract

The implementation stage creates these import-safe entrypoints:

- `validate_inputs.py`
- `gpu_preflight.py`
- `mine_candidates.py`
- `certify_replay.py`
- `evaluate_coverage.py`
- `run_training.py`

Every entrypoint must support `--help`, accept only the arguments declared in
`experiment.yaml`, write atomically below the declared runtime root, and emit a
JSON completion record. Scripts must never edit state directly or access
val120/val200 before a verified T4 promotion record.
