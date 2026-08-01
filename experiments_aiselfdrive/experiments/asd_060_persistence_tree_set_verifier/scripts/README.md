# ASD-060 script contract

The implementation stage creates `validate_inputs.py`, `build_oracle.py`,
`resolve_branch.py`, `gpu_preflight.py`, `build_training_data.py`, and
`train_verifier.py`. Every script is import-safe, deterministic, and writes an
atomic completion JSON. Branch and sealed-cohort checks are mandatory.
