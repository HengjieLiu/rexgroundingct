# ASD-040 script contract

The implementation stage creates `validate_inputs.py`,
`build_censorship_lab.py`, `gpu_preflight.py`, `extract_features.py`,
`run_crossfit.py`, and `run_training.py`.

Training and feature processes must not mount or resolve the evaluator-only
hidden manifest. Only the evaluation process may join prediction trial IDs to
hidden masks. Every script writes an atomic JSON completion marker and refuses
pre-T4 val120/val200 access.
