# ASD-050 script contract

The implementation stage creates `build_quartets.py`, `gpu_preflight.py`,
`evaluate_quartets.py`, and `run_training.py`. Entry points are import-safe,
support `--help`, write atomic completion JSON, and enforce sealed-cohort
access. Flipped arrays are runtime scratch and are never committed.
