# ASD-030 configurations

`protocol.yaml` is the only execution configuration. It freezes candidate
mining, certification, tri-state loss masking, arm schedules, cohorts, and
gates. Implementations must reject command-line overrides that alter a
scientific field; only worker count, device assignment, and resume location may
be operational overrides, and all are recorded in the runtime manifest.
