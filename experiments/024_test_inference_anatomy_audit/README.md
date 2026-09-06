# Exp024: validation-gated test inference with anatomy support

Exp024 is an external-runtime experiment.  It produces six native-resolution
test prediction directories and keeps raw checkpoint outputs, masks, routing
manifests, logs, and reports outside Git:

`/mnt/shengdata1/hengjie/experiments/rexgroundingct/024_test_inference_anatomy_audit`

Execution is complete: all six sets contain 300 files / 582 findings and passed
the native CT-header verification contract.  The b validation gate matched the
published oracle before test inference began.

The coordinator is `scripts/rexgroundingct/run_024_test_inference_anatomy.py`.
Run `preflight`, freeze both prompt-routing manifests, and resolve/hash the
nine retrospective category-oracle checkpoints before launching inference.
The b-test stage must only be launched after the complete validation gate has
written its completion marker.  The oracle is validation-selected and
optimistic; it is not an unbiased test estimate.  Official test TotalSegmentator
source status remains `PASS_PENDING_MANUAL_VISUAL_REVIEW` until a separately
documented human review exists.

No test ground truth is read or created.  Every exported file is checked against
the original CT NIfTI shape, affine, orientation, finding count, and uint8
binary contract.
