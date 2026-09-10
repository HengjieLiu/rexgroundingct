# Top-four test300 d1/d2/d3 packages

Equal float32 sigmoid probabilities, fixed rank order, sum >= 2.0. d2 applies eligible whole-lung 20 mm support; d3 independently applies eligible prompt-selected 20 mm support to d1.

Sixteen-case continuation retained the original smoke outputs and unchanged numerical/export functions. Each package contains 300 native CT files / 582 prompts. No labels, test metrics or upload.

Anatomy review status: PASS_PENDING_MANUAL_VISUAL_REVIEW.

- d1: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/024_test_inference_anatomy_audit/outputs/d1.zip` SHA256 `d785c2268cf6e9fe5a9cedc5194f6100882c17027ca97f8c3660472079bdd5cd`
- d2: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/024_test_inference_anatomy_audit/outputs/d2.zip` SHA256 `2a0c5753efa06d9c1b60658d53d98dc547af4620eae9a1eecb82b30152961b06`
- d3: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/024_test_inference_anatomy_audit/outputs/d3.zip` SHA256 `5341975cac8f4a6c56a81338730f6f90bc93e669fcce2cd43395459ce58d5064`

Reproduce: `python /home/hengjie/code_sync/rexgroundingct/side_experiments/sideexp003_ensemble_method_hub/submission_test300_16.py --job /home/hengjie/code_sync/rexgroundingct/side_experiments/sideexp003_ensemble_method_hub/submission_jobs/s001_top4_test300_d123/job_spec.json --execution /mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp003_ensemble_method_hub/submission_jobs/s001_top4_test300_d123/continuation_16/execution_spec.json`
