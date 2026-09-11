# SideExp005 test300 d11/d12

Complete: two folders, 600 verified files, 582 prompts per variant.

d11 applies frozen collaborator semantic-v1 to d1. d12 applies frozen strict semantic-v2 to d11.
Validation scores did not gate generation. Test labels and metrics are unavailable.
ZIPs were skipped by user request; no upload was performed.

Anatomy review status: PASS_PENDING_MANUAL_VISUAL_REVIEW

| Variant | Bytes | Prediction directory |
| --- | ---: | --- |
| d11 | 269258273 | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/024_test_inference_anatomy_audit/outputs/d11` |
| d12 | 269231106 | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/024_test_inference_anatomy_audit/outputs/d12` |

Finding-level routing and foreground counts: `/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp005_postprocessing_merge/runs/r002_d1_test300_d11_d12_frozen_postprocessing/reports/findings.json`.

Reproduction: `python /mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp005_postprocessing_merge/runs/r002_d1_test300_d11_d12_frozen_postprocessing/source/side_experiments/sideexp005_postprocessing_merge/test300_postprocessing.py run --config /mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp005_postprocessing_merge/runs/r002_d1_test300_d11_d12_frozen_postprocessing/config.json`.
