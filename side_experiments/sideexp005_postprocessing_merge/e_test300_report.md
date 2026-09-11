# SideExp005: e-series test300 outputs

Eight frozen checkpoints, equal sigmoid-probability averaging. e2/e3/e11 derive from e1; e12 derives from e11.
Anatomy review status: PASS_PENDING_MANUAL_VISUAL_REVIEW.
No ZIPs or uploads are created.

Complete: five folders × 300 files, covering 582 prompts per variant. Test labels and scores are unavailable.


## Provenance and timing

Job SHA: `5fee63d1859824ac8f52a6d9eec93ee62ecd6c11020f23f533fc753cf4892700`.
| Phase | Wall minutes |
| --- | ---: |
| postprocessing | 213.71 |

Worker timing totals:

```json
{
  "anatomy_seconds": 14907.504942156374,
  "averaging_seconds": 1940.3030560202897,
  "reading_hash_seconds": 15609.203730270267,
  "scoring_seconds": 0.0,
  "verification_seconds": 10022.658503353596,
  "wall_seconds": 44674.878489904106,
  "writing_seconds": 2007.4140121787786
}
```

Detailed findings, routing and largest changes: `/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp005_postprocessing_merge/runs/r004_top8_test300_frozen_postprocessing/reports`.
Prediction storage: 1.255 GiB.
Reproduction: frozen `e_series_compute.py run --stage test300 --config /mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp005_postprocessing_merge/runs/r004_top8_test300_frozen_postprocessing/config.json`.
