# Side Experiment 002 Logit Export Audit

- Accepted: `0 / 20`
- Complete but unaccepted: `6`
- Incomplete or not started: `14`
- Historical warnings: `0`

Historical reproduction is diagnostic. Acceptance requires complete arrays, exact same-pass storage reproduction, verified array hashes, and unchanged candidate provenance.

| Candidate | Status | Dtype | Cases | Findings | Storage | Masks | Historical | Dice delta | Hits | Logit range | Clips L/H | Provenance | Array hashes | Warnings |
| --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- |
| `exp006_e6d4_e100` | complete_unaccepted | float16 | 200 | 381 | passed | None | passed | -0.00005911 | 281 / 281 | -281.000000 to 76.687500 | 18225488811 / 6004 | match | 200 | - |
| `exp006_e5d4_e100` | complete_unaccepted | float32 | 200 | 381 | passed | None | failed | -0.00016521 | 290 / 290 | -186.875000 to 87.937500 | 15983689825 / 3176 | match | 200 | - |
| `exp007_ddp_e050` | complete_unaccepted | float32 | 200 | 381 | passed | None | failed | -0.00025926 | 287 / 288 | -257.250000 to 89.312500 | 17806366658 / 3786 | match | 200 | - |
| `exp007_ddp_e075` | complete_unaccepted | float16 | 200 | 381 | passed | None | passed | +0.00006959 | 290 / 290 | -275.000000 to 96.875000 | 18925800208 / 8598 | match | 200 | - |
| `exp007_ddp_e100` | complete_unaccepted | float32 | 200 | 381 | passed | None | failed | -0.00021870 | 288 / 289 | -303.000000 to 82.437500 | 17428709983 / 3995 | match | 200 | - |
| `exp008_shared_e080` | running | float16 | 19 | 0 | missing | missing | missing | - | - | -119.000000 to 35.781250 | 943220032 / 67 | match | 19 | - |
| `exp008_shared_e100` | running | float16 | 19 | 0 | missing | missing | missing | - | - | -122.187500 to 38.406250 | 1037861636 / 122 | match | 19 | - |
| `exp008_dual_e080` | complete_unaccepted | float16 | 200 | 381 | passed | None | passed | +0.00001106 | 288 / 288 | -271.000000 to 60.687500 | 13901432815 / 2038 | match | 200 | - |
| `exp008_dual_e100` | running | float16 | 19 | 0 | missing | missing | missing | - | - | -141.375000 to 35.250000 | 1003667380 / 49 | match | 19 | - |
| `exp008_precision_e080` | not_started | - | 0 | 0 | - | - | - | - | - | - | 0 / 0 | - | 0 | - |
| `exp008_precision_e100` | not_started | - | 0 | 0 | - | - | - | - | - | - | 0 / 0 | - | 0 | - |
| `exp008_joint_e080` | running | float32 | 105 | 0 | passed | None | failed | +0.00003362 | 287 / 288 | -152.000000 to 56.812500 | 7254553362 / 1474 | match | 105 | - |
| `exp008_joint_e100` | not_started | - | 0 | 0 | - | - | - | - | - | - | 0 / 0 | - | 0 | - |
| `exp009_baseline_e100` | not_started | - | 0 | 0 | - | - | - | - | - | - | 0 / 0 | - | 0 | - |
| `exp009_s3v1_e100` | not_started | - | 0 | 0 | - | - | - | - | - | - | 0 / 0 | - | 0 | - |
| `exp009_s3v2_e100` | not_started | - | 0 | 0 | - | - | - | - | - | - | 0 / 0 | - | 0 | - |
| `exp009_s3v3_e100` | not_started | - | 0 | 0 | - | - | - | - | - | - | 0 / 0 | - | 0 | - |
| `exp011_native_zscore_e100` | not_started | - | 0 | 0 | - | - | - | - | - | - | 0 / 0 | - | 0 | - |
| `exp011_clip_zscore_e100` | not_started | - | 0 | 0 | - | - | - | - | - | - | 0 / 0 | - | 0 | - |
| `exp011_linear_hu_e100` | not_started | - | 0 | 0 | - | - | - | - | - | - | 0 / 0 | - | 0 | - |
