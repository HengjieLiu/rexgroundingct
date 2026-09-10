# GPU8 test300 launch report

Snapshot: 2026-09-09T02:00:44Z.

Job `j003_top20_test300_fresh_gpu8` is running on `shenggpu8`, branch `gpu8`, supervisor PID `1692263`.
Frozen spec SHA-256: `50ac85fb91a37a748be45e82e27d4ed09862645cdb06258b07f67fdd5bf8e912`.
Source bundle: `14e08295f14402043a1c1dd553194adecd4610a28053b8e313c4b608acf19dc6`.

Wave 1 passed both retained smoke cases on every GPU and is exporting the full test cohort. Published 300-case caches are still pending. The supervisor schedules the remaining four waves automatically.

| Rank | GPU | Model | Completed cases | Post-smoke free MiB |
| ---: | ---: | --- | ---: | ---: |
| 1 | 0 | `exp007_ddp_bs4_e050_a499ad1c` | 5/300 | 44074 |
| 2 | 1 | `exp009_baseline_cont100_e100_50e631f1` | 5/300 | 44074 |
| 3 | 2 | `exp017_ddp_bs4_e050_4f36d9bb` | 5/300 | 44396 |
| 4 | 3 | `exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801` | 5/300 | 43196 |

Validation: 63 SideExp003 tests, 7 preprocessing regression tests, all 300 real CT headers / 582 prompts, and the offline Qwen4B dependency check passed. Segmentation files are hidden from GPU workers. Repository and SideExp003 checks passed; the repository retains 19 pre-existing consistency warnings.

The separate native test preprocessing cache completed with 300 images, zero targets, and 112.80 GiB. The original native train/val and iso07 manifests remain unchanged.
Shared free: 24.02 TiB; local free: 6.35 TiB. The 20 TiB shared reserve remains enforced.

CPU search keeps its selected 5 workers and original supervisor. Current case counts: `{"initial": 200, "k05": 200, "k06": 200, "k07": 200, "k08": 200, "k09": 200, "k10": 2}`. This coordinator currently owns no CPU pause.

The initial startup exported zero cases because the image supplied a conflicting HF_HUB_CACHE default. The launcher now explicitly uses the transferred cache, verified offline. The immutable failed spec/source are retained in `failed_startup_001/`; its logs remain in the sibling runtime directory suffixed `__failed_startup_001`. Replacement cache identities are distinct; no logits or existing cache data were deleted.

The initial 24–30 hour budget remains provisional. Retained smoke cases include the largest workloads and startup costs, so their mean is not a full-cohort ETA. The supervisor records a measured remaining ETA after Wave 1.

Read live runtime state on the gpu8 host; the tool sandbox returned stale external runtime files during verification. The host supervisor and container views agreed on smoke completion.

```bash
python side_experiments/sideexp003_ensemble_method_hub/hub.py cache watch \
  --job j003_top20_test300_fresh_gpu8 --once
```

Runtime: `/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp003_ensemble_method_hub/cache/jobs/j003_top20_test300_fresh_gpu8`. Watch `state.json`, `progress/`, and `supervisor.log`; final `inventory.json` and `report.md` are copied into this tracked job folder when all 20 caches validate.

Main remains at `4617db190f1e6bf93e2093fe7ef4741c2bf9bc7d`. Val200 artifact identities and frozen scoring modules were preserved. Ensemble selection and submission are unchanged.
