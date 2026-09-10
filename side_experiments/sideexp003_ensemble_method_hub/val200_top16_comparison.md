# Val200 seeded search versus uniform top-16 average

Final K=4–16 comparison. All passes evaluated 200 cases. These are optimistic same-val200 diagnostics, not OOF estimates or final ensemble recipes.

The uniform baseline was already computed during the initial pass. Its metrics were rechecked against all 200 unchanged archived case partials (381 unique findings). No fresh model inference or source-logit recomputation was needed. Both methods use the same frozen 16 candidates, average post-sigmoid probabilities, and threshold at >=0.5. Uniform uses all 16 once at 6.25% each; the five seeded searches have scope-specific baskets with replacement.

Dice is on a 0–1 scale; parentheses give hits (finding Dice >=0.1). Bold marks the highest Dice per scope.

| Method / K | All — 381 | 2a — 69 | 2b — 49 | 2c — 60 | 2d — 132 |
|---|---:|---:|---:|---:|---:|
| K=4, initial | 0.357504 (296) | 0.362824 (58) | 0.370373 (38) | 0.420702 (50) | 0.402057 (115) |
| K=5 | 0.360958 (298) | 0.365518 (59) | 0.371852 (38) | 0.426851 (51) | 0.408198 (116) |
| K=6 | 0.365395 (298) | 0.369313 (59) | 0.373111 (38) | 0.426372 (51) | 0.414497 (117) |
| K=7 | 0.365726 (298) | 0.370389 (59) | 0.373078 (38) | 0.427622 (51) | 0.412719 (118) |
| K=8 | 0.366540 (299) | 0.371421 (59) | 0.373256 (37) | 0.428111 (51) | 0.415954 (118) |
| K=9 | 0.366674 (300) | 0.371939 (59) | 0.373457 (37) | 0.427565 (51) | 0.413966 (118) |
| K=10 | 0.367196 (298) | 0.372406 (59) | 0.373734 (38) | 0.428272 (51) | 0.416432 (118) |
| K=11 | 0.367471 (300) | 0.372466 (59) | 0.373679 (38) | 0.427959 (51) | 0.414438 (118) |
| K=12 | 0.367413 (298) | 0.372956 (58) | 0.373773 (37) | 0.428170 (51) | **0.416433** (118) |
| K=13 | 0.367166 (301) | 0.373302 (59) | 0.373800 (37) | 0.427877 (51) | 0.414826 (118) |
| K=14 | 0.368012 (299) | 0.373455 (58) | 0.373914 (37) | **0.428357** (51) | 0.416383 (118) |
| K=15 | 0.367753 (301) | **0.373606** (59) | **0.373941** (37) | 0.428235 (51) | 0.414779 (118) |
| K=16 | **0.368020** (301) | 0.373590 (58) | 0.373878 (37) | 0.428117 (51) | 0.416292 (118) |
| Uniform top 16 | 0.351785 (293) | 0.357965 (59) | 0.365147 (37) | 0.402180 (50) | 0.388951 (114) |

## Seed baskets and additions

| Step | All | 2a | 2b | 2c | 2d |
|---|---|---|---|---|---|
| Initial K=4 seeds | R1, R2, R3, R4 | R4, R3, R15, R2 | R7, R4, R11, R1 | R1, R16, R13, R8 | R1, R3, R2, R5 |
| K=5: add | R8 | R8 | R8 | R11 | R8 |
| K=6: add | R11 | R13 | R3 | R8 | R8 |
| K=7: add | R8 | R11 | R8 | R1 | R11 |
| K=8: add | R1 | R16 | R1 | R16 | R1 |
| K=9: add | R11 | R8 | R3 | R8 | R3 |
| K=10: add | R1 | R1 | R14 | R1 | R9 |
| K=11: add | R11 | R3 | R3 | R11 | R8 |
| K=12: add | R5 | R13 | R7 | R16 | R2 |
| K=13: add | R8 | R8 | R1 | R1 | R3 |
| K=14: add | R1 | R13 | R8 | R8 | R1 |
| K=15: add | R11 | R8 | R1 | R13 | R3 |
| K=16: add | R1 | R13 | R8 | R8 | R9 |
| Uniform top 16 | R1–R16 once each | R1–R16 once each | R1–R16 once each | R1–R16 once each | R1–R16 once each |

## Final K=16 normalized weights

| Checkpoint | All | 2a | 2b | 2c | 2d | Uniform top 16 |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 31.25% | 6.25% | 25% | 25% | 18.75% | 6.25% |
| R2 | 6.25% | 6.25% | 0% | 0% | 12.5% | 6.25% |
| R3 | 6.25% | 12.5% | 18.75% | 0% | 25% | 6.25% |
| R4 | 6.25% | 6.25% | 6.25% | 0% | 0% | 6.25% |
| R5 | 6.25% | 0% | 0% | 0% | 6.25% | 6.25% |
| R6 | 0% | 0% | 0% | 0% | 0% | 6.25% |
| R7 | 0% | 0% | 12.5% | 0% | 0% | 6.25% |
| R8 | 18.75% | 25% | 25% | 31.25% | 18.75% | 6.25% |
| R9 | 0% | 0% | 0% | 0% | 12.5% | 6.25% |
| R10 | 0% | 0% | 0% | 0% | 0% | 6.25% |
| R11 | 25% | 6.25% | 6.25% | 12.5% | 6.25% | 6.25% |
| R12 | 0% | 0% | 0% | 0% | 0% | 6.25% |
| R13 | 0% | 25% | 0% | 12.5% | 0% | 6.25% |
| R14 | 0% | 0% | 6.25% | 0% | 0% | 6.25% |
| R15 | 0% | 6.25% | 0% | 0% | 0% | 6.25% |
| R16 | 0% | 6.25% | 0% | 18.75% | 0% | 6.25% |

## Checkpoint key

| ID | Frozen candidate ID |
|---|---|
| R1 | `exp007_ddp_bs4_e050_a499ad1c` |
| R2 | `exp009_baseline_cont100_e100_50e631f1` |
| R3 | `exp017_ddp_bs4_e050_4f36d9bb` |
| R4 | `exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801` |
| R5 | `exp008_v1_sharedfusion_softguide_e100_61bcec41` |
| R6 | `exp008_v1_dualfusion_softguide_e100_9ba13980` |
| R7 | `exp009_s3v1_fixedrho_suppress_half_quarter_e100_263f27d3` |
| R8 | `exp017_ddp_bs4_e050_a50a69dc` |
| R9 | `exp007_ddp_bs4_e100_d56474a6` |
| R10 | `exp008_v3_dualfusion_softguide_joint_e100_f98374db` |
| R11 | `exp017_ddp_bs4_e075_39aabab2` |
| R12 | `exp008_v2_dualfusion_precision_e100_e67e2bc2` |
| R13 | `exp012_category_2a_replay50_e100_8dcdac04` |
| R14 | `exp007_ddp_bs4_e050_24e8ab93` |
| R15 | `exp017_ddp_bs4_e100_f11d238f` |
| R16 | `exp008_v1_sharedfusion_softguide_e080_ed0cdd0a` |

## Sources

- [Seeded search result](methods/caruana_replacement_seeded_scoped/runs/r001_top16_per_scope_seed4_fullval/result.json)
- [Uniform top-16 result](methods/uniform_global/runs/r003_top16_fresh_val200/result.json)
- [Uniform run specification](methods/uniform_global/runs/r003_top16_fresh_val200/run_spec.json)

Uniform deterministic result SHA-256: `6e35a492b8d3702044ff4e5ebe33e9f1575233b5c320bab93c4abcbad16bf226`.
Seeded deterministic result SHA-256: `c55a29ae3b19920ccc22b0f90eb9b79105cd56cda6c4b2a74b3b131ed5f92e37`.

## Overall greedy versus category-specific greedy at each K

Each cell is **overall greedy / category-specific greedy**. Both searches use full val200: overall greedy optimizes mean Dice across all 381 findings, while category-specific greedy optimizes only that category and starts from its own category-ranked seeds. Entries are Dice (hits); bold marks the higher Dice within each pair.

Uniform top 4 uses overall-ranked R1–R4 at 25% each; it equals the overall greedy K=4 seed. Uniform top 16 uses R1–R16 at 6.25% each. Both average sigmoid probabilities and threshold at >=0.5. Baseline rows contain one Dice (hits) value per category; greedy rows retain the overall / category-specific pair.

| Method / K | 2a — 69 findings | 2b — 49 findings | 2c — 60 findings | 2d — 132 findings |
|---:|---:|---:|---:|---:|
| Uniform top 4 | 0.356342 (58) | 0.369032 (37) | 0.404281 (51) | 0.403234 (115) |
| Uniform top 16 | 0.357965 (59) | 0.365147 (37) | 0.402180 (50) | 0.388951 (114) |
| 4 | 0.356342 (58) / **0.362824** (58) | 0.369032 (37) / **0.370373** (38) | 0.404281 (51) / **0.420702** (50) | **0.403234** (115) / 0.402057 (115) |
| 5 | 0.362199 (59) / **0.365518** (59) | 0.371810 (38) / **0.371852** (38) | 0.409479 (51) / **0.426851** (51) | 0.405781 (116) / **0.408198** (116) |
| 6 | **0.370198** (58) / 0.369313 (59) | 0.372161 (38) / **0.373111** (38) | 0.413468 (51) / **0.426372** (51) | 0.411732 (118) / **0.414497** (117) |
| 7 | **0.371623** (59) / 0.370389 (59) | 0.372415 (38) / **0.373078** (38) | 0.417976 (51) / **0.427622** (51) | 0.410642 (117) / **0.412719** (118) |
| 8 | 0.370450 (59) / **0.371421** (59) | 0.372919 (38) / **0.373256** (37) | 0.416808 (51) / **0.428111** (51) | 0.413868 (118) / **0.415954** (118) |
| 9 | 0.370842 (60) / **0.371939** (59) | 0.372331 (38) / **0.373457** (37) | 0.420259 (51) / **0.427565** (51) | 0.411634 (118) / **0.413966** (118) |
| 10 | 0.369079 (58) / **0.372406** (59) | 0.372384 (38) / **0.373734** (38) | 0.417151 (51) / **0.428272** (51) | 0.414629 (118) / **0.416432** (118) |
| 11 | 0.369816 (59) / **0.372466** (59) | 0.371568 (38) / **0.373679** (38) | 0.420104 (51) / **0.427959** (51) | 0.412501 (117) / **0.414438** (118) |
| 12 | 0.370015 (58) / **0.372956** (58) | 0.371234 (37) / **0.373773** (37) | 0.421160 (51) / **0.428170** (51) | 0.412610 (118) / **0.416433** (118) |
| 13 | 0.370817 (60) / **0.373302** (59) | 0.371584 (38) / **0.373800** (37) | 0.421992 (51) / **0.427877** (51) | 0.412246 (118) / **0.414826** (118) |
| 14 | 0.369719 (58) / **0.373455** (58) | 0.371812 (38) / **0.373914** (37) | 0.422460 (51) / **0.428357** (51) | 0.414267 (118) / **0.416383** (118) |
| 15 | 0.370163 (59) / **0.373606** (59) | 0.371217 (38) / **0.373941** (37) | 0.421814 (51) / **0.428235** (51) | 0.412640 (117) / **0.414779** (118) |
| 16 | 0.368841 (58) / **0.373590** (58) | 0.371222 (38) / **0.373878** (37) | 0.422267 (51) / **0.428117** (51) | 0.413513 (118) / **0.416292** (118) |

Category-specific Dice is higher in all four categories at K=16. Across earlier steps, overall greedy has higher 2a Dice at K=6 and K=7, and higher 2d Dice at K=4. Different fixed seeds and greedy paths mean category-specific search is not guaranteed to dominate at every K.
