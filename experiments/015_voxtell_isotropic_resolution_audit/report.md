# Native Resolution Audit For Isotropic VoxTell Planning

Created: `2026-08-14T06:46:05.618126+00:00`

This audit reads NIfTI headers only. It does not resample CTs or labels, build caches, write predictions, or launch finetuning.

## Scope

| Field | Value |
| --- | --- |
| Splits requested | train, val, test |
| Max cases per split | none |
| Cases audited | 3492 |
| CT missing | 0 |
| CT HU header failures | 0 |
| Metadata shape mismatches | 0 |

## CT Geometry

| Split/group | N | Spacing X min/median/max | Spacing Y min/median/max | Spacing Z min/median/max | Shape Z min/median/max | In-plane equal | Isotropic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2992 | 0.303711 / 0.696596 / 0.976562 | 0.303711 / 0.696596 / 0.976562 | 0.5 / 1.0 / 3.0 | 104.0 / 331.5 / 1005.0 | 2992 | 7 |
| val | 200 | 0.417969 / 0.696158 / 0.923828 | 0.417969 / 0.696158 / 0.923828 | 0.75 / 1.0 / 2.0 | 172.0 / 346.0 / 568.0 | 200 | 0 |
| test | 300 | 0.390625 / 0.710938 / 0.976562 | 0.390625 / 0.710938 / 0.976562 | 0.75 / 1.0 / 2.0 | 167.0 / 363.0 / 617.0 | 300 | 1 |
| train+val | 3192 | 0.303711 / 0.696596 / 0.976562 | 0.303711 / 0.696596 / 0.976562 | 0.5 / 1.0 / 3.0 | 104.0 / 333.0 / 1005.0 | 3192 | 7 |
| all | 3492 | 0.303711 / 0.697266 / 0.976562 | 0.303711 / 0.697266 / 0.976562 | 0.5 / 1.0 / 3.0 | 104.0 / 336.0 / 1005.0 | 3492 | 8 |

## Common CT Spacings

- `train` top `(X, Y, Z)` spacing tuples: `(0.683594, 0.683594, 0.75)`: 312, `(0.683594, 0.683594, 1.5)`: 115, `(0.755859, 0.755859, 0.75)`: 23, `(0.732422, 0.732422, 0.75)`: 22, `(0.740234, 0.740234, 0.75)`: 22, `(0.710938, 0.710938, 0.75)`: 21, `(0.722656, 0.722656, 0.75)`: 21, `(0.800781, 0.800781, 0.75)`: 19
- `train` Z-spacing bins rounded to 0.1 mm: `0.5`: 1, `0.7`: 4, `0.8`: 1284, `1.0`: 624, `1.2`: 332, `1.3`: 9, `1.5`: 685, `2.0`: 48, `3.0`: 5
- `val` top `(X, Y, Z)` spacing tuples: `(0.683594, 0.683594, 0.75)`: 22, `(0.683594, 0.683594, 1.5)`: 7, `(0.763672, 0.763672, 0.75)`: 3, `(0.705078, 0.705078, 0.75)`: 3, `(0.707031, 0.707031, 0.75)`: 3, `(0.732422, 0.732422, 1.5)`: 3, `(0.791016, 0.791016, 0.75)`: 3, `(0.732422, 0.732422, 0.75)`: 2
- `val` Z-spacing bins rounded to 0.1 mm: `0.8`: 93, `1.0`: 38, `1.2`: 22, `1.5`: 46, `2.0`: 1
- `test` top `(X, Y, Z)` spacing tuples: `(0.683594, 0.683594, 0.75)`: 24, `(0.683594, 0.683594, 1.5)`: 9, `(0.785156, 0.785156, 0.75)`: 4, `(0.722656, 0.722656, 0.75)`: 4, `(0.736328, 0.736328, 0.75)`: 4, `(0.769531, 0.769531, 0.75)`: 4, `(0.755859, 0.755859, 0.75)`: 4, `(0.714844, 0.714844, 0.75)`: 3
- `test` Z-spacing bins rounded to 0.1 mm: `0.8`: 142, `1.0`: 69, `1.2`: 30, `1.3`: 3, `1.5`: 55, `2.0`: 1

## Label Audit

| Field | Value |
| --- | ---: |
| Train/val cases | 3192 |
| Train/val label files present | 3192 |
| Train/val label files missing | 0 |
| Test cases | 300 |
| Test labels missing as expected | 300 |
| Label shape mismatches | 0 |
| Label affine mismatches | 3192 |

Released labels are grid-aligned to CT by shape and finding count. Use matched CT headers as the physical spacing source; label NIfTI affine and zoom metadata are not used as physical-resolution truth.

## Isotropic Candidate Impact

| Target | Group | N | Shape Z min/median/max | Voxels/case min/median/max | Image GiB | Dense target GiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 mm | train | 2992 | 179.0 / 337.0 / 754.0 | 14679233.0 / 44032296.0 / 161634226.0 | 507.857 | 328.457 |
| 1 mm | val | 200 | 253.0 / 339.0 / 516.0 | 18171472.0 / 44832158.0 / 79208331.0 | 33.731 | 16.009 |
| 1 mm | test | 300 | 233.0 / 339.0 / 640.0 | 17491725.0 / 45447500.0 / 90250000.0 | 52.468 | 0.0 |
| 1 mm | train+val | 3192 | 179.0 / 337.0 / 754.0 | 14679233.0 / 44048498.0 / 161634226.0 | 541.588 | 344.466 |
| 1 mm | all | 3492 | 179.0 / 338.0 / 754.0 | 14679233.0 / 44226600.0 / 161634226.0 | 594.055 | 344.466 |
| 0.7 mm | train | 2992 | 255.0 / 481.0 / 1077.0 | 42678612.0 / 128310246.5 / 470563917.0 | 1480.583 | 957.575 |
| 0.7 mm | val | 200 | 361.0 / 484.0 / 737.0 | 52954729.0 / 130714605.0 / 230766723.0 | 98.331 | 46.665 |
| 0.7 mm | test | 300 | 333.0 / 484.0 / 914.0 | 51158400.0 / 132375000.0 / 262544940.0 | 152.952 | 0.0 |
| 0.7 mm | train+val | 3192 | 255.0 / 481.0 / 1077.0 | 42678612.0 / 128487910.5 / 470563917.0 | 1578.915 | 1004.239 |
| 0.7 mm | all | 3492 | 255.0 / 482.0 / 1077.0 | 42678612.0 / 128750000.0 / 470563917.0 | 1731.866 | 1004.239 |

## Storage Caveat

Candidate storage estimates are full-FOV, uncompressed tensor estimates. A later cache may differ because crop-to-nonzero, compression, sparse label storage, or changed dtype can reduce or change actual disk use.

## Artifacts

- `native_resolution_summary.json`: machine-readable aggregate summary.
- `native_resolution_cases.csv`: one row per audited CT case.
- `report.md`: this human-readable report.
