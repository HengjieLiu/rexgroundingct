---
created: 2026-07-26
updated: 2026-07-26
status: active
experiment_id: 010_voxtell_public_anatomy_prior_fusion
---

# ReX Prompt Anatomy And TotalSegmentator ROI Audit

## Purpose

This audit identifies the anatomical information explicitly present in the
ReXGroundingCT challenge prompts and translates it into a justified set of
TotalSegmentator outputs and deterministic spatial priors for experiment 010.

The central design constraint is that anatomy must be soft guidance. A target
may be pulmonary, pleural, fissural, airway-centered, chest-wall-adjacent, or
occasionally outside the lung parenchyma. No anatomy mask may be used as an
irreversible crop or hard output restriction.

## Audited Source

- Source:
  `/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json`
- SHA256:
  `3a66087608d5d0177a30c0238845a7e53518fcd82f8f41283a7d14fc02b37e6a`
- Audited field: every value under each case's `findings` mapping.
- Excluded from the count: full reports, impressions, negative-report
  statements, and `anatomical_cot.json`.

The exclusions are deliberate. Full reports and anatomical chain-of-thought
records contain additional anatomical statements or inferred locations that
are not necessarily part of the challenge model input.

| Split | Cases | Prompts | Unique exact prompts | One-finding cases | Maximum findings/case |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 2,992 | 7,687 | 6,148 | 1,056 | 13 |
| Validation | 200 | 381 | 365 | 90 | 5 |
| Test | 300 | 582 | 542 | 136 | 8 |
| **Total** | **3,492** | **8,650** | **6,926** | - | - |

The `8,068` train plus validation prompts agree with the number of labeled
train/validation targets used by the standard VoxTell cache.

## Audit Method

The quantitative audit used case-insensitive prompt-level concept matching.
Each count is the number of prompts containing at least one lexical form for a
concept, not the number of word occurrences. Rows overlap: one prompt can
mention a right lower lobe, posterobasal segment, subpleural position, and
diaphragm simultaneously.

The concept patterns included common morphological and translation variants,
for example:

- pleural: `pleura`, `pleural`, `subpleural`, `juxtapleural`,
  `pleuroparenchymal`, and `hemithorax`;
- airway: `airway`, `bronchial`, `bronchus`, `bronchi`, `peribronchial`,
  `bronchovascular`, and `tree-in-bud`;
- segment direction: apical, superior, basal, inferior, anterior, posterior,
  medial, lateral, and their compound forms;
- central structures: hilar, perihilar, central, mediastinal, paramediastinal,
  paracardiac, pericardiac, and retrocardiac.

Rare structure matches and all prompts in categories `1f` and `2h` were
manually inspected. This remains a lexical audit rather than a clinical NLP
parser, so counts should be interpreted as strong prevalence indicators, not
perfect semantic labels.

## Prompt-Level Anatomy Counts

| Concept | Train | Validation | Test | Total | Total prompts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Laterality | 6,964 | 351 | 543 | 7,858 | 90.84% |
| Lobe or lingula | 4,284 | 243 | 335 | 4,862 | 56.21% |
| Named segment or segments | 2,491 | 150 | 204 | 2,845 | 32.89% |
| Pleural relationship | 1,054 | 68 | 131 | 1,253 | 14.49% |
| Peripheral or subpleural | 967 | 47 | 101 | 1,115 | 12.89% |
| Airway or bronchial relationship | 873 | 30 | 36 | 939 | 10.86% |
| Apical or superior direction | 906 | 53 | 74 | 1,033 | 11.94% |
| Basal or inferior direction | 1,228 | 95 | 93 | 1,416 | 16.37% |
| Anterior or posterior direction | 1,089 | 68 | 75 | 1,232 | 14.24% |
| Medial or lateral direction | 651 | 54 | 55 | 760 | 8.79% |
| Central or hilar relationship | 275 | 6 | 13 | 294 | 3.40% |
| Fissure relationship | 117 | 9 | 13 | 139 | 1.61% |
| Vascular relationship | 62 | 4 | 2 | 68 | 0.79% |
| Mediastinal or paracardiac relationship | 51 | 1 | 3 | 55 | 0.64% |
| Diaphragmatic relationship | 8 | 4 | 4 | 16 | 0.18% |
| Paravertebral, costal, or rib relationship | 7 | 1 | 3 | 11 | 0.13% |
| Chest-wall or subcutaneous relationship | 4 | 0 | 1 | 5 | 0.06% |

Laterality and lobe language dominate the prompt set, but the lower-frequency
relationships are not disposable. They include difficult, valid targets such
as chest-wall masses, subcutaneous emphysema, pleural-space air, diaphragmatic
pleural thickening, and costovertebral lesions.

## Category-Specific Evidence

- Category `1a`, bronchial wall thickening, contains airway language in
  approximately `99.2%` of prompts.
- Category `1b`, bronchiectasis, contains airway language in approximately
  `99.7%` of prompts.
- Category `1e`, micronodules including tree-in-bud, contains airway language
  in approximately `29.9%` and lobe language in approximately `51.0%`.
- Category `2e`, pleural effusion or thickening, contains pleural language in
  approximately `92.4%`.
- Lobe language occurs in approximately `73.0%` of category `2b`,
  `65.0%` of category `2c`, and `54.5%` of category `2d`.
- Peripheral or subpleural language occurs in approximately `32.5%` of
  category `2c` ground-glass prompts.
- Fissure language occurs in approximately `4.3%` of category `2d`
  nodule/mass prompts and also appears in loculated effusion prompts.
- Bilateral language occurs in approximately `84.9%` of category `1c`
  emphysema and `67.9%` of category `1d` septal-thickening prompts.

These distributions argue against selecting a single top-scoring anatomical
region. Bilateral and diffuse prompts require multiple simultaneous regions,
while pleural prompts require regions that extend beyond the lung mask.

## Direct TotalSegmentator Outputs

### Required: `total`

Use the standard `1.5 mm` task for the final scientific cache unless the
planned 20-case comparison shows that `3 mm --fast` is equivalent for the
selected guidance.

Required organ and central-structure labels:

- `lung_upper_lobe_left`
- `lung_lower_lobe_left`
- `lung_upper_lobe_right`
- `lung_middle_lobe_right`
- `lung_lower_lobe_right`
- `trachea`
- `esophagus`
- `heart`
- `aorta`
- `pulmonary_vein`
- `brachiocephalic_trunk`
- `superior_vena_cava`
- `inferior_vena_cava`

Recommended skeletal and chest-boundary labels:

- `rib_left_1` through `rib_left_12`;
- `rib_right_1` through `rib_right_12`;
- `sternum`;
- `costal_cartilages`;
- `vertebrae_T1` through `vertebrae_T12`;
- `vertebrae_C7` and upper lumbar vertebrae visible at the inferior field;
- `spinal_cord`;
- bilateral `clavicula` and `scapula`.

Recommended inferior anatomical anchors:

- `liver`
- `spleen`
- `stomach`
- bilateral adrenal glands

The inferior anchors do not predict a challenge lesion. They help identify the
basal lung and approximate the diaphragmatic interface, which
TotalSegmentator's `total` task does not segment directly.

### Required: `lung_vessels`

Retain all four specialized outputs:

- `lung_airways`
- `lung_airways_wall`
- `lung_arteries`
- `lung_veins`

This task is essential for bronchial wall thickening, bronchiectasis,
tree-in-bud, mucus/secretions, peribronchial findings, bronchovascular
relationships, and central/hilar localization. Its airway output is a combined
tree; it does not name individual lobar or segmental branches.

### Required: `trunk_cavities`

Retain:

- `thoracic_cavity`
- `mediastinum`
- `pericardium`
- `abdominal_cavity`

These compartments provide useful non-lung context for pleural,
paramediastinal, paracardiac, retrocardiac, and diaphragmatic prompts.

### Recommended: `body`

Retain:

- `body_trunc`
- `body_extremities`

The body mask supports a chest-wall and subcutaneous soft-tissue shell. A
simple CT-derived body mask may replace this task if it proves equally stable.

### Optional

The licensed `tissue_types` task could separate subcutaneous fat, torso fat,
and skeletal muscle. Only a few prompts directly need these compartments, so
it should be evaluated after the open anatomy stack rather than made a v1
dependency.

## Required Derived Regions

The following prompt-relevant regions are not direct TotalSegmentator labels:

- whole left lung, whole right lung, and bilateral whole lung;
- lobe-interface and approximate major/minor fissure neighborhoods;
- lobe-relative apical, middle, and basal zones;
- lobe-relative anterior/posterior and medial/lateral zones;
- geometric bronchopulmonary pseudo-segments;
- central and hilar neighborhoods around proximal airways and vessels;
- inner and outer lung-boundary shells;
- apical, costal, mediastinal, diaphragmatic, and fissural pleural regions;
- a diaphragm proxy from the inferior lung surface and superior
  abdominal-organ/cavity surfaces;
- a chest-wall/subcutaneous shell from body, thoracic cavity, ribs, sternum,
  and costal cartilage;
- distance-to-airway, distance-to-pleura, distance-to-fissure,
  distance-to-mediastinum, and distance-to-chest-wall maps.

All distances must be computed in physical millimeters using the image affine
and spacing. Fixed voxel radii are invalid across the heterogeneous source
spacings.

The pseudo-segment design is specified in
`docs/voxtell/bronchopulmonary_segments_and_pseudo_segment_priors.md`.

## Explicit Exclusions

Do not include these outputs in the first anatomy-prior experiment:

- `lung_nodules`
- `pleural_pericard_effusion`
- lesion, opacity, tumor, hemorrhage, or other pathology-predicting tasks

The excluded tasks overlap directly with ReX categories such as pulmonary
nodules/masses and pleural effusion. Using them could be a legitimate external
teacher or ensemble experiment, subject to challenge rules, but it would not
be a clean anatomy-prior experiment and must be reported separately.

## Cache And Fusion Recommendation

1. Preserve native-geometry packed multilabel output rather than independent
   float32 masks.
2. Preserve specialized overlapping masks in a separate packed bit field.
3. Record task version, model resolution, source CT hash, output hash, affine,
   label map, and missing-label flags.
4. Build compact model channels from the cache without regenerating anatomy.
5. Merge repetitive structures such as ribs, vertebrae, and central vessels
   into task-relevant groups for the first model.
6. Give the model named anatomy embeddings and presence flags rather than
   treating every absent mask as a confidently negative structure.
7. Fuse anatomy with zero-initialized residual paths and retain the original
   CT/text path as an unconstrained fallback.
8. Never crop, zero, or intersect lesion predictions using lung or anatomy
   masks.

## Follow-Up Validation

Before anatomy-fusion training:

- complete the val200 TotalSegmentator inventory;
- compare `1.5 mm` and `3 mm` outputs on the same fixed 20 cases;
- measure target containment in lobes, pleural shells, airway neighborhoods,
  fissure neighborhoods, mediastinum, and chest-wall regions;
- stratify containment by category and prompt anatomy term;
- inspect severe-disease cases where public anatomy masks may fail;
- require high-recall soft regions rather than narrow anatomically neat masks;
- confirm every derived region survives source-to-cache and cache-to-model
  orientation transformations.

The final ROI ontology should be selected from this text audit plus measured
target containment. Text frequency alone cannot establish whether a predicted
anatomy mask is geometrically reliable.

The prompt-level estimate and proposed validation policy for applying these
regions as spatial constraints are recorded in
`anatomy_constraint_applicability_val_test.md`.
