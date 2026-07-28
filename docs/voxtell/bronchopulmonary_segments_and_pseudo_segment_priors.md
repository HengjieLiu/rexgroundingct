---
created: 2026-07-26
updated: 2026-07-26
status: active
---

# Bronchopulmonary Segments And Pseudo-Segment Priors

## Purpose

ReXGroundingCT prompts frequently localize findings below the lobe level. This
document defines what bronchopulmonary segments are, what the project can
obtain from TotalSegmentator, how geometric pseudo-segments may be built, and
how those approximations may safely guide VoxTell.

Pseudo-segments are model inputs, not ground-truth anatomical annotations.
They must never become hard crops or output constraints.

## Anatomical Hierarchy

The relevant hierarchy is:

```text
lung -> lobe -> bronchopulmonary segment -> finding
```

A bronchopulmonary segment is a region of a lobe organized around a segmental
bronchus and accompanying pulmonary arterial branch. Segment boundaries are
not generally visible as strong CT surfaces. Fissures separate lobes, not most
segments.

### Common Right-Lung Segments

| Lobe | Segment | Common name |
| --- | --- | --- |
| Right upper | S1 | Apical |
| Right upper | S2 | Posterior |
| Right upper | S3 | Anterior |
| Right middle | S4 | Lateral |
| Right middle | S5 | Medial |
| Right lower | S6 | Superior |
| Right lower | S7 | Medial basal |
| Right lower | S8 | Anterior basal |
| Right lower | S9 | Lateral basal |
| Right lower | S10 | Posterior basal |

### Common Left-Lung Segments

| Lobe | Segment | Common name |
| --- | --- | --- |
| Left upper | S1+S2 | Apicoposterior |
| Left upper | S3 | Anterior |
| Left upper, lingula | S4 | Superior lingular |
| Left upper, lingula | S5 | Inferior lingular |
| Left lower | S6 | Superior |
| Left lower | S7+S8 | Anteromedial basal |
| Left lower | S9 | Lateral basal |
| Left lower | S10 | Posterior basal |

Anatomical variants are common. S1 and S2 or S7 and S8 may be separate or
combined, and translated reports may use informal terms such as `mediobasal`,
`basal segment`, or `superior lower-lobe segment`. Prompt parsing must support
the language that appears in this dataset rather than enforce one perfect
textbook naming scheme.

## ReX Prompt Evidence

In the challenge file audited for experiment 010, prompts containing
`segment` or `segments` occur as follows:

| Split | Segment-bearing prompts |
| --- | ---: |
| Train | 2,491 |
| Validation | 150 |
| Test | 204 |
| **Total** | **2,845 of 8,650, 32.89%** |

Frequently observed forms include:

| Prompt form | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| `posterobasal` | 409 | 31 | 16 | 456 |
| `laterobasal` | 170 | 10 | 5 | 185 |
| `apicoposterior` | 123 | 5 | 5 | 133 |
| `mediobasal` or `anteromediobasal` | 110 | 9 | 4 | 123 |
| `anterobasal` | 66 | 6 | 7 | 79 |
| `lingula` or `lingular` | 508 | 27 | 41 | 576 |

Other common forms include `superior segment`, `anterior segment`, `posterior
segment`, `medial segment`, `lateral segment`, and plural basal segments.

These counts justify a sub-lobar spatial representation. They do not imply
that every occurrence maps cleanly to a canonical segment, and they do not
measure whether a lesion fills the named segment.

## TotalSegmentator Capability Gap

The `total` task directly predicts:

- left upper and lower lobes;
- right upper, middle, and lower lobes;
- trachea and broad surrounding anatomy.

The separate `lung_vessels` task predicts:

- a combined `lung_airways` tree;
- `lung_airways_wall`;
- `lung_arteries`;
- `lung_veins`.

Neither task assigns S1-S10 identities or returns a mask for each named
bronchopulmonary segment. The airway tree is not labeled by main, lobar, and
segmental branch name. TotalSegmentator also does not directly return fissure,
hilum, pleural membrane, or diaphragm masks.

Therefore a five-lobe mask can identify the right lower lobe, but cannot
directly distinguish its posterior basal and lateral basal regions.

## Coordinate Contract

Pseudo-segments must be derived in a known physical coordinate system:

1. Apply the project's validated source-orientation handling.
2. Convert voxel locations through the NIfTI affine into physical coordinates.
3. Define superior/inferior, anterior/posterior, and left/right using physical
   orientation, never raw array dimension numbers.
4. Derive side-specific medial/lateral direction relative to the patient's
   midline.
5. Record the orientation and transform metadata needed to restore every prior
   to model and evaluator geometry.

This is particularly important because the source CTs are LPS-oriented while
the validated VoxTell path performs an explicit LPS-to-RAS-compatible
transformation. Reusing array indices without that contract would exchange
left/right or anterior/posterior regions.

## Geometric Pseudo-Segment Construction

### 1. Start From Validated Lobe Masks

For each lobe:

- remove only clearly spurious disconnected components;
- preserve disease-related holes and deformations;
- compute physical-coordinate bounds and robust coordinate quantiles;
- retain a missing or low-confidence flag when the lobe prediction fails.

The method should not repair a lobe by copying an atlas mask. Such a repair
could falsely suppress real abnormal anatomy.

### 2. Build Continuous Lobe-Relative Coordinates

For every voxel inside a lobe, estimate normalized coordinates:

```text
u_si: superior (+1) to inferior (-1)
u_ap: anterior (+1) to posterior (-1)
u_ml: lateral (+1) to medial (-1)
```

Use robust physical-coordinate quantiles or a lobe-aligned principal frame so
small outliers do not set the full range. The left/right definition of
`u_ml` must be side-aware.

Prefer continuous coordinate fields and soft memberships over hard equal-size
boxes. Smooth transitions let the lesion model override imperfect anatomical
boundaries.

### 3. Form Prompt-Relevant Soft Regions

Examples:

```text
posterobasal lower-lobe prior
  = lower-lobe membership
  * posterior membership
  * inferior membership

anterobasal lower-lobe prior
  = lower-lobe membership
  * anterior membership
  * inferior membership

laterobasal lower-lobe prior
  = lower-lobe membership
  * lateral membership
  * inferior membership

mediobasal lower-lobe prior
  = lower-lobe membership
  * medial membership
  * inferior membership

apicoposterior upper-lobe prior
  = upper-lobe membership
  * superior membership
  * posterior membership
```

The right-middle-lobe medial and lateral priors can use `u_ml`. The lower-lobe
superior-segment prior should emphasize the superior and relatively central
part of that lower lobe rather than the basal surface.

The lingula requires special care. It is part of the left upper lobe, but
TotalSegmentator does not return a separate lingular mask. A geometric lingula
proxy should use the inferior-anterior portion of the left upper lobe and its
relationship to the left heart border. Superior and inferior lingular priors
can then divide that proxy along the superior/inferior axis.

### 4. Add Anatomical Distance Features

Segment words are often accompanied by relative terms such as subpleural,
paramediastinal, paravertebral, or fissural. Combine pseudo-segments with:

- distance to the outer lung boundary;
- distance to lobe interfaces/fissures;
- distance to proximal airways;
- distance to pulmonary vessels;
- distance to mediastinum and heart;
- distance to chest wall and spine;
- distance to the inferior lung/diaphragm proxy.

Distances must be calculated and clipped in millimeters. They should remain
separate features rather than being folded into one irreversible region.

## Airway-Guided Alternative

A more anatomical approximation can use the `lung_airways` tree:

1. skeletonize the airway lumen;
2. identify the tracheal root and left/right main bronchi;
3. identify lobar bronchial branches;
4. detect candidate segmental branches from topology and physical direction;
5. assign branch names with anatomical constraints;
6. partition each lobe by distance to its named segmental branches.

This approach can follow patient-specific anatomy better than geometric bins.
It is also substantially more fragile:

- peripheral branches may be missing at CT resolution;
- mucus, stenosis, collapse, bronchiectasis, and severe disease alter topology;
- adjacent branches can merge in a binary segmentation;
- anatomical variants make fixed branch-order rules unreliable;
- TotalSegmentator provides no branch-name supervision.

An airway-guided method should therefore produce confidence-weighted soft
regions and fall back to geometric priors when topology is incomplete.

## Dedicated Segment Model Alternative

A dedicated bronchopulmonary-segment model could directly predict named
segments. Before adoption it must be checked for:

- public availability and challenge eligibility;
- compatibility with non-contrast CT;
- robustness to severe pulmonary disease;
- explicit left/right segment ontology;
- native geometry and orientation behavior;
- performance on this cohort rather than only healthy anatomy;
- inference cost across all 3,492 challenge CTs.

This is the preferred route if reliable public segment labels are available,
but it is not required for the first anatomy-fusion experiment.

## Prompt-To-Prior Use

Prompt parsing may activate multiple compatible priors:

- `right lower lobe posterobasal` activates right-lower-lobe, posterior, and
  basal memberships;
- `bilateral lower lobes` activates both lower lobes;
- `subpleural ground glass in both lungs` activates bilateral lung and outer
  pleural-shell features without selecting one lobe;
- a prompt without anatomy leaves all spatial guidance optional.

Do not reduce the prompt to one selected region. Many ReX prompts are
bilateral, diffuse, multilobar, or describe multiple foci.

The image/text model must retain:

- the original CT feature path;
- the original text-conditioning path;
- zero-initialized or gated anatomy residuals;
- missing-mask and confidence indicators;
- the ability to predict outside every proposed anatomy region.

## Validation Requirements

Before using pseudo-segments in full training:

1. Visually inspect all common pseudo-segment types on a fixed representative
   set.
2. Confirm orientation with unmistakable left/right and anterior/posterior
   cases.
3. For segment-bearing validation prompts, measure target recall inside a
   deliberately expanded soft pseudo-segment.
4. Compare target recall against the whole-lobe prior; a pseudo-segment must
   add localization without materially losing inclusion.
5. Stratify by segment term, lobe, lesion size, and disease category.
6. Test bilateral and multilobar prompts explicitly.
7. Test cases with missing or distorted airway/lobe predictions.
8. Record the pseudo-segment definition and ontology hash in every experiment
   that consumes it.

A pseudo-segment that looks anatomically precise but excludes true lesions is
worse than a broad lobe prior. The desired first version is a high-recall,
physically meaningful spatial suggestion.

## Project Decision

For the first exp010 anatomy-fusion implementation:

- use TotalSegmentator lobes as direct masks;
- add geometric soft pseudo-segments and continuous lobe-relative coordinate
  maps;
- use the airway tree as a separate feature, not as a required branch parser;
- keep a branch-based or dedicated segment model as a later ablation;
- prohibit hard pseudo-segment cropping or mask intersection.

This provides meaningful sub-lobar localization for the large segment-bearing
prompt subset while preserving the model's ability to correct imperfect
anatomical guidance.
