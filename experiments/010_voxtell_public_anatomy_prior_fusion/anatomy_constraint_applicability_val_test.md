---
created: 2026-07-26
updated: 2026-07-26
status: active
experiment_id: 010_voxtell_public_anatomy_prior_fusion
---

# Val/Test Anatomy-Constraint Applicability

## Purpose

This report estimates how many ReXGroundingCT validation and test prompts could
benefit from prompt-conditioned anatomical guidance or post-processing. It
separates:

- anatomy as an additional soft model input;
- an expanded prompt-selected spatial constraint;
- direct intersection with a raw TotalSegmentator lung or lobe mask.

These uses have different risk. A prompt can clearly name a lobe while its
target extends to the pleura, fissure, diaphragm, or chest wall. The broadest
applicability number must not be interpreted as the number of prompts that are
safe for hard intersection.

## Challenge-Rule Basis

The archived ReXGroundingCT rules state that both tracks permit:

- category information as an auxiliary signal while the exact free-text
  finding remains a direct model input;
- anatomical segmentations such as lobes, lung, and pleura as input channels or
  spatial constraints;
- post-processing that restricts a text-grounded prediction to the anatomical
  region named in the finding.

Project source:
`challenge_info/rexgroundingct_challenge.md`, lines 134-141.

Permission establishes challenge eligibility, not scientific benefit.
Anatomical restriction must still be validated for target recall, Dice, hit
rate, and failure modes.

## Audited Data

- Source:
  `/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json`
- SHA256:
  `3a66087608d5d0177a30c0238845a7e53518fcd82f8f41283a7d14fc02b37e6a`
- Validation: `200` cases and `381` prompts.
- Test: `300` cases and `582` prompts.
- Combined: `500` cases and `963` prompts.

Only the exact `findings` strings and their official category codes were used.
Test masks were not available or inspected, so every test conclusion is an
applicability estimate rather than measured performance.

## Classification Method

### Lexical concepts

Prompt-level matching recognized:

- lobe or lingula;
- lung, pulmonary, or parenchymal;
- right, left, bilateral, or both;
- pleural, subpleural, juxtapleural, or hemithorax;
- airway, bronchial, bronchus, bronchi, tree-in-bud, or tracheal;
- fissural;
- central, hilar, mediastinal, paracardiac, pericardiac, or retrocardiac;
- diaphragmatic, paravertebral, costovertebral, parasternal, chest-wall, and
  subcutaneous relationships.

Counts are prompt-level and concepts overlap.

### Mutually exclusive routing precedence

Each prompt was assigned one primary candidate constraint using this order:

1. Official category `2e` or `2g`: pleural/thoracic shell.
2. Official category `1a` or `1b`: airway neighborhood.
3. Explicit extra-pulmonary body/chest-wall language: body/chest-wall region.
4. Explicit lobe or lingula: expanded named lobe.
5. Explicit lung or laterality: side-specific or whole lung.
6. Otherwise: category-only or spatially ambiguous.

The precedence prevents a pleural prompt that happens to name a lower lobe from
being mislabeled as a raw lower-lobe intersection candidate. It similarly
keeps bronchial-wall targets out of an airway-lumen intersection.

### Boundary-risk rule

A direct lobe/lung candidate was marked as boundary-risk when its prompt
contained a pleural, subpleural, peripheral, fissural, diaphragmatic,
paravertebral, costovertebral, parasternal, or chest-wall relationship.

This is conservative but incomplete. A prompt can omit boundary language even
when its target reaches an anatomical boundary, and a TotalSegmentator mask can
be inaccurate in severe disease.

## Applicability Summary

| Constraint level | Validation | Test | Combined |
| --- | ---: | ---: | ---: |
| Explicit lobe/lung anatomy named | 349/381, 91.6% | 524/582, 90.0% | **873/963, 90.7%** |
| Prompt-selectable expanded anatomy | 374/381, 98.2% | 569/582, 97.8% | **943/963, 97.9%** |
| Conservative raw-mask candidates | 280/381, 73.5% | 386/582, 66.3% | **666/963, 69.2%** |
| No prompt-specific region | 7/381, 1.8% | 13/582, 2.2% | **20/963, 2.1%** |

Definitions:

- **Explicit lobe/lung anatomy named:** a non-pleural-category prompt directly
  contains lobe, lingula, lung, pulmonary, or parenchymal language.
- **Prompt-selectable expanded anatomy:** the prompt/category can choose a
  lobe, lung, airway, pleural, or body region, with spatial expansion where
  appropriate.
- **Conservative raw-mask candidate:** an explicit lobe/lung prompt does not
  contain a recognized boundary-risk term.
- **No prompt-specific region:** category can suggest broad anatomy, but the
  finding text does not select a side, lobe, airway, pleural compartment, or
  body region.

Within the `666` conservative raw-mask candidates, `412` are clean named-lobe
prompts without recognized boundary-risk language. These are the strongest
first cohort for a selective post-processing experiment.

## Primary Constraint Routing

| Primary candidate | Validation | Test | Combined | Recommended form |
| --- | ---: | ---: | ---: | --- |
| Expanded named lobe | 231 | 326 | **557** | Named lobe plus physical margin |
| Side-specific or whole lung | 117 | 197 | **314** | Union of applicable lobes plus margin |
| Airway neighborhood | 14 | 17 | **31** | Airways/wall plus distance neighborhood |
| Pleural/thoracic shell | 12 | 28 | **40** | Inner/outer lung shell within thoracic cavity |
| Extra-pulmonary body/chest wall | 0 | 1 | **1** | Body/chest-wall soft region |
| Category-only or ambiguous | 7 | 13 | **20** | Broad category prior; no prompt-specific hard mask |

The routing bins are mutually exclusive. They differ from raw lexical counts
because airway and pleural categories take precedence over lobe words.

## Boundary-Risk Findings

Of the `873` prompts explicitly naming a lung or lobe, `207` contain
boundary-risk language:

- Validation: `69`.
- Test: `138`.

This explains why the estimated conservative hard-mask cohort is smaller in
test than in validation. Test contains more pleural and boundary-related
language.

Important examples include:

- subpleural ground-glass opacity;
- perifissural or fissure-based nodules;
- pleural effusion extending into a fissure;
- diaphragmatic pleural thickening;
- paravertebral or costovertebral nodules;
- chest-wall or subcutaneous emphysema.

An unmodified lobe intersection can remove exactly the target voxels that make
these prompts anatomically distinctive.

## Proposed Constraint Policy

### Clean named-lobe parenchymal prompts

Candidate policy:

```text
constrained probability = prediction probability * expanded named-lobe mask
```

Evaluate physical lobe-mask margins of:

```text
0 mm, 5 mm, 10 mm, 20 mm
```

The selected margin must be based on validation target recall and prediction
metrics. A `0 mm` intersection is included as an ablation, not as the expected
default.

### Side-specific, bilateral, and diffuse prompts

Use the union of every applicable lobe. Do not select one top-scoring lobe for
prompts containing bilateral, both, multilobar, diffuse, widespread, or
multiple-region language.

For prompts that specify only right or left lung, use the full side-specific
lung. For prompts without laterality, category may activate a bilateral
whole-lung soft prior, but not a prompt-specific hard side constraint.

### Subpleural and peripheral prompts

Use:

```text
named lobe or lung
+ inner boundary shell
+ outer pleural shell
```

The outer shell is necessary because annotations and real disease can extend
beyond the predicted parenchymal boundary.

### Fissural prompts

Use a neighborhood around the interface between adjacent lobe masks. Include
both sides of the interface and a physical margin. Fissure-based nodules and
fissural effusions must not be assigned exclusively to one lobe interior.

### Airway prompts

Category `1a` and `1b` prompts should use:

```text
lung_airways
+ lung_airways_wall
+ distance-to-airway neighborhood
```

Do not intersect with the airway lumen. Bronchial-wall targets lie outside the
lumen, bronchiectatic airways are morphologically abnormal, and peripheral
branches may be missing from the public segmentation.

### Pleural prompts

Categories `2e` and `2g` should use:

```text
thoracic cavity
+ inner and outer lung-boundary shells
+ fissural region
+ side information when available
```

Pleural effusion and pneumothorax primarily occupy pleural space and must never
be restricted to a lung-lobe interior.

### Extra-pulmonary prompts

The test prompt set includes at least one explicit chest-wall/subcutaneous
target. Use body and chest-wall guidance for such prompts. Its existence is
sufficient to reject unconditional whole-lung clipping across the challenge.

### Category-only prompts

The `20` spatially ambiguous prompts can still use a broad category-dependent
soft prior. Examples include unspecified nodules, ground-glass opacities,
tree-in-bud, or atelectatic changes. Category must not invent a side or lobe
that the prompt does not supply.

## Relationship To Pseudo-Segments

Prompts that name posterobasal, anterobasal, laterobasal, mediobasal,
apicoposterior, or lingular regions can add the soft pseudo-segment priors
defined in:

`docs/voxtell/bronchopulmonary_segments_and_pseudo_segment_priors.md`.

Pseudo-segments refine an expanded lobe prior. They are not sufficiently
reliable to replace it or to support a strict pseudo-segment intersection.

## Required Validation Experiment

### Ground-truth containment

For every validation finding and every candidate region/margin, measure:

- fraction of target voxels retained;
- number and fraction of targets retained completely;
- number and fraction retaining at least `95%`, `90%`, and `50%`;
- number of targets removed completely;
- retained background volume;
- results by category, lesion size, and anatomical prompt type.

The containment audit estimates the maximum recall a constraint permits. A
constraint that removes ground truth cannot recover that target regardless of
model quality.

### Prediction-level post-processing

Apply the same policies to fixed pretrained and fine-tuned probability maps,
without rerunning inference. Report:

- global Dice per finding;
- hit rate;
- number of improved, unchanged, and worsened findings;
- false-positive volume removed;
- true-positive volume removed;
- category and prompt-type stratification.

Use the unchanged prediction as the primary control. Select no policy using
test outcomes.

### Suggested acceptance gate

A policy should not become a test default unless it:

- improves validation Dice or hit rate on a predeclared evaluation set;
- retains at least `99%` of aggregate target voxels in its eligible cohort;
- removes no target completely;
- shows no material collapse in pleural, airway, subpleural, small-lesion, or
  diffuse strata;
- remains deterministic from prompt text/category and recorded anatomy masks.

The exact containment threshold may be revised before execution, but it must be
fixed before comparing post-processing variants.

## Recommendation

- Anatomy as a soft input can plausibly help essentially all `963` val/test
  prompts through category- or prompt-conditioned context.
- Prompt-selected expanded constraints are applicable to approximately
  `943/963`, or `97.9%`.
- Direct raw lung/lobe restriction should be treated as a conservative
  candidate for at most `666/963`, or `69.2%`.
- The strongest first hard-restriction cohort is the `412` clean named-lobe
  prompts.
- Do not deploy a global hard constraint.
- Run containment and prediction-level validation before defining the test
  policy.

These are applicability estimates from text and category metadata. They are
not evidence that anatomical restriction has already improved segmentation.
