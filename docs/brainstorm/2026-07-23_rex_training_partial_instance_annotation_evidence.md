---
doc_type: evidence_note
created: 2026-07-23
updated: 2026-07-23
status: active
topic: ReXGroundingCT training annotation completeness
scope: Evidence and implications for counterfactual prompt supervision
primary_sources:
  - literature/challenge_paper.pdf
  - https://arxiv.org/abs/2507.22030
  - https://rexrank.ai/ReXGroundingCT/challenge.html
related_notes:
  - 2026-07-23_voxtell_anatomy_language_capability_gap.md
  - 2026-07-23_counterfactual_prompt_pair_training.md
---

# ReX Training Partial-Instance Annotation Evidence

## Question

What evidence supports the statement that ReXGroundingCT training masks are not
exhaustive, and how broadly can that statement be applied?

This distinction matters for counterfactual training. If an edited prompt is
not listed among a case's findings, it is tempting to train it against a
globally empty mask. That target is only valid if the requested finding is
known to be absent.

## Bottom Line

The primary sources directly establish that ReXGroundingCT uses
**partial-instance annotation in the training split**:

- annotators segmented at most three representative instances for each
  annotated finding, even when more instances were visible;
- validation and test required all visible instances of each finding to be
  segmented.

This proves that a training mask can omit the fourth and later visible
instances of an annotated finding. It does not prove that every unlisted
pathology was present but omitted, nor that every training mask is incomplete.

The precise statement to use is:

> ReXGroundingCT training masks are not necessarily exhaustive at the instance
> level for an annotated finding because no more than three representative
> instances were segmented.

## Direct Paper Evidence

The local paper is:

[`literature/challenge_paper.pdf`](../../literature/challenge_paper.pdf)

The corresponding public paper is:

[ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from
Free-Text Reports](https://arxiv.org/abs/2507.22030)

### 1. Entity Protocol

Section 2.4, "Entity Protocol," on PDF page 4 states that:

- the training split followed a different annotation protocol from validation
  and test;
- training annotators were instructed to segment no more than three
  representative instances of a given entity;
- this limit applied even when additional instances were present;
- validation and test required exhaustive labeling of all visible instances
  belonging to each finding.

This is the strongest direct evidence of training mask incompleteness.

### 2. Dataset Construction Figure

The caption of Figure 3 on PDF page 5 repeats that:

- training was limited to a maximum of three representative instances per
  finding;
- validation and test segmented all visible instances to provide complete
  evaluation ground truth.

### 3. Entity Statistics

The Results section on PDF page 7 reports:

| Split group | Average entities per finding | Interpretation |
| --- | ---: | --- |
| Training | 1.95 | Influenced by the maximum-three protocol |
| Validation and test combined | 3.80 | Reflects more comprehensive annotation |

These averages are supporting evidence. They are not sufficient by themselves
to prove incompleteness, but they are consistent with the documented protocol.

### 4. Limitations Section

The Limitations section on PDF page 8 explicitly describes the training
protocol as a partial-labeling strategy and states that it limits the
completeness of spatial annotations.

This confirms that the three-instance rule is considered a dataset limitation,
not merely a formatting detail.

## Official Challenge Evidence

The official MICCAI challenge page describes the split protocols as:

| Split | Official description |
| --- | --- |
| Training | Partial-instance, up to three instances per finding |
| Validation | Exhaustive, all instances segmented by radiologists |
| Test | Exhaustive, all instances segmented by radiologists |

Source:
[ReXGroundingCT Challenge at MICCAI 2026](https://rexrank.ai/ReXGroundingCT/challenge.html)

The same information is preserved in the repository's source snapshot:

[`challenge_info/rexgroundingct_challenge.md`](../../challenge_info/rexgroundingct_challenge.md)

## Meaning of Finding and Entity

For this dataset:

- a **finding** is a free-text description associated with one mask channel;
- an **entity** is a distinct annotated instance within that finding;
- unique nonzero values in a finding mask distinguish its annotated entities.

For example, a finding such as:

> Multiple nodules in both lungs

can correspond to several separate nodule entities. In training, only up to
three representative nodules may be segmented even if more are visible. The
validation and test protocol instead asks annotators to segment all visible
nodules associated with that finding.

## What the Evidence Supports

The sources support these statements:

1. Some training findings can contain visible but unsegmented additional
   instances.
2. A training finding mask with three entities does not establish that only
   three entities are present.
3. The absence of a same-pathology instance at another location is uncertain
   when the finding may have exceeded the annotation cap.
4. Validation and test masks are more appropriate for exhaustive instance-level
   evaluation.
5. Training objectives should account for partial-instance labels when treating
   unlabeled voxels as background.

## What the Evidence Does Not Support

The sources do not directly establish that:

1. Every training finding mask is incomplete.
2. The selected representative entities have incomplete boundaries.
3. Every pathology visible on a CT is necessarily listed as a finding.
4. A different pathology absent from the finding list is actually present but
   unannotated.
5. An unlisted counterfactual prompt always has a positive target somewhere in
   the image.

The paper's three-instance limit concerns completeness within an annotated
finding. Extending it to entirely different pathology concepts would be an
inference, not a direct dataset fact.

## Counterfactual Examples

### Same pathology, different location

Factual prompt:

> Multiple nodules in the left upper lobe

Counterfactual prompt:

> Multiple nodules in the right lower lobe

If the counterfactual is not listed, a right-lower-lobe nodule could still be
an unsegmented fourth or later instance. The documented entity cap makes a
globally empty target unsafe.

A defensible target is:

- suppress the counterfactual prediction on the known left-upper-lobe GT;
- ignore uncertain voxels elsewhere;
- use a positive target only if a known right-lower-lobe mask exists.

### Different pathology, same location

Factual prompt:

> A nodule in the left upper lobe

Counterfactual prompt:

> Septal thickening in the left upper lobe

The three-instance rule does not demonstrate that unlisted septal thickening is
present. A globally empty target may still be risky because report-derived
findings are not formal proof of visual absence, but that concern is a
conservative modeling assumption rather than direct evidence from the
three-instance protocol.

A defensible target is:

- require the septal-thickening prompt not to select the known nodule mask;
- ignore the remainder of the image unless absence is independently
  established.

### Known positive edited prompt

If the same case contains a labeled finding and mask matching the edited
prompt, the edited prompt should be trained as another positive example. This
is the strongest counterfactual supervision because it requires no absence
assumption.

## Recommended Label Policy

Use three evidence levels:

| Label type | Evidence | Training target |
| --- | --- | --- |
| Known positive | Matching finding and mask exist | Segment the matching mask |
| Partial negative | Edited prompt must not describe the factual GT, but global absence is unknown | Anti-overlap and pairwise ranking on known GT; ignore elsewhere |
| Confirmed absent | Strong independent evidence of absence | Globally empty target with conservative weight |

The default for a generated counterfactual should be **partial negative**, not
confirmed absent.

## Implication for Existing Training

Randomly selecting a prompt that is absent from a case's annotation list and
assigning an all-zero mask can introduce noisy supervision. The risk is
clearest for:

- multiple nodules,
- micronodules,
- multifocal ground-glass opacities,
- scattered linear opacities,
- other findings capable of having more than three spatial instances.

The safer replacement is structured counterfactual pairing with losses
computed on known annotated regions. Confirmed-empty supervision should be a
smaller, separately audited subset.

## Documentation Rule

Future notes and reports should distinguish:

- **documented fact:** training is partial-instance, with at most three
  representative instances per finding;
- **reasonable concern:** an annotation list or report-derived finding list
  may not prove global pathology absence;
- **unsupported claim:** a specific unlisted pathology is present without
  image-level or report-level evidence.

This wording preserves the value of the counterfactual idea without overstating
the annotation evidence.

