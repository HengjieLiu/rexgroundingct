---
created: 2026-07-23
updated: 2026-07-23
status: active
decision_status: "confusion not solved but moving forward"
---

# Sampler Fallback Decision

This note records the unresolved prompt-sampling ambiguity for experiment 002 and
the working decision used for implementation.

## Confusion Not Solved But Moving Forward

The VoxTell paper describes training with `2` positive prompts and `1` negative
prompt per image. That leaves an ambiguity for ReXGroundingCT cases with only
one released positive finding. The public paper/code inspection did not make it
clear whether VoxTell's original pretraining corpus guaranteed at least two
positive targets per sampled image, or whether it had a fallback rule for
one-target images.

ReXGroundingCT does contain many one-finding cases:

- Train: `1056 / 2992` cases have one finding.
- Validation: `90 / 200` cases have one finding.
- Test: `136 / 300` cases have one finding.

This means the fallback is not a rare edge case and must be explicit.

## Patch Sampling Clarification

There are two different meanings of "positive":

- A positive prompt means the finding exists in the case and has a corresponding
  full-volume target channel.
- A positive patch channel means the sampled `192^3` patch actually contains
  nonzero voxels for that finding.

Because training is patch-based, a case-level positive finding can become an
empty target inside a particular patch. Foreground-oversampled patch sampling
reduces this problem, but only for the selected anchor finding. It does not
guarantee that every positive prompt in the same sample is non-empty in the
patch.

For experiment 002, foreground oversampling means:

- With probability `0.85`, choose one selected positive finding as the anchor.
- Sample a foreground voxel from that anchor target.
- Choose patch bounds so that voxel is inside the patch.
- Therefore the anchor positive target should be non-empty in that patch, unless
  the full-volume mask is unexpectedly empty.

For the remaining `0.15` random patches, even positive prompt channels may be
empty in the patch.

## Considered Fallbacks

The first fallback considered was duplicating the single positive prompt so the
sample would still contain `2` positive prompt slots plus `1` negative prompt.
To avoid an exact duplicate target in the same forward pass, the duplicated
positive prompt would have been trained from two different sampled patches.

That option was rejected for the first baseline because it changes the gradient
structure: the same text embedding receives repeated positive supervision from
one case, and the implementation becomes less comparable to a normal
case-patch batch.

The implemented fallback is:

- Multi-finding case: `2` positive prompts + `1` negative prompt.
- One-finding case: `1` positive prompt + `2` negative prompts.
- No duplicate positive prompt is used.
- Negative targets are always empty masks.
- The sample still always has `3` prompt channels for training collation.
- One-finding fallback samples force foreground anchoring on the sole positive
  target so the positive target is non-empty in the patch whenever the
  full-volume mask is non-empty.

## Expected Training-Dynamics Difference

This fallback is not identical to the paper statement. For one-finding cases,
the sample has less positive prompt diversity and more negative supervision than
`2` positive + `1` negative.

If train cases are sampled uniformly, the approximate prompt-slot mix becomes:

- Positive slots: `1056 * 1 + 1936 * 2 = 4928`.
- Negative slots: `1056 * 2 + 1936 * 1 = 4048`.
- Overall slot ratio: about `54.9%` positive and `45.1%` negative.

This differs from the paper's nominal `66.7%` positive and `33.3%` negative mix.
The patch-level non-empty positive ratio will be lower than the case-level ratio
because non-anchor positive prompts can be empty in sampled patches.

For one-finding cases, the forced foreground anchor increases the fraction of
fallback samples with non-empty positive supervision compared with applying the
generic `0.85` foreground-oversampling probability. This is a deliberate
training-dynamics change to avoid many one-finding cases becoming effectively
pure-negative patches.

## Inference Rule

Inference does not use the three-channel training fallback. It remains
challenge-valid:

- Use exactly the actual findings for the case.
- Produce one output channel per actual finding.
- Do not add negative prompts.
- Do not duplicate one-finding prompts.

## Working Decision

Proceed with `1` positive + `2` negatives for one-finding training cases and
label this as an unresolved paper-alignment difference. Revisit only after
baseline results exist or if the paper/authors clarify the original one-target
sampling behavior.
