---
created: 2026-07-27
updated: 2026-07-27
status: active
experiment_id: "009_voxtell_s3_attention_coupling_ablation"
---

# S3 Attention Design

This note records the attention design rationale for experiment 009. It is the
human-readable companion to `codex_execution_spec.md`: the execution spec says
what is being run, while this file explains why the S3 coupling variants exist
and how they are implemented.

This file intentionally does not contain running metric tables. Val20 and
val200 results should stay in the runtime comparison report until final
closeout.

## Current Exp009 Scope

Experiment 009 is a four-arm continuation from experiment 006
`v123_cached_e5_d4` epoch 100:

| Arm | Variant | S3 scales | Purpose |
| --- | --- | --- | --- |
| control | `baseline_cont100` | none | same-source continuation baseline |
| s3v1 | `s3v1_fixedrho_suppress_half_quarter` | `1/2`, `1/4` | mimic original S3-style suppressive gating |
| s3v2 | `s3v2_balanced_feature_half_quarter` | `1/2`, `1/4` | balanced feature boost/suppress coupling |
| s3v3 | `s3v3_logit_residual_half_quarter` | `1/2`, `1/4` | direct logit residual coupling |

The active experiment isolates coupling style, not scale selection. All S3 arms
therefore use the same `1/2 + 1/4` scale set.

The earlier all-scale plan included `1/1 + 1/2 + 1/4`, but the `1/1` S3 arms
OOMed during one-update smoke on 48 GiB GPUs at `192^3`. The likely driver was
the full-resolution 128-channel hidden attention path, not the final single
attention map alone. The current run drops `1/1` rather than silently changing
batch size, patch size, or model width.

Prompt-specific adaptive S3 strength is also deferred. The current run uses
fixed global S3 strength per variant so the ablation remains readable.

## VoxTell Baseline Fusion

VoxTell already has text-image fusion before S3 is added.

The image encoder produces a list of multiscale skip features. A selected
bottleneck image feature map is projected into a sequence and used as the
memory for a transformer decoder. The text embeddings are the transformer
queries. In simplified form:

```text
image_tokens = project_bottleneck(encoder_feature)
text_queries = project_text(text_embedding)
mask_embedding = transformer_decoder(query=text_queries, memory=image_tokens)
```

That is the main cross-attention step: text queries attend to image tokens at
the selected bottleneck resolution.

After this, the per-prompt mask embeddings are projected to decoder channel
spaces and fused with decoder features through dot products. This makes the
decoder text-conditioned at multiple output stages, but the transformer
cross-attention itself is not a full multiscale cross-attention module.

S3 is added on top of this baseline. It does not replace VoxTell's transformer
fusion. It adds explicit voxel-text attention maps at selected decoder skip
scales and tests whether those maps can guide the segmentation path.

## S3 Attention Map

For each enabled scale and each prompt, S3 computes a voxel-text compatibility
map:

```text
A_p,s(x) = sigmoid(exp(logit_scale_s) * cos(V_s(x), T_p) + bias_s)
```

where:

- `p` is the prompt slot.
- `s` is the enabled scale, currently `1/2` or `1/4`.
- `x` is a voxel location at that scale.
- `V_s(x)` is a learned visual embedding from the decoder skip and coarse
  decoder context.
- `T_p` is a learned projection of the prompt-conditioned mask embedding.

In code, the visual embedding is built from the skip tensor and the current
decoder upsampled context:

```text
visual = relu(conv_skip(skip) + conv_context(coarse_context))
visual = normalize(conv_visual(visual))
text = normalize(linear_text(query))
A = sigmoid(scale * einsum(visual, text) + bias)
```

The attention map has one channel per prompt at that scale. During decoding the
implementation processes each prompt separately, stores the per-prompt maps,
and later concatenates the prompt outputs back into the standard VoxTell
prediction shape.

## Why Suppress-Only S3 May Not Improve Dice

The original S3-style gate is suppressive:

```text
skip' = skip * ((1 - rho_eff) + rho_eff * A)
```

Since `A` is in `[0, 1]`, the multiplier is in `[1 - rho_eff, 1]`. With
`rho_fixed = 0.25`, a region can be reduced as low as `0.75x`, but it can never
be boosted above the original skip feature.

This matters for ReXGroundingCT. If the base VoxTell continuation already has
weak logits on a true lesion, a suppress-only gate can mostly help by removing
competing false-positive regions. It cannot directly say "increase evidence
here." That can improve localization behavior or hit count in some cases while
failing to improve Dice, especially for small or low-contrast findings.

The exp009 S3 variants therefore compare three ways to couple attention back
into segmentation.

## S3v1: Fixed-Rho Suppressive Gate

S3v1 is the closest mimic of the original S3 idea:

```text
skip' = skip * ((1 - rho_eff) + rho_eff * A)
rho_eff = rho_fixed * ramp(update)
rho_fixed = 0.25
```

Behavior:

- `A = 1.0` gives multiplier `1.0`.
- `A = 0.5` gives multiplier `1 - 0.5 * rho_eff`.
- `A = 0.0` gives multiplier `1 - rho_eff`.

Strength:

- Suppresses low-attention skip regions.
- Keeps high-attention skip regions unchanged.
- Cannot directly boost true-positive regions.

Why keep it:

- It is the cleanest S3 mimic.
- It is the safest feature modulation arm.
- It tests whether pruning irrelevant skip evidence is enough when added to
  the current VoxTell continuation.

Main risk:

- If Dice is limited by weak true-positive logits rather than false-positive
  clutter, this arm may not help much.

## S3v2: Balanced Feature Modulation

S3v2 changes the feature gate so high attention can boost and low attention can
suppress:

```text
skip' = skip * (1 + alpha_eff * (2A - 1))
alpha_eff = alpha_target * ramp(update)
alpha_target = 0.25
```

Behavior:

- `A = 1.0` gives multiplier `1 + alpha_eff`.
- `A = 0.5` gives multiplier `1.0`.
- `A = 0.0` gives multiplier `1 - alpha_eff`.

At full strength, this gives approximately `0.75x` to `1.25x` modulation.

Why it exists:

- It directly fixes the suppress-only limitation.
- It stays close to the original skip-gating design.
- It tests whether balanced feature modulation is enough to convert useful
  attention into better masks.

Main risk:

- The decoder can still partly ignore multiplicative skip scaling.
- Multiplicative feature changes can be blunt: boosting wrong attention may
  enlarge false positives, while suppressing shifted attention may erase small
  targets.

## S3v3: Direct Zero-Init Logit Residual

S3v3 disables skip gating and uses S3 maps as an explicit output-side residual
signal:

```text
logit' = logit + beta_eff * R([logit, up(A_1/2), up(A_1/4)])
beta_eff = beta_target * ramp(update)
beta_target = 1.0
```

`R` is a tiny residual head:

```text
1x1 Conv3d -> GELU -> 1x1 Conv3d
```

The final convolution in `R` is zero-initialized, so the residual output is
exactly zero at materialization.

Why it exists:

- It gives attention a direct route to raise or lower final mask logits.
- It tests the hypothesis that attention can localize a finding but the normal
  decoder may fail to convert that localization into enough logit mass.
- It separates "learn an attention map" from "force attention through skip
  feature scaling."

Main risk:

- It is the most powerful coupling and therefore the easiest to misuse if the
  attention map is wrong.
- It should eventually be paired with prompt-specific strength or a smaller
  residual policy if it proves unstable.

## No-Impact Initialization And Ramp

All S3 arms are initialized to have no segmentation effect at update 0:

```text
ramp(update) = min(1, update / 2000)
effective_strength = target_strength * ramp(update)
```

At materialization:

- `s3_coupling_scale = 0`.
- S3v1 has `rho_eff = 0`.
- S3v2 has `alpha_eff = 0`.
- S3v3 has `beta_eff = 0`.
- S3v3 also has a zero-initialized final residual convolution.

Therefore, epoch-0 predictions should match the exp006 source model within
numerical tolerance. This is important because it makes later changes
attributable to training with S3 rather than to an accidental initialization
shift.

S3 auxiliary supervision starts from update 1. That allows the attention
projection parameters to learn while the segmentation coupling ramps in slowly.

## Auxiliary Attention Supervision

The main segmentation loss remains the exp006 v123 objective:

```text
L_total = L_v123 + L_s3_aux
```

For the baseline arm, `L_s3_aux` is absent. For S3 arms, the auxiliary loss is
small and applies only to nonempty positive prompt channels. Empty negative
prompt slots are not used to supervise S3 attention.

Current exp009 S3 loss weights:

```text
s3_attention_loss_weight = 0.05
s3_hard_bg_loss_weight = 0.01
s3_fullres_alignment_loss_weight = 0.0
s3_attention_margin = 0.05
s3_hard_background_fraction = 0.01
```

For `1/2` and `1/4`, the target mask is converted into a coarse support target:

```text
Y_s = max_pool3d(adaptive_max_pool3d(Y, attention_shape), kernel=3)
```

The direct attention loss is:

```text
L_direct = (1 - Dice(A, Y_s)) + BCE_probability(A, Y_s)
```

where:

```text
Dice(A, Y_s) = (2 * sum(A * Y_s) + eps) / (sum(A) + sum(Y_s) + eps)
```

The hard-background margin term compares mean attention inside the coarse
positive support with the highest-attention background voxels:

```text
L_hard_bg = relu(margin - mean(A inside Y_s) + mean(top_background(A)))
```

The top background set is the highest `1%` of background attention values at
the given scale.

The disabled full-resolution path exists for `1/1`, where dense supervision is
too expensive and can be dominated by background. Because active exp009 removed
`1/1`, `s3_fullres_alignment_loss_weight` is set to `0.0`.

## What Is Deferred

Several ideas from the broader S3 discussion are intentionally not part of the
current running experiment.

Prompt-specific strength is deferred. ReXGroundingCT includes both tiny focal
lesions and broad diffuse findings. Strong coarse attention may help broad
findings such as consolidation, ground-glass opacity, atelectasis, effusion,
pleural opacity, multifocal disease, bilateral disease, or diffuse wording. The
same strong coarse attention may hurt tiny or focal findings such as nodules,
subcentimeter lesions, millimeter-sized findings, solitary findings, or tiny
focal opacities. A later experiment should use stronger S3 for broad/diffuse
prompts and weak or near-off S3 for small/focal prompts.

Full-resolution S3 is deferred. The first all-scale smoke showed that naive
`1/1` attention is memory-heavy at `192^3`. A future version should use a
memory-reduced design before reintroducing `1/1`, for example lower hidden
width, checkpointing, local/windowed attention, sparse sampled supervision, or
an output-side full-resolution residual that does not keep large hidden volumes
alive during backward.

Scale ablation is deferred. The current run uses `1/2 + 1/4` for every S3 arm
so that the comparison is about coupling style. A later scale experiment can
compare `1/2`, `1/4`, and `1/2 + 1/4` after the best coupling style is chosen.

## General Guidance For This Task

For a text-image fusion segmentation model like VoxTell on ReXGroundingCT,
attention should be added conservatively:

1. Preserve the baseline text-image fusion path.
2. Initialize the new attention path so it has exactly no effect at epoch 0.
3. Add weak attention supervision to shape where attention looks, not to turn
   attention into a second segmentation head.
4. Couple attention at locations that can influence the final mask: decoder
   skips for feature guidance, logits for direct output correction, or both
   only after separate arms show benefit.
5. Keep coarse attention cautious for small lesions and stronger for broad
   diffuse findings.
6. Compare against a same-source continuation baseline, because extra training
   alone can change val20 behavior.

The current exp009 run follows items 1, 2, 3, 4, and 6. Item 5 is deferred to a
prompt-adaptive follow-up if the fixed-strength coupling ablation is
inconclusive.

## Reading The Current Run

The expected interpretation pattern is:

- If S3v1 helps, suppressing irrelevant skip evidence may be enough.
- If S3v2 helps more than S3v1, the model likely needed true boost/suppress
  feature modulation rather than suppress-only pruning.
- If S3v3 helps most, attention localization probably needs a direct logit
  route to affect Dice.
- If all S3 arms trail the baseline, fixed-strength S3 may be too coarse, the
  auxiliary attention target may be misaligned, or attention should be
  prompt-adaptive before it is coupled strongly.

Final decisions should wait for the planned epoch-100 val20 and val200 reports.
