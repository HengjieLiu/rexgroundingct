---
doc_type: research_audit
created: 2026-07-23
updated: 2026-07-23
status: active
topic: ReXGroundingCT VoxTell fine-tuning alignment
scope: Challenge rules, released data, dataset papers, baseline papers, current v123 training, and validation
audit_target:
  experiment: 003_voxtell_rex_ft_rescue_ablation
  variant: v123_opt_poscrop_emptyloss
  checkpoint: epoch 100
  base_model: voxtell_v1.1
  base_model_revision: 0809ac94c5ed594198bf4e85d63245cf222464d7
primary_sources:
  - https://rexrank.ai/ReXGroundingCT/challenge.html
  - https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT
  - https://huggingface.co/mrokuss/VoxTell
  - literature/challenge_paper.pdf
  - literature/challenge_motivation_paper.pdf
  - literature/voxeltell.pdf
related_notes:
  - 2026-07-23_rex_training_partial_instance_annotation_evidence.md
  - 2026-07-23_voxtell_anatomy_language_capability_gap.md
  - 2026-07-23_counterfactual_prompt_pair_training.md
---

# ReXGroundingCT VoxTell Fine-Tuning Alignment Audit

## Executive Conclusion

The current v123 run is a useful direct free-text baseline, and its corrected
orientation, output layout, direct full-prompt input, and primary
quick-evaluation metric are aligned in substance with the MICCAI challenge.
The lowercasing caveat in rank 19 should be resolved before claiming literal
Main-track compliance. Its validation result is:

| Model | Mean global Dice per finding | Hit rate at Dice `>= 0.1` |
| --- | ---: | ---: |
| Public VoxTell v1.1 | 0.225228 | 0.535433 |
| v123, epoch 100 | 0.283343 | 0.690289 |

These all-200 numbers are descriptive, but not a fully split-clean validation
estimate. The current VoxTell model card says v1.1 was trained on all paper
datasets and includes the paper test sets. The VoxTell paper identifies a
50-case ReX benchmark, and those same 50 base-validation cases are contained in
MICCAI val200. Rank 1 separates their metrics from the 150 added cases. The
cleaner v123 number for those added 150 cases is Dice `0.289403` over 266
findings.

However, the current fine-tuning pipeline is not yet well aligned with the
released training supervision or with how VoxTell is used at full-volume
inference. The largest problems are not z-score normalization or a lack of
epochs in isolation. They are:

1. v1.1's published training provenance likely overlaps 50 cases in val200, so
   the current model-selection set is not wholly independent;
2. partial-instance training masks are optimized as if every other voxel were
   confirmed background;
3. randomly sampled "negative" prompts are not verified absent;
4. every v123 positive prompt is trained only on a patch containing its target,
   while inference applies that prompt to many empty sliding-window patches;
5. the materialized schedule never presents 1,710 of 7,687 training findings
   as positive targets;
6. training always groups exactly three prompts, including many synthetic
   negatives, while inference groups a variable number of supplied findings
   that interact through decoder self-attention;
7. a local `192^3` patch receives no full-volume coordinates, anatomical masks,
   or physical spacing, despite location and size being central to the task;
8. the repeated 20-case development evaluation omits several categories,
   including bronchial wall thickening and bronchiectasis.

These issues can plausibly produce exactly the qualitative failures seen in
the visualizations: broad pathology-like activation, wrong side or lobe,
activation around an anatomically similar central structure, and excessive
false positives away from the described finding.

The next serious run should first repair supervision and sampling without
changing the VoxTell network. Anatomy and coordinate guidance should be the
next ablation. More optimizer sweeps on the current schedule would not resolve
the main mismatch.

## Audit Scope

This audit compares five layers of evidence:

1. the live [MICCAI 2026 challenge rules](https://rexrank.ai/ReXGroundingCT/challenge.html);
2. the released
   [ReXGroundingCT dataset card](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT)
   and local JSON files;
3. the ReXGroundingCT dataset paper,
   [arXiv:2507.22030v2](https://arxiv.org/html/2507.22030v2);
4. the older
   [MLHC baseline study](https://proceedings.mlr.press/v298/baharoon25a.html)
   and the [VoxTell paper](https://arxiv.org/abs/2511.11450);
5. the current
   [VoxTell model card](https://huggingface.co/mrokuss/VoxTell), including the
   distinction between v1.0 and v1.1;
6. the current repository code, materialized training schedule, runtime
   manifests, and validation outputs.

The current implementation audited here is:

- configuration:
  [`configs/experiments/003_voxtell_rex_ft_rescue_ablation.json`](../../configs/experiments/003_voxtell_rex_ft_rescue_ablation.json);
- trainer:
  [`scripts/rexgroundingct/train_text_conditioned_voxtell.py`](../../scripts/rexgroundingct/train_text_conditioned_voxtell.py);
- schedule generator:
  [`scripts/rexgroundingct/prepare_training_schedule.py`](../../scripts/rexgroundingct/prepare_training_schedule.py);
- selected model: `v123_opt_poscrop_emptyloss`, epoch 100;
- selected result: quick/global evaluation on all 200 MICCAI validation
  cases at threshold `0.5`.

In this document:

- **Confirmed mismatch** means both the source requirement and current code
  behavior were directly verified.
- **High-confidence risk** means the inputs and implementation were verified,
  but the size of their causal effect still requires an ablation.
- **Comparison trap** means two reported numbers do not measure the same split,
  threshold, or training setup.
- **Opportunity** means the challenge permits a useful signal that the current
  pipeline does not consume.

## Challenge Contract

The current MICCAI challenge contract is:

| Requirement | Consequence for the pipeline |
| --- | --- |
| Main track consumes the exact raw finding | The original prompt must reach the segmentation model directly |
| Fixed category segmentation is disallowed | Category may assist but cannot replace prompt-conditioned grounding |
| Same-category findings in different locations must be separable | The sampler must train location-sensitive distinctions |
| Training is partial-instance | Unannotated voxels are not automatically confirmed background |
| Validation and test are exhaustive | The model is rewarded for retrieving all instances named by a finding |
| Ranking uses Dice per finding per case | Finding exposure and model selection should be finding-aware |
| Hit rate uses Dice `>= 0.1` | VoxTell paper `HIT5` is not the challenge hit metric |
| Instance and distance metrics are also reported | Binary union Dice alone is not a complete evaluation |
| Category and anatomy signals are allowed | Lobe, airway, lung, and pleural priors can be used in both tracks |
| Test inference must be fully automatic | No case-specific manual correction or tuning is allowed |

The released MICCAI splits contain:

| Split | Cases | Finding prompts | Annotation policy |
| --- | ---: | ---: | --- |
| train | 2,992 | 7,687 | partial-instance |
| val | 200 | 381 | exhaustive |
| test | 300 | 582 | exhaustive, masks withheld |

## What the Current Pipeline Actually Does

| Component | Current v123 behavior |
| --- | --- |
| Initialization | public VoxTell v1.1 post-paper checkpoint |
| Image reader | `NibabelIOWithReorient` |
| Target orientation | converts ReX `(F,X,Y,Z)` to VoxTell `(F,Z,Y,X)` |
| Crop | crops nonzero image extent |
| Resampling | none |
| Intensity | per-volume z-score |
| Patch | `192 x 192 x 192` voxels |
| Text | prompt lowercased before frozen Qwen3 embedding |
| Text pooling | one last-token sentence embedding |
| Prompt slots | two positive plus one negative for multi-finding cases |
| One-finding fallback | one positive plus two negatives |
| Prompt interaction | all three prompt queries interact through transformer self-attention |
| v123 crop rule | every selected positive must be nonempty in the same patch |
| Target conversion | all nonzero entity IDs are merged into one binary channel |
| Positive loss | Dice plus weighted BCE |
| Empty-target loss | BCE only, multiplied by `0.5` |
| Augmentation | no spatial or intensity augmentation in this trainer |
| Optimizer | SGD, encoder LR `1e-7`, decoder LR `1e-6` |
| Training scale | 10,000 updates, batch 1, no accumulation, one GPU per model |
| Validation | repeated val20 quick/global checks; val200 at epoch 100 |
| Inference | full-volume sliding window, threshold `0.5`, all case findings grouped together |

The frozen Qwen encoder and precomputed embeddings agree with the VoxTell
paper. The full free-text finding, rather than a fixed category, is directly
embedded, which is aligned with the substance of the Main track. The
exact-as-provided case-normalization question remains open under the challenge
wording. The main modeling problems begin with how masks, patches, negatives,
prompt groups, and validation samples are constructed around that model.

## Ranked Findings

| Rank | Priority | Finding | Classification |
| ---: | --- | --- | --- |
| 1 | P0 | VoxTell v1.1 likely overlaps 50 cases in the current validation set | Split-contamination risk |
| 2 | P0 | Partial-instance labels are treated as exhaustive background | Confirmed mismatch |
| 3 | P0 | Random empty-mask prompts are not verified absent | Confirmed mismatch |
| 4 | P0 | Positive-only crops do not match sliding-window inference | Confirmed mismatch |
| 5 | P0 | The schedule is case-biased and leaves 22.2% of findings unseen | Confirmed mismatch |
| 6 | P0 | Three-prompt training and variable-prompt inference have different query context | Confirmed mismatch |
| 7 | P0 | Local patches lack global anatomy and coordinates | High-confidence risk |
| 8 | P0 | Model selection uses a category-blind 20-case subset | Confirmed mismatch |
| 9 | P1 | Category imbalance and train-to-evaluation distribution shift are uncontrolled | Confirmed mismatch |
| 10 | P1 | Physical scale varies, but spacing and physical coordinates are absent | High-confidence risk |
| 11 | P1 | v1.1-initialized ReX-only adaptation is not the VoxTell paper experiment | Comparison trap |
| 12 | P1 | Training scale, augmentation, precision, and prompt variation differ from VoxTell | Confirmed mismatch |
| 13 | P1 | Focal and non-focal findings use one sampler and one loss recipe | Confirmed mismatch |
| 14 | P1 | Entity identities and challenge secondary metrics are not used | Confirmed limitation |
| 15 | P2 | Prompt and annotation provenance are heterogeneous and noisy | Confirmed data limitation |
| 16 | P2 | Temporal prompts contain information unavailable in one CT | Task-information mismatch |
| 17 | P2 | Universal z-score preprocessing leaves CT-specific HU information unused | Opportunity |
| 18 | P2 | A hard whole-lung constraint would exclude valid pleural targets | Design guardrail |
| 19 | P2 | Lowercasing means the model does not literally consume text as provided | Compliance risk |
| 20 | P2 | Old and current splits, taxonomies, and hit thresholds are easy to mix | Comparison trap |
| 21 | P2 | Repository status does not fully describe completed runtime work | Reproducibility issue |

## 1. VoxTell v1.1 Likely Overlaps the Current Validation Set

**Why this is first:** checkpoint provenance determines whether validation is
an independent estimate at all.

The current
[VoxTell model card](https://huggingface.co/mrokuss/VoxTell) distinguishes two
releases:

| Checkpoint | Model-card provenance |
| --- | --- |
| v1.0 | paper model; strict train/test separation |
| v1.1 | all paper datasets plus additional sources; paper test sets included |

The v123 runtime manifest records the v1.1 Hugging Face snapshot
[`0809ac94c5ed594198bf4e85d63245cf222464d7`](https://huggingface.co/mrokuss/VoxTell/commit/0809ac94c5ed594198bf4e85d63245cf222464d7),
so this is the model-card revision associated with the actual initialization,
not a guessed model name.

It also says v1.1 samples image-text-mask triplets from the instance-focused
datasets with 5% probability. The VoxTell paper's Table 6 identifies a
50-image `ReXGroundingCT Test` row, while its ReX result is reported on 50
cases; Appendix B.3 describes using the official ReX training and validation
splits respectively. Locally:

- all 50 cases from base `dataset.json` val are present in MICCAI val200;
- the other 150 MICCAI validation cases are additions;
- the fixed val20 probe contains 6 of those base 50 cases.

The model card does not enumerate individual ReX filenames. Therefore the
careful conclusion is:

> Unless the VoxTell maintainers clarify otherwise, the original ReX val50
> should be treated as checkpoint-training overlap for v1.1 and every
> descendant initialized from it, including v123.

This is not a challenge-rule violation by itself. External data and pretrained
models are allowed with disclosure. It does mean that val200 as a whole is not
a clean model-selection set for this initialization.

### Measured split-stratified results

The existing prediction files allow the result to be separated without new
inference:

| Model | Validation subset | Cases | Findings | Mean Dice | Hit rate |
| --- | --- | ---: | ---: | ---: | ---: |
| v1.1 | base val50, likely seen | 50 | 115 | 0.213384 | 0.495652 |
| v1.1 | MICCAI-added 150 | 150 | 266 | 0.230348 | 0.552632 |
| v1.1 | all val200 | 200 | 381 | 0.225228 | 0.535433 |
| v123 | base val50, likely seen at initialization | 50 | 115 | 0.269324 | 0.652174 |
| v123 | MICCAI-added 150 | 150 | 266 | 0.289403 | 0.706767 |
| v123 | all val200 | 200 | 381 | 0.283343 | 0.690289 |

The likely seen 50 score *lower*, not higher, than the added 150. Thus there is
no empirical basis to claim that overlap inflated the all-200 score. The
problem is independence and provenance, not a demonstrated direction or size
of bias. The added 150 are cleaner with respect to v1.1 provenance, but they
are not untouched: existing runtime work has already evaluated epoch-100
variants and ensemble thresholds on val200.

### Recommended correction

1. Treat the MICCAI-added 150 cases as the primary local evaluation subset for
   every v1.1-initialized experiment.
2. Keep the base val50 as a separate `checkpoint_seen_or_uncertain` stratum.
3. Recompute all important v123 and pretrained reports with both strata.
4. Run v1.0 as a strict-split checkpoint control, while noting its documented
   known issues.
5. Ask the VoxTell maintainers which ReX case IDs and masks entered v1.1.
6. Describe v1.1 as a broad post-paper checkpoint, not as the paper checkpoint
   or a clean ReX zero-shot model.
7. Disclose checkpoint and external-data provenance in the challenge method.

### Acceptance tests

- Evaluation JSONs explicitly identify `base_val50` and `miccai_added150`.
- Model selection for a v1.1 descendant never silently averages those strata.
- Reports name v1.0 or v1.1 rather than only "pretrained VoxTell."
- A checkpoint provenance manifest records known and uncertain overlap.

## 2. Partial Labels Are Treated as Exhaustive Background

**Why this is near the top:** this changes the mathematical meaning of the target.
The model is punished for predicting visible instances that may be correct but
were not selected for training annotation.

### Source evidence

Section 2.4 of the ReXGroundingCT paper states that training annotators were
instructed to segment no more than three representative instances of a
finding, while validation and test were exhaustive. The paper's Limitations
section explicitly calls this a partial-labeling strategy. The challenge page
repeats the same split distinction.

The released train metadata contains:

| Released `entity_count` | Findings |
| ---: | ---: |
| 1 | 3,231 |
| 2 | 1,976 |
| 3 | 2,271 |
| greater than 3 | 209 |
| total | 7,687 |

The 209 findings above three are an important nuance. The paper and challenge
describe the intended training policy as "up to three," but the released
metadata contains exceptions, with a maximum of 11. Their cause is not
documented in the available sources. Therefore:

- the partial-instance intent is directly established;
- an `entity_count` of three is a censoring-risk indicator, not proof that
  additional instances exist;
- the nominal cap should not be implemented as an absolute metadata invariant.

### Current behavior

The trainer:

1. merges all nonzero entity IDs using `target > 0`;
2. computes Dice and BCE against the resulting binary patch;
3. treats every zero voxel as background;
4. gives extra BCE weight to a radius-10 shell around the annotated mask.

For a finding with omitted fourth or later instances, those omitted instances
are therefore supervised as background whenever they fall in a sampled patch.
Dice also penalizes prediction outside the annotated representative subset.

### Likely effect

- reduced recall for multi-instance findings;
- suppression of additional nodules or opacities at validation/test;
- conflict between training labels and exhaustive challenge ground truth;
- overly conservative masks around representative instances;
- noisy gradients for counterfactual or empty-mask prompts.

### Recommended correction

Implement partial-label-aware targets:

1. preserve integer entity IDs during loading;
2. mark annotated entity voxels as positive;
3. supervise a local boundary region around each annotated entity as
   background, where boundary completeness is more credible;
4. give distant unlabeled voxels an ignore or very low background weight;
5. reserve global empty-mask loss for independently confirmed absences;
6. optionally use teacher predictions or consistency loss on ignored regions,
   but do not convert them directly into hard labels.

### Acceptance tests

- Loss receives explicit `positive`, `known_background`, and `ignore` masks.
- A synthetic omitted fourth component contributes neither positive nor
  negative BCE.
- Findings with `entity_count >= 3` do not receive stronger global background
  penalties than single-instance findings.
- Validation reports Dice and recall stratified by GT entity count.

## 3. Random Negative Prompts Are Not Verified Absent

**Why this is separate from rank 2:** the same partial-label problem is made
worse by a negative sampler that interprets "not listed in this case" as
"absent everywhere in this CT."

### Current behavior

The negative pool is the set of 6,148 unique lowercased training prompts. For
each case, the sampler excludes only exact prompt strings already listed for
that case, then samples another prompt and assigns an all-zero target.

In the materialized v123 schedule:

| Statistic | Value |
| --- | ---: |
| total prompt slots | 30,000 |
| positive slots | 16,260 |
| negative slots | 13,740 |
| negative share | 45.8% |
| negative slots whose category is present elsewhere in the same case | 3,676 |
| same-case category overlap rate among negatives | 26.8% |

Same-category overlap does not prove that a sampled prompt is a false negative.
A nodule in a different lobe can be a valid absent query. It does show that
exact-string exclusion is too weak to establish semantic absence.

One-finding cases make the ratio more aggressive:

- 1,056 of 2,992 training cases have one finding;
- those cases use one positive and two negatives;
- the VoxTell paper's nominal recipe is two positives and one negative.

### Unused evidence

`reports_dataset.json` matches all 2,992 training cases and includes full
reports, negative statements, and findings outside the task scope. A simple
text audit found:

- 12,860 lung/airway/pleural report sentences;
- 4,213 negative-looking sentences;
- explicit negative-looking lung statements in 2,346 training cases.

This regex count is not itself a clinically validated negative label set. It
shows that much stronger evidence is available than exact prompt-string
absence.

### Recommended correction

Use the three-level policy defined in
[`2026-07-23_counterfactual_prompt_pair_training.md`](2026-07-23_counterfactual_prompt_pair_training.md):

| Label type | Target policy |
| --- | --- |
| known positive | supervise the matching released mask |
| partial negative | penalize overlap with a known incompatible GT region; ignore uncertain space |
| confirmed absent | use a globally empty target with conservative weight |

For confirmed absences, require provenance such as:

- an explicit negative report statement;
- an anatomically impossible location;
- a location-specific counterfactual outside the named lobe or side;
- manually reviewed high-confidence examples for initial calibration.

### Acceptance tests

- Every global empty target records a `negative_evidence_type`.
- No empty target is created solely because an exact string is missing.
- A manually reviewed sample estimates the false-negative rate by evidence
  tier.
- Training metrics report each negative tier separately.

## 4. Positive-Only Crops Do Not Match Sliding-Window Inference

**Why this matters:** a focal prompt is applied to the entire CT at inference,
but it is never trained on the many windows where its target is absent.

### Current behavior

For v123:

- all 10,000 events require a positive crop;
- all selected positive findings must have voxels in the same `192^3` patch;
- all 16,260 positive prompt channels are nonempty;
- there are no same-prompt, same-case empty-window events.

The positivity threshold is only one voxel. There is no required fraction of
the component inside the patch, no margin from the patch edge, and no minimum
amount of surrounding anatomical context. A two-positive crop can therefore
technically pass while clipping most of one or both findings.

At full-volume inference, the same prompt is evaluated over a sliding-window
grid. Most windows for a focal nodule, scar, or small opacity do not contain
the target.

A different random prompt with an empty mask is not an adequate substitute.
It teaches "this other sentence is absent here," not "the factual sentence
must remain silent outside its stated anatomy."

### Likely effect

- false positives in anatomically similar regions;
- activation on the wrong lung, lobe, or central structure;
- poor calibration across sliding windows;
- high sensitivity to generic pathology texture and weak location binding.

### Recommended correction

Use a mixed event schedule:

| Event type | Initial share | Purpose |
| --- | ---: | --- |
| same-prompt foreground | 50-60% | learn target appearance and boundaries |
| same-prompt spatial empty | 20-25% | suppress wrong locations at inference |
| paired/counterfactual | 10-15% | bind pathology to side, lobe, and relation |
| confirmed absent prompt | 5-10% | teach true abstention |
| random context | 5-10% | preserve background robustness |

Same-prompt empty windows must respect partial labels. Examples are safest
when the prompt names a location and the patch lies in an incompatible
anatomical region. Diffuse bilateral prompts require a different policy.
Foreground crops should also be component-centered or margin-aware, with the
included fraction of the selected component recorded.

### Acceptance tests

- Schedule manifests count foreground and same-prompt-empty events separately.
- Focal prompts receive known spatial-empty windows.
- Diffuse prompts are not assigned unsafe empty lung patches.
- Positive crops report component coverage and distance to the patch boundary.
- Full-volume false-positive voxel count and wrong-lobe activation are tracked.

## 5. The Schedule Is Not Finding-Aligned

**Why this matters:** the challenge averages over findings, but the schedule
samples cases and then only one or two compatible findings from each case.

### Measured coverage

| Statistic | Value |
| --- | ---: |
| training cases | 2,992 |
| cases eligible under positive-crop geometry | 2,822 |
| multi-finding cases rejected by geometry | 170 |
| unique cases actually sampled | 2,746 |
| eligible cases never sampled | 76 |
| all cases never sampled | 246 |
| training findings | 7,687 |
| unique findings used as a positive | 5,977 |
| findings never used as a positive | 1,710 |
| unseen finding fraction | 22.2% |
| median positive exposures among seen findings | 2 |
| maximum positive exposures | 12 |

The unseen fraction is not confined to one rare class. It ranges from `12.3%`
for category `2h` to `28.0%` for `1e`; category `2a` leaves `305/1,120`
findings (`27.2%`) unseen.

The geometry rule requires two selected positive findings to coexist inside
one patch. This favors co-located findings and rejects spatially separated
ones, even though the challenge explicitly requires separation of findings in
different locations.

Case-uniform sampling also underweights findings in cases with many prompts.
A case with one finding contributes one positive whenever selected. A case
with ten findings contributes only two, so each individual finding has a much
lower chance of exposure.

Repeated categories within a case are not rare:

| Split | Cases with repeated category | Same-category finding pairs |
| --- | ---: | ---: |
| train | 1,061 of 2,992 | 2,713 |
| val | 33 of 200 | 43 |
| test | 52 of 300 | 95 |

The v123 schedule contains 1,386 two-positive events whose two positives share
a category, but it has no explicit pairwise separation objective.

### Recommended correction

1. Make `(case, finding_id)` the primary sampling unit.
2. Generate a coverage-first epoch in which every finding appears at least
   once before any finding repeats.
3. Do not require two positives to share a patch.
4. Use one anchor finding per event, then add an optional paired prompt whose
   role is explicit.
5. Oversample rare categories and difficult morphology/size strata only after
   minimum finding coverage is guaranteed.
6. Add same-category, different-location pairs deliberately.

### Acceptance tests

- `100%` of 7,687 findings receive a positive event in each coverage cycle.
- No case is excluded because two findings cannot fit one patch.
- Exposure histograms are reported by finding, category, entity count, and
  protocol.
- The effective optimization weighting is reported in units of findings, not
  only cases or steps.

## 6. Training and Inference Use Different Prompt Sets

**Why this is high priority:** VoxTell prompt outputs are not computed
independently. Every transformer decoder layer in upstream
[`transformer.py`](../../external/VoxTell/voxtell/model/transformer.py)
applies self-attention across the prompt queries. The mask for one finding can
therefore depend on which other findings are evaluated beside it.

The current training and inference distributions are:

| Property | v123 training | MICCAI inference |
| --- | --- | --- |
| prompts passed together | exactly 3 | all findings for the case |
| prompt-count range | fixed at 3 | val 1-5; test 1-8 |
| mean prompts per case | not applicable | val 1.905; test 1.940 |
| synthetic absent prompts | 45.8% of slots | none intentionally added |
| one-finding case | 1 positive + 2 random negatives | 1 supplied finding |

The train split itself has 1 to 13 findings per case, but the sampler always
reduces them to three slots. The validation runner passes
`sorted_prompts(entry)` as one group, so the number and composition of
companion queries changes at deployment.

This is a confirmed input-distribution mismatch. Its effect size is not yet
measured. Because query self-attention is active, it cannot be assumed harmless.
In particular:

- a prediction may change when the same finding is evaluated alone;
- a one-finding case never sees the two negative companion queries used during
  its training events;
- training query sets contain positive-versus-negative competition, while
  challenge query sets contain only report findings;
- co-occurring prompts can become context shortcuts even though the challenge
  score is defined independently per finding.

### Recommended correction

First run a no-training diagnostic:

1. infer a stratified validation subset with all case prompts together;
2. infer exactly the same findings one prompt at a time;
3. infer them with two fixed, confirmed-absent companion prompts;
4. compare logits, Dice, hit status, and connected components for each finding.

Then choose one of these policies:

- **Prompt-independent policy:** evaluate and train one prompt at a time, or
  mask query self-attention. This most directly matches per-finding scoring.
- **Prompt-set policy:** train with variable prompt counts and realistic
  all-positive groups, plus prompt dropout and permutation invariance.
- **Hybrid policy:** retain grouped inference for efficiency, but add an
  invariance loss between an anchor prompt alone and the same prompt in a
  group.

The simplest initial experiment is one-prompt inference on existing
checkpoints. It requires no model modification and will show whether prompt
context is already affecting the result.

### Acceptance tests

- Predictions are invariant to prompt order.
- Adding an unrelated confirmed-absent prompt does not materially change the
  anchor mask.
- Alone-versus-group Dice between model probability maps is reported.
- Training logs the distribution of prompt counts and positive/negative
  composition.
- The selected inference grouping policy is fixed before final val200
  evaluation.

## 7. Local Patches Lack Global Anatomy and Coordinates

**Why this matters:** the challenge is not merely pathology segmentation. It
requires composition of pathology, side, lobe, segment, and spatial relation.

The current image network receives a local patch with internal positional
encoding, but not:

- the patch origin in the full CT;
- normalized left-right, anterior-posterior, or superior-inferior coordinates;
- lung or lobe masks;
- airway, pleural, fissure, mediastinal, or heart context channels;
- a low-resolution whole-volume view.

This makes "left upper lobe" and "right lower lobe" unnecessarily hard when
the local patch lacks decisive landmarks. It is especially problematic for
bronchial wall thickening, where airway, vessel, mediastinal, and cardiac
boundaries can produce similar local structures.

The text path does not supply an explicit decomposition either. Each complete
sentence becomes one frozen, last-token Qwen embedding. There is no direct loss
requiring the decoder to distinguish the pathology term, laterality, lobe,
segment, morphology, and size attributes. Counterfactual pairs can add this
signal without replacing the raw prompt.

This matters for compositional and multi-region prompts:

- 2,986 of 7,687 training findings (`38.8%`) contain bilateral or both-lung
  wording;
- 798 (`10.4%`) contain the word "and";
- a local crop may show only one side or one named region while retaining the
  full bilateral or multi-location sentence.

Without full-volume position, the same sentence is therefore paired with
locally incomplete evidence, which can teach the model that location words are
optional.

The challenge explicitly permits anatomy masks, category auxiliaries, and
anatomy-constrained postprocessing in both tracks, provided the exact prompt
is still consumed.

### Released anatomy supervision is currently unused

`anatomical_cot.json` contains:

| Split | Cases | Finding CoTs | CoTs containing a box |
| --- | ---: | ---: | ---: |
| train | 2,992 | 7,687 | 5,784 |
| base val | 50 | 115 | 18 |
| base test | 100 | 226 | 34 |

The dataset paper reports CoT finding-alignment accuracy of `0.92` and
adjacency accuracy of `0.75`. These are useful soft signals, not perfect
ground truth.

### Recommended correction order

1. Add three normalized full-volume coordinate channels.
2. Add category as an auxiliary embedding while retaining the exact raw text.
3. Add automatically generated lung, lobe, airway, and pleural context masks.
4. Add a low-resolution whole-volume locator if coordinate channels are not
   sufficient.
5. Train left/right and lobe counterfactual pairs with a pairwise separation
   loss.

Anatomy masks should be soft priors or input channels, not universal hard
clipping. See rank 18.

### Acceptance tests

- Laterality swap test: left-to-right prompt edit moves probability to the
  correct side.
- Lobe swap test: upper-to-lower edit changes the predicted region.
- Pathology swap test: same anatomy but different pathology changes the mask.
- Identical prompt and image with different patch origins produce
  origin-consistent predictions.
- Bronchial categories are evaluated with airway-distance and wrong-organ
  activation diagnostics.

## 8. The 20-Case Development Set Has Major Blind Spots

The repeated `val20` subset contains only 31 findings:

| Included category | Findings |
| --- | ---: |
| 1c | 4 |
| 1e | 2 |
| 2a | 6 |
| 2b | 4 |
| 2c | 6 |
| 2d | 6 |
| 2e | 1 |
| 2g | 1 |
| 2h | 1 |

It contains no `1a` bronchial wall thickening, `1b` bronchiectasis, `1d`
septal thickening, or `1f` other non-focal findings. Category `2f`
honeycombing is absent from the full MICCAI validation set as well.

It also contains 6 cases from the base val50 that should be treated as likely
v1.1 checkpoint-training overlap under rank 1.

This means epoch selection, variant selection, threshold sweeps, and
multi-scale experiments on val20 could not detect the bronchial-wall failure
that motivated this audit.

The runtime also contains epoch-100 val200 ensemble and threshold evaluations.
Thus val200 remains the official development set, but it should not be
described as an untouched confirmation set for the current project.

### Full val200 category diagnostic

These are quick/global metrics at threshold `0.5`. Counts are small for some
categories, so the table is diagnostic rather than a stable ranking.

| Category | Val findings | Pretrained Dice | v123 Dice | v123 hit rate |
| --- | ---: | ---: | ---: | ---: |
| 1a | 3 | 0.0731 | 0.0867 | 0.333 |
| 1b | 11 | 0.0723 | 0.1414 | 0.364 |
| 1c | 17 | 0.0638 | 0.1106 | 0.412 |
| 1d | 6 | 0.1251 | 0.1328 | 0.167 |
| 1e | 11 | 0.1016 | 0.1861 | 0.636 |
| 1f | 4 | 0.0021 | 0.1318 | 0.250 |
| 2a | 69 | 0.1971 | 0.2698 | 0.710 |
| 2b | 49 | 0.2975 | 0.3255 | 0.735 |
| 2c | 60 | 0.3090 | 0.3730 | 0.783 |
| 2d | 132 | 0.2266 | 0.2926 | 0.758 |
| 2e | 11 | 0.4310 | 0.4406 | 0.727 |
| 2g | 1 | 0.1388 | 0.0768 | 0.000 |
| 2h | 7 | 0.0477 | 0.0563 | 0.286 |

The overall gain is real, but it is dominated by common focal categories.
The current validation evidence does not support a claim that v123 has solved
anatomy-sensitive non-focal grounding.

### Recommended correction

1. Build a fixed, stratified development set from the MICCAI-added 150,
   covering every available category, focality, size bin, laterality pattern,
   and entity-count stratum.
2. Keep a locked part of val200 for confirmation rather than repeatedly
   sweeping all 200 cases.
3. Use train-split cross-validation or a curated train holdout for categories
   with too few val cases.
4. Add prompt-perturbation and wrong-anatomy metrics to model selection.
5. Run the full official evaluator, including instance and distance metrics,
   only for selected checkpoints.

## 9. Category Imbalance and Split Shift Are Uncontrolled

The 14 categories are extremely long-tailed, and their proportions differ
between training and challenge evaluation:

| Category | Train n (%) | Val n (%) | Test n (%) |
| --- | ---: | ---: | ---: |
| 1a | 236 (3.07) | 3 (0.79) | 6 (1.03) |
| 1b | 282 (3.67) | 11 (2.89) | 11 (1.89) |
| 1c | 446 (5.80) | 17 (4.46) | 27 (4.64) |
| 1d | 194 (2.52) | 6 (1.57) | 9 (1.55) |
| 1e | 314 (4.08) | 11 (2.89) | 16 (2.75) |
| 1f | 150 (1.95) | 4 (1.05) | 4 (0.69) |
| 2a | 1,120 (14.57) | 69 (18.11) | 113 (19.42) |
| 2b | 1,367 (17.78) | 49 (12.86) | 89 (15.29) |
| 2c | 1,507 (19.60) | 60 (15.75) | 87 (14.95) |
| 2d | 1,743 (22.67) | 132 (34.65) | 190 (32.65) |
| 2e | 237 (3.08) | 11 (2.89) | 27 (4.64) |
| 2f | 16 (0.21) | 0 (0.00) | 0 (0.00) |
| 2g | 18 (0.23) | 1 (0.26) | 1 (0.17) |
| 2h | 57 (0.74) | 7 (1.84) | 2 (0.34) |

The current sampler is case-uniform before its geometry filter. It does not
guarantee category coverage and does not compensate for rare findings.
Meanwhile, the headline mean per finding is dominated by the categories that
are common in val/test, especially `2d`.

This creates two distinct risks:

1. aggregate Dice can improve while rare and non-focal categories remain near
   failure;
2. a training policy tuned to raw train proportions is not matched to the
   challenge finding distribution.

Category balancing must not replace finding coverage. With only 16 train
honeycombing findings and no val/test examples, aggressive oversampling could
mostly memorize annotation style.

### Recommended correction

1. Guarantee one exposure per finding first.
2. Add capped inverse-frequency or square-root category weighting only for
   repeat exposures.
3. Report both challenge-weighted mean Dice and unweighted macro-category
   Dice.
4. Track focal/non-focal and protocol strata separately.
5. Use a curated train holdout for categories absent or too rare in val.
6. Keep category auxiliary inputs, sampling weights, and metric aggregation as
   separate ablations.

## 10. Physical Scale Is Missing From the Model Input

The model uses a no-resampling VoxTell plan. An audit of all 3,192 downloaded
train and val CT headers found:

| Axis spacing | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| x, mm | 0.304 | 0.697 | 0.977 |
| y, mm | 0.304 | 0.697 | 0.977 |
| z, mm | 0.500 | 1.000 | 3.000 |

All audited on-disk volumes use `LPS` axis codes, and the nnU-Net reader
reorients them consistently. The issue here is scale, not orientation.

A `192^3` patch therefore spans approximately:

- x/y: `58` to `188` mm, median about `134` mm;
- z: `96` to `576` mm, median `192` mm.

The network does not receive voxel spacing as input. At the same time, 856 of
7,687 training prompts (`11.1%`) contain an explicit millimeter or centimeter
measurement. The same "4 mm nodule" can occupy a very different number of
voxels across scans.

The v123 boundary shell has the same problem: its radius is fixed at 10 voxels.
With native anisotropic spacing, that shell represents different physical
thicknesses across scans and axes.

VoxTell was designed to tolerate heterogeneous native data, so this is not
proof that no-resampling is wrong. It is a genuine underdetermination for
physical-size language and a likely source of context variability.

### Recommended ablation

Compare:

1. native spacing plus normalized coordinates and explicit spacing metadata;
2. nnU-Net planned target spacing with nearest-neighbor mask resampling;
3. native high-resolution branch plus a standard-spacing low-resolution
   whole-volume branch.

Do not change normalization and spacing simultaneously. That would make the
result uninterpretable.

## 11. The Checkpoint and Data Do Not Reproduce the VoxTell Paper Experiment

The VoxTell paper's instance-focused training pool combines:

- ReXGroundingCT train;
- semantic lesion datasets converted to localized instances using
  TotalSegmentator anatomy;
- location-rich TCIA datasets;
- lung, liver, kidney, brain, and head/neck lesion data.

The non-ReX rows in Table 6 sum to 6,724 images in addition to ReX train. The
separate 50-case ReX row is the held-out benchmark. The paper does not publish
the exact per-source sampling weights. The anatomy-derived prompts are
directly relevant to the location-comprehension gap.

The public model card now states that v1.0, not v1.1, was used for the paper
experiments. v1.1 is a later general-use checkpoint trained on 190 datasets
and about 68,500 volumes, using a 95% semantic and 5% instance-focused sampling
mixture. It includes all paper datasets, additional sources, and paper test
sets.

The reported VoxTell ReX result therefore differs from this experiment in
several ways:

| Property | VoxTell paper | Current audit |
| --- | --- | --- |
| Starting checkpoint | v1.0 paper model | v1.1 post-paper model |
| Checkpoint split policy | strict paper separation | includes paper test sets |
| ReX validation | original 50 cases | MICCAI 200 cases |
| Instance training before current run | extended multi-dataset pool | broader v1.1 mixture including instance data |
| Current adaptation data | extended multi-dataset pool | ReX train only |
| Hit metric | HIT5, Dice `>= 0.05` | challenge hit, Dice `>= 0.1` |

The current MICCAI val200 contains the original base val50 plus 150 additional
cases. The MICCAI test300 contains the original base test100 plus 200
additional cases.

Therefore, the paper's Dice `28.2` and HIT5 `67.8` are context, not a strict
reproduction target for v123. The numerical similarity between `0.2833` and
`28.2` should not be interpreted as reproduction. Current v123 is continued
ReX adaptation of a broader post-paper model, not the first ReX exposure of
the paper checkpoint.

## 12. Training Strategy Differs From Published VoxTell

The general VoxTell paper configuration uses:

- 2,000 epochs x 250 iterations;
- standard nnU-Net augmentation, excluding left-right mirroring;
- SGD at an initial LR reported as `1e-4` in the paper text;
- final global batch 128 on 64 A100 GPUs;
- two positive and one absent negative prompt;
- 85% foreground oversampling;
- default prompt wording 25% of the time and a rephrased variant 75%;
- Dice plus BCE and five-scale deep supervision.

The current v123 run uses:

- 10,000 updates;
- true batch 1 on one GPU;
- no gradient accumulation;
- encoder LR `1e-7`, decoder LR `1e-6`;
- default CUDA autocast, which is FP16 rather than BF16;
- no image augmentation;
- released ReX prompt strings only, lowercased by the embedding path, with no
  rephrasing;
- 45.8% negative slots because of the one-finding fallback.

The VoxTell paper does not disclose a separate, complete optimizer schedule for
its ReX-specific fine-tuning stage. It would therefore be incorrect to claim
that ReX fine-tuning must use the full 500,000-iteration foundation-model
recipe. It is still clear that the current run is a much smaller and less
varied adaptation.

The older MLHC SAT baseline used 40,000 steps, LR `1e-4`, warmup 2,000, full
model fine-tuning, and BF16 after FP16 instability. The completed v123 run did
not show catastrophic FP16 failure, but its successful completion does not
establish equivalence between FP16 and BF16, especially with very small
differential learning rates. SAT is a different model on an older ReX split,
so these settings are evidence that substantial adaptation and precision
checking were used, not a hyperparameter prescription for VoxTell.

### Recommended correction

After ranks 2-6 are fixed:

1. train one model across all four GPUs with DDP;
2. begin with batch 1 per GPU and accumulate to effective batch 16;
3. restore nnU-Net spatial and intensity augmentation except left-right
   mirroring;
4. add validated paraphrases and invariance pairs;
5. compare conservative differential LR against a larger decoder LR;
6. compare BF16 and FP16 AMP stability, gradient norms, and throughput;
7. judge saturation by finding-stratified validation, not training loss alone.

## 13. Focal and Non-Focal Findings Need Different Treatment

The ReX paper reports approximately 79% focal and 21% non-focal findings. The
MLHC baseline study found that non-focal performance could be up to two times
lower than focal performance.

Current v123 applies one recipe to:

- tiny nodules;
- linear scars;
- diffuse emphysema;
- bronchial wall thickening;
- bilateral ground-glass opacity;
- pleural effusion;
- pneumothorax.

These targets differ in useful context, boundary certainty, expected extent,
and safe negative regions. A radius-10 boundary emphasis is not equally
appropriate for a 3 mm nodule and diffuse bilateral emphysema.

There is also a deep-supervision edge case. The trainer creates lower-resolution
targets using nearest-neighbor interpolation. A small positive component can
disappear at a coarse decoder scale; under `empty_bce_only`, that scale is then
treated as a genuinely empty target and receives a suppression loss. The v1234
ablation drops only the lowest-resolution output, while v123 uses all five.
The actual positive-to-empty conversion rate has not been logged.

### Recommended correction

Use the official category as an auxiliary, which is allowed and supplied for
test prompts:

- focal: high-resolution positive anchor, small-lesion weighting, anatomically
  hard same-prompt negatives;
- non-focal: larger or multi-scale context, lower boundary emphasis, extent
  consistency, fewer unsafe empty lung patches;
- pleural: pleural shell and thoracic boundary context;
- bronchial: airway tree and peribronchial context;
- nodule/mass: lung/lobe prior plus size-aware features.

For deep supervision:

- log target non-emptiness at every decoder scale by component size;
- preserve positives with max-pool or occupancy-aware downsampling;
- alternatively ignore a coarse scale when a known positive vanishes;
- compare dropping more coarse outputs for small focal targets.

The raw prompt must remain a direct model input for Main-track eligibility.

## 14. Entity IDs and Secondary Metrics Are Not Used

ReX masks encode separate entities with separate nonzero values. The current
trainer converts every finding channel to a binary union.

This is aligned with the primary global Dice target, which evaluates the union
for a finding. It discards information useful for:

- instance precision, recall, and F1;
- component-level sampling;
- separation of nearby instances;
- equalizing small and large entity exposure;
- diagnosing whether a model merges or fragments findings.

The current v123 result is a quick/global evaluation. Its corrected full
instance and distance metrics are unknown.

### Recommended correction

- preserve entity IDs in the cache;
- sample entities within a finding for crop selection;
- keep the output binary per finding, but add component-aware diagnostics and
  optional separation/coverage losses;
- run the official evaluator before calling a checkpoint challenge-ready.

## 15. Prompt and Annotation Provenance Are Heterogeneous

The released prompts are not raw report sentences. The data pipeline:

1. machine-translated reports from Turkish to English;
2. rewrote reports with GPT-4;
3. extracted and standardized findings with GPT-4;
4. categorized findings with GPT-4o-mini;
5. annotated masks under two training protocols.

The current ReX paper reports:

- descriptor omissions of `0.14` in rewriting and `0.13` in extraction;
- false-negative rates of `0.05` and `0.01`;
- subcategory F1 of `0.92` on a reviewed category-1/2 sample.

Training annotation provenance is also split:

- 1,400 protocol-1 cases: professional annotators with radiologist refinement;
- 1,592 protocol-2 cases: trained medical students supervised by radiologists.

The older MLHC paper reports somewhat different extraction and category
statistics because it describes an earlier 1,914-case dataset version. Those
figures should not be silently transferred to the current release.

The data-construction quality-control stage also excluded:

| Exclusion reason | Findings |
| --- | ---: |
| outside lung/pleura scope | 1,047 |
| not localizable in CT | 449 |
| too diffuse to segment feasibly | 367 |
| normal/benign variation | 121 |

Thus the released finding list is a curated set of segmentable task targets,
not an exhaustive inventory of everything visible in the scan. This is another
reason that "not in the challenge findings" is not equivalent to "visually
absent."

### Recommended correction

- keep protocol and prompt provenance in every training event;
- audit performance by protocol on a train holdout;
- use robust or lower-weight supervision for generated negative labels;
- manually review rare-category counterfactual vocabularies;
- do not train excluded report findings as positives without new masks.

## 16. Temporal Prompts Contain Unavailable Information

An exact-text audit found:

| Prompt feature | Train findings | Share |
| --- | ---: | ---: |
| temporal/comparison wording | 553 | 7.2% |
| bilateral/both-lung wording | 2,986 | 38.8% |
| explicit physical measurement | 856 | 11.1% |
| uncertainty qualifier | 828 | 10.8% |

Fourteen MICCAI val findings contain temporal/comparison wording.

Examples include "newly revealed," "progression," "regression," "stable," and
"compared with the previous examination." The current model receives only the
current CT, so it cannot verify temporal change. It can still localize the
present abnormality, but the temporal clause is not visually identifiable from
one time point.

### Recommended correction

Treat temporal clauses as spatially invariant modifiers:

- pair the full prompt with a version that removes only the temporal clause;
- require prediction consistency;
- retain the exact full prompt at challenge inference;
- do not require a prior scan unless the challenge guarantees one.

This is a safer use of prompt editing than teaching the model to invent
longitudinal reasoning from one volume.

## 17. Universal Z-Score Leaves CT-Specific HU Information Unused

All challenge images are non-contrast chest CT, while VoxTell's preprocessing
was designed for a universal CT/MRI/PET model. The current path:

1. crops to the nonzero image extent;
2. computes one mean and standard deviation over that image;
3. z-scores every voxel;
4. performs no CT clipping, fixed HU normalization, or lung/mediastinal
   windowing.

The downloaded NIfTI files retain HU-like `int16` values. Local examples
include normal air and soft-tissue ranges, but also padding or acquisition
outliers such as `-8192` and maxima above `16000`. Unclipped per-volume z-score
therefore has two CT-specific limitations:

- the network is not given the mean and standard deviation needed to recover
  absolute HU calibration;
- outliers and differences in field of view can change the mapping from HU to
  normalized intensity between cases.

This is an opportunity rather than a proven error. VoxTell v1.1 was pretrained
with this z-score path, so replacing it with raw HU only at inference would
create a serious distribution shift. The current baseline was correct to keep
pretrained normalization while testing the fine-tuning machinery.

For a challenge-specific model, however, attenuation is meaningful for
emphysema, ground-glass opacity, consolidation, fluid, and calcification. Two
training prompts even state explicit HU measurements. Relative contrast alone
may be unnecessarily weak for this CT-only task.

### Recommended correction

Run this only after ranks 2-6 are repaired:

1. baseline: current whole-volume z-score;
2. clip to a broad CT range, then z-score;
3. fixed train-set CT normalization after clipping;
4. lung-window and mediastinal-window auxiliary channels;
5. optional mean/std or spacing metadata supplied to the network.

Do not test unbounded raw HU first, and do not change spacing and intensity
normalization in the same ablation. Full rationale and inspected value ranges
are recorded in
[`docs/voxtell/normalization.md`](../voxtell/normalization.md).

## 18. Whole-Lung Hard Clipping Would Be Incorrect

The task includes lung, airway, and pleural findings. Valid masks can occupy:

- pleural space or pleural surface;
- a pneumothorax region outside aerated lung parenchyma;
- subpleural tissue crossing an imperfect lung boundary;
- central airway or hilar regions;
- tissue adjacent to the diaphragm or mediastinum.

Therefore, "inside a lung parenchyma mask" is not a universal validity rule.

Use category-aware soft priors:

- parenchymal category: dilated lung or lobe mask;
- pleural effusion/thickening: pleural shell and dependent thoracic space;
- pneumothorax: pleural-space prior, not an eroded lung mask;
- bronchial category: airway-centered region;
- paramediastinal/subpleural descriptors: preserve boundary neighborhoods.

## 19. Lowercasing Makes Raw-Text Compliance Non-Literal

The Main track requires the exact finding to be consumed directly and says the
raw prompt is provided as-is. The current semantic path is direct, but it is
not byte-for-byte exact:

- `precompute_text_embeddings.py` converts each prompt to lowercase before
  tokenization;
- upstream `VoxTellPredictor.embed_text_prompts` also lowercases prompts before
  bank lookup or on-the-fly embedding.

No prompt in the released splits collides with another finding from the same
case solely because of lowercasing, and case normalization does not remove any
medical words. This is not evidence that the method would be disqualified.
It is an avoidable ambiguity under the challenge's unusually explicit "exact"
and "as-is" wording.

### Recommended correction

1. Keep an exact original prompt string for tokenization and reporting.
2. If a normalized cache key is needed, store it separately from the text sent
   to Qwen.
3. Regenerate a raw-case embedding bank and compare it with the lowercase bank
   on a fixed subset.
4. Ask the organizers to confirm whether case-only normalization is accepted
   for the Main track.
5. Record any normalization in the submission method description.

The relevant implementations are
[`precompute_text_embeddings.py`](../../scripts/rexgroundingct/precompute_text_embeddings.py)
and upstream
[`predictor.py`](../../external/VoxTell/voxtell/inference/predictor.py).

## 20. Split, Taxonomy, and Metric Comparisons Can Be Misleading

There are three distinct benchmark contexts:

| Context | Train/val/test | Important detail |
| --- | --- | --- |
| MLHC 2025 study | 1,614 / 100 / 200 | older dataset version |
| base ReX release | 2,992 / 50 / 100 | base leaderboard and VoxTell paper |
| MICCAI 2026 | 2,992 / 200 / 300 | current challenge |

Additional traps:

- the older MLHC text labels its category groups in the opposite 1/2 order
  from the current MICCAI codes;
- VoxTell reports HIT5, while the challenge reports hit at Dice `>= 0.1`;
- base leaderboard scores use the 100-case base test, not the 300-case MICCAI
  test;
- the live MICCAI leaderboard evaluates a test subset that is not local
  val200;
- mean Dice per case is secondary; the challenge ranking metric is mean Dice
  per finding.

Every report should state the JSON split file, case count, finding count, Dice
aggregation, hit threshold, and whether the evaluator was quick/global or
official/full.

## 21. Runtime and Repository State Have Drifted

The repository config still says `implemented_pending_launch`, although the
runtime contains completed 100-epoch runs and val200 results. The runtime also
contains many val20 threshold, ensemble, and multi-scale evaluations, plus
epoch-100 val200 ensemble threshold sweeps.

This does not change model accuracy, but it weakens provenance and makes it
easy to audit the planned pipeline instead of the pipeline that actually ran.

The experiment closeout should record:

- completed checkpoint and hash;
- exact schedule and hash;
- current canonical val200 metric;
- thresholds and subsets used for selection;
- whether full official metrics were run;
- deviations from the repository config.

## Items That Are Currently Aligned

The audit did not find a reason to discard the entire pipeline. These parts
should be preserved:

1. The complete free-text content is embedded directly rather than replaced
   by a fixed category; literal case preservation remains to be resolved.
2. Qwen3-Embedding-4B is frozen and precomputed, matching VoxTell's published
   strategy.
3. The instruction wrapper and last-token pooling match upstream VoxTell.
4. Target orientation is corrected before training and predictions are
   exported in ReX `(F,X,Y,Z)` layout.
5. Positive target channels follow numeric finding-key order.
6. Dice plus BCE and deep supervision are broadly consistent with VoxTell.
7. The canonical quick metric is mean global Dice per finding, which is the
   challenge ranking aggregation.
8. Full val200 inference evaluates every supplied finding, not only a category
   label.
9. Per-volume z-score is VoxTell-native preprocessing. The MLHC baseline paper
   explicitly allowed model-native preprocessing. HU/window normalization
   remains a useful ablation, but it is lower priority than the confirmed
   supervision and anatomy problems.

## Recommended Next Pipeline

### Stage 0: Freeze the current baseline

- preserve v123 epoch 100 and its `0.283343` quick/global result;
- mark all-200 metrics as mixed-provenance rather than clean validation;
- make the MICCAI-added 150 result (`0.289403`) the primary v1.1-descendant
  diagnostic;
- retire the current val20 from model selection because 6 of 20 cases belong
  to the likely seen base val50; replace it with a stratified subset drawn
  only from the added 150;
- run v1.0 as a strict-split checkpoint control;
- record the schedule hash and exact threshold;
- compare one-prompt and grouped-prompt inference on a fixed subset;
- run the official full evaluator once to establish instance and distance
  metrics;
- do not use val200 repeatedly for broad hyperparameter search.

### Stage 1: Repair supervision and sampling, no architecture change

Implement together only where logically required:

1. finding-uniform, coverage-first sampling;
2. no two-positive shared-patch requirement;
3. partial-label-aware positive loss;
4. evidence-tiered negatives;
5. same-prompt spatial-empty events;
6. a fixed prompt-context policy: one-prompt training or variable realistic
   groups with an invariance objective;
7. nnU-Net augmentation without left-right mirroring;
8. four-GPU DDP with an effective batch of at least 16.

This stage tests whether the current architecture can improve once the
training signal matches full-volume use.

### Stage 2: Add minimal anatomy, scale, and CT intensity

Run controlled ablations:

1. full-volume coordinate channels;
2. category auxiliary plus exact text;
3. lobe/lung/airway/pleural masks;
4. spacing-aware input or standard-spacing preprocessing;
5. low-resolution whole-volume locator;
6. clipped or fixed-statistics CT normalization as a separate ablation.

Coordinate channels should be tested before a much larger new architecture.

### Stage 3: Add counterfactual prompt pairs

Use:

- known-positive location swaps;
- partial-negative side/lobe swaps;
- pathology swaps in the same anatomy;
- paraphrase invariance;
- temporal-clause invariance.

Report both segmentation metrics and response-to-edit metrics.

### Stage 4: Consider extended instance data

The challenge permits external data with disclosure. Reproduce the useful
part of the VoxTell paper's extended pool:

- lung lesion datasets;
- anatomy-derived lobe localization prompts;
- airway and pleural anatomy;
- carefully harmonized text variants.

This is a later step because the current ReX supervision bugs would otherwise
also affect external data.

## Minimal Ablation Matrix

| ID | Change from previous row | Main question |
| --- | --- | --- |
| A0 | current v123 | frozen reference |
| C0 | split existing metrics into base50 and added150 | what is the cleanest available estimate? |
| C1 | strict-split v1.0 control | how much depends on v1.1 provenance? |
| D0 | one-prompt versus grouped inference, no training | does query context change masks? |
| A1 | finding-uniform full-coverage schedule | does unseen-target coverage matter? |
| A2 | partial-label loss plus safe negatives | does label correctness improve recall? |
| A3 | mixed foreground and same-prompt-empty patches | does wrong-location FP decrease? |
| A4 | prompt-context policy and invariance | does query-set shift matter? |
| A5 | standard augmentation and larger effective batch | was adaptation underpowered? |
| A6 | global coordinates plus category auxiliary | does anatomical compliance improve? |
| A7 | lobe/airway/pleural priors | do anatomy-specific errors improve? |
| A8 | spacing-aware or resampled input | do size and scale prompts improve? |
| A9 | CT-specific normalization | does absolute attenuation help? |
| A10 | counterfactual pair losses | does the model respond correctly to edits? |
| A11 | extended instance-focused data | how much does anatomy-rich external data add? |

Do not combine A1-A4 with A6-A11 in the first run. The ablation should identify
whether the main gain comes from correcting labels, increasing context, or
adding data.

## Go/No-Go Gates

Before a new full 100-epoch run:

- the development split excludes known or likely checkpoint-training overlap;
- schedule covers all 7,687 findings;
- zero cases are rejected by two-positive geometry;
- every global empty target has evidence provenance;
- partial labels have an ignore policy;
- same-prompt foreground and empty-window rates are logged;
- positive crops log component coverage rather than only one-voxel presence;
- known positives cannot silently become supervised empty targets at coarse
  decoder scales;
- prompt-count and positive/negative query composition are logged;
- the alone-versus-group diagnostic has fixed the training/inference policy;
- development evaluation covers all available categories;
- orientation and shape tests still pass.

Before claiming improvement:

- primary metric is mean Dice per finding on locked val data;
- checkpoint-seen and checkpoint-unseen strata are never silently pooled;
- category and focality breakdowns are reported;
- prompt-edit anatomy tests improve, not only aggregate Dice;
- official instance and distance metrics do not collapse;
- threshold was selected without repeatedly tuning on the final val200 result.

Before challenge submission:

- Qwen consumes every exact raw test prompt, with any cache-normalization key
  kept separate;
- any category parser supplements rather than replaces the prompt;
- anatomy constraints are automatic and category-aware;
- 300 output files have the required `(F,H,W,D)` layout;
- external data and pretrained tools are disclosed.

## Quantitative Audit Appendix

### Materialized v123 schedule

| Measure | Value |
| --- | ---: |
| events | 10,000 |
| prompt slots | 30,000 |
| positive slots | 16,260 |
| negative slots | 13,740 |
| multi-finding events | 6,260 |
| one-finding events | 3,740 |
| positive-crop success events | 10,000 |
| unique sampled cases | 2,746 |
| unique positive findings | 5,977 |

### Training prompt composition

| Feature | Count | Percent of 7,687 |
| --- | ---: | ---: |
| temporal/comparison | 553 | 7.2% |
| bilateral/both lungs | 2,986 | 38.8% |
| explicit mm/cm measurement | 856 | 11.1% |
| uncertainty qualifier | 828 | 10.8% |
| contains "and" | 798 | 10.4% |

These are text-pattern counts, not clinically normalized concept counts.

### Auxiliary data availability

| Resource | Train coverage | Current use |
| --- | ---: | --- |
| exact findings and masks | 2,992 cases | yes |
| full reports | 2,992 cases | no |
| anatomical CoT | 7,687 findings | no |
| CoT with at least one box | 5,784 findings | no |
| official category | every train/val/test prompt | visualization only |
| entity IDs | every released train/val mask | discarded by binarization |
| voxel spacing | every CT header | not provided to network |

## Source Notes

### Primary current sources

- [MICCAI challenge page](https://rexrank.ai/ReXGroundingCT/challenge.html):
  tracks, split annotation policies, allowed signals, ranking metric, hit and
  instance thresholds.
- [Hugging Face dataset card](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT):
  released JSON schema, entity-mask semantics, MICCAI split update, and full
  report availability.
- [ReXGroundingCT paper v2](https://arxiv.org/html/2507.22030v2):
  Sections 2.3-2.5, 3, and 5 for extraction quality, annotation protocols,
  partial instances, CoT quality, spacing, and limitations.

### Model and baseline sources

- [VoxTell model card](https://huggingface.co/mrokuss/VoxTell):
  v1.0/v1.1 provenance, v1.1 corpus expansion, test-set inclusion, and
  semantic/instance sampling mixture.
- [VoxTell paper](https://arxiv.org/abs/2511.11450):
  Appendix A for training strategy and Appendix B.3 for the extended
  instance-focused dataset.
- [MLHC baseline paper](https://proceedings.mlr.press/v298/baharoon25a.html):
  older ReX version, CT preprocessing, SAT/BiomedParse/SegVol/MedSAM2
  fine-tuning details, and focal versus non-focal findings.

### Local source copies

- [`literature/challenge_paper.pdf`](../../literature/challenge_paper.pdf)
- [`literature/challenge_motivation_paper.pdf`](../../literature/challenge_motivation_paper.pdf)
- [`literature/voxeltell.pdf`](../../literature/voxeltell.pdf)
- [`challenge_info/SUMMARY.md`](../../challenge_info/SUMMARY.md)
- [`docs/voxtell/preprocessing.md`](../voxtell/preprocessing.md)
- [`docs/voxtell/normalization.md`](../voxtell/normalization.md)

## Final Decision

The current v123 checkpoint should remain the baseline, but the current
all-200 metric must be labeled mixed-provenance. The first action is to make
the MICCAI-added 150 cases the primary local evaluation stratum and run a v1.0
strict-split control. The current fine-tuning schedule should not be the
template for the next full experiment.

The next implementation target should be:

> a finding-uniform, full-coverage, partial-label-aware VoxTell fine-tuning
> pipeline with evidence-tiered negatives and a controlled mixture of
> foreground and same-prompt spatial-empty patches, trained and evaluated
> under a fixed prompt-context policy.

Only after that baseline is stable should global coordinates, anatomy priors,
counterfactual pairs, spacing changes, and extended external data be added.
This order gives the cleanest answer to whether the current model lacks
capacity or has simply been trained with a signal that does not match the
challenge.
