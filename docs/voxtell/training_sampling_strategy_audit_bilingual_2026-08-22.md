---
created: 2026-08-22
updated: 2026-08-22
status: analysis_complete
scope: VoxTell ReXGroundingCT training sampling strategy
---

# VoxTell Training Sampling Strategy Audit
# VoxTell 训练采样策略审计

**Date / 日期:** 2026-08-22
**Scope / 范围:** Current ReXGroundingCT text-conditioned VoxTell training path, with emphasis on Exp017 and the shared sampler.
**Unit of analysis / 分析单位:** One CT case and its findings. A reliable patient identifier is not available. / 一个 CT 病例及其 findings；当前没有可靠的 patient ID。

## 1. Executive conclusion / 总结结论

### 中文

当前 Exp017 的实际训练采样并不是完全随机的。它使用
`--require-positive-crop`，因此每个训练 patch 都必须包含所选阳性 finding
的至少一个 foreground voxel。最新的 40,000-event schedule 中，40,000 个
事件都成功生成了阳性 crop。

但是，当前设计仍然存在四个重要问题：

1. “阳性”只要求至少一个 voxel，监督强度可能太弱；没有控制病灶覆盖率。
2. 一个 event 最多使用两个阳性 finding；更多 finding 只在不同 event 中分批出现，且没有严格的每-finding quota。
3. 192 个多-finding 病例因为找不到可共同放入一个 patch 的阳性 pair，被 positive-crop schedule 排除。
4. “负样本”主要是负提示词的全零 target，不是独立采样的负空间 patch；它依赖病例标注完整性的假设。

最值得优先改进的是：保留阳性强制采样，但增加阳性目标覆盖率要求；为每个 finding 建立采样 quota；无法共同放入 patch 的 finding 分开采样而不是丢弃病例；再增加少量经过解剖和 GT 距离约束的上下文负样本。

### English

The current Exp017 path is not fully random. It uses
`--require-positive-crop`, so every training patch must contain at least one
foreground voxel from each selected positive finding. In the latest 40,000-event
schedule, all 40,000 events produced a successful positive crop.

However, four important weaknesses remain:

1. “Positive” means only one voxel is present; target coverage is not controlled.
2. One event contains at most two positive findings. Additional findings appear across different events, without a strict per-finding quota.
3. 192 multi-finding cases were excluded from the positive-crop schedule because no compatible positive pair could fit into one patch.
4. “Negative samples” are mainly negative prompts with synthetic empty targets, not independently sampled negative spatial patches; this relies on annotation completeness.

The highest-value improvements are to retain forced positive sampling while adding meaningful target-coverage requirements, give every finding an explicit sampling quota, split incompatible findings into separate events instead of dropping cases, and add a small amount of anatomy- and GT-distance-controlled contextual negative sampling.

## 2. Evidence and implementation locations / 证据与代码位置

### 中文

- 共享 sampler：[`train_text_conditioned_voxtell.py`](../../scripts/rexgroundingct/train_text_conditioned_voxtell.py:313)
- Exp017 schedule generation：[`run_017_iso07_hu_ddp_bs4.sh`](../../scripts/rexgroundingct/run_017_iso07_hu_ddp_bs4.sh:175)
- Positive-crop schedule construction：[`prepare_training_schedule.py`](../../scripts/rexgroundingct/prepare_training_schedule.py:132)
- Negative prompt pool：[`train_text_conditioned_voxtell.py`](../../scripts/rexgroundingct/train_text_conditioned_voxtell.py:867)
- Loss 对 empty target 的处理：[`train_text_conditioned_voxtell.py`](../../scripts/rexgroundingct/train_text_conditioned_voxtell.py:982)
- 0.7 mm lung-bbox audit：[`Experiment 019 report`](../../experiments/019_voxtell_iso07_lung_bbox_coverage_audit/report.md)

Exp017 schedule manifest 记录了：

- `24,803` 个多-finding events：2 positive + 1 negative；
- `15,197` 个单-finding events：1 positive + 2 negative；
- `64,803` 个 positive prompt slots；
- `55,197` 个 negative prompt slots；
- `40,000/40,000` 个 positive-crop successes；
- `1,744` 个多-finding病例有兼容 positive pair；
- `192` 个多-finding病例没有兼容 positive pair。

### English

- Shared sampler: [`train_text_conditioned_voxtell.py`](../../scripts/rexgroundingct/train_text_conditioned_voxtell.py:313)
- Exp017 schedule generation: [`run_017_iso07_hu_ddp_bs4.sh`](../../scripts/rexgroundingct/run_017_iso07_hu_ddp_bs4.sh:175)
- Positive-crop schedule construction: [`prepare_training_schedule.py`](../../scripts/rexgroundingct/prepare_training_schedule.py:132)
- Negative prompt pool: [`train_text_conditioned_voxtell.py`](../../scripts/rexgroundingct/train_text_conditioned_voxtell.py:867)
- Empty-target loss handling: [`train_text_conditioned_voxtell.py`](../../scripts/rexgroundingct/train_text_conditioned_voxtell.py:982)
- 0.7 mm lung-bbox audit: [`Experiment 019 report`](../../experiments/019_voxtell_iso07_lung_bbox_coverage_audit/report.md)

The Exp017 schedule manifest records:

- `24,803` multi-finding events: 2 positive + 1 negative;
- `15,197` one-finding events: 1 positive + 2 negatives;
- `64,803` positive prompt slots;
- `55,197` negative prompt slots;
- `40,000/40,000` positive-crop successes;
- `1,744` multi-finding cases with at least one compatible positive pair;
- `192` multi-finding cases with no compatible positive pair.

## 3. Current spatial sampling strategy / 当前空间采样策略

### 3.1 How a positive patch is forced / 如何强制得到阳性 patch

### 中文

对于每个被选中的阳性 target，sampler 会：

1. 读取该 target mask 中所有 foreground voxel 的坐标；
2. 随机选取一个 foreground voxel；
3. 计算能够包含该 voxel 的 patch 起点范围；
4. 在合法起点范围内随机选择 patch start；
5. 提取 `192×192×192` patch；
6. 在训练时再次检查阳性 target 是否为空。

对于坐标 `p`、图像尺寸 `D`、patch 尺寸 `P`，某一维的合法起点大致是：

```text
start ∈ [max(0, p - P + 1), min(p, D - P)]
```

因此，它保证所选 voxel 位于 patch 内。

当前的 `min_positive_voxels=1`，所以只要 patch 中有一个阳性 voxel，检查就通过。

### English

For each selected positive target, the sampler:

1. finds all foreground voxel coordinates in the target mask;
2. selects a foreground voxel;
3. computes the valid patch-start interval containing that voxel;
4. samples a patch start from that interval;
5. extracts a `192×192×192` patch;
6. verifies again during training that the positive target is nonempty.

For coordinate `p`, image dimension `D`, and patch size `P`, the valid start interval along one axis is approximately:

```text
start ∈ [max(0, p - P + 1), min(p, D - P)]
```

This guarantees that the selected voxel lies inside the patch.

The current `min_positive_voxels=1`, so one positive voxel is sufficient for the check to pass.

### 3.2 Is this too weak? / 这个条件是否太弱？

### 中文

是的，作为最低条件它偏弱。它保证了“patch 不是完全空的”，但没有保证“patch 对病灶提供了足够的监督”。例如一个有 10,000 个 voxel 的病灶，如果 patch 只包含其中 2 个 voxel，当前逻辑仍然认为它是有效阳性 patch。

当前逻辑没有保证：

- 整个病灶位于 patch 内；
- patch 包含病灶的最小比例；
- 病灶位于 patch 中心；
- 病灶有足够多的 foreground voxel；
- 多个阳性 finding 都被充分覆盖。

因此，当前采样解决了“完全采不到阳性病灶”的问题，但没有解决“阳性病灶只被擦到一点”的问题。

### English

Yes. As a minimum condition, it is weak. It guarantees that the patch is not completely empty, but not that the patch provides sufficient supervision. For example, a 10,000-voxel lesion with only 2 voxels inside the patch still passes the current rule.

The current rule does not guarantee:

- that the entire lesion is inside the patch;
- a minimum fraction of the target is covered;
- that the lesion is near the patch center;
- a sufficient number of foreground voxels;
- that multiple positive findings are substantially covered.

The current sampler therefore solves “the patch contains no positive lesion at all,” but not “the patch contains too little of the positive lesion.”

## 4. Prompt composition / Prompt 组成方式

### 中文

当前 trainer 强制每个 event 有三个 prompt slot，但这不是模型本身的硬限制。

### 当前 policy

| 病例情况 | 当前 event |
| --- | --- |
| 1 个 finding | 1 个真实阳性 + 2 个负提示词 |
| ≥2 个 finding | 2 个真实阳性 + 1 个负提示词 |
| >2 个 finding | 每次只选择其中 2 个阳性，其他 finding 不参加该 event 的 loss |

对于单-finding 病例，不使用两个相同的阳性 prompt。重复同一个 prompt 只会重复同一个 target，通常不能提供新的空间信息；更合理的是对同一个 finding 生成多个不同空间 crop。

对于多-finding 病例，当前 schedule 选择能够共同放入一个 patch 的阳性 pair。不能共同放入的 finding 不会出现在同一个 event 中。

### English

The current trainer requires three prompt slots per event, but this is not a hard model limitation.

| Case condition | Current event |
| --- | --- |
| 1 finding | 1 real positive + 2 negative prompts |
| ≥2 findings | 2 real positives + 1 negative prompt |
| >2 findings | only 2 positives are selected per event; other findings receive no loss in that event |

For one-finding cases, the same positive prompt is not duplicated twice. Duplicating it would repeat the same target and usually add no new spatial information; multiple spatial crops of the same finding would be more useful.

For multi-finding cases, the schedule selects a positive pair that can fit in one patch. Findings that cannot fit together do not appear in the same event.

### Model versus trainer / 模型与 trainer 的区别

### 中文

VoxTell 模型本身会根据 `num_prompts` 动态循环并生成多个输出，因此理论上可以接受 1、2、3 或更多 prompt。当前限制来自我们自己的 trainer：`sample_from_event` 要求恰好 3 个 slot。

### English

The VoxTell model itself loops over `num_prompts` and can theoretically process 1, 2, 3, or more prompts. The exact-three-slot restriction comes from our trainer: `sample_from_event` requires exactly 3 slots.

## 5. What the negative sample really is / “负样本”实际是什么

### 中文

一次训练 event 只选择一个 CT case。该 case 的 image、target mask 和所有 prompt slot 都属于同一个 case。

负提示词从整个训练集建立的 global prompt pool 中选择，通常来自其他病例的文本。可是，图像不会从其他病例取出；它仍然是当前 case 的 patch。

例如：

```text
病例 A 的图像 patch
阳性提示词：A1
阳性提示词：A2
负提示词：B
```

这里 B 可能是病例 B 中出现过的文本，但 B 的 target 在病例 A 中被直接构造为全零 mask。

因此当前负样本是：

```text
当前病例的图像
+ 当前病例的空间 patch
+ 来自全局 prompt pool 的其他文本
+ 该文本对应的合成全零 target
```

它不是“从另一个病例取一块 CT 作为负 patch”。

### English

Each training event selects one CT case. The image, target masks, and all prompt slots belong to that same case.

Negative prompts are drawn from a global prompt pool built from the training set, so their text often originated from another case. However, the image is never taken from that other case; it remains a patch from the current case.

For example:

```text
Image patch from case A
Positive prompt: A1
Positive prompt: A2
Negative prompt: B
```

Prompt B may have appeared in case B, but its target for case A is directly synthesized as an all-zero mask.

Thus the current negative sample is:

```text
current-case image
+ current-case spatial patch
+ another text prompt from the global pool
+ a synthetic all-zero target for that prompt
```

It is not a CT patch taken from another case.

### Negative-target assumption / 负 target 的假设

### 中文

这个设计隐含了一个假设：

```text
当前病例的标注列表中没有 prompt B
=> 病例中不存在 B
```

代码没有真正检查 prompt B 对应的异常是否在图像中不存在，而是直接创建全零 target。因此，如果存在漏标、同义改写或语义相近的 finding，就可能产生 false-negative supervision。

### English

The design assumes:

```text
prompt B is not listed for the current case
=> finding B is absent from the current case
```

The code does not verify image-level absence of B; it directly creates an all-zero target. Missing annotations, paraphrases, or semantically similar findings can therefore create false-negative supervision.

## 6. Lung-mask and outside-lung behavior / Lung mask 与肺外采样

### 中文

当前训练 sampler 不使用 TotalSegmentator lung mask 进行硬限制。

它使用的是 CT 的 nonzero body crop，所以 patch 可以包含：

- 肺实质；
- 纵隔；
- 胸壁；
- 胸膜；
- 空气或 padding。

对于纯肺实质病灶，这会提供上下文；对于胸腔积液、气胸和胸膜病灶，硬性限制到 lung mask 可能会截断真实 GT。

Exp019 lung-bbox audit 显示，未扩张肺 bbox 只包含 `336/381` 个 validation findings，即 `88.2%`；整体 P95 需要 `3` 个 0.7 mm voxel 的 isotropic expansion，最大需要 `52` 个 voxel / `36.4 mm`。

因此不建议直接使用“patch 必须完全在 lung mask 内”的规则。更安全的是将肺部信息作为 soft prior，例如使用肺 bbox 或肺 mask 的 dilation，并保留外部 margin。

### English

The current training sampler does not hard-constrain patches using the TotalSegmentator lung mask.

It uses the CT nonzero body crop, so a patch may include:

- lung parenchyma;
- mediastinum;
- chest wall;
- pleura;
- air or padding.

This provides useful context for lung findings. For pleural effusion, pneumothorax, and pleural findings, a hard lung-mask restriction could truncate the true GT.

The Exp019 lung-bbox audit found that the unexpanded lung bbox directly contained only `336/381` validation findings, or `88.2%`. The overall P95 isotropic expansion was `3` 0.7 mm voxels, and the maximum was `52` voxels / `36.4 mm`.

Therefore, a rule requiring the entire patch to lie inside the lung mask is not recommended. Lung anatomy should instead be used as a soft prior, such as a lung bbox or dilated lung mask with an external margin.

## 7. Prioritized problems / 按优先级排序的问题

### P0 — Weak positive coverage / 阳性覆盖太弱

### 中文

这是最高优先级问题。当前只要求每个阳性 target 在 patch 中非空，等价于 `min_positive_voxels=1`。应统计并约束 patch 实际覆盖的目标 voxel 数和比例。

### English

This is the highest-priority issue. The current rule only requires each selected positive target to be nonempty in the patch, equivalent to `min_positive_voxels=1`. Actual target voxel count and target-coverage fraction should be measured and constrained.

### P0 — Findings are not quota-balanced / Finding 没有 quota 平衡

### 中文

多-finding 病例每次只选择两个阳性 finding。不同 finding 被采样的次数没有严格保证；更多 finding 只是依赖后续随机 event 再次被选中。

### English

Multi-finding cases use only two positive findings per event. There is no strict guarantee that every finding is sampled equally; additional findings rely on being selected again in later random events.

### P0 — Incompatible cases are dropped / 不兼容病例被排除

### 中文

如果两个阳性 finding 无法共同放入一个 `192³` patch，当前 positive-crop schedule 会把该多-finding病例排除。这个策略避免了非法 crop，但不是理想的数据利用方式。

### English

When two positive findings cannot fit into one `192³` patch, the current positive-crop schedule excludes the multi-finding case. This avoids invalid crops but is not an ideal use of the data.

### P1 — Synthetic negative targets may be false negatives / 合成负 target 可能是 false negative

### 中文

负提示词只要不在当前病例的 prompt 列表中，就被赋予全零 target。这个假设可能受到漏标、同义描述和语义相似 finding 的影响。

### English

Any prompt not listed for the current case receives an all-zero target. This assumption can be affected by missing annotations, paraphrases, and semantically similar findings.

### P1 — No explicit anatomy-aware context policy / 缺少明确的解剖上下文策略

### 中文

当前强制阳性 patch 通常会包含病灶附近上下文，但没有明确控制这些上下文是肺内、胸膜区、胸壁还是大量 padding。

### English

Forced-positive patches usually include local context, but the sampler does not explicitly control whether that context is lung, pleura, chest wall, or mostly padding.

### P2 — Random-crop fallback is unsafe by default / 随机 fallback 默认不够安全

### 中文

trainer 的默认 `require_positive_crop=False`。如果生产 launcher 忘记传该参数，多-finding事件可能退化为约 85% 阳性锚定和约 15% 完全随机 crop；非 anchor 阳性 prompt 也可能在 patch 中为空。

### English

The trainer default is `require_positive_crop=False`. If a production launcher forgets the flag, multi-finding events can revert to roughly 85% anchored crops and 15% fully random crops; a non-anchor positive prompt can also be empty inside the patch.

## 8. Recommended strategy, in priority order / 建议的更好策略及优先级

### P0-A — Positive coverage-aware sampling / 有效阳性覆盖采样

### 中文

保留 `require_positive_crop=True`，但将有效性标准改为：

```text
patch 内 target voxel 数 >= max(8 或 16, full target voxel 数的 10–20%)
```

小结节使用绝对 voxel 阈值，大病灶使用覆盖比例阈值。阈值应在 train split 的目标大小分布上校准，而不是盲目固定。

### English

Keep `require_positive_crop=True`, but define a valid crop using:

```text
target voxels inside patch >= max(8 or 16, 10–20% of full target voxels)
```

Use an absolute threshold for small nodules and a coverage threshold for large lesions. Calibrate the thresholds using the train target-size distribution rather than fixing them blindly.

### P0-B — Finding-centered quota schedule / 以 finding 为中心的 quota schedule

### 中文

不要以“病例 event”为唯一采样单位，而要以 finding 为基本单位。为每个 finding 设置每 epoch 的最小采样次数。

推荐逻辑：

```text
能共同放入 patch 的两个 finding -> 双阳性 event
不能共同放入 patch -> 分别生成单阳性 event
一个病例有 5 个 finding -> 生成多个 event 覆盖所有 finding
```

这样可以同时保留双阳性训练的效率，并避免远距离 finding 被忽略。

### English

Do not use the case event as the only sampling unit; use the finding as the basic unit. Give every finding a minimum number of events per epoch.

Recommended policy:

```text
two findings that fit together -> dual-positive event
findings that do not fit together -> separate single-positive events
case with five findings -> multiple events covering all five findings
```

This retains the efficiency of dual-positive training while preventing distant findings from being under-sampled.

### P0-C — Recover incompatible multi-finding cases / 恢复不兼容多-finding病例

### 中文

不能共同放入一个 patch 时，不要排除整个病例。可以：

- 对每个 finding 生成单阳性 event；
- 使用一个阳性 + 两个 context/negative event；
- 对同一个 finding 生成核心 crop 和上下文 crop。

### English

Do not drop the whole case when findings cannot share a patch. Instead:

- generate single-positive events for each finding;
- use one positive event plus contextual/negative events;
- generate core and context crops for the same finding.

### P1 — Replace weak global negatives with local context negatives / 用局部上下文负样本补充全局负提示词

### 中文

更可靠的负空间样本是：在同一个病例、同一个阳性 prompt 下，选择一个明确不包含该 target 的 patch。

例如：

```text
病例 A 的 finding A
patch 1：包含 A -> 阳性 target
patch 2：距离 A 足够远且不包含 A -> 空 target
```

这比“从另一个病例拿一个文本，然后在当前病例中直接设置全零 mask”更有空间定位意义。

建议只占约 `10–20%` 的训练事件，并通过 GT 距离或 mask exclusion 验证其确实不含 target。

### English

A more reliable spatial negative is a patch from the same case and the same positive prompt that is known not to contain that target.

For example:

```text
case A, finding A
patch 1: contains A -> positive target
patch 2: sufficiently far from A and excludes A -> empty target
```

This has clearer spatial-localization meaning than taking a text prompt from another case and assigning it an all-zero mask in the current case.

Use such events for only about `10–20%` of training and verify target exclusion using GT distance or mask exclusion.

### P1 — Use lung anatomy as a soft prior / 使用肺部解剖作为 soft prior

### 中文

不要硬性要求 patch 完全位于肺 mask 内。可以使用：

- lung bbox + 外扩 margin；
- dilated lung mask；
- lung bbox 与胸膜/胸壁 margin 的联合区域。

Exp019 的最大验证外扩是 52 voxels，因此可以把 `52–64` voxel 作为候选 margin 的起点，再在训练集上验证。

### English

Do not require every patch to lie entirely inside the lung mask. Use instead:

- lung bbox plus an external margin;
- a dilated lung mask;
- a combined lung-bbox and pleural/chest-wall margin region.

Exp019 found a maximum validation expansion of 52 voxels, so `52–64` voxels can be an initial candidate margin, followed by train-set validation.

### P1 — Make production sampling fail closed / 让生产采样 fail closed

### 中文

生产实验应在 config 中明确声明：

```text
sampling_policy: finding_quota_positive_coverage_v1
require_positive_crop: true
```

如果没有显式采样策略，launcher 应拒绝启动，而不是静默使用 `require_positive_crop=False`。

### English

Production experiments should explicitly declare:

```text
sampling_policy: finding_quota_positive_coverage_v1
require_positive_crop: true
```

If no explicit sampling policy is present, the launcher should refuse to start rather than silently using `require_positive_crop=False`.

### P2 — Support variable prompt counts or ignored padding / 支持可变 prompt 数量或 ignored padding

### 中文

VoxTell 模型本身支持可变数量 prompt。trainer 可以改为：

- 每个 event 使用 1–4 个真实 prompt；或
- 保持固定 tensor 长度，但使用 padding prompt 并设置 `ignore_loss_mask`；
- 不要把 padding slot 当作真实负提示词。

对于单-finding 病例，这样就不需要用两个合成负提示词强行填满三个 slot。

### English

The VoxTell model itself supports variable prompt counts. The trainer could:

- use 1–4 real prompts per event; or
- keep a fixed tensor length but use padding prompts with an `ignore_loss_mask`;
- avoid treating padding slots as real negative prompts.

For one-finding cases, this removes the need to fill the remaining slots with synthetic negative prompts.

## 9. Proposed event policy / 建议的 event policy

### 中文

一个更清晰的 policy 可以是：

```text
每个 finding 每 epoch 至少采样 N 次

如果两个 finding 的目标 bbox 和 patch 尺寸兼容：
    生成双阳性 event
否则：
    为每个 finding 生成单阳性 event

单阳性 finding 生成：
    1 个核心阳性 crop
    1 个上下文阳性 crop
    0–1 个同病例局部负 crop

额外加入约 10–20% 的 anatomy-guided context events
```

### English

A clearer policy would be:

```text
sample every finding at least N times per epoch

if two finding bboxes fit within one patch:
    generate a dual-positive event
else:
    generate separate single-positive events

for a single positive finding, generate:
    one lesion-core positive crop
    one lesion-context positive crop
    zero or one same-case local-negative crop

add approximately 10–20% anatomy-guided context events
```

## 10. Required sampler audit metrics / 应记录的采样审计指标

### 中文

每个 event 建议记录：

- case name；
- finding IDs 和 categories；
- positive/negative/padding slot 类型；
- patch start 和 spatial shape；
- 每个 positive target 的 full voxel 数；
- patch 内 voxel 数；
- target coverage fraction；
- target bbox 是否完整位于 patch；
- target center 到 patch center 的距离；
- patch 与 lung mask、expanded lung bbox 的重叠；
- padding/background voxel 比例；
- 每个 finding 的累计采样次数；
- 每个 category 的累计采样次数；
- negative target 的 GT exclusion 验证结果。

### English

Record for every event:

- case name;
- finding IDs and categories;
- positive/negative/padding slot type;
- patch start and spatial shape;
- full voxel count for every positive target;
- voxel count inside the patch;
- target coverage fraction;
- whether the target bbox is fully inside the patch;
- distance from target center to patch center;
- patch overlap with the lung mask and expanded lung bbox;
- padding/background voxel fraction;
- cumulative sampling count for every finding;
- cumulative sampling count for every category;
- GT-exclusion verification for negative targets.

## 11. Recommended ablation order / 建议的 ablation 顺序

### 中文

建议不要一次同时改变所有因素。使用相同 checkpoint、seed、optimizer 和 validation protocol，按以下顺序比较：

1. **Baseline:** 当前 Exp017 策略。
2. **Positive coverage:** 增加最小 voxel 数和 coverage rule。
3. **Finding quota:** 每个 finding 固定最小采样次数，恢复无法组成 pair 的病例。
4. **Core anchor:** 使用病灶核心/中心，而不是随机 foreground voxel。
5. **Context negatives:** 加入 10–20% 同病例、GT 距离约束的局部负 crop。
6. **Soft lung prior:** 使用扩张 lung bbox 或 dilated lung region。
7. **Variable prompts / ignored padding:** 最后再评估 trainer 接口变化。

主要评估指标应包括：

- fixed val200 overall Dice/hit rate；
- 每个 category 的 Dice/hit rate；
- 小结节、大片状病灶、胸膜/气胸 finding 的分层结果；
- target coverage 和 sampler failure rate；
- 训练事件中每个 finding 的实际采样次数。

### English

Do not change all factors at once. Keep the checkpoint, seed, optimizer, and validation protocol fixed and compare in this order:

1. **Baseline:** current Exp017 strategy.
2. **Positive coverage:** add minimum voxel and coverage rules.
3. **Finding quota:** give every finding a minimum sampling count and recover incompatible cases.
4. **Core anchoring:** use lesion-core/center points instead of arbitrary random foreground voxels.
5. **Context negatives:** add 10–20% same-case, GT-distance-controlled local negative crops.
6. **Soft lung prior:** use a dilated lung bbox or dilated lung region.
7. **Variable prompts / ignored padding:** evaluate trainer-interface changes last.

Primary evaluation should include:

- fixed val200 overall Dice/hit rate;
- per-category Dice/hit rate;
- stratified results for small nodules, large/diffuse lesions, and pleural/pneumothorax findings;
- target coverage and sampler failure rate;
- actual per-finding event counts.

## 12. Final recommendation / 最终建议

### 中文

当前 Exp017 已经解决了“完全随机采样导致 patch 没有阳性病灶”的主要问题，但“至少一个 voxel”仍然是一个过弱的阳性质量标准。

建议的第一版改进不是立即把所有 patch 限制到肺 mask，而是：

1. 保留 positive-crop forcing；
2. 增加最小阳性覆盖率；
3. 为每个 finding 建立 quota；
4. 不再因 finding 距离太远而丢弃病例；
5. 用同病例的局部 GT-excluded patch 作为少量空间负样本；
6. 使用扩张肺 bbox 作为 soft anatomy prior；
7. 通过固定 val200 做逐步 ablation。

### English

Current Exp017 has largely solved the problem of completely random patches with no positive lesion, but “at least one voxel” remains too weak as a positive-quality criterion.

The first improved version should not immediately hard-restrict all patches to the lung mask. Instead:

1. retain forced positive cropping;
2. add a minimum positive-target coverage rule;
3. give every finding an explicit quota;
4. stop dropping cases because findings are too far apart;
5. use a small number of same-case GT-excluded local spatial negatives;
6. use an expanded lung bbox as a soft anatomical prior;
7. compare changes incrementally on fixed val200.
