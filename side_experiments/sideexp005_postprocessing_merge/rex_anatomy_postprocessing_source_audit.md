# ReXGrounding anatomical-mask postprocessing 源码核对记录

## 核对范围

- GitHub: HengjieLiu/rexgroundingct，commit `83b12f9155108ecb4f066cfcaa1a659522670c1e`。重点是 exp024 和导入其实现的 exp025；exp022是更早的独立审计。
- ZIP: `RexGrounding_challenge2026-c2-less-semantic-coadapt-v11-freeze.zip`。
- ZIP SHA-256: `7561acb37aa3281f863a4759a40a899d13d1181ff68c309744e8ed411d9c3c7c`。
- 以下 ZIP 摘录直接来自解压后的原文件。每行前缀是原始文件行号，未改动源文件。
- 已运行 ZIP 内置的11条纯文本 parser 测试，并对比了下面11组示例。没有运行 CT 推理或重新计算 Full200 指标。
- GitHub 示例检查使用从连接器读取后逐字摘取的 `route_prompt` 和其常量；并非在用户服务器上运行原仓库。

## 主要结论

两者都是阈值化预测与文本选择的解剖ROI取交集。ZIP v1是 exact-lobe+15mm / side+15mm / whole-lung+10mm；GitHub exp024/025提供相互独立的 whole_lung20 与 fine20，带肺内目标适用性门控；fine不适用时不变。ZIP可选v2只对白名单签名进一步硬裁剪，部署入口要求输入先做过semantic-v1。

## 纯文本示例检查

这是假设输入prompt的函数行为测试，不是实际数据集错误计数。RU/LL等是GitHub内部编码，分别对应RUL/LLL等。

| Prompt | 类别 | ZIP v1 ROI | GitHub fine ROI | GitHub whole ROI |
|---|---|---|---|---|
| nodule in the right upper lobe | 2d | RUL +15mm | RU +20mm | whole lung +20mm |
| opacity predominantly in the left lower lobe | 2b | left lung +15mm | LL +20mm | whole lung +20mm |
| subpleural nodule in the right upper lobe | 2d | right lung +15mm | RU +20mm | whole lung +20mm |
| multifocal nodules in the right upper lobe | 2d | whole lung +10mm | RU +20mm | whole lung +20mm |
| pulmonary nodule | 2d | whole lung +10mm | unchanged | whole lung +20mm |
| right pleural effusion | 2e | right lung +15mm | unchanged | unchanged |
| no opacity in the right upper lobe | 2b | whole lung +10mm | RU +20mm | whole lung +20mm |
| resolved right upper lobe opacity | 2b | whole lung +10mm | RU +20mm | whole lung +20mm |
| RUL nodule | 2d | RUL +15mm | unchanged | whole lung +20mm |
| nodule in the upper lobe of the right lung | 2d | RUL +15mm | R +20mm | whole lung +20mm |
| bilateral upper lobe opacities | 2b | whole lung +10mm | B +20mm | whole lung +20mm |

## GitHub核对的关键文件

以下链接固定到本次核对的commit。
- `scripts/rexgroundingct/run_024_test_inference_anatomy.py`: route_prompt、physical_dilation、apply_support、transform_case_dir。
- `scripts/rexgroundingct/run_025_iso07_anatomy_audit.py`: load_base与ensure_routes复用exp024。
- `configs/experiments/024_test_inference_anatomy_audit.json`: anatomy label IDs、20mm、threshold 0.5。
- `experiments/024_test_inference_anatomy_audit/aggregate_report.md`: b组validation参考结果；不是test分数。
- `experiments/025_iso07_best_anatomy_audit/report.md`: 同一iso07 checkpoint的后处理对照。
- `experiments/022_exp007_official_anatomy_val200_audit/report.md`: 更早的mask/bbox/margin探索审计。

https://github.com/HengjieLiu/rexgroundingct/blob/83b12f9155108ecb4f066cfcaa1a659522670c1e/scripts/rexgroundingct/run_024_test_inference_anatomy.py
https://github.com/HengjieLiu/rexgroundingct/blob/83b12f9155108ecb4f066cfcaa1a659522670c1e/scripts/rexgroundingct/run_025_iso07_anatomy_audit.py
https://github.com/HengjieLiu/rexgroundingct/blob/83b12f9155108ecb4f066cfcaa1a659522670c1e/experiments/024_test_inference_anatomy_audit/aggregate_report.md
https://github.com/HengjieLiu/rexgroundingct/blob/83b12f9155108ecb4f066cfcaa1a659522670c1e/experiments/025_iso07_best_anatomy_audit/report.md

## Z1 — ZIP semantic-v1 冻结推理规则

文件: `scripts/apply_semantic_constraint_best_policy.py`
文件 SHA-256: `bb8d1abd80b2d38bced428037750d40f649238b6a9c7c083d5cad1f06e34f0b4`

### 原文件 L1–L7
```text
   1: #!/usr/bin/env python3
   2: """Apply the frozen semantic exact15 / side15 / whole10 inference policy.
   3: 
   4: GT is optional and is used only after ROI selection for metric reproduction.
   5: The deployable path requires only predictions, finding text, and TotalSegmentator
   6: lung/lobe masks.
   7: """
```
### 原文件 L66–L79
```text
  66: def load_config(path: Path) -> dict[str, Any]:
  67:     config = yaml.safe_load(path.read_text(encoding="utf-8"))
  68:     expected = {
  69:         "high_conf_exact_lobe": ("exact_lobe", 15),
  70:         "ambiguous_lobe": ("side_lung", 15),
  71:         "laterality_only": ("side_lung", 15),
  72:         "bilateral_multifocal": ("whole_lung", 10),
  73:         "nonlocalizable": ("whole_lung", 10),
  74:     }
  75:     for group, (kind, dilation) in expected.items():
  76:         value = config["groups"][group]
  77:         if value["roi"] != kind or int(value["dilation_mm"]) != dilation:
  78:             raise ValueError(f"Frozen policy mismatch for {group}: {value}")
  79:     return config
```
### 原文件 L93–L184
```text
  93: def process_case(task: tuple[str, list[dict[str, Any]], dict[str, Any]]) -> tuple[list[dict], str | None]:
  94:     case, findings, settings = task
  95:     try:
  96:         pred_path = Path(settings["pred_dir"]) / case
  97:         pred, pred_ref = channel_first_mask(
  98:             pred_path, float(settings["threshold"]), bool(settings["prediction_is_probability"])
  99:         )
 100:         ref = geometry_reference(case, pred_ref, Path(settings["ct_dir"]))
 101:         priors = load_priors(case, Path(settings["prior_dir"]), ref)
 102:         sampling = tuple(float(x) for x in nib.affines.voxel_sizes(ref.affine)[:3])
 103:         base_cache: dict[str, np.ndarray] = {
 104:             "whole": priors["whole"],
 105:             "left": priors["left"],
 106:             "right": priors["right"],
 107:         }
 108:         distance_cache: dict[str, np.ndarray] = {}
 109:         roi_cache: dict[tuple[str, int], np.ndarray] = {}
 110: 
 111:         def base(key: str) -> np.ndarray:
 112:             if key not in base_cache:
 113:                 labels = key.removeprefix("lobes:").split("+")
 114:                 base_cache[key] = np.logical_or.reduce([priors[label] for label in labels])
 115:             return base_cache[key]
 116: 
 117:         def dilated(key: str, mm: int) -> np.ndarray:
 118:             cache_key = (key, mm)
 119:             if cache_key not in roi_cache:
 120:                 mask = base(key)
 121:                 if key not in distance_cache:
 122:                     distance_cache[key] = ndimage.distance_transform_edt(~mask, sampling=sampling)
 123:                 roi_cache[cache_key] = mask | (distance_cache[key] <= float(mm))
 124:             return roi_cache[cache_key]
 125: 
 126:         def select_roi(parsed: dict[str, Any]) -> tuple[np.ndarray, str, str, int]:
 127:             group = parsed["group"]
 128:             if group == "high_conf_exact_lobe" and parsed["lobes"]:
 129:                 key = "lobes:" + "+".join(sorted(parsed["lobes"]))
 130:                 return dilated(key, 15), "exact_lobe", "+".join(parsed["lobes"]), 15
 131:             if group in {"high_conf_exact_lobe", "ambiguous_lobe", "laterality_only"}:
 132:                 side = parsed["side"]
 133:                 if side in {"left", "right"}:
 134:                     return dilated(side, 15), "side_lung", side, 15
 135:             return dilated("whole", 10), "whole_lung", "whole", 10
 136: 
 137:         gt = None
 138:         if settings.get("gt_dir"):
 139:             gt, _ = channel_first_mask(Path(settings["gt_dir"]) / case, 0.5, False)
 140:             if gt.shape != pred.shape:
 141:                 raise ValueError(f"GT {gt.shape} != prediction {pred.shape}")
 142:         constrained = np.zeros_like(pred, dtype=np.uint8)
 143:         vv = float(np.prod(nib.affines.voxel_sizes(ref.affine)[:3]))
 144:         rows: list[dict[str, Any]] = []
 145:         for finding in findings:
 146:             idx = int(finding["finding_idx"])
 147:             parsed = semantic_parse(str(finding["prompt"]))
 148:             roi, roi_kind, roi_label, dilation = select_roi(parsed)
 149:             output = pred[idx] & roi
 150:             constrained[idx] = output
 151:             row = {
 152:                 "case_id": case,
 153:                 "finding_id": idx,
 154:                 "finding_text": finding["prompt"],
 155:                 "group": parsed["group"],
 156:                 "side": parsed["side"],
 157:                 "lobes": ";".join(parsed["lobes"]),
 158:                 "selected_roi_kind": roi_kind,
 159:                 "selected_roi_label": roi_label,
 160:                 "dilation_mm": dilation,
 161:                 "pred_volume_mm3": int(output.sum()) * vv,
 162:             }
 163:             if gt is not None:
 164:                 gt_mask = gt[idx]
 165:                 denom = int(output.sum()) + int(gt_mask.sum())
 166:                 dice = 1.0 if denom == 0 else 2.0 * int((output & gt_mask).sum()) / denom
 167:                 row.update({"dice": dice, "hit": dice >= HIT_THRESHOLD})
 168:             rows.append(row)
 169: 
 170:         output_dir = settings.get("output_dir")
 171:         if output_dir:
 172:             output_path = Path(output_dir) / case
 173:             output_path.parent.mkdir(parents=True, exist_ok=True)
 174:             original = nib.load(str(pred_path))
 175:             data: np.ndarray
 176:             if len(original.shape) == 3:
 177:                 data = constrained[0]
 178:             elif original.shape[0] <= 64:
 179:                 data = constrained
 180:             else:
 181:                 data = np.moveaxis(constrained, 0, -1)
 182:             header = original.header.copy()
 183:             header.set_data_dtype(np.uint8)
 184:             nib.save(nib.Nifti1Image(data.astype(np.uint8), original.affine, header), str(output_path))
```

## Z2 — ZIP semantic-v1 文本解析

文件: `scripts/evaluate_semantic_aware_lobe_constraint_part1.py`
文件 SHA-256: `e3495690581bb2720a9c5cf4e06f07d9bec638f7360c0d0d72fa2883139410c2`

### 原文件 L37–L133
```text
  37: EXACT_PATTERNS = {
  38:     "RUL": (
  39:         r"\bright upper lobe\b",
  40:         r"\bupper lobe of (?:the )?right lung\b",
  41:         r"\bRUL\b",
  42:     ),
  43:     "RML": (
  44:         r"\bright middle lobe\b",
  45:         r"\bmiddle lobe of (?:the )?right lung\b",
  46:         r"\bRML\b",
  47:     ),
  48:     "RLL": (
  49:         r"\bright lower lobe\b",
  50:         r"\blower lobe of (?:the )?right lung\b",
  51:         r"\bRLL\b",
  52:     ),
  53:     "LUL": (
  54:         r"\bleft upper lobe\b",
  55:         r"\bupper lobe of (?:the )?left lung\b",
  56:         r"\bLUL\b",
  57:     ),
  58:     "LLL": (
  59:         r"\bleft lower lobe\b",
  60:         r"\blower lobe of (?:the )?left lung\b",
  61:         r"\bLLL\b",
  62:     ),
  63: }
  64: 
  65: AMBIGUOUS_PATTERNS = (
  66:     r"\bpredominant(?:ly)?\b",
  67:     r"\bmainly\b",
  68:     r"\bmostly\b",
  69:     r"\bcentered in\b",
  70:     r"\bgreatest in\b",
  71:     r"\bmost pronounced(?: in)?\b",
  72:     r"\bextending into\b",
  73:     r"\binvolving\b",
  74:     r"\badjacent(?: to)?\b",
  75:     r"\bbasilar\b",
  76:     r"\bbibasilar\b",
  77:     r"\blower lung\b",
  78:     r"\bupper lung\b",
  79:     r"\bperi[- ]?hilar\b",
  80:     r"\binfra[- ]?hilar\b",
  81:     r"\bhilar\b",
  82:     r"\bsubpleural\b",
  83:     r"\bpleural[- ]based\b",
  84:     r"\bperi[- ]?fissural\b",
  85:     r"\bfissural\b",
  86:     r"\blingula\b",
  87:     r"\blingular\b",
  88:     r"\blower lobe predominant\b",
  89:     r"\bupper lobe predominant\b",
  90: )
  91: 
  92: BILATERAL_MULTIFOCAL_PATTERNS = (
  93:     r"\bbilateral(?:ly)?\b",
  94:     r"\bboth lungs?\b",
  95:     r"\bboth (?:the )?(?:upper|lower) lobes?\b",
  96:     r"\b(?:upper|lower) lobes? of both lungs?\b",
  97:     r"\bmultifocal\b",
  98:     r"\bdiffuse(?:ly)?\b",
  99:     r"\bscattered\b",
 100:     r"\bnumerous\b",
 101:     r"\bmultiple bilateral\b",
 102:     r"\bbibasilar\b",
 103: )
 104: 
 105: SIDE_PATTERNS = {
 106:     "left": (
 107:         r"\bleft\b",
 108:         r"\blingula\b",
 109:         r"\blingular\b",
 110:         r"\bLUL\b",
 111:         r"\bLLL\b",
 112:     ),
 113:     "right": (
 114:         r"\bright\b",
 115:         r"\bRUL\b",
 116:         r"\bRML\b",
 117:         r"\bRLL\b",
 118:         r"\bmiddle lobe\b",
 119:     ),
 120: }
 121: 
 122: NEGATION_HISTORY_PATTERNS = (
 123:     r"\bno\b",
 124:     r"\bwithout\b",
 125:     r"\babsent\b",
 126:     r"\bresolved\b",
 127:     r"\bprior\b",
 128:     r"\bprevious\b",
 129:     r"\bhistory of\b",
 130:     r"\bstatus post\b",
 131:     r"\brule out\b",
 132:     r"\bexcluding\b",
 133: )
```
### 原文件 L183–L328
```text
 183: def normalize_text(value: str) -> str:
 184:     return " ".join(str(value or "").strip().split())
 185: 
 186: 
 187: def _unsafe_context(text: str, start: int, end: int, window_tokens: int = 6) -> tuple[bool, str]:
 188:     """Flag cue phrases within six tokens without crossing a strong boundary."""
 189:     low = text.lower()
 190:     left_boundary = max(low.rfind(".", 0, start), low.rfind(";", 0, start), low.rfind(":", 0, start))
 191:     right_candidates = [x for x in (low.find(".", end), low.find(";", end), low.find(":", end)) if x >= 0]
 192:     right_boundary = min(right_candidates) if right_candidates else len(low)
 193:     sentence = low[left_boundary + 1 : right_boundary]
 194:     local_start = start - left_boundary - 1
 195:     local_end = end - left_boundary - 1
 196:     tokens = list(re.finditer(r"[a-z0-9]+", sentence))
 197:     phrase_tokens = [i for i, token in enumerate(tokens) if token.start() < local_end and token.end() > local_start]
 198:     if not phrase_tokens:
 199:         return False, ""
 200:     lo = max(0, min(phrase_tokens) - window_tokens)
 201:     hi = min(len(tokens), max(phrase_tokens) + window_tokens + 1)
 202:     local = " ".join(token.group(0) for token in tokens[lo:hi])
 203:     for pattern in NEGATION_HISTORY_PATTERNS:
 204:         match = re.search(pattern, local, flags=re.I)
 205:         if match:
 206:             return True, match.group(0)
 207:     return False, ""
 208: 
 209: 
 210: def _safe_matches(text: str, patterns: tuple[str, ...]) -> tuple[list[re.Match], list[str]]:
 211:     safe: list[re.Match] = []
 212:     excluded: list[str] = []
 213:     for pattern in patterns:
 214:         for match in re.finditer(pattern, text, flags=re.I):
 215:             unsafe, cue = _unsafe_context(text, *match.span())
 216:             if unsafe:
 217:                 excluded.append(f"{match.group(0)}[{cue}]")
 218:             else:
 219:                 safe.append(match)
 220:     return safe, excluded
 221: 
 222: 
 223: def semantic_parse(prompt: str) -> dict[str, Any]:
 224:     text = normalize_text(prompt)
 225:     matched: list[str] = []
 226:     excluded: list[str] = []
 227:     exact_lobes: set[str] = set()
 228: 
 229:     for lobe, patterns in EXACT_PATTERNS.items():
 230:         safe, unsafe = _safe_matches(text, patterns)
 231:         for match in safe:
 232:             exact_lobes.add(lobe)
 233:             matched.append(match.group(0))
 234:         excluded.extend(unsafe)
 235: 
 236:     bilateral_lobe_patterns = {
 237:         ("LUL", "RUL"): (
 238:             r"\bbilateral(?:ly)? upper lobes?\b",
 239:             r"\bboth (?:the )?upper lobes?\b",
 240:             r"\bupper lobes? of both lungs?\b",
 241:         ),
 242:         ("LLL", "RLL"): (
 243:             r"\bbilateral(?:ly)? lower lobes?\b",
 244:             r"\bboth (?:the )?lower lobes?\b",
 245:             r"\blower lobes? of both lungs?\b",
 246:         ),
 247:     }
 248:     for targets, patterns in bilateral_lobe_patterns.items():
 249:         safe, unsafe = _safe_matches(text, patterns)
 250:         for match in safe:
 251:             exact_lobes.update(targets)
 252:             matched.append(match.group(0))
 253:         excluded.extend(unsafe)
 254: 
 255:     side_hits: dict[str, list[str]] = {"left": [], "right": []}
 256:     for side, patterns in SIDE_PATTERNS.items():
 257:         safe, unsafe = _safe_matches(text, patterns)
 258:         side_hits[side].extend(match.group(0) for match in safe)
 259:         excluded.extend(unsafe)
 260: 
 261:     if exact_lobes & LEFT_LOBES:
 262:         side_hits["left"].append("exact-left-lobe")
 263:     if exact_lobes & RIGHT_LOBES:
 264:         side_hits["right"].append("exact-right-lobe")
 265:     if side_hits["left"] and side_hits["right"]:
 266:         side: str | None = "bilateral"
 267:     elif side_hits["left"]:
 268:         side = "left"
 269:     elif side_hits["right"]:
 270:         side = "right"
 271:     else:
 272:         side = None
 273: 
 274:     bilateral_hits = []
 275:     for pattern in BILATERAL_MULTIFOCAL_PATTERNS:
 276:         safe, unsafe = _safe_matches(text, (pattern,))
 277:         bilateral_hits.extend(match.group(0) for match in safe)
 278:         excluded.extend(unsafe)
 279:     if side == "bilateral":
 280:         bilateral_hits.append("left-and-right-location")
 281: 
 282:     ambiguous_hits = []
 283:     for pattern in AMBIGUOUS_PATTERNS:
 284:         safe, unsafe = _safe_matches(text, (pattern,))
 285:         ambiguous_hits.extend(match.group(0) for match in safe)
 286:         excluded.extend(unsafe)
 287: 
 288:     matched.extend(side_hits["left"])
 289:     matched.extend(side_hits["right"])
 290:     matched.extend(bilateral_hits)
 291:     matched.extend(ambiguous_hits)
 292:     matched = list(dict.fromkeys(matched))
 293:     lobes = [lobe for lobe in LOBE_ORDER if lobe in exact_lobes]
 294: 
 295:     if bilateral_hits:
 296:         group = "bilateral_multifocal"
 297:         side = "bilateral" if side == "bilateral" or any(
 298:             re.search(r"bilateral|both|bibasilar", value, flags=re.I) for value in bilateral_hits
 299:         ) else side
 300:         roi_kind = "whole_lung"
 301:         reason = "bilateral/multifocal/diffuse cue disables side or exact-lobe restriction"
 302:     elif ambiguous_hits:
 303:         group = "ambiguous_lobe"
 304:         roi_kind = "side_lung" if side in {"left", "right"} else "pass_through"
 305:         reason = "soft or anatomically ambiguous location is degraded to laterality"
 306:     elif lobes:
 307:         group = "high_conf_exact_lobe"
 308:         roi_kind = "exact_lobe"
 309:         reason = "unnegated exact-lobe phrase without soft modifier"
 310:     elif side in {"left", "right"}:
 311:         group = "laterality_only"
 312:         roi_kind = "side_lung"
 313:         reason = "unilateral location without a safe exact-lobe phrase"
 314:     else:
 315:         group = "nonlocalizable"
 316:         roi_kind = "pass_through"
 317:         reason = "no reliable unnegated target side or lobe"
 318: 
 319:     if excluded:
 320:         reason += "; excluded context: " + ", ".join(dict.fromkeys(excluded))
 321:     return {
 322:         "group": group,
 323:         "side": side,
 324:         "lobes": lobes,
 325:         "roi_kind": roi_kind,
 326:         "matched_phrases": matched,
 327:         "reason": reason,
 328:     }
```

## Z3 — ZIP anatomy mask生成和几何加载

文件: `scripts/run_totalseg_lung_priors.py`
文件 SHA-256: `8cca74a8d4a840f1b75ba0a6d0c02f7304fa0542d982b610a5a962bf94220459`

### 原文件 L153–L181
```text
 153: def run_totalseg(
 154:     ct_path: Path,
 155:     totalseg_dir: Path,
 156:     device: str,
 157:     fast: bool,
 158:     robust_crop: bool,
 159:     roi_subset: list[str] | None,
 160:     allow_full_task_fallback: bool,
 161: ) -> None:
 162:     cmd = ["TotalSegmentator", "-i", str(ct_path), "-o", str(totalseg_dir), "--device", device]
 163:     if fast:
 164:         cmd.append("--fast")
 165:     if robust_crop:
 166:         cmd.append("--robust_crop")
 167:     if roi_subset:
 168:         cmd += ["--roi_subset", *roi_subset]
 169: 
 170:     try:
 171:         subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
 172:     except subprocess.CalledProcessError as exc:
 173:         if roi_subset and allow_full_task_fallback:
 174:             fallback = ["TotalSegmentator", "-i", str(ct_path), "-o", str(totalseg_dir), "--device", device]
 175:             if fast:
 176:                 fallback.append("--fast")
 177:             if robust_crop:
 178:                 fallback.append("--robust_crop")
 179:             subprocess.run(fallback, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
 180:         else:
 181:             raise RuntimeError(
```
### 原文件 L208–L224
```text
 208: def dilate_mm(mask: np.ndarray, spacing_xyz: tuple[float, float, float], radius_mm: float) -> np.ndarray:
 209:     if radius_mm <= 0:
 210:         return mask.copy()
 211:     if not mask.any():
 212:         return mask.copy()
 213:     # SimpleITK arrays are z, y, x; spacing is x, y, z.
 214:     sampling_zyx = np.asarray((spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]), dtype=float)
 215:     foreground = np.argwhere(mask)
 216:     margin = np.ceil(float(radius_mm) / sampling_zyx).astype(int)
 217:     lower = np.maximum(foreground.min(axis=0) - margin, 0)
 218:     upper = np.minimum(foreground.max(axis=0) + margin + 1, np.asarray(mask.shape))
 219:     crop_slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
 220:     cropped = mask[crop_slices]
 221:     distance_to_foreground = ndimage.distance_transform_edt(~cropped, sampling=sampling_zyx)
 222:     result = mask.copy()
 223:     result[crop_slices] = cropped | (distance_to_foreground <= float(radius_mm))
 224:     return result
```
### 原文件 L268–L301
```text
 268:     reference = sitk.ReadImage(str(ct_path))
 269:     lobe_masks = {name: read_mask_like(totalseg_dir / f"{name}.nii.gz", reference) for name in LUNG_LOBES}
 270:     additional_masks = {
 271:         name: read_mask_like(totalseg_dir / f"{name}.nii.gz", reference)
 272:         for name in additional_rois
 273:     }
 274:     left = np.logical_or.reduce([lobe_masks[name] for name in LEFT_LOBES])
 275:     right = np.logical_or.reduce([lobe_masks[name] for name in RIGHT_LOBES])
 276:     whole = left | right
 277:     if not whole.any():
 278:         raise ValueError("Combined whole lung mask is empty")
 279: 
 280:     rows = []
 281:     masks_to_write = (
 282:         [(name, lobe_masks[name]) for name in LUNG_LOBES]
 283:         + [(name, additional_masks[name]) for name in additional_rois]
 284:         + [
 285:             ("left_lung", left),
 286:             ("right_lung", right),
 287:             ("whole_lung", whole),
 288:         ]
 289:     )
 290:     for label, mask in masks_to_write:
 291:         vox, vol = write_mask(mask, reference, case_dir / f"{label}.nii.gz")
 292:         rows.append({"scan_id": scan_id, "mask": label, "voxels": vox, "volume_mm3": vol})
 293:         if vox == 0:
 294:             # A lobe or auxiliary structure can legitimately be outside a
 295:             # limited-FOV scan. The combined lung mask remains the hard check.
 296:             print(f"{scan_id}: warning: {label} is empty", flush=True)
 297: 
 298:     for radius in DILATION_MM:
 299:         dilated = dilate_mm(whole, reference.GetSpacing(), radius)
 300:         vox, vol = write_mask(dilated, reference, case_dir / f"lung_dilated_{radius}mm.nii.gz")
 301:         rows.append({"scan_id": scan_id, "mask": f"lung_dilated_{radius}mm", "voxels": vox, "volume_mm3": vol})
```

## Z3b — ZIP同shape优先按索引空间处理

文件: `scripts/evaluate_anatomy_prior_diagnostic.py`
文件 SHA-256: `08f2fef1be44af4a5b564e77d7e66dcd4adbc986575be30edcf00124d9be2319`

### 原文件 L163–L193
```text
 163: def load_prior(prior_dir: Path, case_name: str, mask_name: str, target: nib.Nifti1Image) -> np.ndarray:
 164:     scan_id = case_id_from_name(case_name)
 165:     path = prior_dir / scan_id / f"{mask_name}.nii.gz"
 166:     if not path.exists():
 167:         fallback = prior_dir / scan_id / "totalseg" / f"{mask_name}.nii.gz"
 168:         if fallback.exists():
 169:             path = fallback
 170:     if not path.exists():
 171:         raise FileNotFoundError(path)
 172:     prior = nib.load(str(path))
 173:     # ReX GT/prediction masks in this workspace often have an identity affine
 174:     # even though their voxel arrays are already in CT index space. If the shape
 175:     # matches, keep index-space alignment instead of resampling by a misleading
 176:     # mask affine.
 177:     if prior.shape == target.shape:
 178:         return np.asanyarray(prior.dataobj) > 0
 179:     if prior.shape != target.shape or not np.allclose(prior.affine, target.affine, atol=1e-3):
 180:         prior = resample_from_to(prior, target, order=0)
 181:     return np.asanyarray(prior.dataobj) > 0
 182: 
 183: 
 184: def geometry_reference(case_name: str, mask_ref: nib.Nifti1Image, ct_dir: Path | None) -> nib.Nifti1Image:
 185:     if ct_dir is None:
 186:         return mask_ref
 187:     ct_path = ct_dir / case_name
 188:     if not ct_path.exists():
 189:         return mask_ref
 190:     ct = nib.load(str(ct_path))
 191:     if ct.shape == mask_ref.shape:
 192:         return ct
 193:     return mask_ref
```

## Z4 — ZIP semantic-v2 strict部署入口

文件: `scripts/apply_semantic_v2_strict_to_rex_predictions.py`
文件 SHA-256: `940be78b78494646bde4f58e184ff5366b76f8d1fe2fab90133d89cb306803d4`

### 原文件 L1–L7
```text
   1: #!/usr/bin/env python3
   2: """Emit deployable frozen semantic-v2 masks from frozen semantic-v1 masks.
   3: 
   4: The v2 policy is intentionally narrow: it retains semantic-v1 unless a finding
   5: has one of the five scope signatures that was frozen before the step-9000
   6: checkpoint transfer.  It never reads ground-truth masks.
   7: """
```
### 原文件 L34–L107
```text
  34: FROZEN_SIGNATURES = {
  35:     "apical",
  36:     "basal",
  37:     "superior_within_base",
  38:     "laterobasal",
  39:     "subpleural_peripheral+posterior",
  40: }
  41: 
  42: 
  43: def signature(plan: dict) -> str:
  44:     return " OR ".join(
  45:         "+".join(item["name"] for item in group) for group in plan["atom_groups"]
  46:     )
  47: 
  48: 
  49: def case_worker(task: tuple[str, list[dict], dict]) -> tuple[list[dict], str | None]:
  50:     case, findings, settings = task
  51:     try:
  52:         pred_path = Path(settings["pred_dir"]) / case
  53:         pred, pred_ref = channel_first_mask(pred_path, 0.5, False)
  54:         ref = geometry_reference(case, pred_ref, Path(settings["ct_dir"]))
  55:         sampling = tuple(float(item) for item in nib.affines.voxel_sizes(ref.affine)[:3])
  56:         axes = world_axis_info(ref.affine, tuple(int(item) for item in pred.shape[1:]))
  57:         priors = {
  58:             "whole": load_prior(Path(settings["prior_dir"]), case, "whole_lung", ref),
  59:             "whole10": load_prior(Path(settings["prior_dir"]), case, "lung_dilated_10mm", ref),
  60:         }
  61:         for lobe, filename in LOBE_FILES.items():
  62:             priors[lobe] = load_prior(Path(settings["prior_dir"]), case, filename, ref)
  63:         priors["left"] = priors["LUL"] | priors["LLL"]
  64:         priors["right"] = priors["RUL"] | priors["RML"] | priors["RLL"]
  65: 
  66:         out = pred.copy()
  67:         cache: dict[str, np.ndarray] = {}
  68:         rows: list[dict] = []
  69:         for finding in findings:
  70:             finding_id = int(finding["finding_idx"])
  71:             if finding_id >= pred.shape[0]:
  72:                 raise IndexError(f"{case}: finding {finding_id} outside {pred.shape[0]} channels")
  73:             parsed = semantic_parse(str(finding["prompt"]))
  74:             side = parsed["side"] if parsed["side"] in {"left", "right"} else None
  75:             lobes = split_lobes(";".join(parsed["lobes"]))
  76:             if parsed["group"] == "high_conf_exact_lobe" and lobes:
  77:                 strict_base = np.logical_or.reduce([priors[lobe] for lobe in lobes])
  78:                 semantic_roi = ndimage.distance_transform_edt(~strict_base, sampling=sampling) <= 15.0
  79:             elif parsed["group"] in {"high_conf_exact_lobe", "ambiguous_lobe", "laterality_only"} and side:
  80:                 strict_base = priors[side]
  81:                 semantic_roi = ndimage.distance_transform_edt(~strict_base, sampling=sampling) <= 15.0
  82:             else:
  83:                 strict_base = priors["whole"]
  84:                 semantic_roi = priors["whole10"]
  85: 
  86:             # Passing all known lexical clusters is safe: scope_plan retains only
  87:             # patterns actually present in the prompt.
  88:             plan = scope_plan(str(finding["prompt"]), set(SIMPLE_PATTERNS))
  89:             scope_signature = signature(plan)
  90:             selected = bool(plan["hard_eligible"] and scope_signature in FROZEN_SIGNATURES)
  91:             if selected:
  92:                 scope_roi, _ = combine_scope_proxy(
  93:                     plan,
  94:                     semantic_roi,
  95:                     strict_base,
  96:                     priors,
  97:                     axes,
  98:                     sampling,
  99:                     relevant_sides(side, lobes),
 100:                     cache,
 101:                 )
 102:                 out[finding_id] &= scope_roi
 103:             rows.append(
 104:                 {
 105:                     "case_id": case,
 106:                     "finding_id": finding_id,
 107:                     "scope_signature": scope_signature,
```

## Z5 — ZIP v2 signature、作用域和非排他语言

文件: `scripts/audit_segment_v2_scopeaware.py`
文件 SHA-256: `1d12e2f1f3e0243ff92531d1064344372aef4953a4fcca00d017e66c6b9bf989`

### 原文件 L43–L101
```text
  43: # A lexical compound is one anatomical atom, so its component proxies are
  44: # intersected.  Distinct atoms elsewhere in the finding are later unioned.
  45: # name, lexical pattern, proxy components to intersect, discovery clusters
  46: # consumed by the compound.  The final field prevents e.g. "inferior lingular"
  47: # from being added again as a separate inferior-half atom.
  48: COMPOUNDS: list[tuple[str, re.Pattern[str], tuple[str, ...], tuple[str, ...]]] = [
  49:     ("anteromedial_basal", re.compile(r"\bantero\s*-?\s*medial\s+basal\b", re.I), ("basal", "anterior", "medial"), ("basal", "anterior", "medial")),
  50:     ("posterobasal", re.compile(r"\b(?:postero\s*-?\s*basal|posterior\s+basal)\b", re.I), ("basal", "posterior"), ("basal", "posterior")),
  51:     ("anterobasal", re.compile(r"\b(?:antero\s*-?\s*basal|anterior\s+basal)\b", re.I), ("basal", "anterior"), ("basal", "anterior")),
  52:     ("laterobasal", re.compile(r"\b(?:latero\s*-?\s*basal|lateral\s+basal)\b", re.I), ("basal", "lateral"), ("basal", "lateral")),
  53:     ("mediobasal", re.compile(r"\b(?:medio\s*-?\s*basal|medial\s+basal)\b", re.I), ("basal", "medial"), ("basal", "medial")),
  54:     ("apicoposterior", re.compile(r"\bapico\s*-?\s*posterior\b", re.I), ("apical", "posterior"), ("apical", "posterior")),
  55:     ("inferior_lingular", re.compile(r"\b(?:inferior\s+lingular|lingular\s+inferior)\b", re.I), ("lingular",), ("lingular", "inferior_within_base")),
  56:     ("dependent_basal", re.compile(r"\b(?:dependent\s+basal|basal\s+dependent)\b", re.I), ("dependent", "basal"), ("dependent", "basal")),
  57:     ("anteromedial", re.compile(r"\bantero\s*-?\s*medial\b", re.I), ("anterior", "medial"), ("anterior", "medial")),
  58: ]
  59: 
  60: SIMPLE_PATTERNS: dict[str, re.Pattern[str]] = {
  61:     "apical": re.compile(r"\b(?:biapical|apices?|apical)\b", re.I),
  62:     "basal": re.compile(r"\b(?:basal|basilar|lung base|basal levels?)\b", re.I),
  63:     "diaphragmatic": re.compile(r"\b(?:juxtadiaphragmatic|diaphragmatic|adjacent to the diaphragm)\b", re.I),
  64:     "subpleural_peripheral": re.compile(r"\b(?:subpleural|peripheral(?:ly)?|pleural-based|pleural surface)\b", re.I),
  65:     "central": re.compile(r"\b(?:central(?:ly)?|central zones?|central levels?|peri-?hilar|hilar)\b", re.I),
  66:     "anterior": re.compile(r"\banterior\b", re.I),
  67:     "posterior": re.compile(r"\bposterior\b", re.I),
  68:     "medial": re.compile(r"\bmedial\b", re.I),
  69:     "lateral": re.compile(r"\blateral\b", re.I),
  70:     "fissural": re.compile(r"\b(?:peri-?fissural|fissure-based|fissural|fissures?)\b", re.I),
  71:     "lingular": re.compile(r"\b(?:lingula|lingular)\b", re.I),
  72:     "dependent": re.compile(r"\b(?:dependent|dependently)\b", re.I),
  73:     "paramediastinal": re.compile(r"\b(?:paramediastinal|paramediastinum)\b", re.I),
  74:     "superior_within_base": re.compile(r"\b(?:superior|uppermost)\b", re.I),
  75:     "inferior_within_base": re.compile(r"\b(?:inferior|lowermost)\b", re.I),
  76:     "paraspinal_or_costovertebral": re.compile(r"\b(?:paraspinal|paravertebral|costovertebral|adjacent to osteophytes)\b", re.I),
  77:     "retrocardiac": re.compile(r"\b(?:retrocardiac|paracardiac)\b", re.I),
  78: }
  79: 
  80: # These expressions locate an example or dominant component, not the complete
  81: # target extent.  They are retained for analysis but prohibited from hard ROI
  82: # clipping.  "including" is conservatively treated as non-exhaustive.
  83: NONEXCLUSIVE_PATTERNS: dict[str, re.Pattern[str]] = {
  84:     "predominantly": re.compile(r"\b(?:predominantly|primarily|mainly)\b", re.I),
  85:     "most_prominent": re.compile(r"\b(?:most prominent|more prominent|greatest)\b", re.I),
  86:     "largest_anchor": re.compile(r"\blargest\b.*\b(?:in|at|within)\b", re.I),
  87:     "including_nonexhaustive": re.compile(r"\b(?:including|includes?|among)\b", re.I),
  88:     "example_language": re.compile(r"\b(?:such as|for example|one of)\b", re.I),
  89: }
  90: 
  91: DECORATOR_CLUSTERS = {
  92:     "subpleural_peripheral", "central", "fissural", "dependent",
  93:     "diaphragmatic", "paramediastinal",
  94: }
  95: 
  96: STRONG_LIST_CONNECTOR = re.compile(r"(?:;|\band\b|\bor\b)", re.I)
  97: COARSE_LOCATION = re.compile(
  98:     r"\b(?:both\s+lungs?|bilateral(?:ly)?|(?:left|right)\s+(?:upper|middle|lower)\s+lobe|"
  99:     r"(?:upper|middle|lower)\s+lobe\s+of\s+the\s+(?:left|right)\s+lung)\b",
 100:     re.I,
 101: )
```
### 原文件 L232–L348
```text
 232:     # Nearby descriptors form one conjunctive location; explicitly enumerated
 233:     # locations form separate groups whose masks are unioned.  This handles
 234:     # both "fissural ... in the superior segment" (intersection) and
 235:     # "centrally and peripherally" (union).
 236:     positioned = sorted(
 237:         unique_atoms + unique_decorators,
 238:         key=lambda item: item["span"][0] if item["span"] is not None else 10**9,
 239:     )
 240:     atom_groups: list[list[dict[str, Any]]] = []
 241:     for atom in positioned:
 242:         if not atom_groups:
 243:             atom_groups.append([atom])
 244:             continue
 245:         previous = atom_groups[-1][-1]
 246:         if previous["span"] is None or atom["span"] is None:
 247:             atom_groups.append([atom])
 248:             continue
 249:         between = text[previous["span"][1]:atom["span"][0]]
 250:         # A comma is a list separator only after a primary location has
 251:         # already appeared.  This avoids treating measurement punctuation in
 252:         # "peripheral nodule, 5 mm, in the lateral segment" as a scope union.
 253:         is_list = bool(STRONG_LIST_CONNECTOR.search(between)) or (
 254:             "," in between and any(item["kind"] == "atom" for item in atom_groups[-1])
 255:         )
 256:         if is_list:
 257:             atom_groups.append([atom])
 258:         else:
 259:             atom_groups[-1].append(atom)
 260: 
 261:     # A leading distribution/decorator attached to the first listed location
 262:     # scopes the complete following list: "subpleural opacities in A, B and C".
 263:     # A trailing decorator remains local to its own group.
 264:     if len(atom_groups) > 1:
 265:         first_primary_position = next(
 266:             (index for index, item in enumerate(atom_groups[0]) if item["kind"] == "atom"),
 267:             None,
 268:         )
 269:         if first_primary_position is not None:
 270:             leading = [item for item in atom_groups[0][:first_primary_position] if item["kind"] == "decorator"]
 271:             for group in atom_groups[1:]:
 272:                 if leading and any(item["kind"] == "atom" for item in group):
 273:                     present = {item["name"] for item in group}
 274:                     group[:0] = [item for item in leading if item["name"] not in present]
 275: 
 276:     flags = [name for name, pattern in NONEXCLUSIVE_PATTERNS.items() if pattern.search(text)]
 277:     fine_patterns = [pattern for pattern in SIMPLE_PATTERNS.values()] + [item[1] for item in COMPOUNDS]
 278:     unmodified_coarse_clauses = [
 279:         clause.strip()
 280:         for clause in CLAUSE_SPLIT.split(text)
 281:         if COARSE_LOCATION.search(clause)
 282:         and not any(pattern.search(clause) for pattern in fine_patterns)
 283:     ]
 284:     unresolved = sorted(available_clusters & UNRESOLVED)
 285:     known_scope = bool(atom_groups)
 286:     if unresolved and not known_scope:
 287:         mode = "unresolved_location"
 288:         hard_eligible = False
 289:     elif flags:
 290:         mode = "soft_only_nonexclusive"
 291:         hard_eligible = False
 292:     elif unmodified_coarse_clauses:
 293:         mode = "soft_only_incomplete_scope"
 294:         hard_eligible = False
 295:     elif len(atom_groups) > 1:
 296:         mode = "hard_union"
 297:         hard_eligible = True
 298:     elif any(len(group) > 1 for group in atom_groups):
 299:         mode = "hard_intersection"
 300:         hard_eligible = True
 301:     elif known_scope:
 302:         mode = "hard_single"
 303:         hard_eligible = True
 304:     else:
 305:         mode = "unresolved_location"
 306:         hard_eligible = False
 307:     return {
 308:         "scope_mode": mode,
 309:         "hard_eligible": hard_eligible,
 310:         "atoms": unique_atoms,
 311:         "atom_groups": atom_groups,
 312:         "decorators": sorted({item["name"] for item in unique_decorators}),
 313:         "nonexclusive_flags": flags,
 314:         "unresolved_clusters": unresolved,
 315:         "unmatched_discovery_clusters": unmatched_discovery,
 316:         "unmodified_coarse_clauses": unmodified_coarse_clauses,
 317:     }
 318: 
 319: 
 320: def combine_scope_proxy(
 321:     plan: dict[str, Any], semantic_roi: np.ndarray, strict_base: np.ndarray,
 322:     priors: dict[str, np.ndarray], axes: dict[int, tuple[int, np.ndarray, float]],
 323:     sampling: tuple[float, float, float], sides: list[str], case_cache: dict[str, np.ndarray],
 324: ) -> tuple[np.ndarray, list[dict[str, Any]]]:
 325:     if not plan["hard_eligible"]:
 326:         return semantic_roi.copy(), []
 327:     details: list[dict[str, Any]] = []
 328:     group_masks: list[np.ndarray] = []
 329:     for group_index, group in enumerate(plan["atom_groups"]):
 330:         masks_within_group: list[np.ndarray] = []
 331:         for atom in group:
 332:             component_masks: list[np.ndarray] = []
 333:             component_details = []
 334:             for cluster in atom["components"]:
 335:                 mask, name, params = build_proxy(
 336:                     cluster, semantic_roi, strict_base, priors, axes, sampling, sides, case_cache
 337:                 )
 338:                 if mask is None:
 339:                     continue
 340:                 component_masks.append(mask)
 341:                 component_details.append({"cluster": cluster, "proxy": name, "parameters": params})
 342:             if component_masks:
 343:                 masks_within_group.append(np.logical_and.reduce(component_masks))
 344:                 details.append({**atom, "group": group_index, "operation": "component_intersection", "component_details": component_details})
 345:         if masks_within_group:
 346:             group_masks.append(np.logical_and.reduce(masks_within_group))
 347:     scope = np.logical_or.reduce(group_masks) if group_masks else semantic_roi.copy()
 348:     return scope, details
```

## Z6 — ZIP v2几何proxy

文件: `scripts/audit_fine_spatial_modifiers.py`
文件 SHA-256: `b9b73b79c3c0df5f9949c2fc6d79f99ec1e037d02a6525e8cf45a33395f16f8a`

### 原文件 L63–L80
```text
  63: def world_axis_info(affine: np.ndarray, shape: tuple[int, int, int]) -> dict[int, tuple[int, np.ndarray, float]]:
  64:     """Map RAS world axes to voxel axes; reject unverified oblique geometry."""
  65:     result: dict[int, tuple[int, np.ndarray, float]] = {}
  66:     for world_axis in range(3):
  67:         coefficients = np.asarray(affine[world_axis, :3], dtype=float)
  68:         voxel_axis = int(np.argmax(np.abs(coefficients)))
  69:         dominant = abs(float(coefficients[voxel_axis]))
  70:         off_axis = float(np.max(np.abs(np.delete(coefficients, voxel_axis))))
  71:         ratio = off_axis / dominant if dominant > 0 else float("inf")
  72:         if dominant == 0 or ratio > 1e-4:
  73:             raise ValueError(
  74:                 f"Oblique/unresolved affine world axis {world_axis}: coefficients={coefficients.tolist()} ratio={ratio}"
  75:             )
  76:         values = affine[world_axis, voxel_axis] * np.arange(shape[voxel_axis], dtype=np.float32) + affine[world_axis, 3]
  77:         result[world_axis] = (voxel_axis, values.astype(np.float32), ratio)
  78:     if len({value[0] for value in result.values()}) != 3:
  79:         raise ValueError(f"Affine does not provide a one-to-one RAS/voxel axis map: {result}")
  80:     return result
```
### 原文件 L99–L115
```text
  99: def fraction_region(
 100:     anatomical_base: np.ndarray,
 101:     container: np.ndarray,
 102:     axis: tuple[int, np.ndarray, float],
 103:     high: bool,
 104:     fraction: float,
 105:     margin_mm: float,
 106: ) -> np.ndarray:
 107:     voxel_axis, values, _ = axis
 108:     low, upper = axis_extent(anatomical_base, voxel_axis, values)
 109:     if not np.isfinite(low) or upper <= low:
 110:         return np.zeros_like(container)
 111:     if high:
 112:         threshold = upper - fraction * (upper - low) - margin_mm
 113:     else:
 114:         threshold = low + fraction * (upper - low) + margin_mm
 115:     return container & axis_condition(container.shape, voxel_axis, values, threshold, high)
```
### 原文件 L125–L149
```text
 125: def medial_lateral_region(
 126:     priors: dict[str, np.ndarray],
 127:     container: np.ndarray,
 128:     x_axis: tuple[int, np.ndarray, float],
 129:     sides: list[str],
 130:     medial: bool,
 131:     fraction: float = 0.5,
 132:     margin_mm: float = 15.0,
 133: ) -> np.ndarray:
 134:     voxel_axis, values, _ = x_axis
 135:     abs_values = np.abs(values)
 136:     output = np.zeros_like(container)
 137:     for side in sides:
 138:         base = priors[side]
 139:         low, high = axis_extent(base, voxel_axis, abs_values)
 140:         if not np.isfinite(low) or high <= low:
 141:             continue
 142:         if medial:
 143:             threshold = low + fraction * (high - low) + margin_mm
 144:             condition = axis_condition(container.shape, voxel_axis, abs_values, threshold, False)
 145:         else:
 146:             threshold = high - fraction * (high - low) - margin_mm
 147:             condition = axis_condition(container.shape, voxel_axis, abs_values, threshold, True)
 148:         output |= container & base & condition
 149:     return output
```
### 原文件 L159–L211
```text
 159: def build_proxy(
 160:     cluster: str,
 161:     semantic_roi: np.ndarray,
 162:     strict_base: np.ndarray,
 163:     priors: dict[str, np.ndarray],
 164:     axes: dict[int, tuple[int, np.ndarray, float]],
 165:     sampling: tuple[float, float, float],
 166:     sides: list[str],
 167:     case_cache: dict[str, np.ndarray],
 168: ) -> tuple[np.ndarray | None, str, dict[str, Any]]:
 169:     if cluster in {"paraspinal_or_costovertebral", "retrocardiac"}:
 170:         return None, "none", {}
 171:     if cluster in {"apical", "superior_within_base"}:
 172:         return fraction_region(strict_base, semantic_roi, axes[2], True, 0.50, 15.0), "superior_half_plus_15mm", {"fraction": 0.50, "margin_mm": 15}
 173:     if cluster in {"basal", "inferior_within_base"}:
 174:         return fraction_region(strict_base, semantic_roi, axes[2], False, 0.50, 15.0), "inferior_half_plus_15mm", {"fraction": 0.50, "margin_mm": 15}
 175:     if cluster == "diaphragmatic":
 176:         return fraction_region(strict_base, semantic_roi, axes[2], False, 0.35, 15.0), "inferior_35pct_plus_15mm", {"fraction": 0.35, "margin_mm": 15}
 177:     if cluster == "anterior":
 178:         return fraction_region(strict_base, semantic_roi, axes[1], True, 0.50, 15.0), "anterior_half_plus_15mm", {"fraction": 0.50, "margin_mm": 15}
 179:     if cluster in {"posterior", "dependent"}:
 180:         name = "posterior_half_plus_15mm_supine" if cluster == "dependent" else "posterior_half_plus_15mm"
 181:         return fraction_region(strict_base, semantic_roi, axes[1], False, 0.50, 15.0), name, {"fraction": 0.50, "margin_mm": 15, "position_assumption": "supine" if cluster == "dependent" else "none"}
 182:     if cluster in {"medial", "paramediastinal"}:
 183:         fraction = 0.35 if cluster == "paramediastinal" else 0.50
 184:         margin = 10.0 if cluster == "paramediastinal" else 15.0
 185:         return medial_lateral_region(priors, semantic_roi, axes[0], sides, True, fraction, margin), f"medial_{int(fraction*100)}pct_plus_{int(margin)}mm", {"fraction": fraction, "margin_mm": margin}
 186:     if cluster == "lateral":
 187:         return medial_lateral_region(priors, semantic_roi, axes[0], sides, False), "lateral_half_plus_15mm", {"fraction": 0.50, "margin_mm": 15}
 188:     if cluster == "subpleural_peripheral":
 189:         if "boundary_shell" not in case_cache:
 190:             inside_distance = ndimage.distance_transform_edt(priors["whole"], sampling=sampling)
 191:             outside_distance = ndimage.distance_transform_edt(~priors["whole"], sampling=sampling)
 192:             case_cache["boundary_shell"] = (
 193:                 (priors["whole"] & (inside_distance <= 30.0))
 194:                 | (~priors["whole"] & (outside_distance <= 10.0))
 195:             )
 196:         return semantic_roi & case_cache["boundary_shell"], "lung_boundary_inside30_outside10mm", {"inside_shell_mm": 30, "outside_margin_mm": 10}
 197:     if cluster == "central":
 198:         if "central_core" not in case_cache:
 199:             case_cache["central_core"] = priors["whole"] & (ndimage.distance_transform_edt(priors["whole"], sampling=sampling) > 20.0)
 200:         return semantic_roi & case_cache["central_core"], "lung_interior_over_20mm", {"minimum_boundary_distance_mm": 20}
 201:     if cluster == "fissural":
 202:         if "fissure_proxy" not in case_cache:
 203:             case_cache["fissure_proxy"] = build_fissure_proxy(priors, sampling)
 204:         return semantic_roi & case_cache["fissure_proxy"], "within10mm_of_two_lobes", {"lobe_proximity_mm": 10, "minimum_lobes": 2}
 205:     if cluster == "lingular":
 206:         if "lingula_proxy" not in case_cache:
 207:             lul10 = dilate_mm(priors["LUL"], sampling, 10.0)
 208:             inferior = fraction_region(priors["LUL"], lul10, axes[2], False, 0.65, 10.0)
 209:             anterior = fraction_region(priors["LUL"], lul10, axes[1], True, 0.65, 10.0)
 210:             case_cache["lingula_proxy"] = inferior & anterior
 211:         return semantic_roi & case_cache["lingula_proxy"], "inferior_anterior_65pct_LUL_plus10mm", {"inferior_fraction": 0.65, "anterior_fraction": 0.65, "margin_mm": 10}
```

## Z7 — ZIP记录的step6000历史参考值

文件: `scripts/finalize_segment_v2_scopeaware.py`
文件 SHA-256: `f80e5969ee619a2436291ab31b39ecb587d49238521acaf10e5c3ec78b00a33f`

### 原文件 L1–L35
```text
   1: #!/usr/bin/env python3
   2: """Merge segment-v2 audit shards and compare scope-aware vs modifier-v1 conclusions."""
   3: 
   4: from __future__ import annotations
   5: 
   6: import argparse
   7: import json
   8: from pathlib import Path
   9: 
  10: import numpy as np
  11: import pandas as pd
  12: 
  13: 
  14: EXPECTED = {
  15:     "raw": (0.3031991044198736, 274),
  16:     "whole10": (0.31376730948125936, 278),
  17:     "semantic_v1": (0.31848245294040217, 281),
  18: }
  19: 
  20: 
  21: def args_parser() -> argparse.Namespace:
  22:     p = argparse.ArgumentParser(description=__doc__)
  23:     p.add_argument("--shards-dir", type=Path, required=True)
  24:     p.add_argument("--output-dir", type=Path, required=True)
  25:     p.add_argument("--modifier-assignments", type=Path, required=True)
  26:     p.add_argument("--v1-gt-summary", type=Path, required=True)
  27:     p.add_argument("--v1-fp-summary", type=Path, required=True)
  28:     p.add_argument("--reference-leakage", type=Path, required=True)
  29:     p.add_argument("--expected-shards", type=int, required=True)
  30:     return p.parse_args()
  31: 
  32: 
  33: def merge(shards: list[Path], name: str) -> pd.DataFrame:
  34:     frames = []
  35:     for shard in shards:
```

## Z8 — ZIP类别名称

文件: `VoxTell/voxtell/utils/rex_category_prompts.py`
文件 SHA-256: `cd324ef8d5192c740d4b299b6701e17510662afb396a588555e92ec65c82073b`

### 原文件 L12–L27
```text
  12: CATEGORY_NAME_BY_CODE: dict[str, str] = {
  13:     "1a": "Bronchial wall thickening",
  14:     "1b": "Bronchiectasis",
  15:     "1c": "Emphysema (including Centrilobular, Paraseptal, Bullous)",
  16:     "1d": "Septal thickening (including Interlobular, Reticulation)",
  17:     "1e": "Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic)",
  18:     "1f": "Other",
  19:     "2a": "Linear (including subsegmental atelectasis, scarring, fibrosis)",
  20:     "2b": "Atelectasis, consolidation",
  21:     "2c": "Groundglass opacity",
  22:     "2d": "Pulmonary nodules/masses",
  23:     "2e": "Pleural effusion or thickening",
  24:     "2f": "Honeycombing",
  25:     "2g": "Pneumothorax",
  26:     "2h": "Other",
  27: }
```
