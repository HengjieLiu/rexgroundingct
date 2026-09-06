#!/usr/bin/env python3
"""Read sealed Exp022 evidence and render private/repository-safe reports."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from audit_022_official_anatomy import CATEGORIES, REPO, config, now, read, sha, verify_seal, write

TITLES = {'1a': 'Bronchial wall thickening', '1b': 'Bronchiectasis', '1c': 'Emphysema',
          '1d': 'Septal / diffuse attenuation changes', '1e': 'Micronodules / tree-in-bud',
          '1f': 'Other diffuse / heterogeneous findings', '2a': 'Scarring / linear atelectasis',
          '2b': 'Consolidation / atelectatic opacities', '2c': 'Ground-glass opacities',
          '2d': 'Nodules / masses (including atypical prompts)', '2e': 'Pleural effusion / thickening',
          '2f': 'No validation findings', '2g': 'Pneumothorax', '2h': 'Other focal / heterogeneous findings'}
CONTEXT = {
 '1a': 'The three prompts target bronchial/peribronchial walls, not tracheal lumen. No complete bronchial-wall label exists in total. All prompt-routed policies bypass these findings; all_lung is a deliberately unsuitable negative-control diagnostic.',
 '1b': 'Distinguish localized bronchiectasis from bilateral central disease and predominant-site descriptions. Lobes provide approximate pulmonary support, not a bronchial-tree mask; central airways and traction distortion are boundary risks.',
 '1c': 'Many prompts describe bilateral emphysema, but two describe focal bullae. Upper-lobe predominance does not establish exclusivity. Paraseptal disease and apical blebs/bullae require peripheral coverage checks.',
 '1d': 'Diffuse septal changes and mosaic/crazy-paving descriptions dominate; one prompt has internally unusual segment terminology. Broad distribution is retained. Periphery and dense abnormal lung can be absent from anatomy labels.',
 '1e': 'Keep this category separate from 2d. These prompts include bilateral small nodules, centriacinar opacities and tree-in-bud; fine bronchi are unavailable. Segment-specific text supports at most its explicitly named parent lobe.',
 '1f': 'Four different targets: pleural calcification, bilateral upper-lobe nodules, lower-lobe aeration differences, and an occluded main bronchus. The pleural and bronchial targets bypass routing; no single category-wide anatomical rule is justified.',
 '2a': 'Many scar/linear-atelectasis prompts name a lobe or lingula, but apical zones are not automatically upper lobes. Pleuroparenchymal scars may straddle lung boundaries. Fissural thickening is bypassed; inconsistent segment wording is broadened. Including/especially clauses do not exhaust bilateral disease.',
 '2b': 'Consolidation and atelectasis can distort or replace normally aerated lung. Explicit air bronchograms do not make the trachea an appropriate target. Effusions, osteophytes and mediastinum are landmarks, not substitute target masks. Quantify GT and existing TP outside anatomy before accepting a lung gate.',
 '2c': 'This category mixes focal, multifocal and diffuse opacities. Preserve both sides despite predominance and preserve surrounding/halo extent. Subpleural, dependent, diaphragmatic and peribronchovascular descriptions need separate harm accounting.',
 '2d': '132 findings are not 132 isolated solid nodules. Plural bilateral prompts remain bilateral even when the largest lesion has a named lobe. Singular largest-nodule prompts can be localized. Pleural/fissural nodules, ground-glass halos, an air cyst, an apical bleb and an aberrant vascular sequestration description require individual treatment; the vascular target bypasses routing. No exact lymph-node or aberrant-vessel target label is available.',
 '2e': 'Pleural fluid, pleural calcification/thickening and fissural thickening are outside or across the lung interior. Named lobes localize adjacent landmarks rather than establish containment. All prompt-routed candidates are no-op; all_lung explicitly measures the risk of the wrong compartment.',
 '2f': 'No val200 examples. There is no measured evidence or category-specific recommendation.',
 '2g': 'A single right pneumothorax describes pleural air occupying much of the hemithorax. Lung labels cannot represent that air compartment. Upper-lobe predominance is not lobe containment. Prior absence is temporal, not negation of the current target. Prompt routing is no-op; one scan cannot establish generalization.',
 '2h': 'Five cyst prompts, pleural irregularities and architectural distortion are heterogeneous. The pleural target bypasses routing. Cyst support can use the explicitly named lobe, but fissural/peribronchial boundaries remain risks; apical distortion is not an explicit upper-lobe target.'}


def load_evidence():
    c = config(); root = Path(c['runtime_root'])
    _, seal_hash = verify_seal(c)
    source_lock = read(Path(c['source_audit_root']) / 'config/source_lock.json')
    if read(root / 'source_lut.json') != source_lock['lut_provenance']['total_task_label_map']:
        raise ValueError('Report LUT differs from producer-bound sealed mapping')
    review = read(root / 'frozen_review.json')['rows']
    expected = {(r['case'], r['finding_id']): r for r in review}
    evidence = []
    case_records = []
    for path in sorted((root / 'cases').glob('*.json')):
        case = read(path)
        if case['seal_sha256'] != seal_hash:
            raise ValueError('Mixed measurement seals')
        case_records.append(case)
        for f in case['findings']:
            key = case['case'], f['finding_id']
            if key not in expected:
                raise ValueError('Duplicate or unexpected finding')
            r = expected.pop(key)
            if f['id'] != r['id']:
                raise ValueError('Review ID mismatch')
            evidence.append({**r, **f, 'spacing_mm': case['spacing_mm'], 'lung_flags': case['lung_flags']})
    if expected or len(evidence) != c['expected_findings'] or len(case_records) != c['expected_cases']:
        raise ValueError(f'Incomplete evidence: {len(evidence)} findings, {len(case_records)} cases')
    evidence.sort(key=lambda r: r['id'])
    if abs(np.mean([r['baseline']['dice'] for r in evidence]) - c['baseline_dice']) > c['baseline_tolerance']:
        raise ValueError('Aggregate no-op Dice mismatch')
    if sum(r['baseline']['hit'] for r in evidence) != c['baseline_hits']:
        raise ValueError('No-op hit mismatch')
    return c, root, evidence, case_records


def bootstrap(deltas, cases, resamples=2000, seed=20260905):
    names = sorted(set(cases))
    if not names:
        return [None, None]
    sums = np.array([sum(d for d, c in zip(deltas, cases) if c == n) for n in names])
    counts = np.array([cases.count(n) for n in names])
    weights = np.random.default_rng(seed).multinomial(len(names), np.full(len(names), 1 / len(names)), size=resamples)
    means = (weights @ sums) / (weights @ counts)
    return np.quantile(means, [.025, .975]).tolist()


def summarize(rows, policy, intervals=True):
    if not rows:
        return dict(n=0, cases=0, selected=0)
    b = [r['baseline'] for r in rows]
    m = [r['candidates'][policy] for r in rows]
    selected = [i for i, x in enumerate(m) if x['selected']]
    d = [x['dice'] - y['dice'] for x, y in zip(m, b)]
    cases = [r['case'] for r in rows]
    case_means = [np.mean([x['dice'] for x, r in zip(m, rows) if r['case'] == name]) for name in sorted(set(cases))]
    out = dict(n=len(rows), cases=len(set(cases)), selected=len(selected),
               before=float(np.mean([x['dice'] for x in b])), after=float(np.mean([x['dice'] for x in m])),
               delta=float(np.mean(d)), per_case_dice=float(np.mean(case_means)),
               hits_before=sum(x['hit'] for x in b), hits_after=sum(x['hit'] for x in m),
               improved=sum(x > 1e-7 for x in d), worsened=sum(x < -1e-7 for x in d), unchanged=sum(abs(x) <= 1e-7 for x in d),
               hit_gains=sum(x['hit'] and not y['hit'] for x, y in zip(m, b)),
               hit_losses=sum(y['hit'] and not x['hit'] for x, y in zip(m, b)),
               tp_removed=sum(x['tp_removed'] for x in m), fp_removed=sum(x['fp_removed'] for x in m),
               prediction_emptied=sum(x['prediction_emptied'] for x in m),
               precision=float(np.mean([x['precision'] for x in m])), recall=float(np.mean([x['recall'] for x in m])),
               worst_delta=min(d), best_delta=max(d))
    if selected:
        out.update(selected_before=float(np.mean([b[i]['dice'] for i in selected])),
                   selected_after=float(np.mean([m[i]['dice'] for i in selected])),
                   gt_coverage_mean=float(np.mean([m[i]['gt_coverage'] for i in selected])),
                   gt_coverage_min=min(m[i]['gt_coverage'] for i in selected),
                   gt_coverage_below95=sum(m[i]['gt_coverage'] < .95 for i in selected),
                   gt_completely_excluded=sum(m[i]['gt_completely_excluded'] for i in selected))
    else:
        out.update(selected_before=None, selected_after=None, gt_coverage_mean=None,
                   gt_coverage_min=None, gt_coverage_below95=0, gt_completely_excluded=0)
    if intervals:
        out['delta_ci95'] = bootstrap(d, cases)
    return out


def fmt(v, digits=4):
    return '—' if v is None else f'{v:.{digits}f}'


def policy_table(stats):
    lines = ['| Policy | Selected (subset Dice before→after) | Full Dice | Δ Dice [95% CT interval] | Hits | Gain/loss | Better/worse | Mean/min GT retained | GT <95% / excluded | TP/FP removed |',
             '| --- | ---: | ---: | --- | ---: | --- | --- | --- | --- | --- |']
    for key, s in stats.items():
        if not s['n']:
            continue
        lo, hi = s['delta_ci95']
        lines.append(f"| {key} | {s['selected']} ({fmt(s['selected_before'])}→{fmt(s['selected_after'])}) | {s['after']:.4f} | {s['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {s['hits_after']}/{s['n']} | +{s['hit_gains']}/−{s['hit_losses']} | {s['improved']}/{s['worsened']} | {fmt(s['gt_coverage_mean'])}/{fmt(s['gt_coverage_min'])} | {s['gt_coverage_below95']}/{s['gt_completely_excluded']} | {s['tp_removed']}/{s['fp_removed']} |")
    return '\n'.join(lines)


def recommendation(cat, stats):
    if cat == '2f':
        return 'Insufficient evidence: no findings.'
    if cat in {'1a', '2e', '2g'}:
        return 'Too unsupported for hard restriction: keep the prompt-routed no-op; do not substitute a lung or trachea mask for the target compartment.'
    n = next(iter(stats.values()))['n']
    candidates = [(k, s) for k, s in stats.items() if not k.startswith('all_lung') and s['selected']]
    if not candidates:
        return 'Insufficient evidence: no applicable exact target support.'
    best_k, best = max(candidates, key=lambda t: t[1]['delta'])
    safety = 'Some findings still worsen; do not deploy from this result alone.' if best['worsened'] else 'No measured regression under this comparison; this is not a general safety guarantee.'
    label = 'Insufficient evidence; descriptive only' if n < 10 else 'Promising only for a later controlled experiment' if best['delta'] > .001 else 'At most limited benefit; retain no-op pending review'
    return f"{label}. Highest observed prompt-routed Dice is `{best_k}` (Δ {best['delta']:+.4f}; {best['worsened']} worsened; {best['hit_losses']} lost hits). This is a validation-selected description, not a preselected method or held-out estimate. {safety}"


def csv_write(path, rows):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def render():
    c, root, rows, case_records = load_evidence()
    out = root / 'reports'; out.mkdir(exist_ok=True)
    visual_path = out / 'visual_manifest.json'
    inspection_path = out / 'ai_visual_review.json'
    visual_ids = {r['id'] for r in read(visual_path)['records']} if visual_path.exists() else set()
    inspection = read(inspection_path) if inspection_path.exists() else None
    visual_status = inspection['status'] if inspection else 'PENDING_AI_INSPECTION'
    policies = list(rows[0]['candidates'])
    global_stats = {k: summarize(rows, k) for k in policies}
    cats = {cat: {k: summarize([r for r in rows if r['category'] == cat], k) for k in policies} for cat in CATEGORIES}
    private_rows = []
    for r in rows:
        for k, m in r['candidates'].items():
            private_rows.append(dict(id=r['id'], case=r['case'], finding_id=r['finding_id'], category=r['category'],
                                     prompt=r['prompt'], scope=r['scope'], extent=r['extent'], risks=r['risks'],
                                     policy=k, baseline_dice=r['baseline']['dice'], baseline_hit=r['baseline']['hit'], **m))
    csv_write(out / 'per_finding_candidates.csv', private_rows)
    discordance = []
    for r in rows:
        if not r['eligible']:
            continue
        n = r['baseline']['gt_voxels']
        counts = r['all_label_gt_voxels']
        side_applicable = len(r['side_labels']) < 5
        lobe_applicable = r['labels'] != r['side_labels']
        discordance.append(dict(id=r['id'], category=r['category'], case=r['case'], finding_id=r['finding_id'],
                                side_applicable=side_applicable, lobe_applicable=lobe_applicable,
                                contralateral_gt_fraction=sum(counts[i] for i in range(10, 15) if i not in r['side_labels']) / n if n else 0,
                                other_same_side_lobe_gt_fraction=sum(counts[i] for i in r['side_labels'] if i not in r['labels']) / n if n else 0))
    csv_write(out / 'prompt_gt_anatomy_discordance.csv', discordance)
    breakdown = []
    for cat in ['all'] + CATEGORIES:
        group = rows if cat == 'all' else [r for r in rows if r['category'] == cat]
        subgroups = {'all': group}
        subgroups.update({f'extent:{x}': [r for r in group if r['extent'] == x] for x in ['focal', 'multifocal', 'diffuse', 'unclear']})
        subgroups.update({f'locality:{x}': [r for r in group if ('unsupported' if not r['eligible'] else 'whole' if len(r['labels']) == 5 else 'side' if r['scope'] in {'L', 'R'} else 'lobe_union') == x] for x in ['unsupported', 'whole', 'side', 'lobe_union']})
        for risk in sorted({x for r in group for x in r['risks']}):
            subgroups['risk:' + risk] = [r for r in group if risk in r['risks']]
        for name, subset in subgroups.items():
            if subset:
                breakdown.extend(dict(category=cat, subgroup=name, policy=k, **summarize(subset, k, False)) for k in policies)
    csv_write(out / 'subgroup_metrics.csv', breakdown)
    per_case = []
    for case in case_records:
        subset = [r for r in rows if r['case'] == case['case']]
        per_case.extend(dict(case=case['case'], policy=k, **summarize(subset, k, False)) for k in policies)
    csv_write(out / 'per_case_metrics.csv', per_case)
    summary = dict(created_at=now(), baseline_dice=c['baseline_dice'], baseline_hits=c['baseline_hits'],
                   findings=len(rows), cases=len(case_records), global_results=global_stats, categories=cats,
                   recommendations={cat: recommendation(cat, cats[cat]) for cat in CATEGORIES},
                   source_review_status=c['source_review_status'], frozen_review_sha256=sha(root / 'frozen_review.json'),
                   report_script_sha256=sha(__file__), semantic_revisions_after_outcomes=0,
                   visual_status=visual_status, visual_manifest_sha256=sha(visual_path) if visual_path.exists() else None,
                   ai_visual_review_sha256=sha(inspection_path) if inspection_path.exists() else None)
    write(out / 'summary.json', summary)
    write(out / 'all_finding_evidence.json', rows)
    routed = [(k, s) for k, s in global_stats.items() if not k.startswith('all_lung')]
    best_key, best = max(routed, key=lambda t: t[1]['after'])
    raw = global_stats['all_lung:mask:0']; fine = global_stats['prompt_fine:mask:0']
    broad = global_stats['prompt_lung:mask:20']
    executive = [
        '## Main findings', '',
        f"- Indiscriminate exact lung clipping: Dice **{raw['after']:.4f}** (Δ {raw['delta']:+.4f}); {raw['worsened']} findings worsen and {raw['hit_losses']} existing hits are lost. An average gain, if present, is not a safety guarantee.",
        f"- Prompt-routed exact fine-region clipping: Dice **{fine['after']:.4f}** (Δ {fine['delta']:+.4f}); {fine['worsened']} findings worsen and {fine['hit_losses']} hits are lost. Explicit location does not guarantee anatomical-mask containment.",
        f"- Highest observed prompt-routed comparison: `{best_key}`, Dice **{best['after']:.4f}** (Δ {best['delta']:+.4f}); {best['improved']} improve, {best['worsened']} worsen; hit gains/losses +{best['hit_gains']}/−{best['hit_losses']}. This row is selected after looking at val200, not an unbiased improvement estimate or an approved method.",
        f"- A broader fixed comparator, `prompt_lung:mask:20`, reaches **{broad['after']:.4f}**, with {broad['worsened']} measured Dice regressions and {broad['tp_removed']} existing TP voxels removed. Yet its minimum GT coverage is {broad['gt_coverage_min']:.1%}: no regression on current predictions does not prove complete anatomical coverage of target voxels the model already missed.",
        '- Compare exact masks against margins and boxes category by category. Dense/atelectatic lung, pleuroparenchymal boundaries, fissures and unsupported non-lung targets have different failure mechanisms. A broad box can preserve tissue that a lung segmentation omits, but also retains more false positives.',
        '- No policy is adopted. Use this audit to choose a later controlled experiment and an acceptable harm budget after reviewing the individual failures.', '']
    introduction = [
        '# Exp007 val200: official-anatomy benefit–harm audit', '',
        f"Baseline: **{c['baseline_dice']:.10f} finding-mean Dice; {c['baseline_hits']}/381 hits**. Exp007 cont e050 / abs e150, existing threshold-0.5 binary predictions; hit criterion Dice ≥0.1.", '',
        'All 200 cases and 381 findings pass native-grid and saved-baseline equivalence checks. GT instance IDs are collapsed with >0 exactly as in the evaluator. Predictions and GT have legacy finding-first identity headers; anatomical geometry comes from the corresponding native CT. No resampling, inference or source changes.', '',
        '**Research audit only. Official source status remains PASS_PENDING_MANUAL_VISUAL_REVIEW. AI semantic/image inspection here is not the outstanding human review.**', '',
        f'Diagnostic overlay status: {visual_status}. See the separate visual review for observed mechanisms, uncertainty, and limitations.', '',
        *executive,
        '## How to read the comparisons', '',
        '- `all_lung`: indiscriminate whole-lung clipping of every finding, including unsuitable pleural targets; a diagnostic, not a proposed rule.',
        '- `prompt_lung`: whole lung only for semantically eligible targets. `prompt_side`: supported side or both lungs. `prompt_fine`: explicit lobe union where supported, otherwise the broader reviewed scope. Unsupported targets are unchanged.',
        '- `mask:0/5/10/20`: intersection with anatomy or its Euclidean dilation in millimetres. `bbox:0/5/10/20`: native-axis bounding box expanded by that physical margin. Bboxes deliberately include non-lung space and are not equivalent to tissue masks.',
        '- Policies and 381 full-prompt interpretations were frozen before measuring new outcomes. Case IDs are joins, not routing inputs. No case chooses its best policy or margin using GT. This manually reviewed val200 routing table is not yet a validated automatic router for unseen prompts.',
        '- All rows keep the full denominator; selected-subset before/after metrics are in the linked CSV. GT retention refers to selected findings; precision and recall are finding means. TP/FP removals are raw voxel sums across native grids, not physical-volume sums.',
        '- Paired 2,000-draw bootstrap resamples CTs (not findings), seed 20260905; category intervals resample its contributing CTs. Sparse categories are descriptive. These exploratory intervals are not adjusted for multiple comparisons or prior model selection; the cohort is not an untouched holdout.',
        '- Dice changes within 1e-7 count as unchanged. Clipping cannot restore a missed target. Component removal, soft priors, crop refinement and training are not tested.', '',
        '- False positive means outside the released finding annotation, not proven clinically false. Annotation extent/completeness and anatomy-model error are separate uncertainties; these overlays do not adjudicate clinical truth.', '',
        '## Navigation', '',
        ' | '.join(f'[{cat}](#category-{cat})' for cat in CATEGORIES), '',
        '[Every prompt and finding](all_findings.md) · [All finding/policy metrics](per_finding_candidates.csv) · [Locality/extent/risk breakdown](subgroup_metrics.csv) · [Per-case metrics](per_case_metrics.csv) · [Machine summary](summary.json) · [Visual review](visual_review.md)', '',
        '## Full-cohort comparisons', '', policy_table(global_stats), '']
    lines = list(introduction)
    side_n = sum(r['side_applicable'] for r in discordance)
    lobe_n = sum(r['lobe_applicable'] for r in discordance)
    side_bad = sum(r['side_applicable'] and r['contralateral_gt_fraction'] > .05 for r in discordance)
    lobe_bad = sum(r['lobe_applicable'] and r['other_same_side_lobe_gt_fraction'] > .05 for r in discordance)
    lines += ['## Prompt–GT–anatomy disagreements', '',
              f'{side_bad}/{side_n} findings with a unilateral supported region have more than 5% of GT assigned to opposite-lung labels. {lobe_bad}/{lobe_n} findings with a finer lobe union have more than 5% of GT in other same-side lung-lobe labels. These are post-measurement descriptive diagnostics, not routing filters. The 5% display threshold does not modify any policy or score.', '',
              'The visual review confirms several target-versus-support conflicts. A named right-middle-lobe nodule is entirely assigned to the right-upper-lobe label; a named right-upper-lobe cyst is mostly assigned to right-middle-lobe label. A singular left-lower-lobe nodule prompt has released GT components in both lungs. These observations cannot by themselves decide whether prompt wording, annotation extent, or anatomical segmentation is wrong.', '',
              'Conversely, the pneumothorax example has high overlap with lung labels because those labels visibly extend into the prompt-described pleural-air region. High coverage is therefore not proof of an anatomically correct compartment. The corresponding prompt-routed policy remains no-op.', '',
              '[All discordance measurements](prompt_gt_anatomy_discordance.csv) · [Image observations](visual_review.md)', '']
    lines += ['## Remaining harms under the highest-observed routed comparison', '',
              'These findings are not bypassed using their outcomes. They remain in the reported score. Narrow-region error may reflect anatomy-model boundaries, prompt/annotation extent disagreement, or both.', '',
              '| Finding | Category | Baseline→constrained Dice | GT retained | Prompt |', '| --- | --- | --- | ---: | --- |']
    for r in rows:
        m = r['candidates'][best_key]
        if m['delta'] < -1e-7:
            lines.append(f"| [{r['id']:03}](all_findings.md#finding-{r['id']:03}) | {r['category']} | {r['baseline']['dice']:.4f}→{m['dice']:.4f} | {m['gt_coverage']:.1%} | {r['prompt']} |")
    lines.append('')
    for cat in CATEGORIES:
        group = [r for r in rows if r['category'] == cat]
        lines += [f'<a id="category-{cat}"></a>', f'## {cat}: {TITLES[cat]}', '', CONTEXT[cat], '']
        if not group:
            lines += [recommendation(cat, cats[cat]), '']; continue
        lines += [f"{len(group)} findings / {len({r['case'] for r in group})} CTs. Extent: {dict(Counter(r['extent'] for r in group))}. Eligible: {sum(r['eligible'] for r in group)}. Baseline Dice {np.mean([r['baseline']['dice'] for r in group]):.4f}; hits {sum(r['baseline']['hit'] for r in group)}/{len(group)}.", '',
                  policy_table(cats[cat]), '', '**Interpretation:** ' + recommendation(cat, cats[cat]), '',
                  '### Locality, extent and boundary risks', '',
                  'The table below uses the fixed `prompt_fine:mask:0` comparison, not a best-per-subgroup selection. All 32 policies and all risk groups are available in the subgroup CSV.', '',
                  '| Subgroup | N | Δ Dice | Worsened | Hit losses |', '| --- | ---: | ---: | ---: | ---: |']
        for s in breakdown:
            if s['category'] == cat and s['policy'] == 'prompt_fine:mask:0' and s['subgroup'] != 'all':
                lines.append(f"| {s['subgroup']} | {s['n']} | {s['delta']:+.4f} | {s['worsened']} | {s['hit_losses']} |")
        pairs = [(r, k, m) for r in group for k, m in r['candidates'].items()]
        gain = max(pairs, key=lambda t: t[2]['delta']); harm = min(pairs, key=lambda t: t[2]['delta'])
        lines += ['', '### Largest observed changes (diagnostic selections)', '']
        dc = [r for r in discordance if r['category'] == cat]
        lines += [f"Prompt/anatomy discordance above the descriptive 5% GT threshold: {sum(r['side_applicable'] and r['contralateral_gt_fraction'] > .05 for r in dc)} opposite-side findings; {sum(r['lobe_applicable'] and r['other_same_side_lobe_gt_fraction'] > .05 for r in dc)} finer-lobe findings. This is not used to remove cases or alter routes.", '']
        for name, (r, k, m) in [('Gain', gain), ('Harm', harm)]:
            lines += [f"- {name}: [finding {r['id']:03}](all_findings.md#finding-{r['id']:03}), `{k}`, Dice {r['baseline']['dice']:.4f} → {m['dice']:.4f}; GT retained {m['gt_coverage']:.1%}. Global-mean contribution {m['delta']/381:+.6f}. These extrema are selected for failure analysis, not performance estimation."]
        lines += ['', f'[All {cat} prompts](all_findings.md#category-{cat})', '']
    (out / 'report.md').write_text('\n'.join(lines))
    lut = read(root / 'source_lut.json'); lut['0'] = 'unlabeled_background'
    appendix = ['# Every val200 finding: frozen interpretation and measured evidence', '',
                '[Back to category report](report.md)', '', 'Anatomical labels are model predictions, not GT. All-label GT overlap below is diagnostic only and never selects a routed label.', '']
    appendix += ['Scope legend: `N` unsupported/no-op; `W` whole lung with unspecified side; `B` both lungs; `L/R` left/right lung; `U/D` both upper/lower lobes; `LU/LL` left upper/lower lobe; `RU/RM/RL` right upper/middle/lower lobe. `+` is a union, not an intersection. Zone/segment words never create labels that do not exist.', '']
    for cat in CATEGORIES:
        appendix += [f'<a id="category-{cat}"></a>', f'## Category {cat}', '']
        for r in [r for r in rows if r['category'] == cat]:
            b = r['baseline']; total = b['gt_voxels']
            overlaps = [(lut.get(str(i), str(i)), v / total if total else 0) for i, v in enumerate(r['all_label_gt_voxels']) if v]
            overlaps.sort(key=lambda x: -x[1])
            appendix += [f'<a id="finding-{r["id"]:03}"></a>', f'### Finding {r["id"]:03}: {r["case"]} / {r["finding_id"]}', '',
                         '> ' + r['prompt'], '', f"Extent: **{r['extent']}**. Scope: `{r['scope']}`; exact lobe label union: `{r['labels']}`; eligible: {r['eligible']}. Risks: {', '.join(r['risks']) or 'none explicitly described'}. Uncertainty: {', '.join(r['uncertainty']) or 'none explicitly described'}.", '',
                         'Available support: ' + (', '.join(lut[str(i)].replace('_', ' ') for i in r['labels']) if r['labels'] else 'No suitable exact target label; adjacent organ labels are not substitutes.') + '.', '',
                         r['rationale'], '', r['negation_note'], '',
                         'Unavailable anatomy: ' + (' '.join(r['unavailable_anatomy']) or 'No finer lesion-specific anatomy mask; support is not a lesion boundary.'), '',
                         f"Baseline Dice {b['dice']:.6f}; hit {b['hit']}; precision {b['precision']:.4f}; recall {b['recall']:.4f}; GT {b['gt_voxels']} voxels; prediction {b['pred_voxels']} voxels.", '',
                         'GT overlap by official total label: ' + '; '.join(f'{name} {v:.2%}' for name, v in overlaps) + '.', '',
                         '| Policy | Selected | Dice | Δ | GT retained | TP/FP removed | Hit |', '| --- | --- | ---: | ---: | ---: | --- | --- |']
            for k, m in r['candidates'].items():
                appendix.append(f"| {k} | {m['selected']} | {m['dice']:.6f} | {m['delta']:+.6f} | {m['gt_coverage']:.4f} | {m['tp_removed']}/{m['fp_removed']} | {m['hit']} |")
            visual_link = f'[Visual selection/review](visual_review.md#finding-{r["id"]:03})' if r['id'] in visual_ids else 'Not selected for diagnostic overlays; all numeric comparisons above are complete.'
            appendix += ['', visual_link + f' · [Category discussion](report.md#category-{cat})', '']
    (out / 'all_findings.md').write_text('\n'.join(appendix))
    compact = [
        '---', 'created: 2026-09-05', 'status: audit_complete_for_user_review' if inspection else 'status: quantitative_audit_complete_visual_review_pending', '---', '',
        '# Exp022: official anatomy on Exp007 val200', '',
        f"All 200 scans / 381 findings reproduce Exp007 cont e050 / abs e150: **Dice {c['baseline_dice']:.6f}; {c['baseline_hits']}/381 hits**. No source masks, predictions, training or inference changed.", '',
                'Official CT-RATE masks remain pending human visual review. Findings are exploratory on a previously used validation cohort, not held-out gains or deployment approval.', '',
        f'Diagnostic overlay status: {visual_status}.', '',
        *executive,
        '## Full-cohort comparisons', '', policy_table(global_stats), '',
        '## Category conclusions', '',
        '| Category | Findings | Baseline Dice | Recommendation |', '| --- | ---: | ---: | --- |']
    for cat in CATEGORIES:
        group = [r for r in rows if r['category'] == cat]
        compact.append(f"| {cat} | {len(group)} | {fmt(float(np.mean([r['baseline']['dice'] for r in group])) if group else None)} | {recommendation(cat, cats[cat])} |")
    compact += ['', '## Interpretation and evidence', '',
                'The 381 complete prompts were individually interpreted and frozen before new outcomes. Unsupported pleural, central-airway-wall, and aberrant-vessel targets bypass prompt routing. Bilateral distributions are not narrowed to their largest/dominant lesion. Lobe masks are broader parents, not bronchopulmonary segment masks.', '',
                'All policies use fixed 0/5/10/20 mm mask/bbox margins; no GT-selected per-finding routing. Best observed category rows are labeled validation-selected descriptions. Raw TP/FP totals count voxels, not physical volumes. Selected-subset and per-case metrics, all-label GT overlap and all risk/extent/locality breakdowns are in the external report. Sparse categories cannot support confident generalization.', '',
                'External report: `' + str(out / 'report.md') + '`.',
                'All-prompt appendix: `' + str(out / 'all_findings.md') + '`.',
                'Visual review: `' + str(out / 'visual_review.md') + '`.', '',
                'Reproduction: run the measurement entrypoint against its frozen runtime seal, then `report_022_official_anatomy.py`. Source hashes, the preserved original seal and the documented GT-instance-format correction are external. Report-script hash: `' + sha(__file__) + '`.', '',
                '**Stop for user review. No policy is adopted.**', '']
    (out / 'aggregate_report.md').write_text('\n'.join(compact))
    print(json.dumps({k: {x: v[x] for x in ['after', 'delta', 'worsened', 'hit_losses']} for k, v in global_stats.items()}, indent=2))


if __name__ == '__main__':
    render()
