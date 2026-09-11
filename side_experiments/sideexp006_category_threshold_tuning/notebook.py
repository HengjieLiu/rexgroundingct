#!/usr/bin/env python3
"""Execute and export a self-contained review notebook from completed sweep CSVs."""
import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/sideexp006-matplotlib')
os.environ.setdefault('IPYTHONDIR', '/tmp/sideexp006-ipython')
os.environ.setdefault('JUPYTER_RUNTIME_DIR', '/tmp/sideexp006-jupyter')

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from sweep import RUN_ROOT, atomic_json, file_hash, read


def build(root):
    root = Path(root).resolve()
    completion = read(root/'completion.json')
    manifest = read(root/'run_manifest.json')
    if completion['status'] != 'complete' or completion['identity'] != manifest['identity']:
        raise ValueError('Sweep is not complete or source identity differs')
    for name, sha in completion['outputs'].items():
        if file_hash(root/name) != sha:
            raise ValueError(f'Completed output hash mismatch: {name}')
    recipe = manifest['contract']['recipe']
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    notebook = nbformat.v4.new_notebook(metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'sweep_identity': manifest['identity'],
    })
    notebook.cells = [
        md(f"# {recipe.upper()} · category threshold review\n\n"
           "**Fixed val200 · 200 cases · 381 findings · 19 thresholds**\n\n"
           "Raw Dice and hit rate from saved logits. All figures and tables are embedded; "
           "no kernel or model inference is needed to view this notebook.\n\n"
           "This is exploratory tuning on the same validation set. Observed maxima are "
           "descriptive; no category thresholds have been adopted."),
        code(f"RUN_ROOT = {str(root)!r}\n" + '''%matplotlib inline
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

root = Path(RUN_ROOT)
manifest = json.loads((root / 'run_manifest.json').read_text())
summary = json.loads((root / 'summary.json').read_text())
completion = json.loads((root / 'completion.json').read_text())
assert manifest['identity'] == summary['identity'] == completion['identity']
for name, expected_hash in completion['outputs'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected_hash, name
findings = pd.read_csv(root / 'finding_metrics.csv')
categories = pd.read_csv(root / 'category_metrics.csv')
overall = pd.read_csv(root / 'overall_metrics.csv')
comparison = pd.read_csv(root / 'comparison.csv')
assert len(findings) == 7239
assert not findings.duplicated(['case', 'finding_index', 'threshold_pct']).any()
assert findings['case'].nunique() == 200
assert findings[['case', 'finding_index']].drop_duplicates().shape[0] == 381
assert set(findings['threshold_pct']) == set(range(5, 100, 5))
assert len(overall) == 19 and len(categories) == 14 * 19
assert (categories.groupby('threshold_pct')['findings'].sum() == 381).all()
for _, r in overall.iterrows():
    selected = findings[findings.threshold_pct == r.threshold_pct]
    assert abs(selected.dice.mean() - r.dice) < 1e-12
    assert int(selected.hit.sum()) == r.hits
for _, r in categories.iterrows():
    selected = findings[(findings.threshold_pct == r.threshold_pct) & (findings.category == r.category)]
    assert len(selected) == r.findings
    if len(selected):
        assert abs(selected.dice.mean() - r.dice) < 1e-12
        assert int(selected.hit.sum()) == r.hits
baseline = overall[overall.threshold_pct == 50].iloc[0]
if summary.get('expected_baseline'):
    assert abs(baseline.dice - summary['expected_baseline']['dice']) < 1e-10
    assert baseline.hits == summary['expected_baseline']['hits']
plt.rcParams.update({'figure.dpi': 130, 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titleweight': 'bold'})
DICE_COLOR, HIT_COLOR = '#176b9b', '#bd6630'
display(Markdown('**Verification passed:** 7,239 unique finding–threshold records; all category tables recompose exactly.'))
'''),
        md("## Metric definitions and provenance\n\n"
           "- **Dice:** DSC of the raw thresholded mask for each finding; average findings within each category. "
           "Use `(2 × intersection + 1e-6) / (GT voxels + predicted voxels + 1e-6)`; both-empty masks score 1.\n"
           "- **Hit rate:** fraction of findings with Dice ≥ 0.1.\n"
           "- **Threshold:** probability ≥ t, with no connected-component filtering or anatomy postprocessing. "
           "Singleton recipes use the equivalent float32 logit cutoff.\n"
           "- **Storage:** the sweep evaluates saved logits. Float16 storage was validated to preserve "
           "the same-pass 0.50 masks; exact equivalence to unstored float32 outputs at other thresholds is not established.\n"
           "- Instance precision, recall and F1 are deferred. They are different metrics requiring component matching."),
        code('''contract = manifest['contract']
source_rows = [{'model': s['candidate_id'], 'checkpoint SHA256': s['checkpoint_sha256'],
                'cache key': s['cache_key']} for s in contract['sources']]
display(pd.DataFrame(source_rows))
display(Markdown(f"**Run identity:** `{manifest['identity']}`  \\n"
                 f"**Fixed split SHA256:** `{contract['dataset_sha256']}`  \\n"
                 f"**Source commit:** `{contract['code_commit']}`  \\n"
                 f"**Output folder:** `{RUN_ROOT}`"))
references = [{'reference': 'Sweep at 0.50', 'Dice': baseline.dice, 'hits': int(baseline.hits)}]
for key, label in [('expected_baseline', 'Independent expected baseline'), ('historical_reference', 'Historical separate inference')]:
    h = summary.get(key)
    if h:
        references.append({'reference': label, 'Dice': h['dice'], 'hits': h['hits']})
        display(Markdown(label + ' source: `' + h['source'] + '`'))
display(pd.DataFrame(references).style.format({'Dice': '{:.10f}'}))
'''),
        md("## Overall threshold curves\n\nThe dashed line marks 0.50. Stars mark the best observed threshold for each metric."),
        code('''fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for ax, metric, label, color in zip(axes, ['dice', 'hit_rate'], ['Mean finding Dice', 'Hit rate'], [DICE_COLOR, HIT_COLOR]):
    ax.plot(overall.threshold, overall[metric], 'o-', color=color, markersize=4)
    ax.axvline(.5, color='#7b8792', linestyle='--', linewidth=1)
    ranked = overall.assign(distance=(overall.threshold_pct-50).abs()).sort_values(
        [metric, 'distance', 'threshold_pct'], ascending=[False, True, True])
    best = ranked.iloc[0]
    ax.scatter([best.threshold], [best[metric]], marker='*', s=150, color=color, zorder=4)
    ax.set(title=f"{label} · best t={best.threshold:.2f}", xlabel='Probability threshold', ylabel=label,
           xlim=(.025, .975), ylim=(0, 1))
    ax.set_xticks(np.arange(.05, 1, .1))
    ax.grid(alpha=.2)
plt.show()
'''),
        md("## Every official category\n\nBlue: mean finding Dice. Orange: hit rate. "
           "Each panel uses a common 0–1 scale. Categories with fewer than 10 findings are marked **small sample**."),
        code('''fig, axes = plt.subplots(7, 2, figsize=(14, 24), constrained_layout=True)
for ax, (category, group) in zip(axes.flat, categories.groupby('category', sort=False)):
    first = group.iloc[0]
    n = int(first.findings)
    note = ' · SMALL SAMPLE' if 0 < n < 10 else ''
    import textwrap
    title = f"{category} · {first.label}\\nn={n}{note}"
    ax.set(title='\\n'.join(textwrap.fill(line, 52) for line in title.split('\\n')),
           xlabel='Probability threshold', ylabel='Score', xlim=(.025, .975), ylim=(0, 1))
    ax.set_xticks(np.arange(.05, 1, .1))
    if n == 0:
        ax.text(.5, .5, 'No validation findings\\nNo threshold estimate', transform=ax.transAxes,
                ha='center', va='center', fontsize=13, color='#68747c')
        ax.set_facecolor('#f3f5f6')
        continue
    ax.plot(group.threshold, group.dice, 'o-', color=DICE_COLOR, markersize=3, label='Dice')
    ax.plot(group.threshold, group.hit_rate, 's-', color=HIT_COLOR, markersize=3, label='Hit rate')
    ax.axvline(.5, color='#7b8792', linestyle='--', linewidth=1, label='0.50 reference')
    for metric, color in [('dice', DICE_COLOR), ('hit_rate', HIT_COLOR)]:
        best = group.assign(distance=(group.threshold_pct-50).abs()).sort_values(
            [metric, 'distance', 'threshold_pct'], ascending=[False, True, True]).iloc[0]
        ax.scatter([best.threshold], [best[metric]], marker='*', s=95, color=color, zorder=4)
    ax.grid(alpha=.2)
    ax.legend(loc='upper right', fontsize=8)
plt.show()
'''),
        md("## Baseline and observed maxima\n\n"
           "Dice and hit-rate maxima are selected independently. Ties favor the threshold closest to 0.50, "
           "then the lower threshold. Deltas are absolute score differences. Missing values mean no validation findings."),
        code('''table = comparison.copy()
table['sample note'] = table.findings.map(lambda n: 'no data' if n == 0 else ('small sample' if n < 10 else ''))
display(table.style.format({
    'baseline_dice': '{:.6f}', 'baseline_hit_rate': '{:.4f}',
    'best_dice_threshold': '{:.2f}', 'best_dice': '{:.6f}', 'dice_delta': '{:+.6f}',
    'best_hit_rate_threshold': '{:.2f}', 'best_hit_rate': '{:.4f}', 'hit_rate_delta': '{:+.4f}',
}, na_rep='N/A').hide(axis='index'))
'''),
        md("## Review notes and downloadable tables\n\n"
           "Review whether gains are broad or driven by a few findings, particularly in small categories. "
           "The same validation data supplies both the selected maxima and their displayed scores, so these "
           "are tuning estimates. Choosing deployment thresholds or adding instance metrics is a subsequent decision.\n\n"
           "Companion files: `finding_metrics.csv`, `category_metrics.csv`, `overall_metrics.csv`, "
           "`comparison.csv`, `summary.json`, and `run_manifest.json`. "
           "B1, D1, and E1 can use the same presentation with separate source-bound runs."),
    ]
    NotebookClient(notebook, timeout=300, kernel_name='python3',
                   resources={'metadata': {'path': str(root)}}).execute()
    code_cells = [c for c in notebook.cells if c.cell_type == 'code']
    if any(c.execution_count is None or any(o.output_type == 'error' for o in c.outputs) for c in code_cells):
        raise ValueError('Notebook contains an unexecuted or failed cell')
    images = sum('image/png' in o.get('data', {}) for c in code_cells for o in c.outputs)
    if images < 2:
        raise ValueError('Notebook plots are not embedded')
    prefix = f'{recipe}_val200_thresholds'
    ipynb = root/f'{prefix}.executed.ipynb'
    tmp = root/f'.{prefix}.tmp.ipynb'
    nbformat.write(notebook, tmp)
    tmp.replace(ipynb)
    exporter = HTMLExporter(template_name='lab')
    exporter.exclude_input = True
    body, _ = exporter.from_notebook_node(notebook)
    html = root/f'{prefix}.html'
    html.write_text(body)
    verification = dict(status='complete', identity=manifest['identity'],
                        notebook_builder_sha256=file_hash(__file__), code_cells=len(code_cells),
                        error_cells=0, embedded_figures=images,
                        notebook=str(ipynb), notebook_sha256=file_hash(ipynb),
                        html=str(html), html_sha256=file_hash(html))
    atomic_json(root/'notebook_verification.json', verification)
    print(verification)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=RUN_ROOT)
    build(parser.parse_args().run_root)
