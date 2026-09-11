"""Small synthetic end-to-end check of executed outputs and embedded figures."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import sweep as s


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in
                       ['nbformat', 'nbclient', 'nbconvert', 'pandas', 'matplotlib', 'ipykernel']),
                     'Notebook integration test requires the VoxTell image dependencies')
class NotebookTests(unittest.TestCase):
    def test_executed_artifact_and_hash_gate(self):
        import nbformat
        from notebook import build
        with tempfile.TemporaryDirectory(prefix='sideexp006-notebook-') as directory:
            root = Path(directory)
            rows = []
            for i in range(381):
                category = '1a' if i < 3 else ('2g' if i == 3 else '2d')
                for pct in s.PERCENTAGES:
                    g, p, intersection = 100, 200-pct, 50
                    d = s.dice(g, p, intersection)
                    rows.append(dict(case=f'case{i % 200}.nii.gz', finding_index=i//200,
                                     category=category, threshold_pct=pct, threshold=pct/100,
                                     gt_voxels=g, pred_voxels=p, intersection=intersection,
                                     dice=d, hit=int(d >= .1)))
            metrics, comparisons = s.aggregate(rows)
            contract = dict(recipe='d1', sources=[], dataset_sha256='synthetic', code_commit='synthetic')
            s.atomic_json(root/'run_manifest.json', dict(identity='synthetic', contract=contract))
            s.atomic_json(root/'summary.json', dict(identity='synthetic'))
            for name, data in [('finding_metrics', rows), ('comparison', comparisons),
                               ('overall_metrics', [r for r in metrics if r['category'] == 'overall']),
                               ('category_metrics', [r for r in metrics if r['category'] != 'overall'])]:
                s.write_csv(root/f'{name}.csv', data)
            outputs = {p.name: s.file_hash(p) for p in [root/'summary.json', *root.glob('*.csv')]}
            s.atomic_json(root/'completion.json', dict(status='complete', identity='synthetic', outputs=outputs))
            build(root)
            verification = s.read(root/'notebook_verification.json')
            self.assertEqual(verification['error_cells'], 0)
            self.assertEqual(verification['embedded_figures'], 2)
            self.assertEqual(verification['code_cells'], 5)
            nb = nbformat.read(verification['notebook'], as_version=4)
            self.assertTrue(all(c.execution_count for c in nb.cells if c.cell_type == 'code'))
            self.assertIn('data:image/png;base64', Path(verification['html']).read_text())
            (root/'overall_metrics.csv').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                build(root)


if __name__ == '__main__':
    unittest.main()
