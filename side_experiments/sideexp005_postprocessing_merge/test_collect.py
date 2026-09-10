"""Independent completion and artifact checks for aggregate publication."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import collect
from runner import atomic_json, digest, job_digest, sha


class CollectionTests(unittest.TestCase):
    def fixture(self, root):
        cfg = {'runtime_root': str(root)}
        cases = [{'name': f'case{i:03}.nii.gz'} for i in range(200)]
        job = {'config_sha256': digest(cfg), 'cases': cases}
        job['job_sha256'] = job_digest(job)
        atomic_json(root/'job_manifest.json', job)
        inventories, rows, metrics = {}, [], {}
        for variant in collect.VARIANTS:
            files = []
            pred = root/'predictions'/variant; pred.mkdir(parents=True)
            for i, case in enumerate(cases):
                path = pred/case['name']; path.write_bytes(b'verified-fixture')
                n = 2 if i < 181 else 1
                files.append({'path': str(path), 'sha256': sha(path), 'shape': [n, 2, 2, 2], 'dtype': 'uint8'})
                rows.extend({'variant': variant, 'case': case['name'], 'finding_index': j,
                             'gt_voxels': 1, 'pred_voxels': 1, 'intersection_voxels': 1, 'dice': 1., 'hit': True}
                            for j in range(n))
            path = root/'manifests'/(variant+'.json')
            atomic_json(path, {'cases': 200, 'findings': 381, 'files': files})
            inventories[variant] = sha(path)
            metrics[variant] = {'cases': 200, 'findings': 381, 'dice': 1., 'hits': 381, 'case_wise_dice': 1.}
        reports = root/'reports'; reports.mkdir()
        (reports/'report.md').write_text('Verified aggregate fixture.\n')
        done = {'status': 'complete', 'job_sha256': job['job_sha256'], 'files': 1000,
                'finding_evaluations': 1905, 'metrics': metrics, 'changes': {},
                'inventory_hashes': inventories, 'report_sha256': sha(reports/'report.md')}
        atomic_json(root/'completion.json', done)
        atomic_json(reports/'summary.json', done)
        atomic_json(reports/'per_finding.json', rows)
        return cfg

    def test_full_inventory_and_metric_verification(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); cfg = self.fixture(root)
            done, _ = collect.verify_completion(cfg, root)
            self.assertEqual(done['files'], 1000)
            (root/'predictions/d11/case000.nii.gz').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
                collect.verify_completion(cfg, root)

    def test_rejects_report_and_finding_metric_drift(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); cfg = self.fixture(root)
            report = root/'reports/report.md'; original = report.read_bytes()
            report.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'report hash drift'):
                collect.verify_completion(cfg, root)
            report.write_bytes(original)
            path = root/'reports/per_finding.json'; rows = json.loads(path.read_text()); rows[0]['dice'] = .5
            atomic_json(path, rows)
            with self.assertRaisesRegex(ValueError, 'metric/count disagreement'):
                collect.verify_completion(cfg, root)


if __name__ == '__main__':
    unittest.main()
