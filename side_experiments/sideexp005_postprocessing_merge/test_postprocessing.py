"""Source parity and execution-contract tests for SideExp005."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import methods
import runner

runner.scientific()
np, nib = runner.np, runner.nib
import apply_semantic_constraint_best_policy as source_v1
import apply_semantic_v2_strict_to_rex_predictions as source_v2
import run_024_test_inference_anatomy as ours


def synthetic(shape=(48, 52, 44)):
    a = np.zeros(shape, dtype=np.uint8)
    a[8:20, 8:42, 25:38] = 10
    a[8:20, 8:42, 6:25] = 11
    a[29:42, 8:42, 28:38] = 12
    a[29:42, 8:42, 21:28] = 13
    a[29:42, 8:42, 6:21] = 14
    return a


PROMPTS = [
    ("nodule in the right upper lobe", "2d"),
    ("opacity predominantly in the left lower lobe", "2b"),
    ("subpleural nodule in the right upper lobe", "2d"),
    ("multifocal nodules in the right upper lobe", "2d"),
    ("pulmonary nodule", "2d"), ("right pleural effusion", "2e"),
    ("no opacity in the right upper lobe", "2b"),
    ("resolved right upper lobe opacity", "2b"), ("RUL nodule", "2d"),
    ("nodule in the upper lobe of the right lung", "2d"),
    ("bilateral upper lobe opacities", "2b"),
    ("apical pulmonary scarring", "2a"), ("basal opacity", "2b"),
    ("superior segment of the left lower lobe nodule", "2d"),
    ("laterobasal nodule in the right lower lobe", "2d"),
    ("subpleural peripheral opacity in the posterior right lower lobe", "2c"),
    ("opacity including apical scarring", "2a"),
    ("nodules in right upper lobe and right middle lobe", "2d"),
    ("right pneumothorax", "2g"),
]


def findings(prompts=PROMPTS):
    return [{"finding_idx": i, "prompt": p, "category": c} for i, (p, c) in enumerate(prompts)]


class SourceParity(unittest.TestCase):
    def reference(self, root, raw, anatomy, affine, fs):
        case = "synthetic.nii.gz"
        dirs = {k: root / k for k in ("raw", "ct", "priors", "d11", "d12")}
        for d in dirs.values():
            d.mkdir()
        prior_case = dirs["priors"] / "synthetic"
        prior_case.mkdir()
        nib.save(nib.Nifti1Image(raw, affine), dirs["raw"] / case)
        nib.save(nib.Nifti1Image(np.zeros(anatomy.shape, dtype=np.uint8), affine), dirs["ct"] / case)
        priors = methods.anatomy_priors(anatomy)
        sampling = tuple(nib.affines.voxel_sizes(affine))
        priors["whole10"] = methods.whole10(priors, sampling)
        names = {"whole": "whole_lung", "whole10": "lung_dilated_10mm", "left": "left_lung", "right": "right_lung", **source_v1.LOBE_FILES}
        for key, filename in names.items():
            nib.save(nib.Nifti1Image(priors[key].astype(np.uint8), affine), prior_case / (filename + ".nii.gz"))
        settings = {"pred_dir": str(dirs["raw"]), "prior_dir": str(dirs["priors"]), "ct_dir": str(dirs["ct"]),
                    "output_dir": str(dirs["d11"]), "threshold": 0.5, "prediction_is_probability": False, "gt_dir": ""}
        _, error = source_v1.process_case((case, fs, settings))
        self.assertIsNone(error)
        settings.update(pred_dir=str(dirs["d11"]), output_dir=str(dirs["d12"]))
        _, error = source_v2.case_worker((case, fs, settings))
        self.assertIsNone(error)
        return {v: np.asanyarray(nib.load(dirs[v] / case).dataobj) for v in ("d11", "d12")}

    def test_original_deployment_workers_and_ours(self):
        fs = findings()
        anatomy = synthetic()
        rng = np.random.default_rng(43)
        raw = (rng.random((len(fs), *anatomy.shape)) > 0.7).astype(np.uint8)
        for affine in (np.diag([-2., -1.25, 3., 1.]), np.array([[0, -1.25, 0, 30], [2., 0, 0, -15], [0, 0, -3., 80], [0, 0, 0, 1.]])):
            routes = [methods.route_prompt(f["prompt"], f["category"]) for f in fs]
            result, details = methods.postprocess(raw, anatomy, affine, fs, routes)
            with tempfile.TemporaryDirectory() as temp:
                reference = self.reference(Path(temp), raw, anatomy, affine, fs)
            for v in ("d11", "d12"):
                np.testing.assert_array_equal(result[v], reference[v])
            for f, route in zip(fs, routes):
                i = f["finding_idx"]
                np.testing.assert_array_equal(result["d2"][i], ours.apply_support(raw[i], anatomy, affine, route, "whole_lung"))
                np.testing.assert_array_equal(result["d3"][i], ours.apply_support(raw[i], anatomy, affine, route, "fine"))
                if not details[i]["collaborator"]["v2_selected"]:
                    np.testing.assert_array_equal(result["d12"][i], result["d11"][i])
            self.assertTrue(np.all(result["d12"] <= result["d11"]))
            self.assertTrue(all(np.all(v <= raw) for v in result.values()))
            np.testing.assert_array_equal(result["d2"][5], raw[5])
            self.assertLess(result["d11"][5].sum(), raw[5].sum())

    def test_empty_anatomy_matches_source(self):
        anatomy = np.zeros((8, 9, 10), dtype=np.uint8)
        fs = findings([("apical opacity", "2a"), ("right upper lobe nodule", "2d")])
        raw = np.ones((2, *anatomy.shape), dtype=np.uint8)
        affine = np.diag([2., 3., 4., 1.])
        result, _ = methods.postprocess(raw, anatomy, affine, fs, [methods.route_prompt(f['prompt'], f['category']) for f in fs])
        with tempfile.TemporaryDirectory() as temp:
            ref = self.reference(Path(temp), raw, anatomy, affine, fs)
        for v in ref:
            np.testing.assert_array_equal(ref[v], result[v])

    def test_dilation_boundary_and_axis_rejection(self):
        anatomy = synthetic()
        p = methods.anatomy_priors(anatomy)
        sampling = (2., 1.25, 3.)
        expected = methods.ndimage.distance_transform_edt(~p["whole"], sampling=sampling) <= 10.
        np.testing.assert_array_equal(methods.whole10(p, sampling), expected)
        bad = np.eye(4); bad[0, 1] = .2
        with self.assertRaises(ValueError):
            methods.world_axis_info(bad, anatomy.shape)

    def test_wrong_prediction_geometry_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); ct = root/'ct.nii.gz'; pred = root/'pred.nii.gz'
            affine = np.diag([-2., -3., 4., 1.])
            nib.save(nib.Nifti1Image(np.zeros((8, 9, 10), dtype=np.uint8), affine), ct)
            nib.save(nib.Nifti1Image(np.zeros((1, 8, 9, 10), dtype=np.uint8), np.eye(4)), pred)
            case = {'shape': [1, 8, 9, 10], 'affine': affine.tolist(), 'ct_path': str(ct)}
            with self.assertRaisesRegex(ValueError, 'affine mismatch'):
                runner.validate_nifti(pred, case)

    def test_real_frozen_parser_counts(self):
        cfg = runner.read(methods.ROOT / 'config.json')
        cases = runner.read(cfg['inputs']['dataset']['path'])['test']
        count = collections.Counter()
        for case in cases:
            for f in methods.ordered_findings(case):
                parsed = methods.parse_finding(f['prompt'])
                count[parsed['roi_kind']] += 1
                count['v2_selected'] += int(parsed['v2_selected'])
        self.assertEqual(dict(count), cfg['routing_counts'])

    def test_original_transform_case_dir(self):
        # Exercise the deployed d2/d3 directory transformer, not only its helper.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); name = 'train_1_a_1.nii.gz'
            anatomy = synthetic(); affine = np.diag([-2., -1.25, 3., 1.])
            fs = findings(PROMPTS[:6]); raw = np.ones((len(fs), *anatomy.shape), dtype=np.uint8)
            src = root/'raw'; src.mkdir(); ctroot = root/'ct'
            ctpath = ctroot/'dataset/train_fixed/train_1/train_1_a'/name; ctpath.parent.mkdir(parents=True)
            nib.save(nib.Nifti1Image(raw, affine), src/name)
            nib.save(nib.Nifti1Image(np.zeros(anatomy.shape, dtype=np.uint8), affine), ctpath)
            maskpath = root/'anatomy.nii.gz'; nib.save(nib.Nifti1Image(anatomy, affine), maskpath)
            rs = [methods.route_prompt(f['prompt'], f['category']) for f in fs]
            metadata = {'val': [{'name': name, 'findings': {str(f['finding_idx']): f['prompt'] for f in fs},
                                 'categories': {str(f['finding_idx']): f['category'] for f in fs}}]}
            result, _ = methods.postprocess(raw, anatomy, affine, fs, rs)
            with mock.patch.object(ours, 'CT_ROOT', ctroot):
                for variant, policy in [('d2', 'whole_lung'), ('d3', 'fine')]:
                    ours.transform_case_dir(src, root/variant, {(name, i): r for i, r in enumerate(rs)}, policy,
                                            {name: maskpath}, metadata, 'val')
                    np.testing.assert_array_equal(result[variant], np.asanyarray(nib.load(root/variant/name).dataobj))


class ExecutionContracts(unittest.TestCase):
    def test_numeric_order_and_invalid_indices(self):
        case = {'findings': {str(i): str(i) for i in reversed(range(12))}, 'categories': {str(i): '2d' for i in range(12)}}
        self.assertEqual([f['finding_idx'] for f in methods.ordered_findings(case)], list(range(12)))
        case['findings'].pop('4')
        with self.assertRaises(ValueError): methods.ordered_findings(case)

    def test_mixed_precision_chunking_and_hash_failure(self):
        values = [np.array([4., 0., -8., 1., 5., -2., .1], dtype=d) for d in (np.float16, np.float32, np.float16, np.float32)]
        values[1][0] = -1.; values[2][0] = -1.; values[3][0] = -1.
        with tempfile.TemporaryDirectory() as temp:
            refs = []
            for i, a in enumerate(values):
                path = Path(temp)/f'{i}.npy'; np.save(path, a.reshape(1, 1, 1, -1))
                refs.append({'path': str(path), 'sha256': hashlib.sha256(a.tobytes()).hexdigest(), 'dtype': a.dtype.name})
            probs = {str(i): methods.sigmoid_float32(a) for i, a in enumerate(values)}
            from preliminary_ensemble import generate_probability_ensembles
            ref, _ = generate_probability_ensembles(probs, {'uniform': {str(i): 1 for i in range(4)}})['uniform']
            for chunk in (1, 3, 7, 8):
                out = runner.average_logits(refs, (1, 1, 1, 7), chunk, collections.defaultdict(float)).ravel()
                np.testing.assert_array_equal(out, ref >= 2.)
                self.assertEqual(out[1], 1)  # exact 0.5 tie
                self.assertNotEqual(out[0], int(sum(a[0] for a in values) >= 0))
            refs[0]['sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'source array hash'):
                runner.average_logits(refs, (1, 1, 1, 7), 3, collections.defaultdict(float))

    def test_space_lock_and_manifest_tampering(self):
        cfg = {'storage': {'reserve_bytes': 20, 'headroom_bytes': 5}}
        runner.check_space(cfg, free_bytes=25)
        with self.assertRaisesRegex(ValueError, 'insufficient'): runner.check_space(cfg, free_bytes=24)
        with tempfile.TemporaryDirectory() as temp:
            with runner.job_lock(temp):
                with self.assertRaisesRegex(ValueError, 'live'):
                    with runner.job_lock(temp): pass
        job = {'config_sha256': runner.digest({}), 'code': [], 'candidates': [], 'frozen_at_utc': 'x'}
        job['job_sha256'] = runner.job_digest(job)
        job['code'] = ['tampered']
        with self.assertRaisesRegex(ValueError, 'manifest hash'): runner.verify_job({}, job)

    def test_atomic_publication_recovery_and_conflicts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); ct = root/'ct.nii.gz'; affine = np.diag([-2., -3., 4., 1.])
            img = nib.Nifti1Image(np.zeros((8, 9, 10), dtype=np.uint8), affine); nib.save(img, ct)
            case = {'name': 'case.nii.gz', 'shape': [1, 8, 9, 10], 'affine': affine.tolist(), 'ct_path': str(ct)}
            cfg = {'runtime_root': str(root)}; job = {'job_sha256': 'job'}
            dest, staged = runner.artifact_paths(root, case, 'd1'); dest.parent.mkdir(parents=True)
            pred = np.ones(case['shape'], dtype=np.uint8)
            nib.save(nib.Nifti1Image(pred, affine, img.header), staged)
            rec = runner.validate_nifti(staged, case, pred)
            journal = {'status': 'prepared', 'phase': 'baseline', 'case': case['name'], 'job_sha256': 'job',
                       'artifacts': {'d1': {**rec, 'path': str(dest), 'temporary_path': str(staged)}}}
            runner.atomic_json(runner.journal_path(root, case, 'baseline'), journal)
            self.assertEqual(runner.recover_case(cfg, job, case, 'baseline')['status'], 'complete')
            self.assertTrue(dest.exists()); self.assertFalse(staged.exists())
            with dest.open('ab') as stream: stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): runner.recover_case(cfg, job, case, 'baseline')
            runner.journal_path(root, case, 'baseline').unlink()
            with self.assertRaisesRegex(ValueError, 'unowned'): runner.recover_case(cfg, job, case, 'baseline')

    def test_case_and_finding_dice_and_change_counts(self):
        rows = []
        for case, vals in [('a', [1.]), ('b', [0., 0., 0.])]:
            for i, dice in enumerate(vals):
                rows.append({'case': case, 'finding_index': i, 'variant': 'd1', 'category': '2d', 'prompt': 'nodule',
                             'dice': dice, 'hit': dice >= .1, 'gt_voxels': 1, 'pred_voxels': 1, 'intersection_voxels': int(dice)})
        m = runner.metric_summary(rows)
        self.assertEqual(m['dice'], .25); self.assertEqual(m['case_wise_dice'], .5)
        self.assertIsNone(m['categories']['2f']['dice'])
        copy = [{**r, 'variant': 'd11'} for r in rows]
        summary, _ = runner.compare_rows(rows + copy, 'd11')
        self.assertEqual(summary['overall']['unchanged'], 4)
        self.assertEqual(summary['categories']['2f']['findings'], 0)

    def test_relocation_and_frozen_sources(self):
        manifest = runner.read(methods.ROOT/'relocation_manifest.json')
        self.assertFalse((methods.REPO/manifest['source']).exists())
        for name, checksum in manifest['files'].items(): self.assertEqual(runner.sha(methods.ROOT/name), checksum)
        frozen = runner.read(methods.ROOT/'frozen_sources/manifest.json')
        for ref in frozen['files']: self.assertEqual(runner.sha(methods.ROOT/'frozen_sources'/ref['path']), ref['sha256'])

    def test_full_200_case_run_report_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); inputs = root/'inputs'; inputs.mkdir()
            affine = np.diag([-2., -3., 4., 1.]); spatial = (12, 12, 16)
            ctpath = inputs/'ct.nii.gz'
            nib.save(nib.Nifti1Image(np.zeros(spatial, dtype=np.uint8), affine), ctpath)
            anatomy = np.zeros(spatial, dtype=np.uint8)
            for i in range(5): anatomy[2*i:2*i+2, 2:10, 2:14] = 10+i
            ap = inputs/'anatomy.nii.gz'; nib.save(nib.Nifti1Image(anatomy, affine), ap)
            sources, truth = {}, {}
            for f in (1, 2):
                a = np.ones((f, *spatial), dtype=np.float16)
                p = inputs/f'logits{f}.npy'; np.save(p, a)
                sources[f] = [{'path': str(p), 'sha256': hashlib.sha256(a.tobytes()).hexdigest(), 'dtype': 'float16'}] * 4
                p = inputs/f'gt{f}.nii.gz'; nib.save(nib.Nifti1Image(a.astype(np.uint8), affine), p)
                truth[f] = {'path': str(p), 'sha256': runner.sha(p)}
            cases, baseline_rows = [], []
            for i in range(200):
                n = 2 if i < 181 else 1
                fs = findings([('pulmonary nodule', '2d')] * n)
                c = {'name': f'case_{i:03}.nii.gz', 'findings': fs, 'shape': [n, *spatial],
                     'routes': [methods.route_prompt(f['prompt'], f['category']) for f in fs],
                     'affine': affine.tolist(), 'ct_path': str(ctpath),
                     'anatomy': {'path': str(ap), 'sha256': runner.sha(ap)}, 'gt': truth[n], 'sources': sources[n]}
                cases.append(c)
                baseline_rows.extend(runner.score_case(np.ones(c['shape'], dtype=np.uint8), np.ones(c['shape'], dtype=np.uint8), c, 'd1'))
            ref = inputs/'baseline.json'; runner.atomic_json(ref, {'metrics': methods.summarize_rows(baseline_rows)})
            cfg = {'runtime_root': str(root/'runtime'), 'run_id': 'synthetic', 'experiment': 'sideexp005_postprocessing_merge',
                   'workers': 4, 'chunk_elements': 101, 'storage': {'reserve_bytes': 0, 'headroom_bytes': 0},
                   'inputs': {'baseline_result': {'path': str(ref), 'sha256': runner.sha(ref)}},
                   'baseline': {'dice': 1., 'hits': 381, 'tolerance': 1e-12}, 'image_id': 'synthetic',
                   'source_status': 'SYNTHETIC', 'test_generation': 'pending_validation_review', 'environment': {}}
            job = {'config_sha256': runner.digest(cfg), 'code': [], 'candidates': [], 'cases': cases,
                   'smoke_cases': [cases[0]['name'], cases[-1]['name']]}
            job['job_sha256'] = runner.job_digest(job)
            runtime = Path(cfg['runtime_root']); runtime.mkdir()
            runner.atomic_json(runtime/'job_manifest.json', job)
            runner.atomic_json(runtime/'preflight.json', {'job_sha256': job['job_sha256']})
            runner.run(cfg)
            done = runner.read(runtime/'completion.json')
            self.assertEqual((done['files'], done['finding_evaluations']), (1000, 1905))
            self.assertEqual(len(list((runtime/'predictions').rglob('*.nii.gz'))), 1000)
            self.assertEqual(done['metrics']['d1']['dice'], 1.)
            before = {str(p): runner.sha(p) for p in (runtime/'predictions').rglob('*.nii.gz')}
            runner.run(cfg)
            self.assertEqual(before, {str(p): runner.sha(p) for p in (runtime/'predictions').rglob('*.nii.gz')})
            bad = dict(cfg); bad['baseline'] = {'dice': 0., 'hits': 381, 'tolerance': 1e-12}
            recs = [runner.read(runner.journal_path(runtime, c, 'baseline')) for c in cases]
            with self.assertRaisesRegex(ValueError, 'anchor mismatch'): runner.baseline_gate(bad, recs)


if __name__ == '__main__':
    unittest.main()
