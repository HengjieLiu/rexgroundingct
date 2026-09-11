"""Bounded, one-source-pass test300 publication with durable staging proofs."""
from __future__ import annotations
import os
os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
import hashlib
import math
from pathlib import Path
import shutil
import time
import numpy as np
import fresh_cache as fc
import test300_cache as tc

CHUNK = 4 * 1024 * 1024
GIB = 1024**3
PROOF_SCHEMA = 'test300_storage_proof_v1'


def bytes_view(a):
    return memoryview(np.ascontiguousarray(a)).cast('B')


def storage_proof(array):
    tc.require(array.dtype == np.float32 and array.flags.c_contiguous, 'proof requires contiguous float32')
    h32, h16 = hashlib.sha256(), hashlib.sha256()
    mismatch = 0
    flat = array.reshape(-1)
    for start in range(0, array.size, CHUNK):
        a = flat[start:start + CHUNK]
        tc.require(np.isfinite(a).all() and (np.abs(a) <= 30).all(), 'nonfinite/out-of-range proof input')
        b = a.astype(np.float16)
        h32.update(bytes_view(a)); h16.update(bytes_view(b))
        mismatch += int(np.count_nonzero((a >= 0) != (b >= 0)))
    return {'schema': PROOF_SCHEMA, 'float32_sha256': h32.hexdigest(), 'float16_sha256': h16.hexdigest(),
            'float16_mask_mismatch_voxels': mismatch, 'elements': int(array.size), 'finite_clipped': True}


def validate_proof(record, case, job, candidate):
    tc.require(record['job_spec_sha256'] == job['job_spec_sha256'] and record['cache_key'] == candidate['cache_key']
               and record['name'] == case['name'], 'foreign stage proof')
    tc.require(record['dtype'] == 'float32' and record['shape'] == [len(case['findings']), *case['shape']], 'stage proof shape/dtype')
    for key, value in tc.prompt_contract(case).items():
        tc.require(record[key] == value, 'stage prompt drift')
    p = record['storage_proof']
    tc.require(record['storage_proof_sha256'] == fc.json_sha256(p), 'storage proof hash mismatch')
    tc.require(p['schema'] == PROOF_SCHEMA and p['finite_clipped'] is True and p['elements'] == math.prod(record['shape']), 'incomplete storage proof')
    tc.require(p['float32_sha256'] == record['array_sha256'], 'stage/proof hash mismatch')
    tc.require(type(p['float16_mask_mismatch_voxels']) is int and p['float16_mask_mismatch_voxels'] >= 0, 'invalid mask proof')
    for key in ('float32_sha256', 'float16_sha256'):
        tc.require(len(p[key]) == 64 and all(x in '0123456789abcdef' for x in p[key]), 'invalid proof digest')
    return p


def choose_dtype(records):
    return 'float32' if any(r['storage_proof']['float16_mask_mismatch_voxels'] for r in records) else 'float16'


def save_stage(job, candidate, case, array, details, timer):
    with timer.stage('finite_range_clip'):
        tc.require(np.isfinite(array).all(), 'nonfinite inference')
        bounds = [float(array.min()), float(array.max())]
        np.clip(array, -30, 30, out=array)
        array = np.ascontiguousarray(array, dtype=np.float32)
    proof = timer.call('in_memory_storage_proof', storage_proof, array)
    ap, rp = tc.stage_paths(candidate, case['name'])
    tc.require(shutil.disk_usage(job['staging_root']).free >= array.nbytes + GIB, 'local stage full')
    if rp.exists():
        previous = fc.read_json(rp)
        tc.require(previous['job_spec_sha256'] == job['job_spec_sha256'] and previous['cache_key'] == candidate['cache_key'], 'foreign stage')
    timer.call('local_write', tc.atomic_save_npy, ap, array)
    record = {**details, 'name': case['name'], 'shape': list(array.shape), 'dtype': 'float32',
              'job_spec_sha256': job['job_spec_sha256'], 'cache_key': candidate['cache_key'],
              'array_sha256': proof['float32_sha256'], 'storage_proof': proof,
              'storage_proof_sha256': fc.json_sha256(proof), 'unclipped_range': bounds,
              'output_elements': int(array.size), 'npy_bytes': ap.stat().st_size, 'completed_at_utc': fc.utc_now()}
    fc.atomic_write_json(rp, record)
    return record


def read_header(handle):
    version = np.lib.format.read_magic(handle)
    tc.require(version in ((1, 0), (2, 0)), 'unsupported NPY header')
    fn = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = fn(handle)
    tc.require(not fortran and not dtype.hasobject, 'unsafe/noncontiguous NPY')
    return tuple(shape), dtype, handle.tell()


def verify_geometry(record, case):
    import nibabel as nib
    from common import ct_rate_abs_path, CT_ROOT
    path = ct_rate_abs_path(case['name'], CT_ROOT)
    tc.require(Path(record['ct_path']) == path, 'wrong CT path')
    ct = nib.load(str(path))
    tc.require(list(ct.shape) == case['shape'] and np.array_equal(ct.affine, np.asarray(record['affine'])), 'CT geometry drift')


def verify_file(path, shape, dtype, expected_hash):
    """One sequential destination read; hashes and numerical checks share chunks."""
    total = math.prod(shape)
    h = hashlib.sha256()
    read_bytes = 0
    with Path(path).open('rb', buffering=8 * 1024**2) as f:
        got_shape, got_dtype, offset = read_header(f)
        tc.require(got_shape == tuple(shape) and got_dtype == np.dtype(dtype), 'destination header mismatch')
        tc.require(os.fstat(f.fileno()).st_size == offset + total * got_dtype.itemsize, 'destination size mismatch')
        read_bytes += offset
        for start in range(0, total, CHUNK):
            need = min(CHUNK, total - start) * got_dtype.itemsize
            block = f.read(need)
            tc.require(len(block) == need, 'truncated destination')
            values = np.frombuffer(block, dtype=got_dtype)
            tc.require(np.isfinite(values).all() and (np.abs(values) <= 30).all(), 'nonfinite/out-of-range destination')
            h.update(block); read_bytes += len(block)
        tc.require(f.read(1) == b'', 'trailing destination data')
    tc.require(h.hexdigest() == expected_hash, 'destination hash mismatch')
    return read_bytes


class ReadBudget:
    def __init__(self, root):
        self.root = Path(root)
        self.deadline = time.monotonic()
        self.rate = None
        self.cached = {}
        self.last_check = 0
        self.wait_seconds = 0.

    def controls(self):
        now = time.monotonic()
        if now - self.last_check >= .5:
            tc.require(not (self.root / 'control/ABORT').exists(), 'job aborted')
            path = self.root / 'publisher_control.json'
            self.cached = fc.read_json(path) if path.exists() else {'gpu_active': True, 'read_mib_s': 32}
            self.last_check = now
        return self.cached

    def acquire(self, count):
        started = time.monotonic()
        while True:
            d = self.controls()
            if not d.get('gpu_active', True):
                self.deadline = time.monotonic(); self.rate = 0
                break
            if d.get('paused', False):
                time.sleep(.5)
                continue
            rate = max(1., float(d.get('read_mib_s', 32))) * 1024**2
            if self.rate != rate:
                self.deadline = time.monotonic(); self.rate = rate
            delay = self.deadline - time.monotonic()
            if delay <= 0:
                self.deadline = max(self.deadline, time.monotonic()) + count / rate
                break
            time.sleep(min(.5, delay))
        self.wait_seconds += time.monotonic() - started


def publish_one(job, candidate, case, record, dtype, budget=None, progress=None, geometry=verify_geometry):
    proof = validate_proof(record, case, job, candidate)
    tc.require(dtype in ('float16', 'float32'), 'unsupported output dtype')
    tc.require(dtype == 'float32' or proof['float16_mask_mismatch_voxels'] == 0, 'unsafe float16 publication')
    geometry(record, case)
    expected = proof[dtype + '_sha256']
    root = Path(candidate['cache_root']);root.mkdir(parents=True, exist_ok=True)
    path = root / 'cases' / (case['name'] + '.npy')
    manifest = root / 'case_manifests' / (case['name'] + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + job['job_spec_sha256'][:16] + '.tmp')
    started = time.monotonic()
    result = {**record, 'dtype': dtype, 'array_path': str(path), 'array_sha256': expected,
              'same_pass_mask_mismatch_voxels': 0, 'storage_reproduction_status': 'passed',
              'metric_status': tc.METRIC_STATUS}
    if path.exists():
        if manifest.exists():
            old = fc.read_json(manifest)
            tc.require(old['job_spec_sha256'] == job['job_spec_sha256'] and old['cache_key'] == candidate['cache_key']
                       and old['array_sha256'] == expected and old['dtype'] == dtype, 'conflicting published record')
        n = verify_file(path, record['shape'], dtype, expected)
        result.update(npy_bytes=path.stat().st_size, publication_source_bytes=0,
                      destination_verification_bytes=n, reused_verified_destination=True)
        fc.atomic_write_json(manifest, result)
        return result
    count = math.prod(record['shape']); out_size = count * np.dtype(dtype).itemsize
    while shutil.disk_usage(root).free < tc.MIN_SHARED + out_size + GIB:
        tc.require(not (Path(job['runtime_root']) / 'control/ABORT').exists(), 'job aborted')
        if progress:progress(phase='waiting_for_space', case=case['name'])
        time.sleep(30)
    source, _ = tc.stage_paths(candidate, case['name'])
    source_hash, dest_hash = hashlib.sha256(), hashlib.sha256()
    source_bytes = 0
    if progress:progress(phase='publishing', case=case['name'], source_bytes=0)
    last_report = time.monotonic()
    # A failed copy may replace only this job's own temporary destination.
    with source.open('rb', buffering=8 * 1024**2) as src, temporary.open('wb', buffering=8 * 1024**2) as dst:
        shape, source_dtype, offset = read_header(src)
        tc.require(shape == tuple(record['shape']) and source_dtype == np.dtype('float32'), 'source header mismatch')
        tc.require(os.fstat(src.fileno()).st_size == offset + count * 4, 'source size mismatch')
        source_bytes = offset
        np.lib.format.write_array_header_1_0(dst, {'descr': np.lib.format.dtype_to_descr(np.dtype(dtype)), 'fortran_order': False, 'shape': shape})
        for start in range(0, count, CHUNK):
            need = min(CHUNK, count - start) * 4
            if budget:budget.acquire(need)
            block = src.read(need)
            tc.require(len(block) == need, 'truncated source')
            a = np.frombuffer(block, dtype=np.float32)
            tc.require(np.isfinite(a).all() and (np.abs(a) <= 30).all(), 'nonfinite/out-of-range source')
            source_hash.update(block); source_bytes += len(block)
            if dtype == 'float16':
                b = a.astype(np.float16)
                tc.require(np.count_nonzero((a >= 0) != (b >= 0)) == 0, 'conversion mask mismatch')
                encoded = b.tobytes(order='C')
            else:
                encoded = block
            dest_hash.update(encoded);dst.write(encoded)
            if progress and time.monotonic() - last_report >= 5:
                progress(phase='publishing', case=case['name'], source_bytes=source_bytes)
                last_report = time.monotonic()
        tc.require(src.read(1) == b'', 'trailing source data')
        tc.require(source_hash.hexdigest() == proof['float32_sha256'], 'source hash mismatch')
        tc.require(dest_hash.hexdigest() == expected, 'converted hash mismatch')
        dst.flush();os.fsync(dst.fileno())
    copied = time.monotonic()
    if progress:progress(phase='verifying_destination', case=case['name'], source_bytes=source_bytes)
    verified_bytes = verify_file(temporary, record['shape'], dtype, expected)
    tc.require(not path.exists(), 'destination appeared during publication')
    os.replace(temporary, path)
    result.update(npy_bytes=path.stat().st_size, publication_source_bytes=source_bytes,
                  destination_verification_bytes=verified_bytes, publication_seconds=copied-started,
                  destination_verification_seconds=time.monotonic()-copied, published_at_utc=fc.utc_now(),
                  source_read_passes=1)
    fc.atomic_write_json(manifest, result)
    return result


def completed_manifest(job, candidate, cases, records):
    tc.require(len(records) == job['dataset']['cases'] == len(cases), 'incomplete checkpoint')
    tc.require([r['name'] for r in records] == [case['name'] for case in cases], 'case order mismatch')
    tc.require(sum(r['shape'][0] for r in records) == job['dataset']['findings'], 'prompt coverage mismatch')
    tc.require(len({r['dtype'] for r in records}) == 1, 'mixed checkpoint dtype')
    for r, case in zip(records, cases):
        tc.require(r['job_spec_sha256'] == job['job_spec_sha256'] and r['cache_key'] == candidate['cache_key'], 'foreign published case')
        for k,v in tc.prompt_contract(case).items():tc.require(r[k] == v, 'published prompt mismatch')
        tc.require(r['storage_reproduction_status'] == 'passed' and r['same_pass_mask_mismatch_voxels'] == 0, 'missing publication proof')
        tc.require(r.get('destination_verification_bytes') == r['npy_bytes'], 'missing destination readback')
    return {'status':'passed','cases':len(records),'findings':sum(r['shape'][0] for r in records),
            'dtype':records[0]['dtype'],'array_bytes':sum(r['npy_bytes'] for r in records),
            'array_hashes_verified':True,'same_pass_mask_mismatch_voxels':0,'storage_reproduction_status':'passed',
            'metric_status':tc.METRIC_STATUS,'job_spec_sha256':job['job_spec_sha256'],
            'validation_method':'streamed source verification and one destination readback per case'}
