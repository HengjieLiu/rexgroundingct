import unittest
from pathlib import Path
from unittest.mock import patch
import finish_test300_wave2 as g

class Wave2GuardTests(unittest.TestCase):
    def test_identity_and_wave_are_bound(self):
        job={'job_spec_sha256':'ours'}
        state={'pid':123,'wave':2,'job_spec_sha256':'ours'}
        identity={'pid':123,'argv':['python',str((g.old.HERE/'continue_test300.py').resolve()),'run']}
        g.verify(identity,job,state)
        for bad in ({**state,'wave':3},{**state,'pid':124},{**state,'job_spec_sha256':'other'}):
            with self.assertRaises(g.fc.FreshCacheError):g.verify(identity,job,bad)
        with self.assertRaises(g.fc.FreshCacheError):g.verify({'pid':123,'argv':['python','other.py','run']},job,state)

    def test_later_progress_or_container_blocks_guard(self):
        job={'candidates':[{'id':str(i)} for i in range(8)]}
        with patch.object(g.tr,'progress',return_value={'4':{}}):
            with self.assertRaises(g.fc.FreshCacheError):g.assert_no_later(job)
        with patch.object(g.tr,'progress',return_value={}),patch.object(g.old,'docker_command',return_value=('name',[])),patch.object(g.tr,'docker_state',return_value={'State':{}}):
            with self.assertRaises(g.fc.FreshCacheError):g.assert_no_later(job)
        with patch.object(g.tr,'progress',return_value={}),patch.object(g.old,'docker_command',return_value=('name',[])),patch.object(g.tr,'docker_state',return_value=None):
            g.assert_no_later(job)

if __name__=='__main__':unittest.main()
