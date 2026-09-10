import signal
import unittest
from pathlib import Path
from unittest.mock import patch

import finish_test300_wave1 as guard
import fresh_cache as fc


class FinishWaveTests(unittest.TestCase):
    def test_reject_unrelated_process_or_job(self):
        path = Path('/tmp/job.json')
        for argv in (['cpu-search.py', '--job', str(path)],
                     ['resume_test300_no_pause.py', '--job', '/tmp/other.json']):
            with self.assertRaises(fc.FreshCacheError):
                guard.verify_coordinator({'argv': argv}, path)
        guard.verify_coordinator({'argv': ['python', '/repo/resume_test300_no_pause.py', '--job', str(path)]}, path)

    def test_requires_finished_processes_and_strict_cache_validation(self):
        job = {'candidates': [{'id': str(i)} for i in range(4)]}
        records = {str(i): {'status': 'strict_passed'} for i in range(4)}
        states = [{'State': {'Running': False, 'ExitCode': 0}} for _ in range(4)]
        with patch.object(guard.tr, 'completed', return_value=True) as complete:
            states[3]['State']['Running'] = True
            self.assertFalse(guard.all_finished(job, records, states))
            states[3]['State']['Running'] = False
            records['3']['status'] = 'validating'
            self.assertFalse(guard.all_finished(job, records, states))
            records['3']['status'] = 'strict_passed'
            self.assertTrue(guard.all_finished(job, records, states))
            complete.return_value = False
            with self.assertRaises(fc.FreshCacheError):
                guard.all_finished(job, records, states)

    def test_termination_requested_before_scheduler_resumes(self):
        with patch.object(guard, 'send_pidfd') as send, patch.object(
            guard.select, 'select', return_value=([17], [], [])
        ):
            guard.terminate_stopped_coordinator(17)
            self.assertEqual([c.args for c in send.call_args_list],
                             [(17, signal.SIGTERM), (17, signal.SIGCONT)])


if __name__ == '__main__':
    unittest.main()
