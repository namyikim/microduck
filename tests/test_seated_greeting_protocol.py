"""A frozen head, a jump at a waypoint, or a face-plant must not pass."""
import importlib.util
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'src/mjlab_microduck/seated_greeting.py'


class GreetingProtocolTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.exists(), 'Seated greeting protocol has not been implemented')
        spec = importlib.util.spec_from_file_location('greeting_protocol', MODULE)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_left_right_two_nods_and_neutral_finish(self):
        poses = [self.m.greeting_pose(t) for t in (0, 3.5, 6.5, 8, 9, 10, 11, 12, 14)]
        self.assertGreater(poses[1][2], 0.2)
        self.assertLess(poses[2][2], -0.2)
        self.assertAlmostEqual(poses[3][2], 0)
        self.assertGreater(abs(poses[4][1]), 0.1)
        self.assertAlmostEqual(poses[5][1], 0)
        self.assertEqual(poses[4], poses[6])
        self.assertEqual(poses[0], poses[-1])

    def test_targets_are_continuous_and_slow(self):
        previous = self.m.greeting_pose(0)
        for i in range(1, 14001):
            current = self.m.greeting_pose(i / 1000)
            self.assertLessEqual(max(abs(a-b) for a,b in zip(current,previous)) / .001, .42)
            previous = current

    def samples(self):
        return [dict(time_s=i*.02, height_m=.06, upright_cos=1.,
                     head_contact=False, head_error_rad=0., head_actual=self.m.greeting_pose(i*.02),
                     head_target=self.m.greeting_pose(i*.02)) for i in range(700)]

    def test_clean_complete_greeting_passes_sim_checks(self):
        self.assertTrue(self.m.assess_greeting(self.samples())['sim_checks_passed'])

    def test_head_contact_tilt_and_incomplete_episode_fail(self):
        for field,value in [('head_contact',True),('upright_cos',.3),('height_m',.15)]:
            rows=self.samples();rows[250][field]=value
            self.assertFalse(self.m.assess_greeting(rows)['sim_checks_passed'], field)
        self.assertFalse(self.m.assess_greeting(self.samples()[:100])['sim_checks_passed'])

    def test_stationary_head_cannot_pass_on_neutral_frames(self):
        rows=self.samples()
        for r in rows:
            r['head_actual']=[0.,0.,0.,0.]
            r['head_error_rad']=max(abs(v) for v in r['head_target'])
        self.assertFalse(self.m.assess_greeting(rows)['sim_checks_passed'])

    def test_zeroed_commands_and_nonfinite_head_measurements_fail(self):
        rows=self.samples()
        for r in rows:
            r['head_target']=[0.,0.,0.,0.]
            r['head_actual']=[0.,0.,0.,0.]
        self.assertFalse(self.m.assess_greeting(rows)['sim_checks_passed'])
        rows=self.samples();rows[250]['head_actual']=[0.,float('nan'),0.,0.]
        self.assertFalse(self.m.assess_greeting(rows)['sim_checks_passed'])

    def test_missing_and_nonfinite_evidence_fails_closed(self):
        rows=self.samples();rows[250]['head_contact']=None
        self.assertFalse(self.m.assess_greeting(rows)['sim_checks_passed'])
        rows=self.samples();rows[250]['height_m']=float('nan')
        self.assertFalse(self.m.assess_greeting(rows)['sim_checks_passed'])


if __name__ == '__main__':
    unittest.main()
