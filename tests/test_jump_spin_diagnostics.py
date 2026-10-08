"""CPU-only checks: a bow/head-supported pose must not count as flight."""
import importlib.util
import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'scripts' / 'jump_spin_diagnostics.py'
        self.assertTrue(path.exists(), 'Rollout diagnostics are missing')
        spec = importlib.util.spec_from_file_location('diagnostics', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def sample(self, **changes):
        frame = dict(height_m=0.115, vertical_speed_m_s=0.0,
                     feet_contact_count=2, robot_ground_contact=True,
                     spin_progress_deg=0.0, upright_cos=1.0)
        frame.update(changes)
        return frame

    def test_bowing_is_no_observed_takeoff(self):
        result = self.module.summarize_rollout([self.sample(height_m=z) for z in (0.115, 0.10, 0.08)], 0.02)
        self.assertEqual(result['diagnosis'], 'no_observed_takeoff')
        self.assertEqual(result['longest_observed_flight_s'], 0.0)

    def test_head_support_does_not_count_as_airborne(self):
        result = self.module.summarize_rollout([self.sample(feet_contact_count=0, height_m=0.14)], 0.02)
        self.assertEqual(result['diagnosis'], 'no_observed_takeoff')

    def test_flight_duration_is_consecutive_and_spin_is_diagnostic_only(self):
        air = self.sample(feet_contact_count=0, robot_ground_contact=False, height_m=0.18)
        result = self.module.summarize_rollout([air, air, self.sample(), air], 0.02)
        self.assertAlmostEqual(result['longest_observed_flight_s'], 0.04)
        self.assertAlmostEqual(result['observed_flight_s'], 0.06)
        self.assertEqual(result['diagnosis'], 'airborne_without_target_spin')
        result = self.module.summarize_rollout([air, self.sample(spin_progress_deg=360)], 0.02)
        self.assertEqual(result['diagnosis'], 'target_progress_observed_review_landing')
        self.assertNotIn('success', result)

    def test_missing_contact_evidence_is_not_a_pass(self):
        frame = self.sample(robot_ground_contact=None)
        result = self.module.summarize_rollout([frame], 0.02)
        self.assertEqual(result['diagnosis'], 'insufficient_contact_evidence')

    def test_empty_rollout_is_not_a_pass(self):
        result = self.module.summarize_rollout([], 0.02)
        self.assertEqual(result['diagnosis'], 'no_samples')

    def test_notebook_code_cells_are_executable_python(self):
        notebook = json.loads((ROOT / 'MicroDuck_JumpSpin_A100.ipynb').read_text())
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                with self.subTest(cell=index):
                    ast.parse(''.join(cell['source']))


if __name__ == '__main__':
    unittest.main()
