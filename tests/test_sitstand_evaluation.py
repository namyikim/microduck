"""A complete, gentle sit/stand cycle must not pass on a stationary or fallen robot."""

import importlib.util
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "scripts/evaluate_sitstand.py"
    assert path.exists(), "SitStand cycle evaluator is not implemented"
    spec = importlib.util.spec_from_file_location("evaluate_sitstand", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_assessor():
    return load_module().assess_cycle


def samples():
    rows = []
    for step in range(801):
        t = step * 0.02
        blend = min(1, max(0, (t - 2) / 2)) if t < 8 else max(0, 1 - (t - 8) / 2)
        rows.append(
            {
                "time_s": t,
                "height_m": 0.115 - 0.055 * blend,
                "upright_cos": 1.0,
                "head_contact": False,
                "vertical_velocity": 0.0,
            }
        )
    return rows


class TestSitStandEvaluation(unittest.TestCase):
    def test_smoke_gate_allows_bad_motion_but_rejects_execution_failure(self):
        gate = load_module().smoke_evaluation_completed
        summary = {
            "videos_created": 2,
            "trials": [
                {"reasons": ["unstable_sit"]},
                {"reasons": ["head_ground_contact"]},
            ],
        }
        self.assertTrue(gate(summary))
        for reason in (
            "environment_terminated",
            "incomplete_cycle",
            "missing_or_nonfinite_evidence",
        ):
            with self.subTest(reason=reason):
                summary["trials"][0]["reasons"] = [reason]
                self.assertFalse(gate(summary))

    def test_termination_near_end_still_fails(self):
        self.assertFalse(
            load_assessor()(samples()[:-1], terminated=True)["sim_checks_passed"]
        )

    def test_waiting_then_dropping_is_not_a_slow_transition(self):
        rows = samples()
        for row in rows:
            if 2 <= row["time_s"] < 4:
                row["height_m"] = 0.115 if row["time_s"] < 3.5 else 0.06
        self.assertFalse(load_assessor()(rows)["sim_checks_passed"])

    def test_complete_gentle_cycle_passes(self):
        self.assertTrue(load_assessor()(samples())["sim_checks_passed"])

    def test_failures_are_not_hidden(self):
        for kind in (
            "stationary",
            "contact",
            "fallen",
            "incomplete",
            "nonfinite",
            "fast",
        ):
            with self.subTest(kind=kind):
                rows = samples()
                if kind == "stationary":
                    for row in rows:
                        row["height_m"] = 0.115
                elif kind == "contact":
                    rows[200]["head_contact"] = True
                elif kind == "fallen":
                    rows[300]["upright_cos"] = 0.0
                elif kind == "incomplete":
                    rows = rows[:500]
                elif kind == "nonfinite":
                    rows[100]["height_m"] = float("nan")
                elif kind == "fast":
                    for row in rows:
                        if 2 <= row["time_s"] < 4:
                            row["height_m"] = 0.06
                self.assertFalse(load_assessor()(rows)["sim_checks_passed"])


if __name__ == "__main__":
    unittest.main()
