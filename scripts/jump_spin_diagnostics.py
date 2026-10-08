"""Summarize sampled rollout evidence without declaring a policy successful.

No simulation dependencies: regression tests run on an ordinary CPU Python.
Samples precede env.step, so the terminal physics frame is not included.
"""
from __future__ import annotations


def summarize_rollout(samples: list[dict], step_dt: float) -> dict:
    if not samples:
        return {"diagnosis": "no_samples"}
    flight_steps = longest = streak = 0
    missing_contact = False
    for sample in samples:
        contact = sample["robot_ground_contact"]
        feet = sample["feet_contact_count"]
        missing_contact |= contact is None or feet is None
        # Whole-robot contact is essential: feet can lift while the head props
        # up the robot. Height alone must never turn that pose into a jump.
        airborne = contact is False and feet == 0 and sample["height_m"] > 0.125
        streak = streak + 1 if airborne else 0
        flight_steps += int(airborne)
        longest = max(longest, streak)
    progress = max(s["spin_progress_deg"] for s in samples)
    if missing_contact:
        diagnosis = "insufficient_contact_evidence"
    elif not flight_steps:
        diagnosis = "no_observed_takeoff"
    elif progress < 330.0:
        diagnosis = "airborne_without_target_spin"
    else:
        diagnosis = "target_progress_observed_review_landing"
    return {
        "diagnosis": diagnosis,
        "observed_flight_s": flight_steps * step_dt,
        "longest_observed_flight_s": longest * step_dt,
        "min_height_m": min(s["height_m"] for s in samples),
        "max_height_m": max(s["height_m"] for s in samples),
        "max_upward_speed_m_s": max(s["vertical_speed_m_s"] for s in samples),
        "max_task_spin_progress_deg": progress,
        "last_observed_height_m": samples[-1]["height_m"],
        "last_observed_upright_cos": samples[-1]["upright_cos"],
        "last_observed_feet_contact_count": samples[-1]["feet_contact_count"],
    }


def sample_rollout(env) -> dict:
    """Read env 0 before stepping (the renderer runs exactly one environment)."""
    asset = env.scene['robot']

    def scalar(tensor):
        return float(tensor.detach().cpu().item())

    def contacts(name):
        if name not in env.scene.sensors:
            return None
        found = env.scene.sensors[name].data.found[0].reshape(-1)
        return int((found > 0).sum().detach().cpu().item())

    robot_contact = contacts('robot_ground_contact')
    progress = getattr(env, '_jump_spin_max', None)
    q = asset.data.root_link_quat_w[0]
    return {
        'height_m': scalar(asset.data.root_link_pos_w[0, 2] - env.scene.terrain.env_origins[0, 2]),
        'vertical_speed_m_s': scalar(asset.data.root_link_lin_vel_w[0, 2]),
        'yaw_rate_rad_s': scalar(asset.data.root_link_ang_vel_w[0, 2]),
        'upright_cos': scalar(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)),
        'feet_contact_count': contacts('feet_ground_contact'),
        'robot_ground_contact': None if robot_contact is None else robot_contact > 0,
        'head_ground_contact_count': contacts('head_ground_contact'),
        'spin_progress_deg': 0.0 if progress is None else scalar(progress[0]) * 180.0 / 3.141592653589793,
    }
