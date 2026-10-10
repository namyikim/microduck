#!/usr/bin/env python3
"""Preflight rest poses or record a stand → sit → stand cycle without hiding failures."""

import argparse
import itertools
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

TASK = "Mjlab-SitStand-Flat-MicroDuck"
DURATION = 16.0
# Measured BAM open-loop equilibrium, NOT a policy action offset or HOME change.
# 64 noisy starts held 11.37–11.49 cm with max tilt 3.58° on 2026-10-10.
STAND_HOLD_HIP_ACTION = 0.12


def smoke_evaluation_completed(summary, expected_trials=2):
    execution_failures = {
        "environment_terminated",
        "incomplete_cycle",
        "missing_or_nonfinite_evidence",
    }
    trials = summary.get("trials", [])
    return (
        summary.get("videos_created") == expected_trials
        and len(trials) == expected_trials
        and all(
            "reasons" in trial and not execution_failures.intersection(trial["reasons"])
            for trial in trials
        )
    )


def assess_cycle(rows, *, terminated=False):
    reasons = ["environment_terminated"] if terminated else []
    if not rows or any(
        not math.isfinite(r[k])
        for r in rows
        for k in ("time_s", "height_m", "upright_cos", "vertical_velocity")
    ):
        return {
            "sim_checks_passed": False,
            "reasons": ["missing_or_nonfinite_evidence"],
        }
    if (
        rows[0]["time_s"] > 0.021
        or rows[-1]["time_s"] < DURATION - 0.05
        or any(
            b["time_s"] - a["time_s"] > 0.041 or b["time_s"] <= a["time_s"]
            for a, b in itertools.pairwise(rows)
        )
    ):
        reasons.append("incomplete_cycle")
    if any(r["head_contact"] for r in rows):
        reasons.append("head_ground_contact")
    if min(r["upright_cos"] for r in rows) < math.cos(math.radians(45)):
        reasons.append("excessive_transition_tilt")
    # Arrival time alone would accept waiting first, then dropping suddenly.
    # Allow brief overshoot above the training caps (0.05 / 0.08 m/s).
    transition_speeds = [r["vertical_velocity"] for r in rows if r["time_s"] >= 2]
    transition_speeds.extend(
        (b["height_m"] - a["height_m"]) / (b["time_s"] - a["time_s"])
        for a, b in itertools.pairwise(rows)
        if a["time_s"] >= 2 and b["time_s"] > a["time_s"]
    )
    if transition_speeds and min(transition_speeds) < -0.10:
        reasons.append("fast_descent")
    if transition_speeds and max(transition_speeds) > 0.16:
        reasons.append("fast_rise")
    for name, lo, hi, z in [
        ("initial_stand", 1.0, 1.9, 0.115),
        ("sit", 6.0, 7.9, 0.060),
        ("final_stand", 13.0, 16.0, 0.115),
    ]:
        group = [r for r in rows if lo <= r["time_s"] <= hi]
        if not group or any(
            abs(r["height_m"] - z) > 0.015
            or r["upright_cos"] < math.cos(math.radians(15))
            for r in group
        ):
            reasons.append("unstable_" + name)
    durations = {}
    for name, start, end, down in [
        ("sit", 2.0, 8.0, True),
        ("stand", 8.0, 16.0, False),
    ]:
        reached = [
            r["time_s"] - start
            for r in rows
            if start <= r["time_s"] < end
            and (r["height_m"] <= 0.075 if down else r["height_m"] >= 0.100)
        ]
        durations[name] = reached[0] if reached else None
        if not reached or not 1.0 <= reached[0] <= 4.0:
            reasons.append("transition_timing_" + name)
    return {
        "sim_checks_passed": not reasons,
        "reasons": reasons,
        "transition_time_s": durations,
        "max_tilt_deg": math.degrees(
            math.acos(max(-1, min(1, min(r["upright_cos"] for r in rows))))
        ),
        "head_contact_frames": sum(r["head_contact"] for r in rows),
        "peak_vertical_speed_m_s": max(abs(r["vertical_velocity"]) for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if not args.preflight and args.checkpoint is None:
        parser.error("--checkpoint required")
    if args.trials < 1:
        parser.error("--trials must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio.v2 as imageio
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    import mjlab_microduck.jump_spin_plugin  # noqa: F401 — register tasks
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.tasks.microduck_sitstand_env_cfg import (
        SITTING_TARGET_OVERRIDES,
    )

    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = 64 if args.preflight else 1
    cfg.episode_length_s = DURATION + 5
    for name in ("push_robot", "seated_head_jerk", "seated_tip"):
        cfg.events.pop(name, None)
    for name in (
        "push_magnitude",
        "seated_head_jerk_prob",
        "seated_tip_range",
        "head_pose_range",
    ):
        cfg.curriculum.pop(name, None)
    cfg.commands["twist"].resampling_time_range = (100.0, 100.0)
    cfg.commands["twist"].sit_prob = 0.0
    cfg.commands["head_pose"].ranges = ((0.0, 0.0),) * 4
    # SitStand already supplies six zero-padded body-command observations.
    cfg.events["set_ground_state"].params.update(
        sitting_prob=0.0,
        standing_prob=1.0,
        face_up_prob=0.0,
        face_down_prob=0.0,
        sitting_joint_noise_std=0.02,
        sitting_tilt_max=math.radians(2),
    )
    if args.preflight:
        # Check existence of a nominal equilibrium with noisy initial states.
        # Robustness to changed mass/CoM needs the learned feedback policy, not
        # a fixed open-loop control. Keep those variations in training/evaluation.
        for name in (
            "base_com",
            "randomize_com",
            "randomize_head_com",
            "randomize_armature",
            "randomize_mass_inertia",
            "randomize_joint_friction",
            "encoder_bias",
        ):
            cfg.events.pop(name, None)
        for name in ("com_range", "head_com_range"):
            cfg.curriculum.pop(name, None)
        cfg.events["reset_robot_joints"].params["position_range"] = (-0.01, 0.01)
        cfg.events["set_ground_state"].params.update(
            sitting_tilt_max=math.radians(0.5),
            standing_z_min=0.114,
            standing_z_max=0.116,
        )
    cfg.scene.sensors = (
        *cfg.scene.sensors,
        ContactSensorCfg(
            name="head_ground_contact",
            primary=ContactMatch(mode="body", pattern="jaw_soft", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        ),
    )
    cfg.viewer.width = 480
    cfg.viewer.height = 480
    cfg.viewer.distance = 0.7
    cfg.viewer.lookat = (0.0, 0.0, 0.04)
    cfg.viewer.elevation = -15
    raw = ManagerBasedRlEnv(
        cfg=cfg, device="cuda:0", render_mode=None if args.preflight else "rgb_array"
    )
    agent = load_rl_cfg(TASK)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    robot = raw.scene["robot"]
    result = {"task": TASK, "checkpoint": str(args.checkpoint), "trials": []}
    try:
        if args.preflight:
            result.update(
                kind="nominal_equilibrium",
                stand_hold_hip_action=STAND_HOLD_HIP_ACTION,
                initial_joint_offset_range=[-0.01, 0.01],
                initial_tilt_max_deg=0.5,
            )
            for posture in ("stand", "sit"):
                reset = raw.event_manager.get_term_cfg("set_ground_state")
                reset.params.update(
                    sitting_prob=float(posture == "sit"),
                    standing_prob=float(posture == "stand"),
                    sitting_z_min=0.06,
                    sitting_z_max=0.062,
                )
                env.reset()
                target = mdp._servo_default_joint_pos(raw, robot).clone()
                if posture == "sit":
                    for j, v in SITTING_TARGET_OVERRIDES.items():
                        target[:, j] = v
                else:
                    # Keys are servo-layout columns, never simulator joint IDs.
                    target[:, 2] += STAND_HOLD_HIP_ACTION
                    target[:, 11] -= STAND_HOLD_HIP_ACTION
                actions = target - mdp._servo_default_joint_pos(raw, robot)
                tilts = torch.zeros(raw.num_envs, device=raw.device)
                contact = torch.zeros_like(tilts, dtype=torch.bool)
                for _ in range(round(3 / raw.step_dt)):
                    with torch.no_grad():
                        _, _, done, _ = env.step(actions)
                    if bool(done.any()):
                        raise RuntimeError("Preflight reset/NaN")
                    q = robot.data.root_link_quat_w
                    tilts = torch.maximum(
                        tilts,
                        torch.acos((1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)).clamp(-1, 1))
                        * 180
                        / math.pi,
                    )
                    contact |= mdp.seated_greeting_head_contact_cost(raw).bool()
                z = (
                    robot.data.root_link_pos_w[:, 2]
                    - raw.scene.terrain.env_origins[:, 2]
                )
                passed = (
                    (tilts < 15)
                    & (~contact)
                    & ((z - (0.06 if posture == "sit" else 0.115)).abs() < 0.015)
                )
                result["trials"].append(
                    {
                        "posture": posture,
                        "passed": int(passed.sum()),
                        "total": raw.num_envs,
                        "max_tilt_deg": float(tilts.max()),
                        "final_height_range_m": [float(z.min()), float(z.max())],
                    }
                )
            result["passed"] = all(
                x["passed"] / x["total"] >= 0.95 for x in result["trials"]
            )
            (args.output_dir / "preflight.json").write_text(
                json.dumps(result, indent=2)
            )
            print(json.dumps(result), flush=True)
            if not result["passed"]:
                raise RuntimeError("Rest-pose preflight failed; do not train")
            return
        runner = load_runner_cls(TASK)(env, asdict(agent), device="cuda:0")
        runner.load(
            str(args.checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location="cuda:0",
        )
        policy = runner.get_inference_policy(device="cuda:0")
        command = raw.command_manager.get_term("twist")
        for trial in range(1, args.trials + 1):
            env.reset()
            rows = []
            terminated = False
            video = args.output_dir / f"sitstand_trial_{trial:02d}.mp4"
            with imageio.get_writer(str(video), fps=10) as writer:
                for step in range(round(DURATION / raw.step_dt) + 1):
                    t = step * raw.step_dt
                    command.vel_command_b[:] = 0.0
                    command.vel_command_b[:, 0] = float(2 <= t < 8)
                    obs = env.get_observations()
                    q = robot.data.root_link_quat_w[0]
                    rows.append(
                        {
                            "time_s": t,
                            "height_m": float(
                                robot.data.root_link_pos_w[0, 2]
                                - raw.scene.terrain.env_origins[0, 2]
                            ),
                            "upright_cos": float(1 - 2 * (q[1] ** 2 + q[2] ** 2)),
                            "head_contact": bool(
                                mdp.seated_greeting_head_contact_cost(raw)[0]
                            ),
                            "vertical_velocity": float(
                                robot.data.root_link_lin_vel_w[0, 2]
                            ),
                        }
                    )
                    if step % 5 == 0:
                        writer.append_data(raw.render())
                    if step == round(DURATION / raw.step_dt):
                        break
                    with torch.no_grad():
                        _, _, done, _ = env.step(policy(obs))
                    if bool(done.any()):
                        terminated = True
                        break
            assessment = assess_cycle(rows, terminated=terminated)
            assessment.update(trial=trial, video=str(video))
            result["trials"].append(assessment)
            (args.output_dir / f"sitstand_trial_{trial:02d}_trace.json").write_text(
                json.dumps(rows, indent=2)
            )
            print(json.dumps(assessment), flush=True)
        result.update(
            videos_created=len(result["trials"]),
            all_sim_checks_passed=all(t["sim_checks_passed"] for t in result["trials"]),
        )
        (args.output_dir / "evaluation_summary.json").write_text(
            json.dumps(result, indent=2)
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
