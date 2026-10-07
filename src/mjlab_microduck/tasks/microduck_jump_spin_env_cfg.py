"""Experimental MicroDuck standing jump + yaw-spin + two-foot landing task.

Curriculum:
  * 0..1499 PPO iters: target 180°
  * 1500..3999:       target 270°
  * 4000+:            target 360°

Reverse-curriculum reset buckets expose the policy to mid-air and near-landing
states early, then gradually shift probability toward genuine standing starts.
The final policy is intended to start from a normal stand, jump, rotate around
world Z while both feet are airborne, land on both feet, and recover HOME.
"""

import math

from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as common_mdp
from mjlab_microduck.tasks import jump_spin_mdp
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    STAND_Z,
    make_microduck_roulade_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg


EPISODE_LENGTH_S = 4.5
FEET_SENSOR = "feet_ground_contact"

TARGET_INITIAL = math.pi
TARGET_MIDDLE = 1.5 * math.pi
TARGET_FINAL = 2.0 * math.pi
TARGET_MIDDLE_STEP = 1500 * 24
TARGET_FINAL_STEP = 4000 * 24

TARGET_KWARGS = {
    "initial_angle": TARGET_INITIAL,
    "middle_angle": TARGET_MIDDLE,
    "final_angle": TARGET_FINAL,
    "middle_step": TARGET_MIDDLE_STEP,
    "final_step": TARGET_FINAL_STEP,
}

# Symmetric deep crouch/tuck. Servo order is documented in AGENTS.md.
CROUCH_OVERRIDES = {
    2: -0.85,   # left hip_pitch
    3: 1.20,    # left knee
    4: 0.85,    # left ankle
    11: 0.85,   # right hip_pitch
    12: -1.20,  # right knee
    13: -0.85,  # right ankle
}

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]


def make_microduck_jump_spin_env_cfg(play: bool = False):
    # Reuse roulade's ground-contact robot, BAM sim2real DR, 61D observation
    # contract, zero-padded command slots, and NaN guard.
    cfg = make_microduck_roulade_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # Remove all forward-roll-specific objectives.
    for name in list(cfg.rewards.keys()):
        if name.startswith("roulade_"):
            cfg.rewards.pop(name, None)
    for name in ("arrival_damping", "gentle_landing", "angular_momentum"):
        cfg.rewards.pop(name, None)

    # Jump/spin discovery must not be crushed by motion regularizers.
    if "body_ang_vel" in cfg.rewards:
        cfg.rewards["body_ang_vel"].weight = -0.005
    if "action_rate_l2" in cfg.rewards:
        cfg.rewards["action_rate_l2"].weight = -0.02
    if "joint_torque_rate_l2" in cfg.rewards:
        cfg.rewards["joint_torque_rate_l2"].weight = 0.0
    if "self_collisions" in cfg.rewards:
        cfg.rewards["self_collisions"].weight = -0.15

    cfg.rewards["jump_launch"] = RewardTermCfg(
        func=jump_spin_mdp.jump_spin_launch_reward,
        weight=1.5,
        params={
            "sensor_name": FEET_SENSOR,
            "target_vz": 1.2,
            "target_yaw_rate": 8.0,
        },
    )
    cfg.rewards["jump_height"] = RewardTermCfg(
        func=jump_spin_mdp.jump_spin_airborne_height,
        weight=2.0,
        params={
            "sensor_name": FEET_SENSOR,
            "stand_z": STAND_Z,
            "target_height": 0.19,
            "takeoff_min_height": 0.125,
        },
    )
    cfg.rewards["jump_spin_progress"] = RewardTermCfg(
        func=jump_spin_mdp.jump_spin_progress,
        weight=10.0,
        params={
            "sensor_name": FEET_SENSOR,
            "takeoff_min_height": 0.125,
            "max_paid_rate": 25.0,
            **TARGET_KWARGS,
        },
    )
    cfg.rewards["jump_airborne_upright"] = RewardTermCfg(
        func=jump_spin_mdp.jump_spin_airborne_upright,
        weight=1.5,
        params={
            "sensor_name": FEET_SENSOR,
            "takeoff_min_height": 0.125,
            "std": 0.28,
        },
    )
    cfg.rewards["jump_landing"] = RewardTermCfg(
        func=jump_spin_mdp.jump_spin_landing_score,
        weight=8.0,
        params={
            "sensor_name": FEET_SENSOR,
            "target_height": STAND_Z,
            "height_std": 0.025,
            "upright_std": 0.25,
            "pose_std": 0.35,
            "joint_indices": _LEG_JOINTS,
            "takeoff_min_height": 0.125,
            "gate_low_frac": 0.92,
            "gate_high_frac": 0.99,
            **TARGET_KWARGS,
        },
    )
    cfg.rewards["jump_heading_return"] = RewardTermCfg(
        func=jump_spin_mdp.jump_spin_heading_return,
        weight=3.0,
        params={
            "sensor_name": FEET_SENSOR,
            "std": 0.20,
            "takeoff_min_height": 0.125,
            "gate_low_frac": 0.94,
            "gate_high_frac": 0.995,
            **TARGET_KWARGS,
        },
    )
    cfg.rewards["jump_wrong_axis"] = RewardTermCfg(
        func=jump_spin_mdp.jump_spin_wrong_axis_cost,
        weight=-0.03,
        params={
            "sensor_name": FEET_SENSOR,
            "takeoff_min_height": 0.125,
        },
    )
    cfg.rewards["jump_horizontal_drift"] = RewardTermCfg(
        func=jump_spin_mdp.jump_spin_horizontal_drift_cost,
        weight=-0.5,
    )
    cfg.rewards["jump_landing_impact"] = RewardTermCfg(
        func=jump_spin_mdp.jump_spin_landing_impact_cost,
        weight=-1.0,
        params={"sensor_name": FEET_SENSOR, "max_down_speed": 0.35},
    )

    # Replace roulade reset with jump reverse curriculum.
    cfg.events.pop("set_roulade_state", None)
    if play:
        standing_prob, airborne_prob, landing_prob = 1.0, 0.0, 0.0
        stand_crouch = (0.0, 0.05)
    else:
        standing_prob, airborne_prob, landing_prob = 0.30, 0.50, 0.20
        stand_crouch = (0.2, 0.9)

    cfg.events["set_jump_spin_state"] = EventTermCfg(
        func=jump_spin_mdp.reset_jump_spin_state,
        mode="reset",
        params={
            "standing_prob": standing_prob,
            "airborne_prob": airborne_prob,
            "landing_prob": landing_prob,
            "stand_z": STAND_Z,
            "standing_crouch_factor_range": stand_crouch,
            "airborne_z_range": (0.16, 0.24),
            "airborne_progress_range": (math.radians(45.0), math.radians(315.0)),
            "airborne_yaw_rate_range": (4.0, 14.0),
            "airborne_vz_range": (-0.2, 0.8),
            "landing_z_range": (0.125, 0.17),
            "landing_progress_range": (math.radians(300.0), math.radians(355.0)),
            "landing_yaw_rate_range": (0.0, 4.0),
            "landing_vz_range": (-0.8, -0.05),
            "crouch_overrides": CROUCH_OVERRIDES,
            "joint_noise_std": 0.03 if not play else 0.0,
        },
    )

    # Replace roulade-specific curricula, keep its proven CoM/head-CoM DR ramps.
    for name in (
        "roulade_spawn_mix",
        "action_rate_weight",
        "arrival_damping_weight",
        "torque_rate_weight",
        "gentle_landing_weight",
    ):
        cfg.curriculum.pop(name, None)

    if not play:
        cfg.curriculum["jump_spin_spawn_mix"] = CurriculumTermCfg(
            func=common_mdp.event_param_curriculum,
            params={
                "event_name": "set_jump_spin_state",
                "param_stages": [
                    {
                        "step": 0,
                        "params": {
                            "standing_prob": 0.30,
                            "airborne_prob": 0.50,
                            "landing_prob": 0.20,
                            "standing_crouch_factor_range": (0.2, 0.9),
                        },
                    },
                    {
                        "step": 2500 * 24,
                        "params": {
                            "standing_prob": 0.50,
                            "airborne_prob": 0.35,
                            "landing_prob": 0.15,
                            "standing_crouch_factor_range": (0.0, 0.65),
                        },
                    },
                    {
                        "step": 5000 * 24,
                        "params": {
                            "standing_prob": 0.70,
                            "airborne_prob": 0.20,
                            "landing_prob": 0.10,
                            "standing_crouch_factor_range": (0.0, 0.35),
                        },
                    },
                    {
                        "step": 7500 * 24,
                        "params": {
                            "standing_prob": 0.85,
                            "airborne_prob": 0.10,
                            "landing_prob": 0.05,
                            "standing_crouch_factor_range": (0.0, 0.15),
                        },
                    },
                ],
            },
        )

        cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
            func=common_mdp.reward_weight,
            params={
                "reward_name": "action_rate_l2",
                "weight_stages": [
                    {"step": 0, "weight": -0.02},
                    {"step": 4000 * 24, "weight": -0.05},
                    {"step": 7000 * 24, "weight": -0.10},
                ],
            },
        )
        cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
            func=common_mdp.reward_weight,
            params={
                "reward_name": "joint_torque_rate_l2",
                "weight_stages": [
                    {"step": 0, "weight": 0.0},
                    {"step": 5000 * 24, "weight": -2e-4},
                    {"step": 7500 * 24, "weight": -5e-4},
                ],
            },
        )

    return cfg


MicroduckJumpSpinRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Direction-specific positive-yaw trick: left/right symmetry would
        # transform the commanded spin into the opposite direction.
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="jump_spin",
    run_name="jump_spin",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10_000,
)
