"""MDP helpers for the experimental MicroDuck jump-spin task.

The task is deliberately state-based rather than clock-driven:
stand -> take off -> rotate around world Z while fully airborne -> land on both
feet -> recover the standing pose.

Training uses reverse-curriculum reset buckets (standing / mid-air / landing)
so the policy can learn landing and aerial control before it has discovered a
complete take-off from a standing start.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as common_mdp


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _state(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not hasattr(env, "_jump_spin_accum"):
        z = torch.zeros(env.num_envs, device=env.device)
        env._jump_spin_accum = z.clone()
        env._jump_spin_max = z.clone()
        env._jump_spin_paid = z.clone()
        env._jump_spin_launch_paid = z.clone()
        env._jump_spin_airborne_latch = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env._jump_spin_start_yaw = z.clone()
        env._jump_spin_last_update_step = -1
    return env._jump_spin_accum, env._jump_spin_max, env._jump_spin_paid


def _root_height(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
    return torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2],
        nan=0.0,
    )


def _feet_found(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor | None:
    if sensor_name not in env.scene.sensors:
        return None
    found = env.scene.sensors[sensor_name].data.found
    return found.view(found.shape[0], -1) > 0


def _any_foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    found = _feet_found(env, sensor_name)
    if found is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return found.any(dim=-1)


def _both_feet_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    found = _feet_found(env, sensor_name)
    if found is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if found.shape[1] >= 2:
        return found.sum(dim=-1) >= 2
    return found.any(dim=-1)


def curriculum_target_angle(
    env: ManagerBasedRlEnv,
    initial_angle: float = math.pi,
    middle_angle: float = 1.5 * math.pi,
    final_angle: float = 2.0 * math.pi,
    middle_step: int = 1500 * 24,
    final_step: int = 4000 * 24,
) -> float:
    """180° -> 270° -> 360° curriculum based on env steps."""
    step = int(env.common_step_counter)
    if step < middle_step:
        return initial_angle
    if step < final_step:
        return middle_angle
    return final_angle


def _completion_gate(
    env: ManagerBasedRlEnv,
    low_frac: float,
    high_frac: float,
    **target_kwargs,
) -> torch.Tensor:
    _state(env)
    target = curriculum_target_angle(env, **target_kwargs)
    frac = env._jump_spin_max / max(target, 1e-6)
    t = torch.clamp((frac - low_frac) / max(high_frac - low_frac, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return smooth * env._jump_spin_airborne_latch.float()


def _update_spin(
    env: ManagerBasedRlEnv,
    asset: Entity,
    sensor_name: str,
    takeoff_min_height: float,
) -> None:
    """Integrate positive WORLD-Z yaw only while both feet are off the ground."""
    _state(env)
    step = int(env.common_step_counter)
    if step == env._jump_spin_last_update_step:
        return

    contact = _any_foot_contact(env, sensor_name)
    height = _root_height(env, asset)
    airborne = (~contact) & (height > takeoff_min_height)

    # Once a genuine take-off occurred, remember it for landing gates.
    env._jump_spin_airborne_latch |= airborne

    omega_z = torch.nan_to_num(asset.data.root_link_ang_vel_w[:, 2], nan=0.0)
    forward_rate = torch.clamp(omega_z, min=0.0)
    delta = forward_rate * env.step_dt
    delta = delta * airborne.float() * env._jump_spin_airborne_latch.float()

    env._jump_spin_accum = env._jump_spin_accum + delta
    env._jump_spin_max = torch.maximum(env._jump_spin_max, env._jump_spin_accum)
    env._jump_spin_last_update_step = step


def reset_jump_spin_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    standing_prob: float = 0.30,
    airborne_prob: float = 0.50,
    landing_prob: float = 0.20,
    stand_z: float = 0.115,
    standing_crouch_factor_range: tuple[float, float] = (0.2, 0.9),
    airborne_z_range: tuple[float, float] = (0.16, 0.24),
    airborne_progress_range: tuple[float, float] = (
        math.radians(45.0),
        math.radians(315.0),
    ),
    airborne_yaw_rate_range: tuple[float, float] = (4.0, 14.0),
    airborne_vz_range: tuple[float, float] = (-0.2, 0.8),
    landing_z_range: tuple[float, float] = (0.125, 0.17),
    landing_progress_range: tuple[float, float] = (
        math.radians(300.0),
        math.radians(355.0),
    ),
    landing_yaw_rate_range: tuple[float, float] = (0.0, 4.0),
    landing_vz_range: tuple[float, float] = (-0.8, -0.05),
    crouch_overrides: dict[int, float] | None = None,
    joint_noise_std: float = 0.03,
    **target_kwargs,
) -> None:
    """Reset into standing, mid-air spin, or near-landing states.

    Progress ranges describe the final-angle stage and scale with the current
    target. Otherwise 300..355 degree landing spawns start *past* the early
    180/270 degree goals and never practice the remaining part of the turn.
    """
    if env_ids is None or len(env_ids) == 0:
        return

    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    _state(env)

    total = max(standing_prob + airborne_prob + landing_prob, 1e-6)
    u = torch.rand(num, device=env.device) * total
    is_stand = u < standing_prob
    is_air = (u >= standing_prob) & (u < standing_prob + airborne_prob)
    is_land = ~(is_stand | is_air)

    def uniform(lo: float, hi: float) -> torch.Tensor:
        return torch.rand(num, device=env.device) * (hi - lo) + lo

    # Progress represented by the physical yaw offset relative to a random
    # starting heading. This makes reverse-curriculum states geometrically
    # consistent with the accumulated rotation used by the reward.
    progress = torch.zeros(num, device=env.device)
    target = curriculum_target_angle(env, **target_kwargs)
    final_angle = target_kwargs.get("final_angle", 2.0 * math.pi)
    progress_scale = target / final_angle
    air_progress = uniform(*airborne_progress_range) * progress_scale
    land_progress = uniform(*landing_progress_range) * progress_scale
    progress = torch.where(is_air, air_progress, progress)
    progress = torch.where(is_land, land_progress, progress)

    start_yaw = uniform(-math.pi, math.pi)
    yaw = start_yaw + progress

    half = yaw * 0.5
    quat = torch.stack(
        [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)],
        dim=1,
    )

    z = torch.full((num,), stand_z, device=env.device)
    z = torch.where(is_air, uniform(*airborne_z_range), z)
    z = torch.where(is_land, uniform(*landing_z_range), z)

    env.sim.data.qpos[env_ids, 2] = z + env.scene.terrain.env_origins[env_ids, 2]
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0

    vz = torch.zeros(num, device=env.device)
    vz = torch.where(is_air, uniform(*airborne_vz_range), vz)
    vz = torch.where(is_land, uniform(*landing_vz_range), vz)
    env.sim.data.qvel[env_ids, 2] = vz

    omega_z = torch.zeros(num, device=env.device)
    omega_z = torch.where(is_air, uniform(*airborne_yaw_rate_range), omega_z)
    omega_z = torch.where(is_land, uniform(*landing_yaw_rate_range), omega_z)
    # Free-joint rotational qvel is body-frame. Spawn orientation is upright,
    # so body Z == world Z here.
    env.sim.data.qvel[env_ids, 5] = omega_z

    servo_ids = common_mdp._servo_joint_ids(env, asset)

    if crouch_overrides:
        factors = torch.zeros(num, device=env.device)
        stand_factor = uniform(*standing_crouch_factor_range)
        air_factor = uniform(0.65, 1.0)
        land_factor = uniform(0.0, 0.45)
        factors = torch.where(is_stand, stand_factor, factors)
        factors = torch.where(is_air, air_factor, factors)
        factors = torch.where(is_land, land_factor, factors)

        for joint_idx, target in crouch_overrides.items():
            col = 7 + servo_ids[joint_idx]
            home = env.sim.data.qpos[env_ids, col]
            env.sim.data.qpos[env_ids, col] = home + factors * (target - home)

    if joint_noise_std > 0.0:
        cols = torch.tensor(
            [7 + j for j in servo_ids], device=env.device, dtype=torch.long
        )
        noise = torch.randn(num, len(cols), device=env.device) * joint_noise_std
        env.sim.data.qpos[env_ids.unsqueeze(1), cols.unsqueeze(0)] += noise

    env._jump_spin_accum[env_ids] = progress
    env._jump_spin_max[env_ids] = progress
    env._jump_spin_paid[env_ids] = progress
    env._jump_spin_launch_paid[env_ids] = 0.0
    env._jump_spin_start_yaw[env_ids] = start_yaw
    env._jump_spin_airborne_latch[env_ids] = is_air | is_land
    env._jump_spin_last_update_step = -1


def jump_spin_launch_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_vz: float = 1.2,
    target_yaw_rate: float = 8.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay new launch momentum, including upward pushes before yaw is learned.

    The old product gave a pure vertical push zero signal. Half of the score
    now comes from upward speed alone; the other half encourages simultaneous
    yaw. Pay only increases beyond the episode's best score, so repeated
    grounded bobbing cannot farm a per-step launch reward. RewardManager
    multiplies by step_dt: total unweighted launch payout is at most one.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _state(env)
    contact = _any_foot_contact(env, sensor_name)
    not_taken_off = ~env._jump_spin_airborne_latch
    vz = torch.clamp(asset.data.root_link_lin_vel_w[:, 2], min=0.0)
    wz = torch.clamp(asset.data.root_link_ang_vel_w[:, 2], min=0.0)
    vertical = torch.clamp(vz / target_vz, 0.0, 1.0)
    yaw = torch.clamp(wz / target_yaw_rate, 0.0, 1.0)
    score = vertical * (0.5 + 0.5 * yaw) * contact.float() * not_taken_off.float()
    paid = env._jump_spin_launch_paid
    delta = torch.clamp(score - paid, min=0.0)
    env._jump_spin_launch_paid = torch.maximum(paid, score)
    return delta / env.step_dt


def jump_spin_airborne_height(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    stand_z: float,
    target_height: float = 0.19,
    takeoff_min_height: float = 0.125,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Dense jump-height reward, active only after both feet leave the ground."""
    asset: Entity = env.scene[asset_cfg.name]
    _update_spin(env, asset, sensor_name, takeoff_min_height)
    contact = _any_foot_contact(env, sensor_name)
    height = _root_height(env, asset)
    score = torch.clamp(
        (height - stand_z) / max(target_height - stand_z, 1e-6), 0.0, 1.0
    )
    return score * (~contact).float() * env._jump_spin_airborne_latch.float()


def jump_spin_progress(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    takeoff_min_height: float = 0.125,
    max_paid_rate: float = 25.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **target_kwargs,
) -> torch.Tensor:
    """Pay only new positive yaw accumulated while fully airborne."""
    asset: Entity = env.scene[asset_cfg.name]
    _update_spin(env, asset, sensor_name, takeoff_min_height)
    _, max_accum, paid = _state(env)

    target = curriculum_target_angle(env, **target_kwargs)
    frontier = torch.clamp(max_accum, max=target)
    delta = torch.clamp(frontier - torch.clamp(paid, max=target), min=0.0)
    delta = torch.clamp(delta, max=max_paid_rate * env.step_dt)

    # Excess rate is intentionally forfeited rather than paid later.
    env._jump_spin_paid = torch.maximum(paid, frontier)
    return delta / (env.step_dt * max(target, 1e-6))


def jump_spin_airborne_upright(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    takeoff_min_height: float = 0.125,
    std: float = 0.28,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Keep the trunk vertical while yawing in flight."""
    asset: Entity = env.scene[asset_cfg.name]
    _update_spin(env, asset, sensor_name, takeoff_min_height)
    contact = _any_foot_contact(env, sensor_name)
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    score = torch.exp(-tilt_sq / (std * std))
    return score * (~contact).float() * env._jump_spin_airborne_latch.float()


def jump_spin_wrong_axis_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    takeoff_min_height: float = 0.125,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize roll/pitch angular velocity in flight; yaw is intentionally free."""
    asset: Entity = env.scene[asset_cfg.name]
    _update_spin(env, asset, sensor_name, takeoff_min_height)
    contact = _any_foot_contact(env, sensor_name)
    omega_xy = asset.data.root_link_ang_vel_w[:, :2]
    return torch.sum(omega_xy * omega_xy, dim=-1) * (~contact).float()


def jump_spin_horizontal_drift_cost(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    v_xy = asset.data.root_link_lin_vel_w[:, :2]
    return torch.sum(v_xy * v_xy, dim=-1)


def jump_spin_landing_score(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    height_std: float = 0.025,
    upright_std: float = 0.25,
    pose_std: float = 0.35,
    joint_indices: list[int] | None = None,
    takeoff_min_height: float = 0.125,
    gate_low_frac: float = 0.92,
    gate_high_frac: float = 0.99,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **target_kwargs,
) -> torch.Tensor:
    """Dominant terminal annuity: nearly-complete spin + both feet + stable stand."""
    asset: Entity = env.scene[asset_cfg.name]
    _update_spin(env, asset, sensor_name, takeoff_min_height)
    both = _both_feet_contact(env, sensor_name).float()
    gate = _completion_gate(env, gate_low_frac, gate_high_frac, **target_kwargs)
    if joint_indices is None:
        joint_indices = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
    stand = common_mdp.standing_composite_score(
        env,
        target_height=target_height,
        height_std=height_std,
        upright_std=upright_std,
        pose_std=pose_std,
        joint_indices=joint_indices,
        asset_cfg=asset_cfg,
    )
    return stand * both * gate


def jump_spin_heading_return(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    std: float = 0.20,
    takeoff_min_height: float = 0.125,
    gate_low_frac: float = 0.94,
    gate_high_frac: float = 0.995,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **target_kwargs,
) -> torch.Tensor:
    """At the final 360° stage, reward landing facing the original heading."""
    asset: Entity = env.scene[asset_cfg.name]
    _update_spin(env, asset, sensor_name, takeoff_min_height)
    target = curriculum_target_angle(env, **target_kwargs)
    if target < 1.9 * math.pi:
        return torch.zeros(env.num_envs, device=env.device)

    q = asset.data.root_link_quat_w
    yaw = torch.atan2(
        2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
        1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2),
    )
    err = torch.atan2(
        torch.sin(yaw - env._jump_spin_start_yaw),
        torch.cos(yaw - env._jump_spin_start_yaw),
    )
    heading = torch.exp(-((err / std) ** 2))
    both = _both_feet_contact(env, sensor_name).float()
    gate = _completion_gate(env, gate_low_frac, gate_high_frac, **target_kwargs)
    return heading * both * gate


def jump_spin_landing_impact_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    max_down_speed: float = 0.35,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize hard downward velocity while the feet are in contact after take-off."""
    asset: Entity = env.scene[asset_cfg.name]
    _state(env)
    both = _both_feet_contact(env, sensor_name).float()
    downward = torch.clamp(-asset.data.root_link_lin_vel_w[:, 2] - max_down_speed, min=0.0)
    return downward * downward * both * env._jump_spin_airborne_latch.float()
