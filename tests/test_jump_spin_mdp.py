"""Tensor regressions for takeoff discovery and reverse-curriculum resets."""
import math
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import jump_spin_mdp as mdp


class Scene(dict):
    pass


def make_env(num_envs=4, step=0):
    asset = SimpleNamespace(data=SimpleNamespace(
        root_link_lin_vel_w=torch.zeros(num_envs, 3),
        root_link_ang_vel_w=torch.zeros(num_envs, 3),
    ))
    scene = Scene(robot=asset)
    scene.terrain = SimpleNamespace(env_origins=torch.zeros(num_envs, 3))
    scene.sensors = {'feet': SimpleNamespace(data=SimpleNamespace(found=torch.ones(num_envs, 2)))}
    env = SimpleNamespace(scene=scene, num_envs=num_envs, device='cpu',
                          common_step_counter=step, step_dt=0.02,
                          sim=SimpleNamespace(data=SimpleNamespace(
                              qpos=torch.zeros(num_envs, 21), qvel=torch.zeros(num_envs, 20))))
    mdp._state(env)
    return env, asset


def test_upward_push_receives_launch_signal_without_yaw():
    env, asset = make_env()
    asset.data.root_link_lin_vel_w[:, 2] = torch.tensor([0.0, 0.3, 0.6, 1.2])
    reward = mdp.jump_spin_launch_reward(env, 'feet')
    assert reward[0] == 0
    assert torch.all(reward[1:] > 0)
    assert torch.all(reward[2:] > reward[1:-1])


def test_launch_cannot_be_farmed_by_repeating_the_same_push():
    env, asset = make_env()
    asset.data.root_link_lin_vel_w[:, 2] = 1.2
    asset.data.root_link_ang_vel_w[:, 2] = 8.0
    first = mdp.jump_spin_launch_reward(env, 'feet')
    second = mdp.jump_spin_launch_reward(env, 'feet')
    assert torch.all(first > 0)
    assert torch.all(second == 0)
    assert torch.all(first * env.step_dt <= 1.0)


def test_yaw_without_upward_motion_and_post_takeoff_give_no_launch_bonus():
    env, asset = make_env()
    asset.data.root_link_ang_vel_w[:, 2] = 8.0
    assert torch.all(mdp.jump_spin_launch_reward(env, 'feet') == 0)
    asset.data.root_link_lin_vel_w[:, 2] = 1.2
    env._jump_spin_airborne_latch[:] = True
    assert torch.all(mdp.jump_spin_launch_reward(env, 'feet') == 0)


@pytest.mark.parametrize('step,low,high', [
    (0, 150.0, 177.5), (1500 * 24, 225.0, 266.25), (4000 * 24, 300.0, 355.0),
])
def test_reverse_landing_progress_tracks_current_target(monkeypatch, step, low, high):
    env, _ = make_env(num_envs=128, step=step)
    monkeypatch.setattr(mdp.common_mdp, '_servo_joint_ids', lambda env, asset: list(range(14)))
    mdp.reset_jump_spin_state(env, torch.arange(128), standing_prob=0, airborne_prob=0,
                             landing_prob=1, joint_noise_std=0)
    degrees = env._jump_spin_accum * 180.0 / math.pi
    assert torch.all(degrees >= low)
    assert torch.all(degrees <= high)
    assert torch.all(env._jump_spin_accum < mdp.curriculum_target_angle(env))


def test_reset_clears_launch_budget_for_only_reset_environments(monkeypatch):
    env, asset = make_env()
    monkeypatch.setattr(mdp.common_mdp, '_servo_joint_ids', lambda env, asset: list(range(14)))
    asset.data.root_link_lin_vel_w[:, 2] = 1.2
    mdp.jump_spin_launch_reward(env, 'feet')
    mdp.reset_jump_spin_state(env, torch.tensor([0]), standing_prob=1, airborne_prob=0,
                             landing_prob=0, joint_noise_std=0)
    reward = mdp.jump_spin_launch_reward(env, 'feet')
    assert reward[0] > 0
    assert torch.all(reward[1:] == 0)
