"""Prevent falling/head support from being rewarded as a seated greeting."""
import importlib
from pathlib import Path
from types import SimpleNamespace
import math
import torch
import pytest
from mjlab_microduck.tasks import mdp
from mjlab_microduck.seated_greeting import greeting_pose


def config(play=False):
    path=Path(__file__).parents[1]/'src/mjlab_microduck/tasks/microduck_seated_greeting_env_cfg.py'
    assert path.exists(), 'Seated greeting task is not implemented'
    module=importlib.import_module('mjlab_microduck.tasks.microduck_seated_greeting_env_cfg')
    return module.make_microduck_seated_greeting_env_cfg(play=play)


def test_only_seated_starts_and_no_forced_jerks():
    for play in (True,False):
        cfg=config(play)
        reset=cfg.events['set_ground_state'].params
        assert reset['sitting_prob']==1 and reset['standing_prob']==0
        assert cfg.commands['twist'].sit_prob==1
        assert not {'push_robot','seated_head_jerk','seated_tip'} & cfg.events.keys()
        assert 'head_pose_range' not in cfg.curriculum
        assert cfg.rewards['greeting_head_contact'].weight<0
        assert 'greeting_failed' in cfg.terminations
        for group in ('actor','critic'):
            from mjlab_microduck.tasks.microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg
            assert list(cfg.observations[group].terms)==list(make_microduck_sitstand_env_cfg(play=play).observations[group].terms)
        assert 'expand_bam_friction_fields' in cfg.events


def test_torch_commands_match_rendering_protocol_and_reset_to_neutral():
    assert hasattr(mdp,'SeatedGreetingCommand'), 'Greeting command is not implemented'
    t=torch.arange(0,14,.02)
    cmd=object.__new__(mdp.SeatedGreetingCommand)
    cmd._command=torch.zeros(len(t),4)
    cmd._gains=torch.ones(len(t),1)
    cmd._small_offsets=torch.zeros(len(t),2)
    cmd._env=SimpleNamespace(episode_length_buf=t/.02,step_dt=.02)
    cmd._update_command()
    assert torch.allclose(cmd.command,torch.tensor([greeting_pose(float(x)) for x in t]),atol=1e-6)


def fake_env():
    data=SimpleNamespace(root_link_quat_w=torch.tensor([[1.,0,0,0],[1.,0,0,0],[0.,1,0,0]]),
                         root_link_pos_w=torch.tensor([[0.,0,.06]]*3))
    class Scene(dict): pass
    scene=Scene(robot=SimpleNamespace(data=data))
    scene.terrain=SimpleNamespace(env_origins=torch.zeros(3,3))
    scene.sensors={'head_ground_contact':SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[0],[1],[0]])))}
    return SimpleNamespace(scene=scene)


def test_head_support_and_flop_are_failures_missing_sensor_raises():
    assert hasattr(mdp,'seated_greeting_failed'), 'Failure gate is not implemented'
    env=fake_env()
    assert mdp.seated_greeting_failed(env).tolist()==[False,True,True]
    assert mdp.seated_greeting_head_contact_cost(env).tolist()==[0.,1.,0.]
    del env.scene.sensors['head_ground_contact']
    with pytest.raises(KeyError): mdp.seated_greeting_failed(env)


def test_real_command_build_and_episode_reset():
    env=SimpleNamespace(num_envs=2,device='cpu',step_dt=.02,episode_length_buf=torch.tensor([175,325]))
    cmd=config(play=True).commands['head_pose'].build(env)
    cmd._resample_command(torch.tensor([0,1]))
    cmd._update_command()
    assert cmd.command[0,2]>.2 and cmd.command[1,2]<-.2
    env.episode_length_buf.zero_()
    cmd._update_command()
    assert torch.equal(cmd.command,torch.zeros(2,4))
