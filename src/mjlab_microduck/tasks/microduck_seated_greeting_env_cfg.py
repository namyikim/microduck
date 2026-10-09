"""Seated left/right look + two nods. Starts and finishes seated; no sit/stand transition.

Reuses SitStand's measured seated keyframe, BAM, DR, noise and 61D contract.
The twist posture flag stays 1; this policy is NOT a stand/walk replacement.
"""
from copy import deepcopy
import math
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import (
    make_microduck_sitstand_env_cfg, MicroduckSitStandRlCfg,
)
from mjlab_microduck.seated_greeting import DURATION_S

TASK_ID = 'Mjlab-SeatedGreeting-Flat-MicroDuck'
EXPERIMENT_NAME = 'seated_greeting_v1'


def make_microduck_seated_greeting_env_cfg(play=False):
    cfg = make_microduck_sitstand_env_cfg(play=play, rough=False)
    cfg.episode_length_s = DURATION_S
    cfg.commands['twist'].sit_prob = 1.
    cfg.commands['twist'].resampling_time_range = (100.,100.)
    cfg.commands['head_pose'] = mdp.SeatedGreetingCommandCfg(
        ranges=((-0.02,.02),(0.,math.radians(8)),(-math.radians(15),math.radians(15)),(-.02,.02)),
        resampling_time_range=(100.,100.), randomize_amplitude=not play,
    )
    cfg.events['set_ground_state'].params.update(
        sitting_prob=1., standing_prob=0., face_up_prob=0., face_down_prob=0.,
        sitting_joint_noise_std=.02 if not play else 0.,
        sitting_tilt_max=math.radians(2) if not play else 0.,
        sitting_z_min=.06, sitting_z_max=.062 if not play else .06,
    )
    for name in ('push_robot','seated_head_jerk','seated_tip'):
        cfg.events.pop(name,None)
    for name in ('push_magnitude','seated_head_jerk_prob','seated_tip_range','head_pose_range','rise_speed_weight'):
        cfg.curriculum.pop(name,None)
    cfg.rewards.pop('rise_bootstrap',None)
    cfg.rewards['head_pose_tracking'] = RewardTermCfg(func=mdp.seated_greeting_tracking,weight=2.)
    cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
        name='head_ground_contact',
        primary=ContactMatch(mode='body',pattern='jaw_soft',entity='robot'),
        secondary=ContactMatch(mode='body',pattern='terrain'),
        fields=('found',),reduce='none',num_slots=1,
    ))
    cfg.rewards['greeting_head_contact'] = RewardTermCfg(func=mdp.seated_greeting_head_contact_cost,weight=-10.)
    cfg.terminations['greeting_failed'] = TerminationTermCfg(func=mdp.seated_greeting_failed,time_out=False)
    # New narrow task: retain static physical randomization and conservative initial CoM ranges.
    for name in ('com_range','head_com_range'):
        cfg.curriculum.pop(name,None)
    return cfg


MicroduckSeatedGreetingRlCfg = deepcopy(MicroduckSitStandRlCfg)
MicroduckSeatedGreetingRlCfg.experiment_name = EXPERIMENT_NAME
MicroduckSeatedGreetingRlCfg.run_name = 'seated-greeting'
MicroduckSeatedGreetingRlCfg.max_iterations = 3000
MicroduckSeatedGreetingRlCfg.save_interval = 100
