"""Register namyikim's experimental MicroDuck tasks."""

from mjlab.tasks.registry import register_mjlab_task

from mjlab_microduck.tasks import MicroduckOnPolicyRunner
from mjlab_microduck.tasks.microduck_jump_spin_env_cfg import (
    MicroduckJumpSpinRlCfg,
    make_microduck_jump_spin_env_cfg,
)


register_mjlab_task(
    task_id="Mjlab-JumpSpin-Flat-MicroDuck",
    env_cfg=make_microduck_jump_spin_env_cfg(),
    play_env_cfg=make_microduck_jump_spin_env_cfg(play=True),
    rl_cfg=MicroduckJumpSpinRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
