[![Open JumpSpin In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/namyikim/microduck/blob/main/MicroDuck_JumpSpin_A100.ipynb)\n\n# MicroDuck Jump Spin 360

Experimental RL task for a standing jump followed by an airborne yaw rotation
and a controlled two-foot landing.

## Task

```text
stand
  -> crouch / load legs
  -> take off
  -> both feet airborne
  -> yaw spin
  -> both-feet landing
  -> stable HOME stand
```

Task ID:

```text
Mjlab-JumpSpin-Flat-MicroDuck
```

## Curriculum

The requested final maneuver is difficult enough that a fixed 360° goal from
iteration zero is likely to stall. The environment therefore raises the target:

- 0–1499 PPO iterations: 180°
- 1500–3999: 270°
- 4000+: 360°

It also uses reverse-curriculum reset buckets. Early training includes many
mid-air and near-landing states; later training shifts to normal standing starts.

## Smoke test

```bash
uv run train Mjlab-JumpSpin-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max_iterations 5
```

## A100 training

A conservative first run:

```bash
uv run train Mjlab-JumpSpin-Flat-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 10000
```

The task saves checkpoints every 100 PPO iterations.

## What to monitor

Do not judge the run by total reward alone. Watch these terms independently:

- `jump_height`: genuine flight is being discovered.
- `jump_spin_progress`: yaw accumulates only while both feet are airborne.
- `jump_airborne_upright`: spin stays on the vertical axis.
- `jump_landing`: nearly-complete rotations finish on two feet in HOME.
- `jump_heading_return`: at the 360° stage the final heading returns to the
  take-off heading.
- `jump_wrong_axis`: roll/pitch tumbling should stay low.
- `jump_landing_impact`: catches hard crash landings.

## Hardware warning

This is an experimental dynamic maneuver. Validate it in simulation and inspect
landing impact, joint limits, torque/velocity margins, and repeated-failure
behavior before considering a real-robot deployment. A policy that looks good
in a few simulation rollouts is not sufficient evidence for safe hardware use.
