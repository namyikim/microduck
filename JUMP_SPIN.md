[![Open JumpSpin In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/namyikim/microduck/blob/main/MicroDuck_JumpSpin_A100.ipynb)

# MicroDuck Jump Spin 360

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


## Automatic rollout videos

The JumpSpin Colab notebook automatically renders evaluation videos after
training reaches the target iteration count.

It selects the latest local checkpoint and runs five independent play-mode
episodes from the normal standing start. The headless renderer writes:

```text
MyDrive/microduck-jump-spin-training/
└── videos/
    ├── jump_spin_trial_01.mp4
    ├── jump_spin_trial_02.mp4
    ├── jump_spin_trial_03.mp4
    ├── jump_spin_trial_04.mp4
    ├── jump_spin_trial_05.mp4
    └── evaluation_summary.json
```

The summary records the checkpoint, per-episode reward, episode length, and
the observed maximum airborne spin-progress accumulator. The notebook also
embeds the generated MP4 files directly below the training cell.

You can rerender manually from any local checkpoint:

```bash
uv run python scripts/render_jump_spin.py \
  --checkpoint logs/rsl_rl/jump_spin/<run>/model_9900.pt \
  --output-dir /tmp/jump-spin-videos \
  --trials 5
```


## Hugging Face release

학습 완료 후 모델 검증과 Hugging Face 배포 절차는 [JUMP_SPIN_HF_RELEASE.md](JUMP_SPIN_HF_RELEASE.md)를 따릅니다. 실제 업로드 전에 rollout 영상의 수동 검토가 필요합니다.

## 학습은 끝났지만 점프하지 않는 경우

`model_9999.pt`는 학습 진행 횟수를 나타내며 동작 성공을 의미하지 않습니다.
기존 평가 파일은 Drive의 `MyDrive/microduck-training/videos/evaluation_summary.json`,
학습 로그는 `MyDrive/microduck-training/training.log`에 있습니다.

기존 checkpoint를 진단하려면 최신 Colab 노트북에서 Drive 연결·설정·설치 셀을
실행한 뒤 **기존 checkpoint만 재평가 (재학습 없음)** 셀을 실행하세요.
학습 시작 셀을 실행할 필요가 없습니다. 여러 실험이 있으면 `EVAL_CHECKPOINT`로
평가할 파일을 직접 지정하세요.

새 결과는 `videos/diagnostic_eval/`에 저장됩니다. 요약 JSON의
`rollout_diagnostics`에는 몸통 높이, 상승 속도, 실제 공중 시간, 마지막 관측 자세와
발 접촉 수가 기록됩니다. 각 `*_trace.json`에는 프레임별 진단값이 있습니다.
`no_observed_takeoff`는 관측 구간에서 이륙하지 못했다는 뜻이고,
`airborne_without_target_spin`은 공중 상태가 있었지만 목표 회전량에 못 미쳤다는 뜻입니다.
머리 등 로봇의 다른 부위가 바닥에 닿으면 실제 공중 시간으로 세지 않습니다.

진단은 자동 reset 직전의 최종 물리 프레임을 놓칠 수 있습니다. 기존 회전 누적값은
양의 world-Z 각속도를 적분하므로 순회전 360° 또는 안정적 착지의 증명이 아닙니다.
영상과 진단·학습 로그를 함께 확인한 뒤 재학습 방향을 결정해야 합니다.
