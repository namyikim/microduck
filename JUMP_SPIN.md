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
  --checkpoint logs/rsl_rl/jump_spin_launch_v2/<run>/model_9900.pt \
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

## 2026-10-08 실패 분석과 launch v2

제공된 `model_9999.pt` 평가 요약에서는 5회 모두 225스텝(4.5초)을 실행했지만
관측 회전량은 모두 0°였습니다. 제공된 로그의 마지막 100 iteration 평균은
`jump_launch=0.00076`, `jump_spin_progress=0.06979`,
`action_rate_l2=-0.94698`이었습니다. 이는 목표 동작을 획득했다는 증거가 아닙니다.
로그에는 초기 0–12회와 마지막 9100–9999회가 있으므로 전체 학습 경과를 복원할 수는 없습니다.

코드 및 CPU 텐서 테스트에서 확인한 두 문제를 수정했습니다.

- 이륙 보상의 `vertical * yaw`는 회전 없는 상승에 항상 0을 반환했습니다.
  v2는 상승 자체에도 신호를 주고 회전이 동반되면 추가 점수를 줍니다.
  한 에피소드에서 이전 최고 점수를 넘은 증가분만 지급하므로 같은 동작을 반복해서
  보상을 계속 받지 못합니다. `step_dt` 적분 후 최대 지급량은 reward weight 이하입니다.
- 착지 보조 시작의 300–355° 범위는 초기 180°/270° 목표를 이미 넘었습니다.
  v2는 시작 진행량을 현재 목표에 비례시킵니다. 180° 단계의 착지 시작은
  150–177.5°, 270° 단계는 225–266.25°, 360° 단계는 기존과 동일합니다.

새 실험 이름은 `jump_spin_launch_v2`입니다. checkpoint는
`MyDrive/microduck-training/logs/rsl_rl/jump_spin_launch_v2/`에 저장됩니다.
기존 `jump_spin`의 9999번 checkpoint를 새 실험에 복사하거나 자동 resume하지 마세요.
보상 변경만으로 기존 모델의 가중치가 개선되지는 않습니다.

노트북 기본값은 **10,000회 전체 학습과 마지막 영상 평가 자동 실행**입니다.
새 실험은 기존 스크립트의 64환경/5회 smoke test를 먼저 거칩니다. 회전 목표와
시작 조건의 커리큘럼도 자동으로 진행되어 단계마다 셀을 수동 실행할 필요가 없습니다.
짧은 진단만 원하면 선택적으로 `TARGET_ITERS = 500`을 사용할 수 있습니다.
이륙이 계속 0이면 같은 설정의 장기 학습을 반복하지 말고 진단 결과를 확인하세요.
학습 횟수는 성공을 보장하지 않으며, 360° 성공은 실제 재학습·영상 검증 전까지 미확인입니다.

로컬 Mac에서는 mjlab/CUDA를 실행하지 못해 임시 PyTorch 2.2.2 환경에서
시뮬레이터 import만 대체한 순수 텐서 테스트를 실행했습니다. Colab의 고정 버전
PyTorch 2.9.1은 변경하지 않았습니다. 전체 config 테스트는 Linux CI,
물리 smoke test와 재학습은 Colab에서 별도로 확인해야 합니다.
