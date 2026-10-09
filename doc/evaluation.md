# 평가 영상과 실패 진단

[README로 돌아가기](../README.md) · [JumpSpin 설계](jump-spin.md) · [배포](release.md)

## 학습 후 자동 평가

JumpSpin 노트북은 목표 학습 횟수에 도달하면 최신 로컬 체크포인트로 정상 서기 상태에서 시작하는 독립 에피소드 5개를 평가합니다. 중간에 세션이 끊기면 학습을 재개해 목표에 도달한 뒤 평가가 실행됩니다.

기본 노트북의 저장 위치는 다음과 같습니다.

```text
MyDrive/microduck-training/videos/
├── jump_spin_trial_01.mp4
├── jump_spin_trial_02.mp4
├── jump_spin_trial_03.mp4
├── jump_spin_trial_04.mp4
├── jump_spin_trial_05.mp4
├── evaluation_summary.json
└── *_trace.json
```

별도 CLI 실행 스크립트에서는 기존 영상을 보존하기 위해 `videos/jump_spin_launch_v2/` 같은 하위 폴더를 사용할 수 있습니다. 실행 로그에 출력된 저장 위치가 기준입니다.

## 영상에서 확인할 것

1. 정상적인 양발 서기에서 출발하는지 확인합니다.
2. 양발이 실제로 지면에서 떨어지고, 머리 등 다른 부위로 몸을 지지하지 않는지 확인합니다.
3. 수직축을 중심으로 약 360° 회전하는지 확인합니다.
4. 양발로 착지하고 안정적으로 서는지 확인합니다.
5. 착지 충격과 관절 한계 동작이 과도하지 않은지 여러 영상에서 확인합니다.

`model_9999.pt`의 번호는 학습 진행 횟수이며 성공 점수가 아닙니다. 실제 로봇에 적용하기 전에는 반복 실패 양상과 충격·관절·구동기 여유도까지 검토해야 합니다.

## 기존 체크포인트만 다시 평가

최신 JumpSpin 노트북에서 Drive 연결·설정·설치 셀까지만 실행하고 **기존 checkpoint만 재평가 (재학습 없음)** 셀을 사용합니다. 학습 시작 셀은 실행하지 않습니다.

- `RUN_EXISTING_CHECKPOINT_EVAL = True`로 바꿉니다.
- 기본 평가 대상은 이전 `jump_spin` 실험입니다. v2를 평가하려면 `EVAL_EXPERIMENT_NAME = EXPERIMENT_NAME`으로 바꿉니다.
- 특정 파일을 평가하려면 `EVAL_CHECKPOINT`에 정확한 경로를 지정합니다.
- 결과는 기본적으로 `videos/diagnostic_eval/`에 저장됩니다.

명령행에서 직접 평가할 수도 있습니다. 아래 `CHECKPOINT`는 실제 파일 경로로 바꿉니다.

```bash
CHECKPOINT="/path/to/model_9999.pt"
uv run python scripts/render_jump_spin.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir /tmp/jump-spin-videos \
  --trials 5
```

## JSON과 학습 로그 해석

`evaluation_summary.json`에는 체크포인트, 에피소드 보상·길이, 관측 회전 진행량이 기록됩니다. `rollout_diagnostics`에는 몸통 높이, 상승 속도, 실제 공중 시간, 마지막 관측 자세와 발 접촉 수가 포함됩니다. `*_trace.json`은 프레임별 기록입니다.

| 진단값 | 의미 |
|---|---|
| `no_observed_takeoff` | 관측 구간에서 실제 이륙을 확인하지 못함 |
| `airborne_without_target_spin` | 공중 상태는 있었지만 목표 회전량에 미달 |

진단은 자동 리셋 직전 마지막 물리 프레임을 놓칠 수 있습니다. 기존 회전 누적값은 양의 world-Z 각속도를 적분하므로 순회전 360°나 안정적인 착지의 증명이 아닙니다. 영상·진단·학습 로그를 함께 확인합니다.

동작이 고개만 숙이고 끝난다면 먼저 실제 체크포인트의 이륙 여부와 주 보상 항목을 확인합니다. 총 보상 상승만 보고 같은 설정으로 장기 학습을 반복하지 않습니다. 문제를 공유할 때는 평가 요약, 영상, `training.log`의 마지막 부분과 체크포인트 경로를 함께 제공하면 됩니다.
