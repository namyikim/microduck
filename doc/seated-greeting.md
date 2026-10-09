# 앉아서 두리번거리기와 인사 / Seated greeting

## 한국어

현재 목표는 **앉은 상태에서 왼쪽 보기 → 오른쪽 보기 → 정면 → 두 번 끄덕이기 → 앉아서 정지**입니다.
JumpSpin 최종 모델은 머리 접촉과 쓰러짐으로 실패했으며 이 학습에 재사용하지 않습니다.

[Colab 노트북 열기](https://colab.research.google.com/github/namyikim/microduck/blob/main/MicroDuck_SeatedGreeting_A100.ipynb)

### 내일 실행할 순서

1. Colab에서 A100 GPU를 선택합니다.
2. 설정·Drive 연결·설치 셀을 실행합니다.
3. 준비가 되면 설정 셀의 `START_TRAINING=False`를 `True`로 바꾸고 설정 셀을 다시 실행합니다.
4. 학습 셀을 실행합니다. 자세 검증 → GPU 검증 학습 → 공식 ONNX 검증 → 본학습 → 영상 5개가 순서대로 실행됩니다.

노트북은 기본적으로 학습이 꺼져 있습니다. 예약 실행을 만들지 않으며, 기존 JumpSpin 자동 점검도 재개하지 않습니다.
CLI 런타임을 새로 만들 때는 [agent.md](../agent.md)에 따라 항상 `colab run --keep`을 사용합니다.
기존 런타임의 설치·Drive 연결 후 다음 명령으로 같은 파이프라인을 실행할 수 있습니다.

```bash
uv run python scripts/train_seated_greeting_colab.py --start-training
```

### 학습 범위와 검증

- Task: `Mjlab-SeatedGreeting-Flat-MicroDuck`
- Experiment: `seated_greeting_v1`, 기본 4,096환경·총 3,000회. 이 횟수는 초기 예산이지 성공 보장이 아닙니다.
- 기존 SitStand **환경 설계**를 재사용합니다. 기본 실행은 새 정책을 처음부터 학습하며 기존 SitStand/JumpSpin 가중치를 자동으로 불러오지 않습니다.
- 이미 앉은 상태에서 시작합니다. 앉기·일어서기·보행은 이 정책의 범위 밖입니다.
- 약 14초: 처음 2초 정지, 좌우 약 15°, 정면 복귀, 약 8° 끄덕임 2회, 마지막 2초 정지.
- 머리 명령은 cosine easing으로 연결합니다. 정책의 액션에는 새 필터를 추가하지 않습니다.
- BAM 액추에이터, 기존 앉은 자세, 관측 61차원/출력 14차원, 관측 정규화를 유지합니다.
- 머리 접촉이나 45° 초과 몸통 기울기, 앉은 높이의 큰 이탈은 학습 에피소드를 실패 종료합니다.
- 평가에서는 쓰러진 동작도 끝까지 기록하고, 머리 접촉 없음·기울기 20° 이내·앉은 높이 4~8.5cm·각 인사 구간의 명령 추적 오차를 검사합니다. 가만히 버티는 모델은 좌우 보기와 끄덕임 구간 검사에 실패합니다.

장시간 학습 전에 **64환경에서 앉은 목표 자세를 3초 유지한 뒤 머리 동작 전체를 수행하는 물리 검증**을 통과해야 합니다. 기울기·높이·머리 접촉 중 하나라도 기준을 벗어나면 중단하고 `preflight.json`을 확인합니다. 이어서 64환경·5회 학습 및 공식 ONNX 내보내기·추론 검증을 수행합니다.

GPU 물리 검증과 실제 학습 결과는 아직 확인 전입니다. 코드 테스트 통과와 실제 동작 성공은 다릅니다.

### 저장 위치와 재개

Drive의 `microduck-training/seated_greeting_v1/` 안에 저장됩니다.

| 파일 | 용도 |
|---|---|
| `training.log` | 본학습 iteration·보상·예상 시간 |
| `logs/rsl_rl/seated_greeting_v1/<run>/model_*.pt` | 체크포인트 |
| `evaluations/<UTC 실행시각>/preflight.json` | 앉은 자세와 머리 동작 물리 검증 |
| 같은 폴더의 `policy.onnx` | 정규화가 포함된 최종 ONNX |
| 같은 폴더의 `seated_greeting_trial_01.mp4` ~ `05.mp4` | 평가 영상 |
| 같은 폴더의 `evaluation_summary.json`, `*_trace.json` | 동작 구간별 검사·시계열 |
| `status.json` | 현재 단계와 실패 원인 |

같은 노트북을 다시 실행하면 같은 실험의 가장 높은 체크포인트에서 재개합니다. 체크포인트 숫자와 내부 iteration을 확인하고, 이미 목표에 도달했다면 본학습을 생략합니다. 각 평가를 새 폴더에 저장해 이전 결과와 섞이지 않습니다.

### 실제 로봇에 적용하기 전

이 정책은 **앉은 자세 전용**입니다. `twist=[1,0,0]`과 시간에 따른 4차원 머리 명령을 받아야 합니다. 일반 보행 정책이나 항상 같은 명령을 주는 일회성 동작으로 바로 교체할 수 없습니다. 서 있는 로봇을 이 정책으로 전환하지 마세요. 실제 배포 시에는 검증된 앉기 정책과 머리 명령 실행기를 연결하고 동작 범위를 확인해야 합니다. 노트북은 로봇에 배포하지 않습니다.

## English

The target is a **seated greeting**: look left, look right, return to center, nod twice, then rest seated. The failed JumpSpin model is not reused.

Open the notebook on a later day, select A100, mount Drive, install dependencies, then explicitly set `START_TRAINING=True`. The default does not start training. No new schedule is created. Always use `colab run --keep` when creating a CLI runtime.

The new `Mjlab-SeatedGreeting-Flat-MicroDuck` task builds on the SitStand environment and its measured seated pose, BAM actuators and normalized 61D observations / 14D actions. It trains a new policy from scratch in `seated_greeting_v1`. The 14-second command uses small eased yaw (±15°) and pitch (8°) motions; the actor's actions remain unfiltered. It starts seated and does not teach sitting down, standing up or walking.

Before the default 4,096-environment / 3,000-iteration run, the pipeline requires a 64-environment seated hold and head-motion physics preflight, then a 64-environment / 5-iteration smoke run and official normalized ONNX shape/inference checks. Failed preflight stops training. GPU physics and training are not yet verified; passing code tests does not establish a successful behavior.

Five videos, per-gesture tracking/contact/tilt checks, traces and ONNX are saved under `MyDrive/microduck-training/seated_greeting_v1/evaluations/<UTC timestamp>/`. Checkpoints resume only within this experiment. Inspect videos even when simulation checks pass.

Hardware use requires an already-seated entry pose, posture flag `[1,0,0]`, and the time-varying head commands. This is not a drop-in walking or constant-command trick policy. Deployment is a separate task.
