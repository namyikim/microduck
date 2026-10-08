# MicroDuck RL — Colab / JumpSpin Extension

이 저장소는 공식 [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)을 기반으로,
Google Colab A100 학습, Google Drive checkpoint 복구, **JumpSpin 360** 실험 task,
자동 rollout 평가, Hugging Face 배포 준비 기능을 추가한 저장소입니다.

> 공식 프로젝트의 기본 구조, 기존 task, 설치법, sim2real 구성, BAM actuator, ONNX 배포 방식 등
> **원본에서 가져온 내용은 공식 upstream README를 참고하세요.**
>
> - Upstream repository: [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)
> - Imported snapshot 정보: [UPSTREAM.md](UPSTREAM.md)

이 README는 원본 설명을 반복하지 않고 **이 저장소에서 추가하거나 변경한 부분**을 중심으로 설명합니다.

---

## 바로 실행

### JumpSpin 360 — Google Colab A100

[![Open JumpSpin In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/namyikim/microduck/blob/main/MicroDuck_JumpSpin_A100.ipynb)

### 일반 MicroDuck 학습 — Google Colab A100

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/namyikim/microduck/blob/main/MicroDuck_Colab_A100.ipynb)

---

## 이 저장소에서 추가한 기능

### 1. Google Colab A100 학습 환경

공식 MicroDuck RL 코드를 Google Colab A100에서 바로 실행할 수 있도록 별도 notebook과 실행 스크립트를 추가했습니다.

주요 파일:

```text
MicroDuck_Colab_A100.ipynb
MicroDuck_JumpSpin_A100.ipynb
COLAB.md
scripts/setup_colab.sh
scripts/train_colab.sh
scripts/progress_filter.py
```

추가 기능:

- A100 GPU 환경 자동 확인
- Python 3.12 + `uv` 환경 설치
- 저장소 자동 clone / pull
- 실시간 training log 출력
- PPO iteration 진행률을 퍼센트로 표시
- Google Drive에 training log 저장
- Colab 세션 종료 후 checkpoint 자동 resume 지원

자세한 내용: [COLAB.md](COLAB.md)

---

### 2. Google Drive checkpoint / resume 강화

Colab 런타임이 종료되면 `/content` 아래의 파일이 사라지기 때문에,
학습 checkpoint를 Google Drive에 지속적으로 백업하도록 변경했습니다.

현재 기본 Drive 루트:

```text
MyDrive/microduck-training/
```

JumpSpin checkpoint:

```text
MyDrive/microduck-training/logs/rsl_rl/jump_spin/
```

주요 동작:

- JumpSpin task는 checkpoint를 **100 PPO iterations마다 생성**
- Colab 학습 중 Google Drive로 약 **10초 주기 동기화**
- Drive 전체에서 `model_XXXX.pt`를 검색
- **iteration 숫자가 가장 큰 checkpoint를 자동 선택**
- checkpoint 내부의 `iter` 값도 확인
- resume 시 선택된 checkpoint와 iteration을 로그에 명확히 표시
- 수동 복구용 checkpoint도 지원

수동 복구 파일은 다음처럼 둘 수 있습니다.

```text
MyDrive/microduck-training/logs/rsl_rl/jump_spin/manual_recovery/model_XXXX.pt
```

다음 Colab 실행에서는 가장 높은 숫자의 checkpoint부터 이어서 학습합니다.

---

## JumpSpin 360

이 저장소에서 새로 추가한 실험적 RL task입니다.

Task ID:

```text
Mjlab-JumpSpin-Flat-MicroDuck
```

목표 동작:

```text
stand
  -> crouch
  -> jump
  -> both feet airborne
  -> yaw spin
  -> two-foot landing
  -> stable stand
```

### Curriculum

바로 360°를 학습시키는 대신 단계적으로 난도를 높입니다.

| PPO iteration | 회전 목표 |
|---:|---:|
| 0–1499 | 180° |
| 1500–3999 | 270° |
| 4000+ | 360° |

또한 reverse curriculum을 사용해서 초기에는 공중/착지 상태를 많이 경험시키고,
후반으로 갈수록 실제 **standing start** 비율을 높입니다.

기본 목표 학습량:

```text
10,000 PPO iterations
```

상세 설명: [JUMP_SPIN.md](JUMP_SPIN.md)

---

## JumpSpin에서 추가한 reward / curriculum

JumpSpin을 위해 다음과 같은 별도 reward 항목을 추가했습니다.

```text
jump_launch
jump_height
jump_spin_progress
jump_airborne_upright
jump_landing
jump_heading_return
jump_wrong_axis
jump_horizontal_drift
jump_landing_impact
```

핵심은 단순히 몸을 회전시키는 것이 아니라:

- 실제로 양발이 지면에서 떨어지고
- world Z축 중심으로 회전하고
- 목표 회전을 완료하고
- 양발로 착지하고
- 다시 안정적으로 서는 것

까지 하나의 성공 동작으로 학습시키는 것입니다.

관련 코드:

```text
src/mjlab_microduck/jump_spin_plugin.py
src/mjlab_microduck/tasks/jump_spin_mdp.py
src/mjlab_microduck/tasks/microduck_jump_spin_env_cfg.py
```

---

## 자동 rollout 영상 평가

JumpSpin 학습이 완료되면 Colab notebook에서 최신 checkpoint를 사용해 자동으로 평가 영상을 생성할 수 있습니다.

기본적으로 정상 standing 상태에서 여러 회의 독립 rollout을 수행합니다.

생성 파일:

```text
MyDrive/microduck-training/videos/
├── jump_spin_trial_01.mp4
├── jump_spin_trial_02.mp4
├── jump_spin_trial_03.mp4
├── jump_spin_trial_04.mp4
├── jump_spin_trial_05.mp4
└── evaluation_summary.json
```

평가 시 확인할 항목:

- 정상적인 두 발 standing 상태에서 시작하는지
- 발이 실제로 지면에서 떨어지는지
- 회전축이 주로 수직축인지
- 약 360° 회전을 달성하는지
- 양발 착지하는지
- 착지 후 자세를 회복하는지
- 과도한 충격이나 관절 한계 동작이 없는지

Renderer:

```text
scripts/render_jump_spin.py
```

---

## Hugging Face 배포 준비

JumpSpin 학습 완료 후 바로 업로드하지 않고,
**평가 → ONNX 변환 → release candidate 생성 → 수동 검토 → Hugging Face 업로드**
순서로 배포하도록 별도 도구를 추가했습니다.

관련 파일:

```text
scripts/prepare_jump_spin_release.py
scripts/publish_jump_spin_hf.py
JUMP_SPIN_HF_RELEASE.md
```

Release candidate에는 다음이 포함됩니다.

```text
hf_release_candidate/
├── policy.onnx
├── manifest.json
├── README.md
├── release_report.json
├── evaluation_summary.json
└── replay.mp4
```

검증 항목:

- checkpoint iteration 일관성
- 안전한 공식 exporter 경로를 통한 ONNX 변환
- observation normalizer를 ONNX graph에 포함
- ONNX input/output shape: **61 → 14**
- CPU smoke test
- NaN / inf 검사
- constant-output 여부 검사
- MicroDuck policy manifest 검사
- rollout 평가 결과 기록

실제 Hugging Face 업로드는 다음 옵션이 없으면 실행되지 않습니다.

```text
--confirm-reviewed
```

새 Hugging Face repository는 기본적으로 **private**로 생성하도록 구성했습니다.

전체 절차: [JUMP_SPIN_HF_RELEASE.md](JUMP_SPIN_HF_RELEASE.md)

---

## 이 저장소의 주요 추가 파일

```text
.
├── COLAB.md
├── JUMP_SPIN.md
├── JUMP_SPIN_HF_RELEASE.md
├── MicroDuck_Colab_A100.ipynb
├── MicroDuck_JumpSpin_A100.ipynb
├── UPSTREAM.md
├── scripts/
│   ├── setup_colab.sh
│   ├── train_colab.sh
│   ├── progress_filter.py
│   ├── render_jump_spin.py
│   ├── prepare_jump_spin_release.py
│   └── publish_jump_spin_hf.py
├── src/mjlab_microduck/
│   ├── jump_spin_plugin.py
│   └── tasks/
│       ├── jump_spin_mdp.py
│       └── microduck_jump_spin_env_cfg.py
└── .github/workflows/
    ├── import-upstream.yml
    └── jump-spin-tests.yml
```

공식 upstream에서 가져온 나머지 파일들의 설명은
[공식 microduck_rl repository](https://github.com/pollen-robotics/microduck_rl)를 참고하세요.

---

## Upstream 동기화

`.github/workflows/import-upstream.yml`을 통해
공식 `pollen-robotics/microduck_rl:develop` snapshot을 다시 가져올 수 있습니다.

동기화 시 이 저장소에서 추가한 다음 기능들은 보존하도록 구성했습니다.

- Colab notebook / training script
- Google Drive checkpoint/resume 도구
- JumpSpin task / reward / curriculum
- JumpSpin test
- rollout video renderer
- Hugging Face release 도구 및 문서

가져온 upstream commit은 [UPSTREAM.md](UPSTREAM.md)에 기록합니다.

---

## CI

JumpSpin 관련 변경은 GitHub Actions에서 기본 검사를 수행합니다.

주요 검사:

- JumpSpin task registry 등록
- JumpSpin config test
- renderer Python compile
- Hugging Face release tool Python compile

Workflow:

```text
.github/workflows/jump-spin-tests.yml
```

---

## 원본 프로젝트

이 저장소에 포함된 MicroDuck RL의 기본 기능과 구조는 아래 공식 프로젝트를 기반으로 합니다.

- **MicroDuck RL:** https://github.com/pollen-robotics/microduck_rl
- **MicroDuck:** https://github.com/pollen-robotics/microduck
- **mjlab:** https://github.com/mujocolab/mjlab
- **BAM:** https://github.com/Rhoban/bam

원본 프로젝트의 일반적인 설치법, 기존 task 목록, robot model, actuator model,
sim2real 설명은 공식 upstream 문서를 기준으로 확인해주세요.

---

## License

원본 프로젝트의 라이선스를 따릅니다.

- Code: Apache License 2.0
- 3D model files: Creative Commons BY-SA-NC

자세한 내용은 [LICENSE](LICENSE)를 참고하세요.
