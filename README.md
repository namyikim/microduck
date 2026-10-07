# MicroDuck Colab A100 Trainer

Google Colab A100에서 Pollen Robotics의 공식 MicroDuck RL을 학습하기 위한 실행용 저장소입니다.

[Open in Google Colab](https://colab.research.google.com/github/namyikim/microduck/blob/main/MicroDuck_Colab_A100.ipynb)

이 저장소는 공식 RL 코드를 복제해서 별도로 유지하지 않습니다. Colab 실행 시
`pollen-robotics/microduck_rl`의 최신 `develop` 브랜치를 clone하고,
A100 학습 + Google Drive checkpoint backup/resume 기능만 추가합니다.

## 기본 학습 설정

- Task: `Mjlab-Velocity-Flat-MicroDuck`
- GPU: Google Colab A100
- Parallel environments: `4096`
- Target PPO iterations: `6000`
- Checkpoint interval: upstream 기본값 `250` iterations
- Drive sync interval: `60` seconds
- W&B: `offline` 기본값
- 최초 실행: `64 env × 5 iterations` smoke test

공식 MicroDuck 문서에서는 walking policy를 4096 environments로 학습하며,
usable gait까지 약 1~2시간, gait/curriculum-heavy 학습은 대략 4000~6000 iterations를
하나의 일반적인 학습 범위로 안내합니다.

## 실행 방법

1. 위의 **Open in Google Colab** 링크를 엽니다.
2. **Runtime > Change runtime type > A100 GPU**를 선택합니다.
3. 노트북 셀을 위에서부터 실행합니다.
4. Google Drive 연결을 승인합니다.
5. 학습 셀을 실행합니다.

기본 결과 저장 위치:

```
MyDrive/microduck-training/
└── logs/
    └── rsl_rl/
        └── velocity/
            └── <training-run>/
                ├── model_*.pt
                └── ...
```

## Colab 세션 종료 후 자동 resume

Colab의 `/content`는 세션 종료 시 사라질 수 있기 때문에 학습 로그와 checkpoint를
Google Drive에 주기적으로 복사합니다.

다음 세션에서 노트북을 다시 실행하면:

1. Drive의 이전 `logs/`를 `/content/microduck_rl/logs/`로 복원
2. `logs/rsl_rl/velocity/`에서 가장 최신 `model_*.pt` 탐색
3. checkpoint 번호를 기준으로 완료 iteration 추정
4. 목표 `6000`까지 남은 iteration만 추가 실행
5. 학습 중 60초마다 Drive로 동기화

checkpoint 파일은 쓰기가 끝난 것으로 판단되는 파일만 임시 파일로 복사한 뒤 rename하여
Drive에 저장하므로, 세션이 종료되는 순간의 부분 checkpoint가 최신 정상 파일을 덮어쓰는
위험을 줄였습니다.

## 파일 구성

```
.
├── MicroDuck_Colab_A100.ipynb
├── README.md
├── .gitignore
└── scripts/
    ├── setup_colab.sh
    └── train_colab.sh
```

### `scripts/setup_colab.sh`

- GPU 확인
- `uv` 설치
- Python 3.12 설치
- 공식 `pollen-robotics/microduck_rl` clone/update
- `uv sync --frozen`
- PyTorch CUDA 확인
- MicroDuck task registry 확인

### `scripts/train_colab.sh`

- Drive checkpoint 복원
- 64 env / 5 iteration smoke test
- 최신 checkpoint 자동 탐색
- 4096 env 본 학습
- checkpoint 자동 resume
- Drive 주기적 backup

## 기본값 변경

노트북 설정 셀:

```python
TASK_ID = "Mjlab-Velocity-Flat-MicroDuck"
EXPERIMENT_NAME = "velocity"
NUM_ENVS = 4096
TARGET_ITERS = 6000
SYNC_INTERVAL = 60
DRIVE_ROOT = "/content/drive/MyDrive/microduck-training"
SMOKE_TEST = True
```

다른 MicroDuck task를 학습할 때는 해당 task의 `experiment_name`도 맞춰야
자동 checkpoint 탐색이 올바른 경로를 사용합니다.

## 수동 실행

```bash
git clone https://github.com/namyikim/microduck.git /content/microduck_colab
bash /content/microduck_colab/scripts/setup_colab.sh
```

Drive가 `/content/drive`에 mount된 뒤:

```bash
TASK_ID=Mjlab-Velocity-Flat-MicroDuck \
EXPERIMENT_NAME=velocity \
NUM_ENVS=4096 \
TARGET_ITERS=6000 \
DRIVE_ROOT=/content/drive/MyDrive/microduck-training \
bash /content/microduck_colab/scripts/train_colab.sh
```

## 학습 완료 후

공식 upstream의 도구를 사용해서 policy를 검증하고 ONNX로 export할 수 있습니다.

```bash
cd /content/microduck_rl

uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run_id>

uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
  --wandb-run-path <entity/project/run_id>
```

ONNX export는 observation normalizer를 그래프에 포함하는 공식 경로를 사용하는 것이
중요합니다.

## Upstream

- https://github.com/pollen-robotics/microduck
- https://github.com/pollen-robotics/microduck_rl
- https://github.com/mujocolab/mjlab
