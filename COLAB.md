# Google Colab A100 학습

이 저장소는 `pollen-robotics/microduck_rl`의 소스 snapshot과
Colab A100 학습/재시작 도구를 함께 관리합니다.

## 바로 실행

[Open in Google Colab](https://colab.research.google.com/github/namyikim/microduck/blob/main/MicroDuck_Colab_A100.ipynb)

Colab에서 **Runtime > Change runtime type > A100 GPU**를 선택한 뒤 위에서부터 실행하세요.

기본값:

- Task: `Mjlab-Velocity-Flat-MicroDuck`
- Parallel environments: `4096`
- Target PPO iterations: `6000`
- Checkpoint: upstream runner 기본값 `250` iterations
- Drive sync: `60` seconds
- Backup: `MyDrive/microduck-training/logs/`
- W&B: offline 기본값

## 세션이 끊겨도 이어서 학습

학습 로그와 `model_*.pt`는 Google Drive에 주기적으로 복사됩니다.
다음 Colab 세션에서 노트북을 다시 실행하면 최신 checkpoint를 찾아
목표 iteration까지 이어서 학습합니다.

## 현재 저장소 구조

이 저장소 자체가 학습 소스입니다. Colab에서 별도로
`pollen-robotics/microduck_rl`를 clone하지 않습니다.

```text
/content/microduck/
├── src/mjlab_microduck/
├── scripts/
├── tests/
├── pyproject.toml
├── uv.lock
├── MicroDuck_Colab_A100.ipynb
└── COLAB.md
```

## Upstream 동기화

`.github/workflows/import-upstream.yml`은 공식
`pollen-robotics/microduck_rl:develop`을 가져와 현재 저장소에 반영합니다.

동기화 시 사용자 전용 파일은 보존합니다.

- `COLAB.md`
- `MicroDuck_Colab_A100.ipynb`
- `scripts/setup_colab.sh`
- `scripts/train_colab.sh`
- `.github/workflows/import-upstream.yml`

가져온 upstream commit은 `UPSTREAM.md`에 기록합니다.


## 학습 진행률 표시

학습 로그의 `Learning iteration X/Y`를 읽어 다음과 같은 퍼센트 진행바를 함께 표시합니다.

```text
[MicroDuck progress] [██████████░░░░░░░░░░░░░░░░░░░░]  33.3%  (2000/6000)
```

RSL-RL 자체 로그의 `ETA`도 계속 표시되므로 퍼센트와 예상 남은 시간을 함께 확인할 수 있습니다.

## 실시간 학습 로그

학습 셀은 line-by-line streaming으로 실행되고, Python/RSL-RL 출력도
unbuffered 모드로 강제합니다. 따라서 다음과 같은 로그가 셀 아래에 즉시 나타납니다.

```text
Learning iteration 1234/6000
[MicroDuck progress] [██████░░░░░░░░░░░░░░░░░░░░░░]  20.6%  (1235/6000)
ETA: ...
```

동시에 동일한 로그를 Google Drive에도 저장합니다.

```text
MyDrive/microduck-training/training.log
```

기존 버전의 노트북을 이미 실행 중이라 출력이 비어 있다면, 현재 셀을 중지한 뒤
저장소를 최신으로 pull하고 다시 실행하면 최신 checkpoint에서 자동 resume됩니다.
