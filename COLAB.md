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
