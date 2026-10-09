# 학습과 재개

[README로 돌아가기](../README.md) · [평가와 진단](evaluation.md)

## Colab 노트북으로 실행

- [JumpSpin 360 노트북](https://colab.research.google.com/github/namyikim/microduck/blob/main/MicroDuck_JumpSpin_A100.ipynb)
- [일반 보행 노트북](https://colab.research.google.com/github/namyikim/microduck/blob/main/MicroDuck_Colab_A100.ipynb)

A100 GPU 런타임을 선택하고, 설정·Drive 연결·저장소 설치 셀을 순서대로 실행한 뒤 학습 셀을 실행합니다. 설치는 이 저장소를 `/content/microduck`에 받아 Python 3.12와 `uv.lock`의 의존성을 사용합니다. 기본 W&B 모드는 `offline`입니다.

| 설정 | JumpSpin | 일반 보행 |
|---|---|---|
| `TASK_ID` | `Mjlab-JumpSpin-Flat-MicroDuck` | `Mjlab-Velocity-Flat-MicroDuck` |
| `EXPERIMENT_NAME` | `jump_spin_launch_v2` | 선택한 task 설정에서 결정 |
| `NUM_ENVS` | 4096 | 4096 |
| `TARGET_ITERS` | 10000 | 6000 |
| `SYNC_INTERVAL` | 10초 | 60초 |
| 평가 영상 | 학습 종료 후 5개 자동 생성 | JumpSpin 자동 평가는 제공하지 않음 |

`TARGET_ITERS`는 추가 실행 횟수가 아니라 도달할 총 학습 횟수입니다. 새 학습은 `SMOKE_TEST = True`로 64개 환경·5회 검증을 먼저 수행합니다. JumpSpin의 짧은 진단만 필요하면 `TARGET_ITERS = 500`으로 설정할 수 있지만, 이 시점은 최종 360° 단계가 아닙니다.

## Colab CLI 사용 규칙

CLI 런타임은 **반드시 `colab run --keep`으로 시작**합니다. 규칙 원문은 [agent.md](../agent.md)에 있습니다.

아래는 별도로 준비한 Python 실행 스크립트에 적용하는 명령 형식입니다. `YOUR_RUNNER.py`는 저장소에 포함된 파일명이 아니며, 설치·Drive 연결·학습·평가 절차를 구성한 실제 로컬 파일로 바꿔야 합니다.

```bash
colab run --keep --gpu A100 --session microduck-training \
  --timeout 43200 YOUR_RUNNER.py
```

`--keep`은 스크립트 종료 후 CLI가 런타임을 자동 삭제하지 않게 합니다. Colab의 세션 제한·사용량 제한을 해제하지 않으며, 로컬 연결이 끊겨도 무제한 실행을 보장하지 않습니다. `--timeout`은 CLI의 대기 시간 설정입니다.

이미 `--keep`으로 만든 세션에는 다음 명령을 사용할 수 있습니다.

```bash
colab sessions
colab usage
colab drivemount --session microduck-training
colab exec --session microduck-training --timeout 43200 -f YOUR_NEXT_SCRIPT.py
```

`YOUR_NEXT_SCRIPT.py`도 실제 준비한 파일로 바꿉니다. `exec`에는 `--keep` 옵션이 없습니다. 새 런타임에서는 Google Drive 승인이 다시 필요할 수 있습니다. 런타임 종료는 사용자가 요청했을 때만 수행합니다.

## 로그와 백업

Drive 기본 루트는 `MyDrive/microduck-training/`입니다.

```text
microduck-training/
├── training.log
├── logs/rsl_rl/jump_spin_launch_v2/<run>/model_*.pt
└── videos/
```

학습 셀의 `Learning iteration X/Y`, `[MicroDuck progress]`, `ETA`에서 진행 상황을 확인합니다. 같은 학습 로그가 Drive의 `training.log`에 기록됩니다. Drive 미리보기는 새로고침이 필요할 수 있습니다. 로컬 CLI 출력을 파일로 저장했다면 `tail -f <로그 파일>`로 볼 수 있습니다.

JumpSpin 모델은 100회마다 저장하고, 기본 10초 주기로 Drive에 복사합니다. 동기화 주기와 모델 생성 주기는 다릅니다. 저장·복사가 끝나기 전에 런타임이 종료되면 마지막 백업 이후의 진행은 잃을 수 있습니다.

## 세션 종료 후 복원

노트북에서 Drive 연결과 설치를 다시 완료한 뒤 학습 셀을 실행합니다. 스크립트는 Drive 로그를 복원하고 **해당 실험 폴더 안에서** `model_숫자.pt`의 숫자가 가장 큰 체크포인트를 고릅니다. 내부 `iter` 값도 읽어 남은 횟수를 계산합니다.

수동으로 복구할 파일은 다음 경로에 둘 수 있습니다.

```text
MyDrive/microduck-training/logs/rsl_rl/jump_spin_launch_v2/manual_recovery/model_XXXX.pt
```

로그의 `Loading model checkpoint from:`에서 실제 선택된 파일을 확인하세요. `latest_checkpoint_*.txt`는 참고용이며 더 높은 번호의 파일보다 우선하지 않습니다. 기존 `jump_spin` 모델을 `jump_spin_launch_v2` 폴더에 복사하지 마세요. 서로 다른 보상 설정의 실험입니다.

## 로컬 GPU 환경에서 실행

아래 명령은 호환되는 Linux/CUDA 환경과 `uv`가 준비된 경우의 예시입니다. GPU가 없는 Mac에서 실행하는 학습 방법은 아닙니다.

```bash
git clone https://github.com/namyikim/microduck.git
cd microduck
uv sync --python 3.12 --frozen
uv run list-envs
uv run train Mjlab-JumpSpin-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
uv run train Mjlab-JumpSpin-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 10000
```

로컬 `train` 명령은 노트북의 Drive 백업·자동 영상 생성 절차를 포함하지 않습니다. 영상 생성은 [평가 가이드](evaluation.md)를 따릅니다.
