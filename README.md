# MicroDuck RL · Greeting & SitStand

[한국어](#한국어) · [English](#english)

## 한국어

### 목적과 배경

이 프로젝트는 소형 이족보행 로봇 **MicroDuck이 앉은 자세를 유지하면서 좌우를 보고, 정면에서 두 번 고개를 끄덕이는 인사 동작**을 학습하는 프로젝트입니다. 실제 로봇의 충격 부담을 줄이기 위해 공중 회전에서 작은 머리 움직임으로 목표를 변경했습니다.

공식 [MicroDuck RL](https://github.com/pollen-robotics/microduck_rl)의 SitStand 환경을 바탕으로 Colab 학습, Drive 백업·재개, 자세 검증과 자동 평가를 구성했습니다. 앉은 인사는 3,000회 학습 후 시뮬레이션 평가 영상 5개와 CPU ONNX 시험 5개를 통과했습니다. 다음 목표는 **서기 → 천천히 앉기 → 유지 → 다시 서기**입니다. 실제 로봇 시험은 하지 않았습니다.

이전 JumpSpin 실험은 머리 접촉과 쓰러짐으로 실패했습니다. 기록과 코드는 보존하며 새 학습에는 해당 모델을 사용하지 않습니다.

### 실행 방법

**다음 동작:** [천천히 앉았다 일어서기 학습](doc/sitstand.md)은 기존 SitStand 환경에서 별도 정책을 학습합니다. 아래 노트북은 완료된 앉은 인사 실험을 재현할 때 사용합니다.

1. **[앉은 인사 학습 노트북 열기](https://colab.research.google.com/github/namyikim/microduck/blob/main/MicroDuck_SeatedGreeting_A100.ipynb)**
2. Colab에서 **A100 GPU**를 선택하고 Drive 연결·설치 셀을 실행합니다.
3. 학습할 준비가 되면 설정 셀의 **`START_TRAINING=True`**로 바꾸고 학습 셀을 실행합니다. 기본값은 `False`입니다.

앉은 자세와 머리 움직임의 물리 검증, 64환경·5회 검증 학습, 공식 ONNX 검증을 통과하면 **4,096환경·총 3,000회** 본학습과 평가 영상 5개 생성을 진행합니다. 같은 실험만 자동 재개하며, 횟수 도달이 동작 성공을 뜻하지는 않습니다. [학습 범위·검증·결과 경로](doc/seated-greeting.md)

일반 보행 학습은 [기본 Colab 노트북](https://colab.research.google.com/github/namyikim/microduck/blob/main/MicroDuck_Colab_A100.ipynb)을 사용하세요. GPU 설정, 로컬 실행, CLI 사용법은 [학습 가이드](doc/colab.md)에 있습니다. **Colab CLI로 런타임을 시작할 때는 `colab run --keep`을 반드시 사용합니다.**

### 진행 상황과 결과 확인

Google Drive의 **내 드라이브 → `microduck-training`**에서 확인합니다.

| 항목 | 저장 위치 |
|---|---|
| 진행 횟수·예상 남은 시간 | `seated_greeting_v1/training.log`의 `Learning iteration`, `ETA` |
| 학습 모델 | `seated_greeting_v1/logs/rsl_rl/seated_greeting_v1/<실행 폴더>/model_*.pt` |
| 평가 영상·요약 | `seated_greeting_v1/evaluations/<실행시각>/`의 MP4와 JSON |

로그 미리보기는 새로고침이 필요할 수 있습니다. 새 정책은 이미 앉은 상태에서 시작하며 앉기·일어서기·보행은 포함하지 않습니다.

### 상세 문서

| 문서 | 내용 |
|---|---|
| [학습과 재개](doc/colab.md) | 설정, CLI의 `--keep`, Drive 백업, 체크포인트 복원, 로컬 실행 |
| [앉은 인사](doc/seated-greeting.md) | 목표, 검증 결과, 실행 방법 |
| [앉았다 일어서기](doc/sitstand.md) | 다음 동작, 사전 검사, 실행 명령과 결과 경로 |
| [JumpSpin 설계](doc/jump-spin.md) | 목표 동작, 커리큘럼, 보상, 기존 실패 분석과 v2 변경 |
| [평가와 진단](doc/evaluation.md) | 영상 확인, 기존 모델 재평가, 진단 JSON 해석 |
| [ONNX와 Hugging Face 배포](doc/release.md) | 모델 내보내기, 검토용 패키지, 업로드 |
| [파일 정리와 유지보수](doc/maintenance.md) | 이전 체크포인트 정리, 코드 위치, CI, upstream 관리 |

기반 프로젝트: [MicroDuck](https://github.com/pollen-robotics/microduck) · [mjlab](https://github.com/mujocolab/mjlab) · [BAM](https://github.com/Rhoban/bam). 가져온 소스 버전은 [UPSTREAM.md](UPSTREAM.md), 코드 라이선스는 [Apache 2.0](LICENSE)을 참고하세요. 로봇 모델 등 자산은 원본의 별도 라이선스를 확인하세요.

---

## English

### Purpose and background

This project trains **MicroDuck to stay seated, look left and right, and nod twice**. The target has shifted from airborne spinning to small head movements to reduce impact exposure on the physical robot.

The new task builds on the official [MicroDuck RL](https://github.com/pollen-robotics/microduck_rl) SitStand environment, with Colab training, Drive backup/resume, pose preflight and automatic evaluation. The seated greeting completed 3,000 iterations and passed five simulation evaluations and five CPU ONNX trials. The next target is **stand → slowly sit → hold → stand**. No physical robot testing has been performed. The previous JumpSpin experiment failed through head contact and collapse; its model is not reused.

### How to run

**Next motion:** [Gentle sit/stand training](doc/sitstand.md) trains a separate policy using the existing SitStand task. The notebook below reproduces the completed seated greeting experiment.

1. **[Open the seated greeting notebook](https://colab.research.google.com/github/namyikim/microduck/blob/main/MicroDuck_SeatedGreeting_A100.ipynb).**
2. Select **A100 GPU**, mount Drive, and run installation.
3. When ready, set **`START_TRAINING=True`** and run the training cell. The default is `False`.

Seated-pose/head-motion preflight, a 64-environment / 5-iteration smoke run and official ONNX validation must pass before the **4,096-environment / 3,000-iteration** main run and five evaluation videos. Resume stays within the new experiment. The iteration target is not a success guarantee. [Details and results](doc/seated-greeting.md)

For walking training, use the [general Colab notebook](https://colab.research.google.com/github/namyikim/microduck/blob/main/MicroDuck_Colab_A100.ipynb). GPU setup, local commands, and CLI usage are covered in the [training guide](doc/colab.md). **Always start Colab CLI runtimes with `colab run --keep`.**

### Progress and results

Look under **My Drive → `microduck-training`** in Google Drive.

| Item | Location |
|---|---|
| Iteration count and estimated time remaining | `Learning iteration` and `ETA` in `seated_greeting_v1/training.log` |
| Checkpoints | `seated_greeting_v1/logs/rsl_rl/seated_greeting_v1/<run>/model_*.pt` |
| Evaluation videos and summary | MP4 files and JSON under `seated_greeting_v1/evaluations/<timestamp>/` |

The Drive preview may need refreshing. The new policy starts seated; sitting down, standing up and walking are outside its scope.

### Detailed documentation

The guides below are maintained in Korean, with command examples and original technical identifiers.

| Guide | Contents |
|---|---|
| [Training and resume](doc/colab.md) | Configuration, CLI `--keep`, Drive backups, recovery, local commands |
| [Seated greeting](doc/seated-greeting.md) | Target, checks, execution and results |
| [Gentle sit/stand](doc/sitstand.md) | Next motion, preflight, commands and output paths |
| [JumpSpin design](doc/jump-spin.md) | Target maneuver, curriculum, rewards, prior failure analysis, v2 changes |
| [Evaluation and diagnostics](doc/evaluation.md) | Video review, checkpoint evaluation, diagnostic JSON |
| [ONNX and Hugging Face release](doc/release.md) | Export, review package, upload |
| [Cleanup and maintenance](doc/maintenance.md) | Old checkpoints, code map, CI, upstream management |

Related projects: [MicroDuck](https://github.com/pollen-robotics/microduck) · [mjlab](https://github.com/mujocolab/mjlab) · [BAM](https://github.com/Rhoban/bam). See [UPSTREAM.md](UPSTREAM.md) for source provenance and [Apache 2.0](LICENSE) for the code license. Check the original licenses for robot models and other assets.
