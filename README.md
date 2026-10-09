# MicroDuck RL · JumpSpin 360

[한국어](#한국어) · [English](#english)

## 한국어

### 목적과 배경

이 프로젝트는 소형 이족보행 로봇 **MicroDuck이 제자리에서 점프하고, 공중에서 수직축을 중심으로 360° 회전한 뒤 양발로 착지해 안정적으로 서는 동작**을 강화학습으로 배우도록 하는 프로젝트입니다. 시뮬레이션에서 동작을 검증하고, 최종적으로 실제 로봇에서 사용할 정책을 만드는 것이 목표입니다.

공식 [MicroDuck RL](https://github.com/pollen-robotics/microduck_rl)을 기반으로 Google Colab A100 학습, Google Drive 백업·재개, JumpSpin 학습 환경, 자동 평가 영상과 ONNX 내보내기·배포 준비 도구를 추가했습니다. Colab 세션이 끊겨도 저장된 체크포인트에서 학습을 이어갈 수 있도록 구성했습니다.

**현재 JumpSpin은 실험 단계입니다.** 학습 완료나 `model_9999.pt` 생성만으로 목표 동작의 성공을 의미하지 않습니다. 최종 영상과 평가 결과로 실제 점프·회전·착지를 확인해야 합니다.

### 실행 방법

1. **[JumpSpin 학습 노트북 열기](https://colab.research.google.com/github/namyikim/microduck/blob/main/MicroDuck_JumpSpin_A100.ipynb)**
2. Colab의 **런타임 → 런타임 유형 변경 → A100 GPU**를 선택합니다. 설정·Drive 연결·설치 셀을 위에서부터 실행하고 Drive 접근을 승인합니다.
3. **학습 시작 / 자동 resume** 셀을 실행합니다. 처음에는 짧은 검증 학습을 하고, 이후 본학습과 마지막 평가 영상 생성을 자동으로 진행합니다. 회전 목표는 180° → 270° → 360°로 자동 전환됩니다.

기본 설정은 **4,096개 병렬 환경 · 총 10,000회 학습 · 평가 영상 5개**입니다. 같은 노트북을 다시 실행하면 `jump_spin_launch_v2` 실험의 최신 체크포인트에서 이어갑니다. 배포용 패키지 생성은 학습 후 별도 단계입니다.

일반 보행 학습은 [기본 Colab 노트북](https://colab.research.google.com/github/namyikim/microduck/blob/main/MicroDuck_Colab_A100.ipynb)을 사용하세요. GPU 설정, 로컬 실행, CLI 사용법은 [학습 가이드](doc/colab.md)에 있습니다. **Colab CLI로 런타임을 시작할 때는 `colab run --keep`을 반드시 사용합니다.**

### 진행 상황과 결과 확인

Google Drive의 **내 드라이브 → `microduck-training`**에서 확인합니다.

| 항목 | 저장 위치 |
|---|---|
| 진행 횟수·예상 남은 시간 | `training.log`의 `Learning iteration`, `ETA` |
| 학습 모델 | `logs/rsl_rl/jump_spin_launch_v2/<실행 폴더>/model_*.pt` |
| 평가 영상·요약 | `videos/`의 MP4와 `evaluation_summary.json` |

로그 미리보기는 새로고침이 필요할 수 있습니다. 별도 CLI 실행 스크립트에서는 결과를 `videos/jump_spin_launch_v2/`처럼 하위 폴더에 저장할 수 있으므로 실행 로그의 출력 경로를 확인하세요.

### 상세 문서

| 문서 | 내용 |
|---|---|
| [학습과 재개](doc/colab.md) | 설정, CLI의 `--keep`, Drive 백업, 체크포인트 복원, 로컬 실행 |
| [JumpSpin 설계](doc/jump-spin.md) | 목표 동작, 커리큘럼, 보상, 기존 실패 분석과 v2 변경 |
| [평가와 진단](doc/evaluation.md) | 영상 확인, 기존 모델 재평가, 진단 JSON 해석 |
| [ONNX와 Hugging Face 배포](doc/release.md) | 모델 내보내기, 검토용 패키지, 업로드 |
| [파일 정리와 유지보수](doc/maintenance.md) | 이전 체크포인트 정리, 코드 위치, CI, upstream 관리 |

기반 프로젝트: [MicroDuck](https://github.com/pollen-robotics/microduck) · [mjlab](https://github.com/mujocolab/mjlab) · [BAM](https://github.com/Rhoban/bam). 가져온 소스 버전은 [UPSTREAM.md](UPSTREAM.md), 코드 라이선스는 [Apache 2.0](LICENSE)을 참고하세요. 로봇 모델 등 자산은 원본의 별도 라이선스를 확인하세요.

---

## English

### Purpose and background

This project trains the small bipedal robot **MicroDuck to jump in place, rotate 360° about the vertical axis while airborne, land on both feet, and return to a stable standing pose** using reinforcement learning. The goal is to validate the maneuver in simulation and ultimately produce a policy for the physical robot.

Built on the official [MicroDuck RL](https://github.com/pollen-robotics/microduck_rl) project, this repository adds Google Colab A100 training, Google Drive backup and resume, a JumpSpin environment, automatic evaluation videos, and tools for ONNX export and release preparation. Saved checkpoints allow training to resume after a Colab session ends.

**JumpSpin is currently experimental.** Completing training or producing `model_9999.pt` does not establish success. Inspect the final videos and evaluation results to verify takeoff, rotation, and landing.

### How to run

1. **[Open the JumpSpin training notebook](https://colab.research.google.com/github/namyikim/microduck/blob/main/MicroDuck_JumpSpin_A100.ipynb).**
2. In Colab, select **Runtime → Change runtime type → A100 GPU**. Run the configuration, Drive connection, and installation cells in order, granting Drive access when prompted.
3. Run the **training / automatic resume** cell. It performs a short validation run on the first start, then continues through training and final video evaluation. The rotation target advances automatically from 180° to 270° to 360°.

Defaults are **4,096 parallel environments · 10,000 total training iterations · five evaluation videos**. Rerunning the notebook resumes the latest checkpoint in the `jump_spin_launch_v2` experiment. Release package preparation is a separate step after training.

For walking training, use the [general Colab notebook](https://colab.research.google.com/github/namyikim/microduck/blob/main/MicroDuck_Colab_A100.ipynb). GPU setup, local commands, and CLI usage are covered in the [training guide](doc/colab.md). **Always start Colab CLI runtimes with `colab run --keep`.**

### Progress and results

Look under **My Drive → `microduck-training`** in Google Drive.

| Item | Location |
|---|---|
| Iteration count and estimated time remaining | `Learning iteration` and `ETA` in `training.log` |
| Checkpoints | `logs/rsl_rl/jump_spin_launch_v2/<run>/model_*.pt` |
| Evaluation videos and summary | MP4 files and `evaluation_summary.json` under `videos/` |

The Drive log preview may need refreshing. Custom CLI runners may save results in a subfolder such as `videos/jump_spin_launch_v2/`; check the output path printed by the runner.

### Detailed documentation

The guides below are maintained in Korean, with command examples and original technical identifiers.

| Guide | Contents |
|---|---|
| [Training and resume](doc/colab.md) | Configuration, CLI `--keep`, Drive backups, recovery, local commands |
| [JumpSpin design](doc/jump-spin.md) | Target maneuver, curriculum, rewards, prior failure analysis, v2 changes |
| [Evaluation and diagnostics](doc/evaluation.md) | Video review, checkpoint evaluation, diagnostic JSON |
| [ONNX and Hugging Face release](doc/release.md) | Export, review package, upload |
| [Cleanup and maintenance](doc/maintenance.md) | Old checkpoints, code map, CI, upstream management |

Related projects: [MicroDuck](https://github.com/pollen-robotics/microduck) · [mjlab](https://github.com/mujocolab/mjlab) · [BAM](https://github.com/Rhoban/bam). See [UPSTREAM.md](UPSTREAM.md) for source provenance and [Apache 2.0](LICENSE) for the code license. Check the original licenses for robot models and other assets.
