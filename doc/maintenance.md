# 파일 정리와 저장소 유지보수

[README로 돌아가기](../README.md) · [학습과 재개](colab.md)

## 이전 학습 파일 정리

[별도 정리 노트북 열기](https://colab.research.google.com/github/namyikim/microduck/blob/main/MicroDuck_Cleanup_Old_Checkpoints.ipynb)

GPU 없이 실행할 수 있습니다. 기존 `jump_spin` 실험에서 지정한 최종 체크포인트 하나를 남기고 다른 `model_*.pt` 파일만 정리합니다.

1. Drive를 연결하고 정리 대상 폴더와 보존할 체크포인트를 확인합니다.
2. 기본값인 `DELETE = False`로 삭제 대상과 예상 확보 용량을 미리 확인합니다.
3. 실제로 삭제하려면 `DELETE = True`로 바꾸고 설정 셀과 정리 셀을 다시 실행합니다.

기본 보존 파일은 `2026-10-08_12-32-56_colab-a100/model_9999.pt`입니다. 이 파일을 남기는 것은 이전 실험 보존을 위한 것이며, 성공한 동작이라는 의미는 아닙니다. 정리 노트북은 기본적으로 v2 실험이나 로그·영상을 삭제하지 않습니다. 진행 중인 학습의 복원 파일을 삭제 대상으로 바꾸지 마세요.

## 주요 파일

| 위치 | 역할 |
|---|---|
| [JumpSpin 노트북](../MicroDuck_JumpSpin_A100.ipynb) | 학습·백업·자동 평가 |
| [일반 Colab 노트북](../MicroDuck_Colab_A100.ipynb) | 기본 task 학습 |
| [설치 스크립트](../scripts/setup_colab.sh) | GPU 확인, Python·의존성 설치 |
| [학습 스크립트](../scripts/train_colab.sh) | Drive 복원·백업, 학습 재개 |
| [진행률 출력](../scripts/progress_filter.py) | PPO iteration 진행 표시 |
| [평가 렌더러](../scripts/render_jump_spin.py) | 영상과 진단 출력 |
| [진단 함수](../scripts/jump_spin_diagnostics.py) | rollout 진단값 계산 |
| [릴리스 준비](../scripts/prepare_jump_spin_release.py) | ONNX 검증·패키지 생성 |
| [Hugging Face 업로드](../scripts/publish_jump_spin_hf.py) | 검토한 패키지 업로드 |
| [AGENTS.md](../AGENTS.md) · [agent.md](../agent.md) | 개발 규칙과 Colab CLI `--keep` 필수 규칙 |

기반 프로젝트에서 가져온 정책별 기술 문서는 기존 [docs/](../docs/)에 있습니다. 이 저장소의 사용자 가이드는 `doc/`에서 관리합니다. 이전 `COLAB.md`, `JUMP_SPIN.md`, `JUMP_SPIN_HF_RELEASE.md`는 새 문서로 연결하는 안내 파일입니다.

## 검증과 CI

GPU 장기 학습 전에는 64개 환경·5회 검증을 먼저 수행합니다. 설정·MDP·진단 테스트는 다음 명령으로 실행합니다.

```bash
uv run --with pytest pytest -q \
  tests/test_jump_spin_cfg.py \
  tests/test_jump_spin_mdp.py \
  tests/test_jump_spin_diagnostics.py
```

[JumpSpin CI](../.github/workflows/jump-spin-tests.yml)는 관련 경로 변경 시 task 등록, 도구 Python 컴파일, 설정 테스트를 실행합니다. 실제 GPU 학습과 최종 동작 검증을 대신하지 않습니다.

## Upstream 소스 관리

공식 원본은 [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)의 `develop`이며, 가져온 commit과 시각은 [UPSTREAM.md](../UPSTREAM.md)에 기록합니다.

[Import upstream 워크플로](../.github/workflows/import-upstream.yml)는 소스 snapshot을 통째로 교체하고 명시된 확장 파일을 복원하는 구조입니다. **현재 보존 목록에는 새 `doc/` 문서와 모든 후속 수정이 포함되어 있지 않습니다.** 실행 전에 보존·복원 목록과 README 생성 절차를 검토해야 하며, 현재 README나 사용자 가이드가 자동 보존된다고 가정하면 안 됩니다. 이 문서 정리는 해당 워크플로의 실행 동작을 변경하지 않습니다.

## 기반 프로젝트와 라이선스

- [MicroDuck RL](https://github.com/pollen-robotics/microduck_rl): 학습 환경 기반
- [MicroDuck](https://github.com/pollen-robotics/microduck): 실제 로봇 런타임
- [mjlab](https://github.com/mujocolab/mjlab): MuJoCo Warp 기반 학습 프레임워크
- [BAM](https://github.com/Rhoban/bam): 구동기 모델

코드는 [Apache License 2.0](../LICENSE)을 따릅니다. 로봇 모델 등 개별 자산에는 원본의 별도 라이선스가 적용될 수 있으므로 자산 출처에서 확인합니다.
