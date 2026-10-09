# ONNX 내보내기와 Hugging Face 배포

[README로 돌아가기](../README.md) · [평가와 진단](evaluation.md)

## 배포 순서

학습 → 영상 평가 → 공식 ONNX 내보내기 → 검토용 패키지 생성 → 수동 검토 → Hugging Face 업로드 순서입니다. 학습 완료가 자동 업로드나 실제 로봇 동작 성공을 뜻하지 않습니다.

정책은 정규화를 포함한 61차원 관측 입력과 14차원 관절 출력을 유지해야 합니다. 공식 exporter를 사용해야 관측 정규화가 ONNX에 포함됩니다. 체크포인트를 임의의 변환 코드로 내보내지 않습니다.

## 1. 최종 모델과 영상 선택

정상 서기에서 출발해 실제 이륙·수직축 회전·양발 착지·자세 회복이 이루어지는지 [평가 가이드](evaluation.md)에 따라 확인합니다. 회전 누적값만으로 승인하지 않습니다.

모델은 `MyDrive/microduck-training/logs/rsl_rl/jump_spin_launch_v2/` 아래에 있습니다. 평가에 사용한 모델과 배포할 모델이 같은지 확인하세요.

## 2. 검토용 패키지 생성

아래 `CHECKPOINT`는 실제 평가한 파일 경로로 바꿉니다. 별도 영상 하위 폴더를 사용했다면 `VIDEOS`도 맞춰야 합니다.

```bash
cd /content/microduck
CHECKPOINT="/content/drive/MyDrive/microduck-training/logs/rsl_rl/jump_spin_launch_v2/YOUR_RUN/model_9999.pt"
VIDEOS="/content/drive/MyDrive/microduck-training/videos"

uv run python scripts/prepare_jump_spin_release.py \
  --checkpoint "$CHECKPOINT" \
  --evaluation-summary "$VIDEOS/evaluation_summary.json" \
  --video "$VIDEOS/jump_spin_trial_01.mp4" \
  --repo YOUR_HF_USERNAME/microduck-jump-spin \
  --output-dir /content/drive/MyDrive/microduck-training/hf_release_candidate
```

이 명령은 업로드하지 않습니다. 체크포인트 iteration, 공식 exporter의 정규화 포함, ONNX 61→14 형태, CPU 추론의 유한값·출력 변화, manifest를 확인하고 평가 통계를 기록합니다.

```text
hf_release_candidate/
├── policy.onnx
├── manifest.json
├── README.md
├── release_report.json
├── evaluation_summary.json  # 제공한 경우
└── replay.mp4              # 제공한 경우
```

`READY_FOR_MANUAL_REVIEW`는 수동 검토를 기다리는 상태이며 자동 승인이 아닙니다.

## 3. 검토 후 업로드

Hugging Face 인증은 `huggingface_hub`의 로그인·토큰 방식을 사용합니다. 토큰을 저장소나 노트북 소스에 넣지 않습니다.

```bash
uv run python scripts/publish_jump_spin_hf.py \
  --package-dir /content/drive/MyDrive/microduck-training/hf_release_candidate \
  --repo YOUR_HF_USERNAME/microduck-jump-spin \
  --confirm-reviewed \
  --tag v1
```

업로더는 `--confirm-reviewed` 없이는 실행되지 않습니다. 새 저장소는 기본 private이며, 공개하려면 `--public`을 명시합니다. 기존 저장소의 `policy.onnx` 또는 `manifest.json` 교체에는 `--force`가 필요합니다.

최종 패키지는 정규화를 포함한 정책, MicroDuck schema-2 manifest, 모델 카드와 평가 자료를 담습니다. 실제 로봇 연결·실행은 [MicroDuck 런타임 프로젝트](https://github.com/pollen-robotics/microduck)의 절차를 따릅니다.
