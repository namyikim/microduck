# JumpSpin Hugging Face release

This release flow is intentionally split into **prepare/review** and **upload**.

The Hub upload must happen only after the completed policy has been visually checked.

## 1. Finish training

JumpSpin is configured for 10,000 PPO iterations. The final 360-degree target is already active
from iteration 4,000, and the reset curriculum becomes increasingly standing-start heavy through
the later stages. Do not publish an intermediate checkpoint just because it can rotate.

Keep the final checkpoint in Google Drive under:

```text
MyDrive/microduck-training/logs/rsl_rl/jump_spin_launch_v2/
```

## 2. Render evaluation rollouts

The JumpSpin Colab notebook automatically renders rollout videos after training reaches the target.
The videos and `evaluation_summary.json` are stored under the Drive training root.

Review at least these points manually:

- starts from a normal two-foot standing pose,
- leaves the ground rather than pivoting on one foot,
- rotation is predominantly about world Z,
- reaches approximately one full turn,
- lands on both feet,
- remains upright and recovers instead of merely touching down,
- no obviously violent joint-limit or impact behavior.

The numeric spin-progress check is only a screening aid. Visual approval is mandatory.

## 3. Build a release candidate — no upload

Example from Colab:

```bash
cd /content/microduck

uv run python scripts/prepare_jump_spin_release.py \
  --checkpoint /content/drive/MyDrive/microduck-training/logs/rsl_rl/jump_spin_launch_v2/manual_recovery/model_10000.pt \
  --evaluation-summary /content/drive/MyDrive/microduck-training/videos/evaluation_summary.json \
  --video /content/drive/MyDrive/microduck-training/videos/jump_spin_trial_01.mp4 \
  --repo YOUR_HF_USERNAME/microduck-jump-spin \
  --output-dir /content/drive/MyDrive/microduck-training/hf_release_candidate
```

This step does **not** upload anything. It creates:

```text
hf_release_candidate/
├── policy.onnx
├── manifest.json
├── README.md
├── release_report.json
├── evaluation_summary.json     # when supplied
└── replay.mp4                  # when supplied
```

Checks performed:

- checkpoint can be loaded and iteration metadata is coherent,
- ONNX is exported through the project exporter with the observation normalizer baked in,
- model graph is 61 observations -> 14 actions,
- ONNX CPU smoke-run is finite and non-constant,
- model manifest validates,
- rollout spin-progress statistics are summarized when evaluation data exists.

The status can become `READY_FOR_MANUAL_REVIEW`, but that is **not** automatic approval.

## 4. Upload only after review

Authentication follows the normal `huggingface_hub` login/token mechanism. Do not put a token
in the repository or notebook source.

The uploader refuses to run without the explicit review flag:

```bash
uv run python scripts/publish_jump_spin_hf.py \
  --package-dir /content/drive/MyDrive/microduck-training/hf_release_candidate \
  --repo YOUR_HF_USERNAME/microduck-jump-spin \
  --confirm-reviewed \
  --tag v1
```

New repositories are **private by default**. Add `--public` only when the release is ready to be
publicly visible.

For an existing repository, the script refuses to replace `policy.onnx` or `manifest.json`
unless `--force` is explicitly provided.

## 5. Expected Hugging Face model contents

The policy repository is compatible with the existing MicroDuck policy publishing contract:

- `policy.onnx` — deployable policy with normalization baked in,
- `manifest.json` — MicroDuck schema-2 policy manifest,
- `README.md` — Hugging Face model card,
- `replay.mp4` — representative rollout when included,
- evaluation/release JSON files — traceability for the release candidate.

The current Hugging Face Hub Python API supports model repository creation and folder upload via
`HfApi.create_repo()` / `HfApi.upload_folder()`; interrupted folder uploads can be rerun and
already committed data is skipped/deduplicated.
