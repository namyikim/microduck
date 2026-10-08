#!/usr/bin/env python3
"""Build a reviewable Hugging Face release candidate for MicroDuck JumpSpin.

This script NEVER uploads anything. It:
1. validates a local RSL-RL checkpoint,
2. exports it to ONNX through the project's safe exporter (normalizer baked in),
3. validates/smoke-runs the ONNX graph,
4. stages manifest/model-card metadata,
5. carries over rollout evidence when provided,
6. writes release_report.json with a manual-review gate.

Actual Hub upload is handled separately by publish_jump_spin_hf.py and requires
an explicit --confirm-reviewed flag.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

import mjlab_microduck.jump_spin_plugin  # noqa: F401
from mjlab_microduck.export import ExportConfig, run_export
from mjlab_microduck.publish import manifest as manifest_lib

TASK_ID = "Mjlab-JumpSpin-Flat-MicroDuck"
DEFAULT_DURATION_S = 4.5
DEFAULT_REPO = "YOUR_HF_USERNAME/microduck-jump-spin"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("jump_spin_hf_release"))
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--evaluation-summary", type=Path, default=None)
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--min-progress-deg", type=float, default=330.0)
    p.add_argument("--min-progress-pass-rate", type=float, default=0.8)
    return p.parse_args()


def checkpoint_iteration(checkpoint: Path) -> tuple[int, int | None]:
    try:
        filename_iter = int(checkpoint.stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Expected model_<N>.pt, got: {checkpoint.name}") from exc

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stored = payload.get("iter")
    stored_iter = int(stored) if stored is not None else None
    return filename_iter, stored_iter


def load_eval(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation summary not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    filename_iter, stored_iter = checkpoint_iteration(checkpoint)
    if stored_iter is not None and abs(stored_iter - filename_iter) > 1:
        raise RuntimeError(
            f"Checkpoint iteration mismatch: filename={filename_iter}, stored iter={stored_iter}"
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    onnx_path = output_dir / manifest_lib.POLICY_FILE
    result = run_export(
        TASK_ID,
        ExportConfig(
            onnx_file=str(onnx_path),
            checkpoint_file=str(checkpoint),
            num_envs=1,
        ),
    )

    shape = manifest_lib.check_onnx(onnx_path)
    manifest_lib.smoke_run_onnx(onnx_path)

    eval_data = load_eval(args.evaluation_summary)
    progress = [float(x) for x in eval_data.get("observed_max_spin_progress_deg", [])]
    pass_count = sum(x >= args.min_progress_deg for x in progress)
    pass_rate = (pass_count / len(progress)) if progress else None

    replay_path = None
    if args.video is not None:
        video = args.video.expanduser().resolve()
        if not video.is_file() or video.suffix.lower() != ".mp4":
            raise FileNotFoundError(f"Expected an MP4 video: {video}")
        replay_path = output_dir / manifest_lib.REPLAY_FILE
        shutil.copy2(video, replay_path)

    training = {
        "repo": "namyikim/microduck",
        **manifest_lib.git_provenance(),
        "task_id": TASK_ID,
        "checkpoint": result.checkpoint_iteration or filename_iter,
        "source_file": checkpoint.name,
    }

    eval_block = {
        "manual_video_review_required": True,
        "evaluation_summary_supplied": bool(eval_data),
        "trials": len(progress),
        "spin_progress_threshold_deg": args.min_progress_deg,
        "spin_progress_pass_count": pass_count if progress else None,
        "spin_progress_pass_rate": pass_rate,
        "observed_max_spin_progress_deg": progress or None,
    }

    policy_manifest = manifest_lib.build_manifest(
        name="jump-spin",
        kind="episodic",
        description=(
            "MicroDuck standing jump with an in-air yaw spin and two-foot recovery, "
            "trained with the JumpSpin curriculum."
        ),
        duration_s=DEFAULT_DURATION_S,
        chain=False,
        entry_pose="standing",
        training=training,
        eval=eval_block,
    )
    manifest_lib.validate_manifest(policy_manifest)

    (output_dir / "manifest.json").write_text(
        manifest_lib.dump_manifest(policy_manifest), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        manifest_lib.render_readme(policy_manifest, args.repo), encoding="utf-8"
    )

    auto_progress_ok = (
        pass_rate is not None
        and len(progress) >= 5
        and pass_rate >= args.min_progress_pass_rate
    )
    report = {
        "status": "READY_FOR_MANUAL_REVIEW" if auto_progress_ok else "NEEDS_REVIEW_OR_MORE_EVAL",
        "upload_performed": False,
        "manual_review_required": True,
        "task": TASK_ID,
        "checkpoint": str(checkpoint),
        "filename_iteration": filename_iter,
        "stored_iteration": stored_iter,
        "onnx": str(onnx_path),
        "onnx_shape": {
            "input_name": shape.input_name,
            "output_name": shape.output_name,
            "obs_len": shape.obs_len,
            "action_len": shape.action_len,
        },
        "evaluation": eval_block,
        "repo_id_for_model_card": args.repo,
        "replay_mp4": str(replay_path) if replay_path else None,
        "next_step": (
            "Visually inspect the rollout videos. Only after approval run "
            "scripts/publish_jump_spin_hf.py with --confirm-reviewed."
        ),
    }
    (output_dir / "release_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if args.evaluation_summary is not None:
        shutil.copy2(args.evaluation_summary, output_dir / "evaluation_summary.json")

    print(f"[JumpSpin release] checkpoint : {checkpoint.name}")
    print(f"[JumpSpin release] iteration  : {stored_iter if stored_iter is not None else filename_iter}")
    print(f"[JumpSpin release] ONNX       : {shape.obs_len} -> {shape.action_len}, smoke OK")
    if pass_rate is not None:
        print(
            f"[JumpSpin release] progress   : {pass_count}/{len(progress)} "
            f">= {args.min_progress_deg:.0f} deg ({pass_rate:.1%})"
        )
    print(f"[JumpSpin release] package    : {output_dir}")
    print("[JumpSpin release] upload     : NOT PERFORMED")
    print("[JumpSpin release] status     : manual video review required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
