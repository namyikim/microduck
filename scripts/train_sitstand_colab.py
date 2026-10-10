#!/usr/bin/env python3
"""Opt-in SitStand Colab run: rest-pose preflight, smoke + export, training, five videos."""

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from evaluate_sitstand import smoke_evaluation_completed
from train_seated_greeting_colab import latest_checkpoint

TASK = "Mjlab-SitStand-Flat-MicroDuck"
EXPERIMENT = "microduck_sitstand"


def export_verified(checkpoint, output, min_iteration):
    import torch

    import mjlab_microduck.jump_spin_plugin  # noqa: F401 — register tasks
    from mjlab_microduck.export import ExportConfig, run_export
    from mjlab_microduck.publish.manifest import check_onnx, smoke_run_onnx

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload["iter"]) < min_iteration:
        raise RuntimeError("Checkpoint is earlier than target")
    if not all(
        bool(torch.isfinite(v).all())
        for v in payload["actor_state_dict"].values()
        if torch.is_tensor(v)
    ):
        raise RuntimeError("Non-finite actor checkpoint")
    run_export(
        TASK,
        ExportConfig(
            checkpoint_file=str(checkpoint),
            onnx_file=str(output),
            num_envs=1,
            device="cuda:0",
        ),
    )
    check_onnx(output)
    smoke_run_onnx(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-training", action="store_true")
    parser.add_argument("--target-iters", type=int, default=6000)
    parser.add_argument(
        "--validation-summary",
        type=Path,
        help="Wait for a preceding greeting ONNX validation before full training",
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive/microduck-training/sitstand_v1"),
    )
    args = parser.parse_args()
    if not args.start_training:
        print("준비 모드입니다. --start-training을 지정하면 학습합니다.")
        return
    if args.target_iters < 5 or args.num_envs < 1:
        parser.error("target-iters >= 5 and num-envs >= 1 required")
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Mount Google Drive first")
    repo = Path(__file__).resolve().parents[1]
    root = args.drive_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = root / "evaluations" / stamp
    out.mkdir(parents=True)
    env = dict(
        os.environ,
        WANDB_MODE="offline",
        WANDB_SILENT="true",
        MUJOCO_GL="egl",
        PYTHONUNBUFFERED="1",
    )
    env.pop("MICRODUCK_WARM_START", None)

    def status(state, **extra):
        (root / "status.json").write_text(
            json.dumps(
                dict(
                    state=state,
                    task=TASK,
                    target_iterations=args.target_iters,
                    output_dir=str(out),
                    updated_at_utc=datetime.now(UTC).isoformat(),
                    **extra,
                ),
                indent=2,
            )
        )

    def run(cmd):
        print("RUN:", " ".join(map(str, cmd)), flush=True)
        subprocess.run(list(map(str, cmd)), cwd=repo, env=env, check=True)

    try:
        status("preflight")
        run(
            [
                "uv",
                "run",
                "python",
                "scripts/evaluate_sitstand.py",
                "--preflight",
                "--output-dir",
                out,
            ]
        )
        status("smoke_test")
        name = "sitstand-smoke-" + stamp
        run(
            [
                "uv",
                "run",
                "train",
                TASK,
                "--env.scene.num-envs",
                "64",
                "--agent.max_iterations",
                "5",
                "--agent.experiment-name",
                "sitstand_smoke",
                "--agent.run-name",
                name,
            ]
        )
        candidates = list(
            (repo / "logs/rsl_rl/sitstand_smoke").glob("*" + name + "/model_4.pt")
        )
        if len(candidates) != 1:
            raise RuntimeError("Fresh smoke checkpoint missing")
        export_verified(candidates[0], out / "smoke_policy.onnx", 4)
        # Exercise two resets and the video path before committing GPU hours.
        run(
            [
                "uv",
                "run",
                "python",
                "scripts/evaluate_sitstand.py",
                "--checkpoint",
                candidates[0],
                "--output-dir",
                out / "smoke_evaluation",
                "--trials",
                "2",
            ]
        )
        smoke = json.loads(
            (out / "smoke_evaluation/evaluation_summary.json").read_text()
        )
        if not smoke_evaluation_completed(smoke):
            raise RuntimeError(
                "Smoke evaluation terminated or produced incomplete/non-finite evidence"
            )
        # The barely trained smoke policy is expected to fail the motion criteria.
        if args.validation_summary:
            status("waiting_for_onnx_validation")
            deadline = time.monotonic() + 7200
            while not args.validation_summary.is_file():
                if time.monotonic() > deadline:
                    raise RuntimeError("Timed out waiting for ONNX validation")
                time.sleep(15)
            validation = json.loads(args.validation_summary.read_text())
            if (
                not validation["numerical_parity_passed"]
                or not validation["all_cpu_trials_passed"]
            ):
                raise RuntimeError("Preceding greeting ONNX validation did not pass")
        status("training")
        env.update(
            TASK_ID=TASK,
            EXPERIMENT_NAME=EXPERIMENT,
            NUM_ENVS=str(args.num_envs),
            TARGET_ITERS=str(args.target_iters),
            DRIVE_ROOT=str(root),
            REPO_DIR=str(repo),
            SYNC_INTERVAL="10",
            SMOKE_TEST="0",
            RUN_NAME="gentle-sitstand",
        )
        run(["bash", "scripts/train_colab.sh"])
        cp = latest_checkpoint(repo / "logs/rsl_rl" / EXPERIMENT)
        if cp is None:
            raise RuntimeError("No final checkpoint")
        status("exporting", checkpoint=str(cp))
        export_verified(cp, out / "policy.onnx", args.target_iters - 1)
        status("evaluating", checkpoint=str(cp))
        run(
            [
                "uv",
                "run",
                "python",
                "scripts/evaluate_sitstand.py",
                "--checkpoint",
                cp,
                "--output-dir",
                out,
                "--trials",
                "5",
            ]
        )
        summary = json.loads((out / "evaluation_summary.json").read_text())
        if summary["videos_created"] != 5 or any(
            not Path(t["video"]).is_file() or Path(t["video"]).stat().st_size == 0
            for t in summary["trials"]
        ):
            raise RuntimeError("Five videos missing")
        status(
            "completed",
            checkpoint=str(cp),
            sim_checks_passed=summary["all_sim_checks_passed"],
        )
        print("RESULT", out, "SIM_CHECKS", summary["all_sim_checks_passed"], flush=True)
    except BaseException as exc:
        status("failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
