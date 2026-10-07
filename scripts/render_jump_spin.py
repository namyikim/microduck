#!/usr/bin/env python3
"""Render fixed-length JumpSpin rollouts from a local RSL-RL checkpoint.

Designed for headless Google Colab:
- no interactive viewer/server
- one environment
- records complete play-mode episodes
- writes several MP4 trials plus evaluation_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Must be set before MuJoCo/mjlab creates a renderer.
os.environ.setdefault("MUJOCO_GL", "egl")

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

# Explicit import guarantees registration even outside the normal console scripts.
import mjlab_microduck.jump_spin_plugin  # noqa: F401


TASK_ID = "Mjlab-JumpSpin-Flat-MicroDuck"


def checkpoint_number(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render MicroDuck JumpSpin rollout videos from a checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove leftovers from an interrupted previous render. Stable trial files
    # are overwritten below only after fresh videos have completed.
    for stale in output_dir.glob("jump-spin-episode-*.mp4"):
        stale.unlink(missing_ok=True)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args.trials < 1:
        raise ValueError("--trials must be >= 1")

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[JumpSpin video] task       : {TASK_ID}")
    print(f"[JumpSpin video] checkpoint : {checkpoint}")
    print(f"[JumpSpin video] device     : {device}")
    print(f"[JumpSpin video] trials     : {args.trials}")
    print(f"[JumpSpin video] output     : {output_dir}")

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)

    # A video trial must always start from the real deployment condition.
    # play=True already sets the JumpSpin reset distribution to 100% standing.
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    recorder = VideoRecorder(
        raw_env,
        video_folder=output_dir,
        episode_trigger=lambda episode: episode < args.trials,
        video_length=None,
        name_prefix="jump-spin",
        disable_logger=False,
    )
    env = RslRlVecEnvWrapper(recorder, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    obs = env.get_observations()
    reward_sum = 0.0
    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    current_steps = 0
    max_progress_deg: list[float] = []
    current_max_progress = 0.0

    # Guard against a broken termination/timeout configuration.
    max_total_steps = int(args.trials * env.unwrapped.max_episode_length * 1.25) + 100
    total_steps = 0

    try:
        while recorder.video_count < args.trials and total_steps < max_total_steps:
            # Capture progress before stepping because the environment may auto-reset
            # immediately on timeout, which resets the task accumulator.
            progress = getattr(env.unwrapped, "_jump_spin_max", None)
            if progress is not None:
                current_max_progress = max(
                    current_max_progress,
                    float(progress[0].detach().cpu().item()),
                )

            with torch.no_grad():
                actions = policy(obs)
            obs, rewards, dones, _ = env.step(actions)

            reward_sum += float(rewards[0].detach().cpu().item())
            current_steps += 1
            total_steps += 1

            if bool(dones[0].detach().cpu().item()):
                episode_rewards.append(reward_sum)
                episode_steps.append(current_steps)
                max_progress_deg.append(current_max_progress * 180.0 / 3.141592653589793)
                idx = len(episode_rewards)
                print(
                    f"[JumpSpin video] trial {idx}/{args.trials}: "
                    f"reward={reward_sum:.3f}, "
                    f"observed_progress≈{max_progress_deg[-1]:.1f}°, "
                    f"steps={current_steps}"
                )
                reward_sum = 0.0
                current_steps = 0
                current_max_progress = 0.0

        if recorder.video_count < args.trials:
            raise RuntimeError(
                f"Only {recorder.video_count}/{args.trials} videos were completed "
                f"within {max_total_steps} steps."
            )
    finally:
        env.close()

    # Rename recorder filenames into stable trial names.
    generated = sorted(output_dir.glob("jump-spin-episode-*.mp4"))
    videos: list[str] = []
    for i, source in enumerate(generated[: args.trials], start=1):
        dest = output_dir / f"jump_spin_trial_{i:02d}.mp4"
        if dest.exists():
            dest.unlink()
        shutil.move(str(source), str(dest))
        videos.append(str(dest))

    summary = {
        "task": TASK_ID,
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": checkpoint_number(checkpoint),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "trials_requested": args.trials,
        "videos_created": len(videos),
        "episode_rewards": episode_rewards[: args.trials],
        "episode_steps": episode_steps[: args.trials],
        "observed_max_spin_progress_deg": max_progress_deg[: args.trials],
        "videos": videos,
        "note": (
            "observed_max_spin_progress_deg is a diagnostic read from the task "
            "accumulator and may miss the terminal frame because the vectorized "
            "environment auto-resets on timeout."
        ),
    }
    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[JumpSpin video] complete")
    print(f"[JumpSpin video] summary: {summary_path}")
    for path in videos:
        print(f"[JumpSpin video] video  : {path}")


if __name__ == "__main__":
    main()
