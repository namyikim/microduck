#!/usr/bin/env python3
"""Upload a reviewed JumpSpin release package to Hugging Face.

Safety gate: upload is refused unless --confirm-reviewed is provided.
The package should first be created by prepare_jump_spin_release.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

from mjlab_microduck.publish import manifest as manifest_lib


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package-dir", required=True, type=Path)
    p.add_argument("--repo", required=True, help="<user-or-org>/microduck-jump-spin")
    p.add_argument("--confirm-reviewed", action="store_true")
    p.add_argument("--public", action="store_true", help="Create a new repo as public; private is default.")
    p.add_argument("--force", action="store_true", help="Allow replacing files in an existing repo.")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_reviewed:
        raise SystemExit(
            "Refusing upload: visually review the completed JumpSpin rollouts first, "
            "then rerun with --confirm-reviewed."
        )

    package = args.package_dir.expanduser().resolve()
    required = ["policy.onnx", "manifest.json", "README.md", "release_report.json"]
    missing = [name for name in required if not (package / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Release package is incomplete; missing: {missing}")

    shape = manifest_lib.check_onnx(package / "policy.onnx")
    manifest_lib.smoke_run_onnx(package / "policy.onnx")

    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    manifest_lib.validate_manifest(manifest)
    if manifest.get("name") != "jump-spin":
        raise RuntimeError(f"Unexpected manifest policy name: {manifest.get('name')!r}")

    report = json.loads((package / "release_report.json").read_text(encoding="utf-8"))
    if report.get("upload_performed"):
        print("[JumpSpin HF] note: release_report says this package was uploaded before.")

    api = HfApi()
    exists = api.repo_exists(args.repo, repo_type="model")
    if exists and not args.force:
        existing = set(api.list_repo_files(args.repo, repo_type="model"))
        protected = {"policy.onnx", "manifest.json"}
        if existing & protected:
            raise SystemExit(
                f"{args.repo} already contains release files. "
                "Use a new repo/version or pass --force after review."
            )

    api.create_repo(
        repo_id=args.repo,
        repo_type="model",
        private=not args.public,
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=str(package),
        commit_message=(
            f"publish JumpSpin checkpoint {manifest.get('training', {}).get('checkpoint', 'unknown')}"
        ),
    )

    if args.tag:
        api.create_tag(
            repo_id=args.repo,
            repo_type="model",
            tag=args.tag,
            tag_message=f"JumpSpin {args.tag}",
        )

    print(f"[JumpSpin HF] ONNX shape : {shape.obs_len} -> {shape.action_len}")
    print(f"[JumpSpin HF] uploaded   : {getattr(commit, 'commit_url', args.repo)}")
    if not args.public:
        print("[JumpSpin HF] visibility : private")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
