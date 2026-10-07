#!/usr/bin/env bash
set -euo pipefail

TASK_ID="${TASK_ID:-Mjlab-Velocity-Flat-MicroDuck}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-velocity}"
NUM_ENVS="${NUM_ENVS:-4096}"
TARGET_ITERS="${TARGET_ITERS:-6000}"
SYNC_INTERVAL="${SYNC_INTERVAL:-60}"
RUN_NAME="${RUN_NAME:-colab-a100}"
UPSTREAM_DIR="${UPSTREAM_DIR:-/content/microduck_rl}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/microduck-training}"
SMOKE_TEST="${SMOKE_TEST:-1}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
  echo "ERROR: $UPSTREAM_DIR 가 없습니다. setup_colab.sh를 먼저 실행하세요." >&2
  exit 1
fi

if [[ ! -d /content/drive/MyDrive ]]; then
  echo "ERROR: Google Drive가 /content/drive/MyDrive 에 mount되지 않았습니다." >&2
  exit 1
fi

cd "$UPSTREAM_DIR"

LOCAL_LOGS="$UPSTREAM_DIR/logs"
EXPERIMENT_DIR="$LOCAL_LOGS/rsl_rl/$EXPERIMENT_NAME"
BACKUP_LOGS="$DRIVE_ROOT/logs"

mkdir -p "$LOCAL_LOGS" "$EXPERIMENT_DIR" "$BACKUP_LOGS"

echo "== Restore previous logs from Drive =="
if [[ -d "$BACKUP_LOGS/rsl_rl" ]]; then
  rsync -a "$BACKUP_LOGS/" "$LOCAL_LOGS/"
fi

find_latest_checkpoint() {
  python3 - "$EXPERIMENT_DIR" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
best = None

if root.exists():
    for p in root.rglob("model_*.pt"):
        m = re.fullmatch(r"model_(\d+)\.pt", p.name)
        if not m:
            continue
        iteration = int(m.group(1))
        item = (iteration, p.stat().st_mtime, p)
        if best is None or item[:2] > best[:2]:
            best = item

if best is not None:
    print(best[2])
PY
}

sync_once() {
  mkdir -p "$BACKUP_LOGS"

  # Log/config files can be copied normally.
  rsync -a --exclude='model_*.pt' "$LOCAL_LOGS/" "$BACKUP_LOGS/" || true

  # Checkpoints are copied only after they have been stable for a short time.
  # A .partial file is atomically renamed after the copy completes.
  while IFS= read -r -d '' src; do
    rel="${src#"$LOCAL_LOGS/"}"
    dst="$BACKUP_LOGS/$rel"
    mkdir -p "$(dirname "$dst")"
    tmp="$dst.partial"
    cp -f "$src" "$tmp" && mv -f "$tmp" "$dst"
  done < <(find "$LOCAL_LOGS" -type f -name 'model_*.pt' -mmin +0.25 -print0 2>/dev/null)
}

sync_loop() {
  while true; do
    sync_once
    sleep "$SYNC_INTERVAL"
  done
}

LATEST="$(find_latest_checkpoint || true)"

if [[ -z "$LATEST" && "$SMOKE_TEST" == "1" ]]; then
  echo
  echo "== Smoke test: 64 envs / 5 iterations =="
  uv run train "$TASK_ID" \
    --env.scene.num-envs 64 \
    --agent.run-name smoke-a100 \
    --agent.max_iterations 5

  # Smoke-test artifacts must never be selected as the main resume checkpoint.
  find "$EXPERIMENT_DIR" -mindepth 1 -maxdepth 1 -type d -name '*_smoke-a100' \
    -exec rm -rf {} + 2>/dev/null || true

  echo "Smoke test passed."
fi

LATEST="$(find_latest_checkpoint || true)"
DONE=0
LOAD_RUN=""
LOAD_CHECKPOINT=""

if [[ -n "$LATEST" ]]; then
  FILE="$(basename "$LATEST")"
  RUN_DIR="$(basename "$(dirname "$LATEST")")"
  N="${FILE#model_}"
  N="${N%.pt}"

  # rsl_rl checkpoint naming is iteration-based; model_3249.pt corresponds
  # to approximately 3250 completed updates.
  DONE=$((10#$N + 1))
  LOAD_RUN="$RUN_DIR"
  LOAD_CHECKPOINT="$FILE"

  echo
  echo "Latest checkpoint : $LATEST"
  echo "Completed approx. : $DONE iterations"
fi

if (( DONE >= TARGET_ITERS )); then
  echo
  echo "Target already reached: $DONE >= $TARGET_ITERS"
  sync_once
  exit 0
fi

REMAINING=$((TARGET_ITERS - DONE))

echo
echo "== Training configuration =="
echo "Task              : $TASK_ID"
echo "Experiment        : $EXPERIMENT_NAME"
echo "Environments      : $NUM_ENVS"
echo "Target iterations : $TARGET_ITERS"
echo "Already done      : $DONE"
echo "Run now           : $REMAINING"
echo "Drive backup      : $BACKUP_LOGS"
echo "Sync interval     : $SYNC_INTERVAL sec"
echo "WANDB_MODE        : $WANDB_MODE"

sync_loop &
SYNC_PID=$!

cleanup() {
  kill "$SYNC_PID" >/dev/null 2>&1 || true
  sync_once
}
trap cleanup EXIT INT TERM

CMD=(
  uv run train "$TASK_ID"
  --env.scene.num-envs "$NUM_ENVS"
  --agent.max_iterations "$REMAINING"
  --agent.run-name "$RUN_NAME"
)

if [[ -n "$LOAD_CHECKPOINT" ]]; then
  CMD+=(
    --agent.resume True
    --agent.load-run "$LOAD_RUN"
    --agent.load-checkpoint "$LOAD_CHECKPOINT"
  )
  echo
  echo "Resume from: $LOAD_RUN/$LOAD_CHECKPOINT"
else
  echo
  echo "Starting a fresh training run."
fi

echo
printf 'Command: '
printf '%q ' "${CMD[@]}"
echo
echo

"${CMD[@]}"

echo
echo "Training command finished."
