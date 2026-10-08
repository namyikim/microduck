#!/usr/bin/env bash
set -euo pipefail

TASK_ID="${TASK_ID:-Mjlab-Velocity-Flat-MicroDuck}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
NUM_ENVS="${NUM_ENVS:-4096}"
TARGET_ITERS="${TARGET_ITERS:-6000}"
SYNC_INTERVAL="${SYNC_INTERVAL:-10}"
DRIVE_VERIFY_INTERVAL="${DRIVE_VERIFY_INTERVAL:-10}"
RUN_NAME="${RUN_NAME:-colab-a100}"
REPO_DIR="${REPO_DIR:-/content/microduck}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/microduck-training}"
SMOKE_TEST="${SMOKE_TEST:-1}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED="1"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "ERROR: $REPO_DIR 가 없습니다. setup_colab.sh를 먼저 실행하세요." >&2
  exit 1
fi

if [[ ! -d /content/drive/MyDrive ]]; then
  echo "ERROR: Google Drive가 /content/drive/MyDrive 에 mount되지 않았습니다." >&2
  exit 1
fi

cd "$REPO_DIR"

# Prefer the explicit experiment name supplied by the Colab notebook.
# Importing task plugins can print warnings/patch banners to stdout; capturing that
# output in command substitution previously corrupted EXPERIMENT_NAME and caused
# checkpoint resume to search a bogus directory.
if [[ -n "$EXPERIMENT_NAME" ]]; then
  CONFIG_EXPERIMENT_NAME="$EXPERIMENT_NAME"
else
  CONFIG_EXPERIMENT_NAME="$(uv run python - "$TASK_ID" <<'PY'
import contextlib
import sys
from importlib.metadata import entry_points

task_id = sys.argv[1]
# Keep plugin banners/warnings out of stdout because stdout is captured by bash.
with contextlib.redirect_stdout(sys.stderr):
    for ep in entry_points(group="mjlab.tasks"):
        try:
            ep.load()
        except Exception:
            pass
    from mjlab.tasks.registry import load_rl_cfg
    cfg = load_rl_cfg(task_id)

print(cfg.experiment_name)
PY
)"
fi

if [[ -z "$CONFIG_EXPERIMENT_NAME" || "$CONFIG_EXPERIMENT_NAME" == *$'\n'* ]]; then
  echo "ERROR: Invalid experiment_name resolved for task: $TASK_ID" >&2
  printf 'Resolved value: %q\n' "$CONFIG_EXPERIMENT_NAME" >&2
  exit 1
fi

EXPERIMENT_NAME="$CONFIG_EXPERIMENT_NAME"

LOCAL_LOGS="$REPO_DIR/logs"
EXPERIMENT_DIR="$LOCAL_LOGS/rsl_rl/$EXPERIMENT_NAME"
BACKUP_LOGS="$DRIVE_ROOT/logs"
mkdir -p "$LOCAL_LOGS" "$EXPERIMENT_DIR" "$BACKUP_LOGS"

TASK_KEY="$(printf '%s' "$TASK_ID" | tr -cs 'A-Za-z0-9._-' '_')"
LATEST_MANIFEST="$DRIVE_ROOT/latest_checkpoint_${TASK_KEY}.txt"

echo "Task config experiment: $EXPERIMENT_NAME"
echo "Checkpoint search dir: $BACKUP_LOGS/rsl_rl/$EXPERIMENT_NAME"

echo "== Restore previous logs from Drive =="
if [[ -d "$BACKUP_LOGS/rsl_rl" ]]; then
  rsync -a --exclude='*.partial' "$BACKUP_LOGS/" "$LOCAL_LOGS/"
fi

echo "== Drive checkpoint candidates for $EXPERIMENT_NAME =="
find "$BACKUP_LOGS/rsl_rl/$EXPERIMENT_NAME" -type f -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr \
  | head -n 10 \
  | cut -d' ' -f2- || true

if [[ -f "$LATEST_MANIFEST" ]]; then
  echo "Task checkpoint manifest: $(cat "$LATEST_MANIFEST")"
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
        item = (int(m.group(1)), p.stat().st_mtime, p)
        if best is None or item[:2] > best[:2]:
            best = item
if best is not None:
    print(best[2])
PY
}

sync_checkpoint() {
  local src="$1"
  local rel dst tmp before_size before_mtime after_size after_mtime

  rel="${src#"$LOCAL_LOGS/"}"
  dst="$BACKUP_LOGS/$rel"
  tmp="$dst.partial"
  mkdir -p "$(dirname "$dst")"

  before_size="$(stat -c '%s' "$src" 2>/dev/null || echo -1)"
  before_mtime="$(stat -c '%Y' "$src" 2>/dev/null || echo -1)"

  if [[ -f "$dst" ]] &&
     [[ "$(stat -c '%s' "$dst" 2>/dev/null || echo -2)" == "$before_size" ]] &&
     [[ "$(stat -c '%Y' "$dst" 2>/dev/null || echo -2)" == "$before_mtime" ]]; then
    return 0
  fi

  rm -f "$tmp"
  cp -p "$src" "$tmp" || { rm -f "$tmp"; return 0; }

  after_size="$(stat -c '%s' "$src" 2>/dev/null || echo -3)"
  after_mtime="$(stat -c '%Y' "$src" 2>/dev/null || echo -3)"

  if [[ "$before_size" != "$after_size" || "$before_mtime" != "$after_mtime" ]] ||
     [[ "$(stat -c '%s' "$tmp" 2>/dev/null || echo -4)" != "$after_size" ]]; then
    rm -f "$tmp"
    return 0
  fi

  mv -f "$tmp" "$dst"
  if [[ -f "$dst" ]]; then
    echo "[Drive checkpoint] $(basename "$src") -> $dst"
  else
    echo "ERROR: Drive checkpoint copy failed: $dst" >&2
    return 1
  fi
}

sync_once() {
  mkdir -p "$BACKUP_LOGS"
  rsync -a --exclude='model_*.pt' --exclude='*.partial' "$LOCAL_LOGS/" "$BACKUP_LOGS/" || true
  while IFS= read -r -d '' src; do
    sync_checkpoint "$src"
  done < <(find "$LOCAL_LOGS" -type f -name 'model_*.pt' -print0 2>/dev/null)

  local latest rel
  latest="$(find_latest_checkpoint || true)"
  if [[ -n "$latest" && -f "$latest" ]]; then
    rel="${latest#"$LOCAL_LOGS/"}"
    printf '%s\n' "$rel" > "$LATEST_MANIFEST.tmp"
    mv -f "$LATEST_MANIFEST.tmp" "$LATEST_MANIFEST"
  fi
}

sync_loop() {
  while true; do
    sync_once
    sleep "$SYNC_INTERVAL"
  done
}

# Always choose the checkpoint with the highest model_<iteration>.pt number.
# The manifest is informational only; it must never override a newer manually
# restored/uploaded checkpoint.
LATEST="$(find_latest_checkpoint || true)"
if [[ -f "$LATEST_MANIFEST" ]]; then
  MANIFEST_REL="$(cat "$LATEST_MANIFEST" 2>/dev/null || true)"
  if [[ -n "$MANIFEST_REL" ]]; then
    echo "Task checkpoint manifest (informational): $MANIFEST_REL"
  fi
fi

if [[ -z "$LATEST" && "$SMOKE_TEST" == "1" ]]; then
  echo "== Smoke test: 64 envs / 5 iterations =="
  uv run train "$TASK_ID" \
    --env.scene.num-envs 64 \
    --agent.run-name smoke-a100 \
    --agent.max_iterations 5

  find "$EXPERIMENT_DIR" -mindepth 1 -maxdepth 1 -type d -name '*_smoke-a100' \
    -exec rm -rf {} + 2>/dev/null || true
  echo "Smoke test passed."
fi

if [[ -z "$LATEST" ]]; then
  LATEST="$(find_latest_checkpoint || true)"
fi
DONE=0
LOAD_RUN=""
LOAD_CHECKPOINT=""

if [[ -n "$LATEST" ]]; then
  FILE="$(basename "$LATEST")"
  RUN_DIR="$(basename "$(dirname "$LATEST")")"
  N="${FILE#model_}"
  N="${N%.pt}"

  # Read the iteration stored inside the checkpoint instead of trusting only the filename.
  CKPT_ITER="$(uv run python - "$LATEST" <<'PY'
import sys
import torch
p = sys.argv[1]
d = torch.load(p, map_location="cpu", weights_only=False)
print(int(d.get("iter", -1)))
PY
)"
  if [[ "$CKPT_ITER" =~ ^[0-9]+$ ]]; then
    DONE="$CKPT_ITER"  # RSL-RL stores the exact current_learning_iteration
  else
    DONE=$((10#$N))
  fi

  # mjlab treats load_run/load_checkpoint as regular expressions.
  # Our generated run/checkpoint names only use safe characters, so anchor the
  # literal names directly. Avoid shell double-escaping which previously made
  # Python regex look for a backslash that is not present in the filename.
  LOAD_RUN="^${RUN_DIR}$"
  LOAD_CHECKPOINT="^model_${N}[.]pt$"

  echo
  echo "============================================================"
  echo " RESUME CHECKPOINT FOUND"
  echo "============================================================"
  echo "Checkpoint path     : $LATEST"
  echo "Checkpoint filename : $FILE"
  echo "Checkpoint iter     : $DONE"
  echo "Resume run dir      : $RUN_DIR"
  echo "============================================================"
  echo
fi

if (( DONE >= TARGET_ITERS )); then
  echo "Target already reached: $DONE >= $TARGET_ITERS"
  sync_once
  exit 0
fi

REMAINING=$((TARGET_ITERS - DONE))

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
  echo "Resume requested from: $RUN_DIR/$FILE"
  echo "IMPORTANT: mjlab must print '[INFO]: Loading model checkpoint from:' below."
else
  echo
  echo "============================================================"
  echo " WARNING: NO CHECKPOINT FOUND - STARTING FROM ZERO"
  echo "============================================================"
  echo "Drive search root: $BACKUP_LOGS/rsl_rl/$EXPERIMENT_NAME"
  echo
fi

printf 'Command: '
printf '%q ' "${CMD[@]}"
echo

# Stream logs to Colab immediately and save the same output to Google Drive.
# PYTHONUNBUFFERED=1 prevents Python/RSL-RL stdout from being block-buffered by the pipe.
TRAIN_LOG_DIR="$DRIVE_ROOT"
TRAIN_LOG="$TRAIN_LOG_DIR/training.log"
mkdir -p "$TRAIN_LOG_DIR"

echo "Live log file       : $TRAIN_LOG"
echo "If the Colab output is collapsed, open this file in Google Drive."

"${CMD[@]}" 2>&1 \
  | python3 -u "$REPO_DIR/scripts/progress_filter.py" \
  | tee -a "$TRAIN_LOG"

echo "Training command finished."
