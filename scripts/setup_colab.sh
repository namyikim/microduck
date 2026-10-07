#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO="${UPSTREAM_REPO:-https://github.com/pollen-robotics/microduck_rl.git}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-develop}"
UPSTREAM_DIR="${UPSTREAM_DIR:-/content/microduck_rl}"

echo "== GPU =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: GPU runtime이 아닙니다. Colab에서 A100 GPU를 선택하세요." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo "== Basic tools =="
if ! command -v rsync >/dev/null 2>&1; then
  apt-get -qq update
  apt-get -qq install -y rsync
fi

echo
echo "== Install uv =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo
echo "== Python 3.12 =="
uv python install 3.12

echo
echo "== Clone/update official MicroDuck RL =="
if [[ -d "$UPSTREAM_DIR/.git" ]]; then
  git -C "$UPSTREAM_DIR" fetch origin "$UPSTREAM_BRANCH"
  git -C "$UPSTREAM_DIR" checkout "$UPSTREAM_BRANCH"
  git -C "$UPSTREAM_DIR" pull --ff-only origin "$UPSTREAM_BRANCH"
else
  rm -rf "$UPSTREAM_DIR"
  git clone --branch "$UPSTREAM_BRANCH" --depth 1 "$UPSTREAM_REPO" "$UPSTREAM_DIR"
fi

echo
echo "== Install locked dependencies =="
cd "$UPSTREAM_DIR"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
uv sync --python 3.12 --frozen

echo
echo "== CUDA / Torch check =="
uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch CUDA is not available.")

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}, {p.total_memory / 1024**3:.1f} GiB")
PY

echo
echo "== MicroDuck task registry check =="
uv run list-envs | grep -E 'MicroDuck|microduck' | head -60 || true

echo
echo "Setup complete: $UPSTREAM_DIR"
