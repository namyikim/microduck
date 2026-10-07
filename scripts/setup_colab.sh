#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/microduck}"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "ERROR: $REPO_DIR is not a git repository." >&2
  echo "Clone https://github.com/namyikim/microduck.git first." >&2
  exit 1
fi

echo "== GPU =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: Colab GPU runtime을 선택하세요." >&2
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
echo "== Update namyikim/microduck =="
git -C "$REPO_DIR" pull --ff-only origin main

echo
echo "== Install locked dependencies =="
cd "$REPO_DIR"
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
echo "== MicroDuck tasks =="
uv run list-envs | grep -E 'MicroDuck|microduck' | head -60 || true

echo
echo "Setup complete: $REPO_DIR"
