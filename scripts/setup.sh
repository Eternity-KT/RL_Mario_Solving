#!/usr/bin/env bash
# Cài môi trường cho cả nhóm (Linux/macOS/WSL).  Chạy từ gốc dự án:
#   bash scripts/setup.sh          # CPU
#   GPU=1 bash scripts/setup.sh    # máy có GPU NVIDIA
# Yêu cầu: Python 3.10-3.12 và trình biên dịch C++ (Ubuntu: sudo apt install build-essential).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3.11}

[ -x .venv/bin/python ] || "$PY" -m venv .venv
source .venv/bin/activate
python --version
pip install --upgrade pip
if [ "${GPU:-0}" = "1" ]; then pip install torch; else pip install torch --index-url https://download.pytorch.org/whl/cpu; fi
pip install -e .
pip install pandas matplotlib
python -c "import torch, gymnasium, nes_py, gym_super_mario_bros, numpy; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| numpy', numpy.__version__)"
pytest -q --no-header -rN | tail -n 3 || true
echo "Xong. Kích hoạt: source .venv/bin/activate"
