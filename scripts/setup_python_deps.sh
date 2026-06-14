#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/setup_python_deps.sh
# Installs Python dependencies into the conda env named 'rk3588'.

ENV_NAME="rk3588"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REQ_FILE="$REPO_ROOT/requirements.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda 未找到，请先安装 Anaconda/Miniconda 并确保 conda 命令可用。"
  exit 1
fi

# Make 'conda activate' available in non-interactive shells
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "无法找到 conda 初始化脚本：$CONDA_BASE/etc/profile.d/conda.sh"
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda 环境 '$ENV_NAME' 不存在，正在创建 (python=3.10)..."
  conda create -y -n "$ENV_NAME" python=3.10
fi

echo "激活环境 $ENV_NAME ..."
conda activate "$ENV_NAME"

if [ ! -f "$REQ_FILE" ]; then
  echo "requirements.txt 未找到：$REQ_FILE"
  exit 1
fi

echo "升级 pip 并安装依赖（来自 $REQ_FILE）..."
python -m pip install -U pip setuptools wheel
python -m pip install -r "$REQ_FILE"

echo "完成：环境 '$ENV_NAME' 已安装依赖。"
