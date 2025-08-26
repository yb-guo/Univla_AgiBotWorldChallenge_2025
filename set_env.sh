#!/bin/bash
set -euo pipefail

ENV_NAME="ptca"
PYTHON_VER="3.10"

echo " ######## Oral in env.sh ########"
echo "当前 Python: $(which python)"
echo "当前 pip: $(which pip)"
echo " ########"

# -------------------------------
# Helper: 在环境中安装 pip 包
# -------------------------------
run_pip() {
    python3 -m pip install --upgrade pip setuptools wheel "$@"
}

# -------------------------------
# 安装必要包
# -------------------------------
run_pip torch==2.2.0+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
run_pip tensorboard wandb

if [ -f "requirements.txt" ]; then
    run_pip -r requirements.txt
fi

# =======================
#  配置 PATH
# =======================
echo "🔧 添加 ~/.local/bin 到 PATH ..."
export PATH="$HOME/.local/bin:$PATH"
if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
echo "✅ PATH 配置完成"
echo " ######## Environment ready ########"
echo "当前 Python: $(which python)"
echo "当前 pip: $(which pip)"
echo "当前 PATH: $PATH"