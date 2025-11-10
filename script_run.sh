#!/usr/bin/env bash
# 自动切换到脚本所在目录（避免路径错乱）
cd "$(dirname "$0")"

# 激活 conda 环境
# 注意：source conda.sh 是 Linux/macOS/WSL 下 conda 的正确用法
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base

# 运行训练
python src/train.py --config configs/base/config.yaml
