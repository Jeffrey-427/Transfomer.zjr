Transformer EN↔DE (IWSLT17) — Course Assignment
1. 项目简介

手工实现 Encoder–Decoder Transformer，在 IWSLT17 英↔德 双向翻译任务上完成基线与消融。训练过程中记录 CE（Loss）/PPL/BLEU，控制台实时显示进度与 ETA，并保存曲线与对比图。

2. 环境与依赖

Python 3.10（建议）/ CUDA 可选

依赖：torch, tqdm, sentencepiece, sacrebleu, pyyaml, pandas, matplotlib

Windows / PowerShell

# 创建并进入虚拟环境
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt


Linux / macOS

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

3. 数据与预处理(原始数据以及处理后的数据在data文件夹中)

数据集：IWSLT2017 EN↔DE（已在本机准备）

分词：SentencePiece（Unigram），联合词表，vocab_size=16000，包含自定义语言标签 <2en>、<2de>

处理产物：${DATA_ROOT}/processed/iwslt17/bi/

快速准备（已处理好的同学仍需设置环境变量）

# Windows 设置数据根目录
$env:DATA_ROOT="D:\iwslt17"


（如需从头处理一次数据与训练分词器，运行）

python src\prepare_iwslt_local.py `
  --root $env:DATA_ROOT `
  --out-dir $env:DATA_ROOT\processed\iwslt17\bi `
  --vocab-size 16000 `
  --dev-year 2010 `
  --test-year 2010 `
  --emit-tokenized


训练脚本会使用 ${DATA_ROOT}\processed\iwslt17\bi\spm.model 并自动以 sp.get_piece_size() 对齐模型 vocab_size。每个 epoch 会在控制台与 CSV 中写入 CE/PPL/BLEU，避免单独评测脚本。

4. 训练与评测（最小命令）
Windows（PowerShell）
# 训练：双向 EN↔DE（基线）
python src\train.py --config configs\base\config.yaml

# 消融（任选其一或全部）
python src\train.py --config configs\ablation\shallow.yaml      # 变浅：2/2 层 + 减小 FFN
python src\train.py --config configs\ablation\high_drop.yaml    # 高 Dropout：更强正则

Linux / macOS
export DATA_ROOT=/data/iwslt17

# 训练：双向 EN↔DE（基线）
python src/train.py --config configs/base/config.yaml

# 消融
python src/train.py --config configs/ablation/shallow.yaml
python src/train.py --config configs/ablation/high_drop.yaml


训练期间每个 epoch 自动计算：[Train] CE/PPL、[Dev] en-de & de-en CE/BLEU，并写入 runs/<exp>/epoch_metrics.csv；结束后保存 best.ckpt 与曲线图。

5. 结果（基线）

训练/验证指标与样例翻译会随 epoch 自动打印并写入 runs/<exp_name>/epoch_metrics.csv。

曲线图：runs/<exp_name>/loss_curve_epoch.png、runs/<exp_name>/ppl_curve_epoch.png。

对比图（基线 vs 消融）：运行

python src\plot_compare_min.py


生成 runs\run1\compare\compare_val_loss.png / compare_val_ppl.png / compare_bleu.png。

6. 消融实验

Shallow：num_encoder_layers=2, num_decoder_layers=2, dim_ff 减小。

High-Dropout：dropout=0.5。

每个消融只改动单一因素，其余与基线一致，保证对比公平。

7. 复现实验

硬件：NVIDIA RTX 系列（单卡即可）/ CPU 也可运行（较慢）

OS：Windows 11 / WSL2 Ubuntu 22.04 / Linux

Python：3.10

随机种子：427

一键复现（Windows 示例）

# 克隆
git clone https://github.com/Jeffrey-427/Transformer-IWSLT17.git
cd Transformer-IWSLT17

# 环境
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 数据根
$env:DATA_ROOT="D:\iwslt17"

# 训练基线
python src\train.py --config configs\base\config.yaml

# 画对比图
python src\plot_compare_min.py

8.注意事项（踩坑速查）

CUDA illegal memory access：90% 因 token 越界（<2en>/<2de> 未写入词表或 id≥vocab）。本项目已在前向前检查，并强制 vocab_size = sp.get_piece_size()。

GPU 不可用：确认 torch.cuda.is_available()，不要把单卡系统设成 CUDA_VISIBLE_DEVICES=1。

曲线“直线段”：本项目按 epoch 粒度记录验证点，折线连点属正常；绘图脚本已支持去重和平滑。

9.许可

MIT（数据集版权归原提供方所有，仅用于教学与学术研究）