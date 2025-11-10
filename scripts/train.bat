@echo off
setlocal EnableDelayedExpansion

echo ============================
echo  One-Click Transformer Run
echo ============================

REM ---- Check conda ----
conda --version >NUL 2>&1
IF ERRORLEVEL 1 (
  echo [ERROR] Conda 未安装 或 未加入 PATH
  pause
  exit /b 1
)

REM ---- Ensure directory structure ----
if not exist src mkdir src
if exist transformer.py move /Y transformer.py src\transformer.py >NUL
if not exist src\transformer.py (
  echo [ERROR] 缺少 src\transformer.py
  pause
  exit /b 1
)

if not exist configs mkdir configs
if not exist data mkdir data
if not exist tokenizer mkdir tokenizer
if not exist results mkdir results
if not exist scripts mkdir scripts

REM ---- Create environment if not exist ----
set ENV_NAME=transformer
for /f "tokens=1" %%i in ('conda env list ^| findstr /i " %ENV_NAME% "') do set FOUND=1
if not defined FOUND (
  echo [Info] 创建 Conda 环境...
  conda create -n %ENV_NAME% -y python=3.10
)

echo [Info] 安装依赖 (自动使用 CPU 版)
conda run -n %ENV_NAME% python -m pip install --upgrade pip
conda run -n %ENV_NAME% pip install torch torchvision torchaudio tqdm pyyaml sentencepiece sacrebleu pandas matplotlib

REM *(如果你是 GPU，请替换上面这行 torch 为 CUDA 对应版本)*

REM ---- Write configs ----
>configs\base.yaml (
  echo data:
  echo   spm_model: data/processed/spm.model
  echo   train_bi: data/processed/train_bi.jsonl
  echo   dev_en_de: data/dev_en_de.jsonl
  echo   dev_de_en: data/dev_de_en.jsonl
  echo.
  echo special_ids:
  echo   pad_id: 0
  echo   bos_id: 2
  echo   eos_id: 3
  echo   lang_tag_en: "<2en>"
  echo   lang_tag_de: "<2de>"
  echo.
  echo model:
  echo   d_model: 128
  echo   nhead: 4
  echo   dim_ff: 512
  echo   num_encoder_layers: 2
  echo   num_decoder_layers: 2
  echo   dropout: 0.1
  echo   share_embeddings: true
  echo   tie_softmax: true
  echo   max_src_len: 64
  echo   max_tgt_len: 64
  echo.
  echo train:
  echo   batch_size: 32
  echo   epochs: 3
  echo   amp: false
  echo   label_smoothing: 0.1
  echo   grad_clip: 1.0
  echo   warmup_steps: 4000
  echo   lr_factor: 1.0
  echo   seed: 427
  echo   save_dir: results/run1
)

REM ---- Write minimal dataset + tokenizer builder ----
>scripts\prepare_data.py (
  echo import os, json, sentencepiece as spm
  echo os.makedirs("data", exist_ok=True)
  echo pairs=[("Hello!","Hallo!"),("How are you?","Wie geht es dir?"),("Good morning.","Guten Morgen."),("Thank you very much.","Vielen Dank."),("I love machine learning.","Ich liebe maschinelles Lernen.")]
  echo train=[]
  echo for s,t in pairs:
  echo     train.append({"src":s,"tgt":t,"src_lang":"en","tgt_lang":"de"})
  echo     train.append({"src":t,"tgt":s,"src_lang":"de","tgt_lang":"en"})
  echo with open("data/train_bi.jsonl","w",encoding="utf-8") as f:
  echo     for i in train: f.write(json.dumps(i,ensure_ascii=False)+"\n")
  echo with open("data/dev_en_de.jsonl","w",encoding="utf-8") as f:
  echo     for s,t in pairs: f.write(json.dumps({"src":s,"tgt":t,"src_lang":"en","tgt_lang":"de"},ensure_ascii=False)+"\n")
  echo with open("data/dev_de_en.jsonl","w",encoding="utf-8") as f:
  echo     for s,t in pairs: f.write(json.dumps({"src":t,"tgt":s,"src_lang":"de","tgt_lang":"en"},ensure_ascii=False)+"\n")
  echo with open("data/corpus.txt","w",encoding="utf-8") as f:
  echo     for s,t in pairs: f.write(s+"\n"); f.write(t+"\n")
  echo spm.SentencePieceTrainer.Train(input="data/corpus.txt",model_prefix="tokenizer/spm",vocab_size=800,pad_id=0,unk_id=1,bos_id=2,eos_id=3,user_defined_symbols=["<2en>","<2de>"])
)

echo [Info] 生成数据与分词器...
conda run -n %ENV_NAME% python src\prepare_iwslt_local.py

echo [Info] 开始训练...
conda run -n %ENV_NAME% python train.py --config configs\base\config.yaml

echo.
echo ✅ 完成：结果已保存到 results\run1
pause
