import os, math
import pandas as pd
import matplotlib.pyplot as plt

# === 这里按你的实际路径改一下 ===
BASELINE_DIR = r"D:\iwslt17\runs\run1"
SHALLOW_DIR  = r"D:\iwslt17\runs\shallow"
HIGH_DROP_DIR= r"D:\iwslt17\runs\high_drop"

RUNS = {
    "Baseline": BASELINE_DIR,
    "Shallow": SHALLOW_DIR,
    "High-Dropout": HIGH_DROP_DIR,
}

OUT_DIR = os.path.join(BASELINE_DIR, "compare")
os.makedirs(OUT_DIR, exist_ok=True)

def _get(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols:
            return df[cols[name.lower()]]
    return None

def load_run(run_name, run_dir):
    csv_path = os.path.join(run_dir, "epoch_metrics.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{run_name} 缺少 {csv_path}")
    df = pd.read_csv(csv_path)
    df = pd.read_csv(csv_path)
    df = df.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last")
    # （可选）提示有哪些重复
    dups = df["epoch"][df["epoch"].duplicated(keep=False)]
    if len(dups) > 0:
        print(f"[{run_name}] duplicate epochs found:", list(sorted(set(dups))))

    # 统一列名（兼容 train_ce / train_loss；dev_ce_*；dev_bleu_*）
    epoch = _get(df, ["epoch"])
    train_ce = _get(df, ["train_ce", "train_loss"])

    # 验证集 CE（两方向）
    dev_ce_ed = _get(df, ["dev_ce_en-de", "dev_ce_en_de", "val_ce_en-de"])
    dev_ce_de = _get(df, ["dev_ce_de-en", "dev_ce_de_en", "val_ce_de-en"])
    if dev_ce_ed is None or dev_ce_de is None:
        # 若写过 val_loss 列就直接用
        val_loss = _get(df, ["val_loss"])
        if val_loss is None:
            raise ValueError(f"{run_name} 缺少 dev_ce 或 val_loss 列")
    else:
        val_loss = 0.5 * (dev_ce_ed + dev_ce_de)

    # ppl = exp(val_loss)
    val_ppl = val_loss.apply(lambda x: math.exp(min(float(x), 20.0)))

    # BLEU（两方向平均；若缺失则返回 None）
    bleu_ed = _get(df, ["dev_bleu_en-de", "bleu_en-de", "en-de_bleu"])
    bleu_de = _get(df, ["dev_bleu_de-en", "bleu_de-en", "de-en_bleu"])
    if bleu_ed is not None and bleu_de is not None:
        bleu = 0.5 * (bleu_ed + bleu_de)
    else:
        bleu = None

    # 末轮摘要
    tail = {
        "epoch": int(epoch.iloc[-1]),
        "val_loss": float(val_loss.iloc[-1]),
        "val_ppl": float(val_ppl.iloc[-1]),
        "bleu": (None if bleu is None else float(bleu.iloc[-1])),
    }

    return {
        "name": run_name,
        "epoch": epoch,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "bleu": bleu,
        "tail": tail,
    }

# 读取所有 run
curves = []
for name, d in RUNS.items():
    curves.append(load_run(name, d))

# 画图：val loss
plt.figure()
for c in curves:
    plt.plot(c["epoch"], c["val_loss"], label=c["name"])
plt.xlabel("epoch"); plt.ylabel("val loss (CE)"); plt.title("Validation Loss (EN↔DE avg)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "compare_val_loss.png"), dpi=160); plt.close()

# 画图：val ppl
plt.figure()
for c in curves:
    plt.plot(c["epoch"], c["val_ppl"], label=c["name"])
plt.xlabel("epoch"); plt.ylabel("val PPL"); plt.title("Validation PPL (EN↔DE avg)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "compare_val_ppl.png"), dpi=160); plt.close()

# 画图：BLEU（若缺失就跳过该曲线/整张图）
has_bleu = [c for c in curves if c["bleu"] is not None]
if len(has_bleu) >= 1:
    plt.figure()
    for c in has_bleu:
        plt.plot(c["epoch"], c["bleu"], label=c["name"])
    plt.xlabel("epoch"); plt.ylabel("BLEU"); plt.title("Dev BLEU (EN↔DE avg, greedy)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "compare_bleu.png"), dpi=160); plt.close()

# 打印末轮摘要
print("== Last-epoch summary ==")
for c in curves:
    t = c["tail"]
    print(f"{c['name']}: epoch={t['epoch']} | val_loss={t['val_loss']:.4f} | "
          f"val_ppl={t['val_ppl']:.1f} | BLEU={t['bleu'] if t['bleu'] is not None else 'N/A'}")
print(f"Saved figures to: {OUT_DIR}")
