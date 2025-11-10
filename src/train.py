# train.py
from __future__ import annotations
import os, json, math, random, argparse, csv
from typing import List, Dict
import yaml
import sentencepiece as spm
import sacrebleu
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformer import Transformer, TransformerConfig
from tqdm import tqdm
import time
import pandas as pd
import matplotlib.pyplot as plt


@torch.no_grad()
def eval_bleu_loader(model, loader, sp, special, lang_tag_id, max_len, device):
    import sacrebleu
    model.eval()
    BOS, EOS = special["bos_id"], special["eos_id"]
    hyps, refs = [], []
    for batch in loader:
        src = batch["src_ids"].to(device)
        src_pad = batch["src_pad"].to(device)
        ys = model.generate_greedy(src, src_pad, lang_tag_id, BOS, EOS, max_len=max_len)  # [B,T]
        outs = ys[:, 2:].tolist()  # 去掉 <2xx>, <BOS>
        for i, seq in enumerate(outs):
            if EOS in seq:
                seq = seq[:seq.index(EOS)]
            hyps.append(sp.decode(seq))
            refs.append(batch["raw_tgt"][i])
    return sacrebleu.corpus_bleu(hyps, [refs]).score

# ---------- Utils ----------
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def subsequent_mask(sz: int, device) -> torch.Tensor:
    return torch.triu(torch.ones((sz, sz), dtype=torch.bool, device=device), diagonal=1)

def noam_lr(step: int, d_model: int, warmup: int, factor: float) -> float:
    if step <= 0: step = 1
    return factor * (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))

# ---------- Data ----------
class JSONLDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
    def __len__(self): return len(self.lines)
    def __getitem__(self, idx):
        obj = json.loads(self.lines[idx])
        # 返回原始字符串，便于日志打印 & 快速 BLEU
        return obj["src"], obj["tgt"], obj.get("src_lang",""), obj.get("tgt_lang","")

def build_collate(sp: spm.SentencePieceProcessor, special, max_src_len, max_tgt_len, lang2id):
    PAD, BOS, EOS = special["pad_id"], special["bos_id"], special["eos_id"]
    def collate(batch):
        raw_src = []
        raw_tgt = []
        src_tok, tgt_in_tok, tgt_out_tok = [], [], []
        for src, tgt, _, tgt_lang in batch:
            raw_src.append(src); raw_tgt.append(tgt)
            s = sp.encode(src, out_type=int)[:max_src_len-1] + [EOS]
            y = sp.encode(tgt, out_type=int)[:max_tgt_len-2]
            lang_tag = lang2id["en"] if tgt_lang == "en" else lang2id["de"]
            tin = [lang_tag, BOS] + y
            tout = [BOS] + y + [EOS]
            src_tok.append(s); tgt_in_tok.append(tin); tgt_out_tok.append(tout)

        B = len(batch)
        S = max(len(s) for s in src_tok)
        T = max(len(t) for t in tgt_in_tok)
        src_ids = torch.full((B,S), PAD, dtype=torch.long)
        tgt_in  = torch.full((B,T), PAD, dtype=torch.long)
        tgt_out = torch.full((B,T), PAD, dtype=torch.long)

        src_len = []; tgt_len = []
        for i in range(B):
            s, tin, tout = src_tok[i], tgt_in_tok[i], tgt_out_tok[i]
            src_ids[i,:len(s)] = torch.tensor(s)
            tgt_in[i,:len(tin)] = torch.tensor(tin)
            tgt_out[i,:len(tout)] = torch.tensor(tout)
            src_len.append(len(s)); tgt_len.append(len(tin))

        src_len = torch.tensor(src_len, dtype=torch.long)
        tgt_len = torch.tensor(tgt_len, dtype=torch.long)
        src_pad = (torch.arange(S).unsqueeze(0) >= src_len.unsqueeze(1))
        tgt_pad = (torch.arange(T).unsqueeze(0) >= tgt_len.unsqueeze(1))
        tgt_sub = subsequent_mask(T, device=torch.device("cpu"))
        return {"src_ids": src_ids, "tgt_in": tgt_in, "tgt_out": tgt_out,
                "src_pad": src_pad, "tgt_pad": tgt_pad, "tgt_sub": tgt_sub,
                "raw_src": raw_src, "raw_tgt": raw_tgt}
    return collate

# ---------- Quick BLEU on current batch ----------
@torch.no_grad()
def quick_batch_bleu_and_sample(model, batch, sp, special, max_len, device, sample_size=8):
    """在当前训练 batch 上做快速 greedy 解码，返回 BLEU 分数与一个样例 (src, ref, hyp)。"""
    model.eval()
    PAD, BOS, EOS = special["pad_id"], special["bos_id"], special["eos_id"]

    src = batch["src_ids"].to(device)
    src_pad = batch["src_pad"].to(device)
    tgt_in = batch["tgt_in"].to(device)     # 为了拿到语言标签
    raw_src = batch["raw_src"]
    raw_tgt = batch["raw_tgt"]

    B = src.size(0)
    K = min(sample_size, B)

    hyps, refs = [], []
    hyp_first = None

    # 逐条生成（因为每条的 lang tag 可能不一致）
    for i in range(K):
        lang_tag_id = int(tgt_in[i, 0].item())
        ys = model.generate_greedy(src[i:i+1], src_pad[i:i+1],
                                   lang_tag_id, BOS, EOS, max_len=max_len)  # [1, T]
        out_ids = ys[0, 2:].tolist()   # 去掉 <2xx> 和 <BOS>
        if EOS in out_ids:
            out_ids = out_ids[:out_ids.index(EOS)]
        hyp = sp.decode(out_ids)
        ref = raw_tgt[i]  # 直接用原始目标文本
        hyps.append(hyp); refs.append(ref)
        if hyp_first is None:
            hyp_first = hyp
    # sacreBLEU 计算（小样本，速度很快）
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score if hyps else 0.0
    sample = (raw_src[0], refs[0], hyp_first if hyp_first is not None else "")
    model.train()
    return bleu, sample

# ---------- CE on loader ----------
@torch.no_grad()
def evaluate_ce(model: Transformer, loader: DataLoader, loss_fn, device):
    model.eval()
    total, ntok = 0.0, 0
    for batch in loader:
        src = batch["src_ids"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)
        src_pad = batch["src_pad"].to(device)
        tgt_pad = batch["tgt_pad"].to(device)
        tgt_sub = batch["tgt_sub"].to(device)

        mem = model.encode(src, src_pad)
        logits = model.decode(tgt_in, mem, tgt_sub, tgt_pad, src_pad)
        V = logits.size(-1)
        flat_logit = logits.reshape(-1, V)
        flat_tgt = tgt_out.reshape(-1)
        tok_loss = loss_fn(flat_logit, flat_tgt)
        mask = (flat_tgt != special["pad_id"])
        total += tok_loss[mask].sum().item()
        ntok += mask.sum().item()
    # 循环结束后：
    return total / max(ntok, 1)


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer & vocab
    sp = spm.SentencePieceProcessor(model_file=cfg["data"]["spm_model"])
    vocab_size = sp.get_piece_size()
    special = cfg["special_ids"]
    lang2id = {
        "en": sp.piece_to_id(special["lang_tag_en"]),
        "de": sp.piece_to_id(special["lang_tag_de"]),
    }

    # datasets & loaders
    ds_train = JSONLDataset(cfg["data"]["train_bi"])
    ds_dev_en_de = JSONLDataset(cfg["data"]["dev_en_de"])
    ds_dev_de_en = JSONLDataset(cfg["data"]["dev_de_en"])

    collate = build_collate(sp, special,
                            cfg["model"]["max_src_len"], cfg["model"]["max_tgt_len"],
                            lang2id)

    train_loader = DataLoader(ds_train, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, collate_fn=collate, drop_last=True)
    dev_loader_en_de = DataLoader(ds_dev_en_de, batch_size=cfg["train"]["batch_size"],
                                  shuffle=False, collate_fn=collate)
    dev_loader_de_en = DataLoader(ds_dev_de_en, batch_size=cfg["train"]["batch_size"],
                                  shuffle=False, collate_fn=collate)

    # model
    model_cfg = TransformerConfig(
        vocab_size=vocab_size,
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        dim_ff=cfg["model"]["dim_ff"],
        num_encoder_layers=cfg["model"]["num_encoder_layers"],
        num_decoder_layers=cfg["model"]["num_decoder_layers"],
        dropout=cfg["model"]["dropout"],
        share_embeddings=cfg["model"]["share_embeddings"],
        tie_softmax=cfg["model"]["tie_softmax"],
        pad_id=special["pad_id"],
        max_src_len=cfg["model"]["max_src_len"],
        max_tgt_len=cfg["model"]["max_tgt_len"],
    )
    model = Transformer(model_cfg).to(device)

    # optim & loss
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=special["pad_id"],
                                  label_smoothing=cfg["train"]["label_smoothing"],
                                  reduction='none')  # 逐元素损失，后面手动做“掩码平均”
    amp_on = cfg["train"]["amp"] and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=amp_on)

    # logging & save dir
    save_dir = cfg["train"]["save_dir"]; os.makedirs(save_dir, exist_ok=True)
    metrics_csv = os.path.join(save_dir, "train_metrics.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["step", "epoch", "loss", "ppl", "batch_bleu"])

    hist_steps, hist_loss, hist_ppl, hist_bleu = [], [], [], []
    best_dev = math.inf
    global_step = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        steps_per_epoch = len(train_loader)
        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}/{cfg['train']['epochs']}",
                    dynamic_ncols=True, mininterval=0.3)

        # —— 本 epoch 的累计（按 token）——
        epoch_loss_sum = 0.0
        epoch_tok_sum = 0
        step_ema = None

        for i, batch in enumerate(train_loader, start=1):
            iter_start = time.time()

            src = batch["src_ids"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)
            src_pad = batch["src_pad"].to(device)
            tgt_pad = batch["tgt_pad"].to(device)
            tgt_sub = batch["tgt_sub"].to(device)

            with torch.amp.autocast('cuda', enabled=amp_on):
                mem = model.encode(src, src_pad)
                logits = model.decode(tgt_in, mem, tgt_sub, tgt_pad, src_pad)

                V = logits.size(-1)
                flat_logit = logits.reshape(-1, V)
                flat_tgt = tgt_out.reshape(-1)
                tok_loss = loss_fn(flat_logit, flat_tgt)  # [B*T]
                mask = (flat_tgt != special["pad_id"])
                loss = tok_loss[mask].mean()  # —— 平均每 token 损失 ——

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            global_step += 1
            lr = noam_lr(global_step, model_cfg.d_model, cfg["train"]["warmup_steps"], cfg["train"]["lr_factor"])
            for pg in opt.param_groups: pg["lr"] = lr
            scaler.step(opt);
            scaler.update();
            opt.zero_grad(set_to_none=True)

            # —— 累计到 epoch 统计 ——
            epoch_loss_sum += float(tok_loss[mask].sum().item())
            epoch_tok_sum += int(mask.sum().item())

            # —— ETA（仅显示速度 & 本 epoch 进度）——
            step_time = max(1e-6, time.time() - iter_start)
            step_ema = step_time if (step_ema is None or global_step < 20) else (0.9 * step_ema + 0.1 * step_time)
            eta_epoch = (steps_per_epoch - i) * step_ema
            pbar.set_postfix({"lr": f"{lr:.2e}", "ETA(e)": f"{int(eta_epoch // 60)}m{int(eta_epoch % 60)}s"})
            pbar.update(1)

        pbar.close()

        # —— Epoch 级训练指标 ——
        train_ce = epoch_loss_sum / max(epoch_tok_sum, 1)
        train_ppl = math.exp(min(train_ce, 20.0))
        print(f"[Train] epoch {epoch} | CE={train_ce:.4f} | PPL={train_ppl:.1f}")

        # —— Dev CE（双向） ——
        dev_en_de = evaluate_ce(model, dev_loader_en_de, loss_fn, device)
        dev_de_en = evaluate_ce(model, dev_loader_de_en, loss_fn, device)
        print(f"[Dev-CE] en-de={dev_en_de:.4f} | de-en={dev_de_en:.4f}")

        # —— Dev BLEU（双向，各跑一次）——
        lang_id_de = sp.piece_to_id(special["lang_tag_de"])
        lang_id_en = sp.piece_to_id(special["lang_tag_en"])
        bleu_en_de = eval_bleu_loader(model, dev_loader_en_de, sp, special, lang_id_de, cfg["infer"]["max_gen_len"],
                                      device)
        bleu_de_en = eval_bleu_loader(model, dev_loader_de_en, sp, special, lang_id_en, cfg["infer"]["max_gen_len"],
                                      device)
        print(f"[Dev-BLEU] en-de={bleu_en_de:.2f} | de-en={bleu_de_en:.2f}")

        # —— 记录到 CSV（按 epoch）——
        with open(os.path.join(save_dir, "epoch_metrics.csv"), "a", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            if epoch == 1 and fcsv.tell() == 0:
                w.writerow(["epoch", "train_ce", "train_ppl", "dev_ce_en-de", "dev_ce_de-en", "dev_bleu_en-de",
                            "dev_bleu_de-en"])
            w.writerow([epoch, f"{train_ce:.6f}", f"{train_ppl:.2f}", f"{dev_en_de:.6f}", f"{dev_de_en:.6f}",
                        f"{bleu_en_de:.2f}", f"{bleu_de_en:.2f}"])

        # —— 挑 best 并保存 ——（以双向 CE 平均为准）
        avg_dev = 0.5 * (dev_en_de + dev_de_en)
        if avg_dev < best_dev:
            best_dev = avg_dev
            ckpt = os.path.join(save_dir, "best.ckpt")
            torch.save({"model": model.state_dict(), "cfg": model_cfg.__dict__}, ckpt)
            print(f"Saved best to {ckpt}")

    # === 训练全部结束后，按 epoch 画曲线 ===
    csv_path = os.path.join(save_dir, "epoch_metrics.csv")
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            # Loss(=CE) 曲线
            plt.figure()
            plt.plot(df["epoch"], df["train_ce"])
            plt.xlabel("epoch");
            plt.ylabel("train CE");
            plt.title("Train CE per epoch")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "loss_curve_epoch.png"), dpi=160)
            plt.close()

            # PPL 曲线
            plt.figure()
            plt.plot(df["epoch"], df["train_ppl"])
            plt.xlabel("epoch");
            plt.ylabel("train PPL");
            plt.title("Train PPL per epoch")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "ppl_curve_epoch.png"), dpi=160)
            plt.close()

            print(f"[Plot] saved to {save_dir}")
        else:
            print(f"[Plot] skip: {csv_path} not found.")
    except Exception as e:
        print(f"[Plot] failed: {e}")

    print("Training done.")

    # ---- plot curves ----
    # Loss 曲线
    plt.figure()
    plt.plot(hist_steps, hist_loss)
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=160)
    plt.close()

    # PPL 曲线
    plt.figure()
    plt.plot(hist_steps, hist_ppl)
    plt.xlabel("step"); plt.ylabel("ppl"); plt.title("Training Perplexity")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ppl_curve.png"), dpi=160)
    plt.close()

    # 也把最后一次 batch BLEU 曲线存一份（可选）
    plt.figure()
    plt.plot(hist_steps, hist_bleu)
    plt.xlabel("step"); plt.ylabel("batch BLEU"); plt.title("Training batch BLEU (quick)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "batch_bleu_curve.png"), dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
