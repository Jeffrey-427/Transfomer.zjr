#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os, io, json, html, random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import xml.etree.ElementTree as ET

try:
    import sentencepiece as spm
    _SPM_OK = True
except Exception as e:
    _SPM_OK = False
    _SPM_ERR = e


# ---------- 小工具 ----------
def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def read_lines_clean(path: str) -> List[str]:
    """读取 train.tags.*：去空行、去以 '<' 开头的标签行。"""
    out = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:       # 跳过空白
                continue
            if s.startswith('<'):  # 跳过标签
                continue
            out.append(s)
    return out

def parse_xml_bitext(path_src_xml: str, path_tgt_xml: str) -> List[Tuple[str, str]]:
    """按顺序收集 <seg>，再 zip 对齐。"""
    def collect(p: str) -> List[str]:
        try:
            root = ET.parse(p).getroot()
        except ET.ParseError as e:
            raise RuntimeError(f"XML parse error for {p}: {e}")
        segs = []
        for seg in root.iter('seg'):
            txt = ''.join(seg.itertext()).strip()
            txt = html.unescape(txt)
            if txt:
                segs.append(txt)
        return segs

    s = collect(path_src_xml); t = collect(path_tgt_xml)
    if len(s) != len(t):
        n = min(len(s), len(t))
        print(f"[WARN] XML length mismatch: {os.path.basename(path_src_xml)} vs {os.path.basename(path_tgt_xml)}; truncate to {n}")
        s, t = s[:n], t[:n]
    return list(zip(s, t))

def dump_jsonl(pairs: List[Tuple[str, str]], out_path: str, src_lang: str, tgt_lang: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, 'w', encoding='utf-8') as f:
        for a, b in pairs:
            f.write(json.dumps({"src": a, "tgt": b, "src_lang": src_lang, "tgt_lang": tgt_lang}, ensure_ascii=False) + "\n")

def tokenized_jsonl(spm_model: str, inp_jsonl: str, out_jsonl: str, eos: bool = True) -> None:
    """用 SentencePiece 将 src/tgt 转为 ids；若 eos=True，则末尾加 EOS(id=2)。"""
    if not _SPM_OK:
        raise RuntimeError(f"sentencepiece 未安装: {_SPM_ERR}")
    sp = spm.SentencePieceProcessor(model_file=spm_model)
    eos_id = 2
    with open(inp_jsonl, 'r', encoding='utf-8') as fin, open(out_jsonl, 'w', encoding='utf-8') as fout:
        for line in fin:
            obj = json.loads(line)
            src_ids = sp.encode(obj["src"], out_type=int)
            tgt_ids = sp.encode(obj["tgt"], out_type=int)
            if eos:
                src_ids.append(eos_id)
                tgt_ids.append(eos_id)
            obj["src_ids"] = src_ids
            obj["tgt_ids"] = tgt_ids
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------- 主流程 ----------
def find_required_files(root: str, pair: str, dev_year: int, test_year: int) -> Dict[str, Optional[str]]:
    src, tgt = pair.split('-')
    base = os.path.join(root, pair)
    train_src = os.path.join(base, f"train.tags.{pair}.{src}")
    train_tgt = os.path.join(base, f"train.tags.{pair}.{tgt}")
    dev_src_xml = os.path.join(base, f"IWSLT17.TED.dev{dev_year}.{pair}.{src}.xml")
    dev_tgt_xml = os.path.join(base, f"IWSLT17.TED.dev{dev_year}.{pair}.{tgt}.xml")
    test_src_xml = os.path.join(base, f"IWSLT17.TED.tst{test_year}.{pair}.{src}.xml")
    test_tgt_xml = os.path.join(base, f"IWSLT17.TED.tst{test_year}.{pair}.{tgt}.xml")
    def ok(p: str) -> Optional[str]: return p if os.path.exists(p) else None
    return dict(
        train_src=ok(train_src), train_tgt=ok(train_tgt),
        dev_src_xml=ok(dev_src_xml), dev_tgt_xml=ok(dev_tgt_xml),
        test_src_xml=ok(test_src_xml), test_tgt_xml=ok(test_tgt_xml),
    )

def load_train_pairs(train_src: str, train_tgt: str) -> List[Tuple[str, str]]:
    s = read_lines_clean(train_src); t = read_lines_clean(train_tgt)
    if len(s) != len(t):
        n = min(len(s), len(t))
        print(f"[WARN] train length mismatch: {os.path.basename(train_src)} vs {os.path.basename(train_tgt)}; truncate to {n}")
        s, t = s[:n], t[:n]
    return list(zip(s, t))

def train_sentencepiece(corpus_texts: List[str], out_prefix: str, vocab_size: int) -> str:
    """训练共享分词器；固定特殊 id：PAD=0, BOS=1, EOS=2, UNK=3；加入 <2en>/<2de> 标签。"""
    if not _SPM_OK:
        raise RuntimeError(f"sentencepiece 未安装: {_SPM_ERR}")
    txt_path = f"{out_prefix}.txt"
    ensure_dir(os.path.dirname(out_prefix))
    with io.open(txt_path, "w", encoding="utf-8") as f:
        for line in corpus_texts:
            if line.strip():
                f.write(line.strip() + "\n")
    spm.SentencePieceTrainer.Train(
        input=txt_path,
        model_prefix=out_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="unigram",
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,
        user_defined_symbols=",".join(["<2en>", "<2de>"]),
    )
    return f"{out_prefix}.model"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="原始 IWSLT 根目录，例如：D:\\...\\data")
    ap.add_argument("--out-dir", type=str, default=None, help="输出目录，默认 <root>\\processed\\iwslt17\\bi")
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--dev-year", type=int, default=2010)
    ap.add_argument("--test-year", type=int, default=2010)
    ap.add_argument("--lang-pairs", type=str, default="en-de,de-en", help="要处理的方向，逗号分隔")
    ap.add_argument("--emit-tokenized", action="store_true", help="额外输出 *.tok.jsonl（分词后 ids，末尾加 EOS）")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    random.seed(args.seed)
    root = os.path.normpath(args.root)
    out_dir = os.path.normpath(args.out_dir) if args.out_dir else os.path.join(root, "processed", "iwslt17", "bi")
    ensure_dir(out_dir)

    # 1) 逐方向读取并写出 dev/test；训练样本汇总到 bi_train
    bi_train: List[Tuple[str, str, str, str]] = []  # (src, tgt, src_lang, tgt_lang)
    stats: Dict[str, int] = {}

    for pair in args.lang_pairs.split(","):
        pair = pair.strip()
        if not pair:
            continue
        src_lang, tgt_lang = pair.split("-")
        f = find_required_files(root, pair, args.dev_year, args.test_year)

        if not f["train_src"] or not f["train_tgt"]:
            print(f"[WARN] 缺少 {pair} 的 train 文件：{os.path.join(root,pair)} 下的 train.tags.{pair}.*")
        else:
            tp = load_train_pairs(f["train_src"], f["train_tgt"])
            for s, t in tp:
                bi_train.append((s, t, src_lang, tgt_lang))
            stats[f"train_{pair}"] = len(tp)

        if f["dev_src_xml"] and f["dev_tgt_xml"]:
            dev_pairs = parse_xml_bitext(f["dev_src_xml"], f["dev_tgt_xml"])
            dump_jsonl(dev_pairs, os.path.join(out_dir, f"dev_{pair}.jsonl"), src_lang, tgt_lang)
            stats[f"dev_{pair}"] = len(dev_pairs)
        else:
            print(f"[WARN] 缺少 {pair} 的 dev{args.dev_year} XML，跳过。")

        if f["test_src_xml"] and f["test_tgt_xml"]:
            test_pairs = parse_xml_bitext(f["test_src_xml"], f["test_tgt_xml"])
            dump_jsonl(test_pairs, os.path.join(out_dir, f"test_{pair}.jsonl"), src_lang, tgt_lang)
            stats[f"test_{pair}"] = len(test_pairs)
        else:
            print(f"[WARN] 缺少 {pair} 的 tst{args.test_year} XML，跳过。")

    # 2) 合并双向训练集
    random.shuffle(bi_train)
    bi_out = os.path.join(out_dir, "train_bi.jsonl")
    with open(bi_out, "w", encoding="utf-8") as f:
        for s, t, sl, tl in bi_train:
            f.write(json.dumps({"src": s, "tgt": t, "src_lang": sl, "tgt_lang": tl}, ensure_ascii=False) + "\n")
    stats["train_bi"] = len(bi_train)

    # 3) 训练共享 SentencePiece（语料=bi_train 的 src+tgt 合并）
    print(f"[INFO] 训练 SentencePiece（vocab={args.vocab_size}）...")
    spm_prefix = os.path.join(out_dir, "spm")
    spm_model = train_sentencepiece([x[0] for x in bi_train] + [x[1] for x in bi_train], spm_prefix, args.vocab_size)
    print(f"[INFO] 分词器已保存：{spm_model}")

    # 4) 可选：输出分词后的 *.tok.jsonl
    if args.emit_tokenized:
        for name in ["train_bi.jsonl", "dev_en-de.jsonl", "dev_de-en.jsonl", "test_en-de.jsonl", "test_de-en.jsonl"]:
            p = os.path.join(out_dir, name)
            if os.path.exists(p):
                out_tok = p.replace(".jsonl", ".tok.jsonl")
                print(f"[INFO] Tokenizing {name} -> {os.path.basename(out_tok)}")
                tokenized_jsonl(spm_model, p, out_tok, eos=True)

    # 5) 统计摘要
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("[DONE] 输出目录：", out_dir)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
