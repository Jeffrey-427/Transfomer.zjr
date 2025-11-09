# transformer.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.dk = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        return x.view(B, L, self.nhead, self.dk).transpose(1, 2)  # [B,H,L,dk]

    def forward(
        self,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,       # [Lq, Lk], True=mask
        key_padding_mask: Optional[torch.Tensor] = None # [B, Lk], True=pad
    ) -> torch.Tensor:
        B, Lq, D = q.shape
        Lk = k.shape[1]

        qh = self._shape(self.q_proj(q), B, Lq)
        kh = self._shape(self.k_proj(k), B, Lk)
        vh = self._shape(self.v_proj(v), B, Lk)

        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.dk)  # [B,H,Lq,Lk]

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        ctx = torch.matmul(attn, vh)                                   # [B,H,Lq,dk]
        ctx = ctx.transpose(1, 2).contiguous().view(B, Lq, D)          # [B,Lq,D]
        return self.o_proj(ctx)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dim_ff, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask: Optional[torch.Tensor]):
        h = self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x),
                           attn_mask=None, key_padding_mask=src_key_padding_mask)
        x = x + self.drop1(h)
        h2 = self.ffn(self.ln2(x))
        x = x + self.drop2(h2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.drop2 = nn.Dropout(dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dim_ff, dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x, mem, tgt_sub_mask, tgt_key_padding_mask, src_key_padding_mask):
        h = self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x),
                           attn_mask=tgt_sub_mask, key_padding_mask=tgt_key_padding_mask)
        x = x + self.drop1(h)
        h2 = self.cross_attn(self.ln2(x), mem, mem,
                             attn_mask=None, key_padding_mask=src_key_padding_mask)
        x = x + self.drop2(h2)
        h3 = self.ffn(self.ln3(x))
        x = x + self.drop3(h3)
        return x


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int
    nhead: int
    dim_ff: int
    num_encoder_layers: int
    num_decoder_layers: int
    dropout: float = 0.1
    share_embeddings: bool = True
    tie_softmax: bool = True
    pad_id: int = 0
    max_src_len: int = 128
    max_tgt_len: int = 128


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.src_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        if cfg.share_embeddings:
            self.tgt_embed = self.src_embed
        else:
            self.tgt_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)

        self.pos = PositionalEncoding(cfg.d_model, max_len=4096)
        self.enc_layers = nn.ModuleList([EncoderLayer(cfg.d_model, cfg.nhead, cfg.dim_ff, cfg.dropout)
                                         for _ in range(cfg.num_encoder_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(cfg.d_model, cfg.nhead, cfg.dim_ff, cfg.dropout)
                                         for _ in range(cfg.num_decoder_layers)])
        self.ln_enc = nn.LayerNorm(cfg.d_model)
        self.ln_dec = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_softmax:
            self.out_proj.weight = self.tgt_embed.weight

    def encode(self, src_ids, src_key_padding_mask):
        x = self.pos(self.src_embed(src_ids))
        for layer in self.enc_layers:
            x = layer(x, src_key_padding_mask)
        return self.ln_enc(x)

    def decode(self, tgt_in, mem, tgt_sub_mask, tgt_key_padding_mask, src_key_padding_mask):
        x = self.pos(self.tgt_embed(tgt_in))
        for layer in self.dec_layers:
            x = layer(x, mem, tgt_sub_mask, tgt_key_padding_mask, src_key_padding_mask)
        x = self.ln_dec(x)
        return self.out_proj(x)

    @torch.no_grad()
    def generate_greedy(
        self,
        src_ids: torch.Tensor,
        src_pad_mask: torch.Tensor,
        lang_tag_id: int,
        bos_id: int,
        eos_id: int,
        max_len: int = 128,
    ) -> torch.Tensor:
        B = src_ids.size(0)
        mem = self.encode(src_ids, src_pad_mask)
        ys = torch.full((B, 2), bos_id, dtype=torch.long, device=src_ids.device)
        ys[:, 0] = lang_tag_id  # [<2xx>, <bos>]
        for _ in range(max_len):
            L = ys.size(1)
            sub_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=src_ids.device), diagonal=1)
            tgt_pad_mask = torch.zeros((B, L), dtype=torch.bool, device=src_ids.device)
            logits = self.decode(ys, mem, sub_mask, tgt_pad_mask, src_pad_mask)
            next_token = logits[:, -1, :].argmax(-1)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            if (next_token == eos_id).all():
                break
        return ys
