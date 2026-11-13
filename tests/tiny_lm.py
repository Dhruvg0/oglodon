"""A from-scratch causal LM that is NOT a Oglodon model, with switchable bugs, to test ``oglodon.verify``."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn


class _BadGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * 2.0

    @staticmethod
    def backward(ctx, g):
        return g * 3.0  # should be 2.0


class TinyLM(nn.Module):
    def __init__(self, vocab=40, d=32, heads=4, layers=2, max_len=64, bug=None, plain=False):
        super().__init__()
        self.bug, self.plain, self.heads = bug, plain, heads
        self.config = SimpleNamespace(vocab_size=vocab, max_position_embeddings=max_len)
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.qkv = nn.ModuleList(nn.Linear(d, 3 * d, bias=False) for _ in range(layers))
        self.out = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(layers))
        self.ff = nn.ModuleList(
            nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for _ in range(layers)
        )
        self.norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.head = nn.Linear(d, vocab, bias=False)
        if bug == "dead_param":
            self.unused = nn.Linear(d, d)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        labels=None,
        past_key_values=None,
        use_cache=False,
    ):
        B, T = input_ids.shape
        past_len = 0 if not past_key_values else past_key_values[0][0].shape[2]
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + T, device=input_ids.device).expand(B, T)
        x = self.tok(input_ids) + self.pos(position_ids)
        S = past_len + T
        i = torch.arange(T, device=x.device)[:, None] + past_len
        j = torch.arange(S, device=x.device)[None, :]
        allowed = (j <= i) if self.bug != "leak" else torch.ones(T, S, dtype=torch.bool, device=x.device)
        allowed = allowed[None, None].expand(B, 1, T, S)
        if attention_mask is not None and self.bug != "ignore_mask":
            km = attention_mask.bool()
            if km.shape[1] < S:
                km = torch.cat([torch.ones(B, S - km.shape[1], dtype=torch.bool, device=km.device), km], 1)
            allowed = allowed & km[:, None, None, :]
        new_past = []
        for n, (qkv, out, ff, norm) in enumerate(zip(self.qkv, self.out, self.ff, self.norm, strict=True)):
            h = norm(x)
            q, k, v = qkv(h).chunk(3, -1)
            sp = lambda t: t.view(B, -1, self.heads, t.shape[-1] // self.heads).transpose(1, 2)  # noqa: E731
            q, k, v = sp(q), sp(k), sp(v)
            if past_key_values:
                pk, pv = past_key_values[n]
                if self.bug == "cache_bug":
                    k, v = torch.cat([k, pk], 2), torch.cat([v, pv], 2)  # wrong order
                else:
                    k, v = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
            new_past.append((k, v))
            sc = (q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
            sc = sc.masked_fill(~allowed, torch.finfo(sc.dtype).min)
            a = sc.softmax(-1) @ v
            x = x + out(a.transpose(1, 2).reshape(B, T, -1))
            x = x + ff(x)
        if self.bug == "batch_leak":
            x = x + 0.5 * x.mean(0, keepdim=True)
        if self.bug == "bad_backward":
            x = _BadGrad.apply(x) / 2.0
        logits = self.head(x)
        if self.plain:
            return logits
        loss = None
        if labels is not None:
            if self.bug == "no_shift":
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
                )
            else:
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100
                )
        return SimpleNamespace(logits=logits, loss=loss, past_key_values=new_past if use_cache else None)
