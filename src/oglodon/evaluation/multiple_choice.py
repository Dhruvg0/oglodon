"""Log-likelihood multiple choice (the scoring used by HellaSwag / ARC / PIQA-style benchmarks)."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..verify.adapter import CausalLMAdapter
from ._util import model_device


def continuation_logprob(
    model: Any,
    context_ids: Sequence[int],
    continuation_ids: Sequence[int],
    *,
    device: str | torch.device | None = None,
) -> float:
    """Sum of log P(continuation | context) in nats. Over-long contexts are truncated from the left."""
    return _logprobs(model, [(list(context_ids), list(continuation_ids))], device)[0]


@torch.no_grad()
def _logprobs(model: Any, pairs: list[tuple[list[int], list[int]]], device: Any = None) -> list[float]:
    adapter = CausalLMAdapter(model)
    dev = model_device(model, device)
    limit = adapter.max_positions
    seqs: list[tuple[list[int], int]] = []  # (ids, number of leading context tokens)
    for ctx, cont in pairs:
        if not cont:
            raise ValueError("a choice tokenised to zero tokens")
        if not ctx:
            raise ValueError("an empty context cannot be scored (the first token has no prediction)")
        ids = ctx + cont
        if limit is not None and len(ids) > limit:
            if len(cont) >= limit:
                raise ValueError(f"a continuation of {len(cont)} tokens does not fit in {limit} positions")
            ids = ids[-limit:]
        seqs.append((ids, len(ids) - len(cont)))
    width = max(len(i) for i, _ in seqs)
    x = torch.zeros(len(seqs), width, dtype=torch.long)
    for r, (ids, _) in enumerate(seqs):
        x[r, : len(ids)] = torch.tensor(ids)  # right padding: causality keeps earlier positions unaffected
    was = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    logp = F.log_softmax(adapter.logits(x.to(dev)).float(), dim=-1)
    if hasattr(model, "train"):
        model.train(was)
    out = []
    for r, (ids, n_ctx) in enumerate(seqs):
        tgt = torch.tensor(ids[n_ctx:], device=dev)
        pos = torch.arange(n_ctx - 1, len(ids) - 1, device=dev)  # logits at t predict token t+1
        out.append(float(logp[r, pos, tgt].sum()))
    return out


@dataclass
class MultipleChoiceResult:
    n: int
    acc: float  # argmax of total log-likelihood
    acc_norm: float  # argmax of log-likelihood per character of the choice (as in lm-evaluation-harness)
    acc_token_norm: float  # per token
    stderr: float  # binomial standard error of ``acc``
    predictions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            k: getattr(self, k) for k in ("n", "acc", "acc_norm", "acc_token_norm", "stderr", "predictions")
        }


def evaluate_multiple_choice(
    model: Any,
    examples: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    device: str | torch.device | None = None,
    keep_predictions: bool = True,
) -> MultipleChoiceResult:
    """Score ``{"context": str, "choices": [str, ...], "label": int}`` examples by continuation likelihood.

    The context and every choice are tokenised separately (``tokenizer.encode``) and joined, so put a leading
    space in the choices if your text convention needs one. Chance level is ``1 / len(choices)``; a model that
    is not much above it has not learned the task, whatever the other metrics say.
    """
    if not examples:
        raise ValueError("no examples")
    hits = hits_norm = hits_tok = 0
    preds: list[dict[str, Any]] = []
    for i, ex in enumerate(examples):
        choices, label = ex["choices"], int(ex["label"])
        if len(choices) < 2 or not 0 <= label < len(choices):
            raise ValueError(f"example {i}: need >= 2 choices and a label inside them")
        ctx = list(tokenizer.encode(ex["context"]))
        conts = [list(tokenizer.encode(c)) for c in choices]
        lp = _logprobs(model, [(ctx, c) for c in conts], device)
        by_char = [lp[k] / max(len(choices[k]), 1) for k in range(len(choices))]
        by_tok = [lp[k] / len(conts[k]) for k in range(len(choices))]
        best, best_n, best_t = (max(range(len(choices)), key=v.__getitem__) for v in (lp, by_char, by_tok))
        hits += best == label
        hits_norm += best_n == label
        hits_tok += best_t == label
        if keep_predictions:
            preds.append({"index": i, "label": label, "pred": best, "logprobs": lp})
    n = len(examples)
    acc = hits / n
    return MultipleChoiceResult(n, acc, hits_norm / n, hits_tok / n, math.sqrt(acc * (1 - acc) / n), preds)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read one JSON object per line (blank lines ignored)."""
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON: {e}") from e
    return out
