"""Weight-only int8 quantization in pure PyTorch (no deprecated ``torch.ao`` API, no extra dependency).

Each ``nn.Linear`` inside the Transformer blocks is replaced by :class:`Int8Linear`: the weights are stored as int8
with a float32 scale per output channel (or per group of input channels, which is more accurate when a row has
outliers) and are dequantized on the fly in every forward pass. Activations stay in floating point.

What you get: roughly 4x smaller linear weights in memory and on disk (embeddings, norms and biases stay float, so the
total shrinks less on models whose vocabulary is large). What you do not get: speed. Dequantizing on the fly is not
faster than float matmuls on CPU. The result is **inference only** (the int8 weights have no gradients), and it is
lossy: use :func:`quantization_report` to measure the damage on your own text before you rely on a quantized model.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

QUANT_FILE = "quantization.json"
SCHEME = "int8-weight-only"


class Int8Linear(nn.Module):
    """``nn.Linear`` with int8 weights ``(out, in)`` and float32 scales ``(out, in // group_size)``."""

    qweight: torch.Tensor
    scale: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self, in_features: int, out_features: int, bias: bool, group_size: int | None = None
    ) -> None:
        super().__init__()
        group = in_features if group_size is None else group_size
        if in_features % group:
            raise ValueError(f"group_size {group} does not divide in_features {in_features}")
        self.in_features, self.out_features, self.group_size = in_features, out_features, group
        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features, in_features // group, dtype=torch.float32))
        self.register_buffer("bias", torch.zeros(out_features) if bias else None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int | None = None) -> Int8Linear:
        out = cls(linear.in_features, linear.out_features, linear.bias is not None, group_size)
        w = linear.weight.detach().float()
        g = out.group_size
        grouped = w.view(out.out_features, -1, g)
        scale = grouped.abs().amax(dim=-1) / 127.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))  # an all-zero group stays exactly zero
        q = torch.round(grouped / scale[..., None]).clamp_(-127, 127)  # symmetric: -128 is never used
        out.qweight.copy_(q.view_as(w).to(torch.int8))
        out.scale.copy_(scale)
        if linear.bias is not None:
            assert out.bias is not None
            out.bias.copy_(linear.bias.detach())
        return out.to(linear.weight.device)

    def dequantize_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        w = (
            self.qweight.view(self.out_features, -1, self.group_size).to(dtype)
            * self.scale.to(dtype)[..., None]
        )
        return w.view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.dequantize_weight(x.dtype), bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, group_size={self.group_size}, int8"


def is_quantized(model: nn.Module) -> bool:
    return any(isinstance(m, Int8Linear) for m in model.modules())


def _set_module(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, new)


def quantize_int8(
    model: Any,
    group_size: int | None = None,
    include_lm_head: bool = False,
    skip: Iterable[str] = (),
    inplace: bool = False,
) -> Any:
    """Return a copy of ``model`` whose block linears are int8 (``inplace=True`` modifies ``model`` instead).

    ``group_size=None`` uses one scale per output channel; a smaller group (for example 64) costs a little more
    memory for scales and is more accurate when a weight row has outliers. The output projection (``lm_head``) is
    only quantized with ``include_lm_head=True`` and only when it is not tied to the embedding (quantizing a tied
    matrix would silently untie it). ``skip`` lists module names to leave in float. Embeddings, norms and biases always
    stay float. The returned model has ``quantization_info`` describing what was done.
    """
    if is_quantized(model):
        raise ValueError("the model is already quantized")
    cfg = model.config
    if include_lm_head and cfg.tie_embeddings:
        raise ValueError(
            "include_lm_head needs tie_embeddings=False (the output matrix is shared with the embedding)"
        )
    skip = set(skip)
    if group_size is not None:
        widths = {
            m.in_features
            for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and (n.startswith("blocks.") or (include_lm_head and n == "lm_head"))
        }
        if group_size < 1 or any(w % group_size for w in widths):
            common = math.gcd(*widths) if widths else 1
            raise ValueError(
                f"group_size {group_size} must divide the input width of every quantized layer "
                f"(widths {sorted(widths)}); the largest size that fits all of them is {common}, or any divisor of it"
            )
    target = model if inplace else copy.deepcopy(model)
    done: list[str] = []
    for name, module in list(target.named_modules()):
        wanted = name.startswith("blocks.") or (include_lm_head and name == "lm_head")
        if isinstance(module, nn.Linear) and wanted and name not in skip:
            _set_module(target, name, Int8Linear.from_linear(module, group_size))
            done.append(name)
    if not done:
        raise ValueError("no linear layers were quantized")
    target.quantization_info = {"scheme": SCHEME, "group_size": group_size, "modules": done}
    return target


def apply_quantized_structure(model: Any, info: dict[str, Any]) -> Any:
    """Swap the listed linears for empty :class:`Int8Linear` modules (their buffers come from a checkpoint)."""
    if info.get("scheme") != SCHEME:
        raise ValueError(f"unknown quantization scheme {info.get('scheme')!r}")
    for name in info["modules"]:
        lin = model.get_submodule(name)
        if not isinstance(lin, nn.Linear):
            raise ValueError(f"{name} is not a linear layer in this model")
        _set_module(
            model,
            name,
            Int8Linear(lin.in_features, lin.out_features, lin.bias is not None, info["group_size"]),
        )
    model.quantization_info = dict(info)
    return model


def write_quantization_info(model: Any, path: str | Path) -> None:
    info = getattr(model, "quantization_info", None)
    if info is not None:
        (Path(path) / QUANT_FILE).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def read_quantization_info(path: str | Path) -> dict[str, Any] | None:
    f = Path(path) / QUANT_FILE
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def model_nbytes(model: nn.Module) -> int:
    """Bytes held by parameters and buffers (shared storage counted once)."""
    seen: set[int] = set()
    total = 0
    for t in [*model.parameters(), *model.buffers()]:
        if t is not None and t.untyped_storage().data_ptr() not in seen:
            seen.add(t.untyped_storage().data_ptr())
            total += t.numel() * t.element_size()
    return total


@torch.no_grad()
def quantization_report(
    reference: Any, quantized: Any, tokens: torch.Tensor | list[int], block_size: int | None = None
) -> dict[str, Any]:
    """Measure what quantization changed, on ``tokens`` (your own text, tokenized).

    Returns the size of both models, per-layer weight error, and on the text: perplexity of each model, the
    mean KL divergence from the reference's next-token distribution to the quantized one, the share of positions
    where the most likely token is unchanged, and the largest logit difference.
    """
    from .evaluation.perplexity import perplexity

    ids = torch.as_tensor(tokens).long().flatten()
    if ids.numel() < 2:
        raise ValueError("need at least one full window of tokens (two or more tokens)")
    block = min(block_size or reference.config.max_seq_len, reference.config.max_seq_len, ids.numel())
    n_win = ids.numel() // block
    if n_win < 1:
        raise ValueError("need at least one full window of tokens")
    x = ids[: n_win * block].view(n_win, block)
    dev = next(reference.parameters()).device
    was = (reference.training, quantized.training)
    reference.eval()
    quantized.eval()
    kl_sum, top1, count = 0.0, 0, 0
    max_diff = 0.0
    for i in range(0, n_win, 8):
        batch = x[i : i + 8].to(dev)
        a, b = reference(batch).float(), quantized(batch).float()
        max_diff = max(max_diff, float((a - b).abs().max()))
        la, lb = F.log_softmax(a, -1), F.log_softmax(b, -1)
        kl_sum += float((la.exp() * (la - lb)).sum(-1).sum())
        top1 += int((a.argmax(-1) == b.argmax(-1)).sum())
        count += batch.numel()
    layer_err = {}
    for name, m in quantized.named_modules():
        if isinstance(m, Int8Linear):
            ref = reference.get_submodule(name).weight.detach().float()
            diff = ref - m.dequantize_weight(torch.float32)
            layer_err[name] = float(diff.norm() / ref.norm().clamp_min(1e-12))
    p_ref = perplexity(reference, x.flatten(), block_size=block)
    p_q = perplexity(quantized, x.flatten(), block_size=block)
    reference.train(was[0])
    quantized.train(was[1])
    return {
        "bytes_reference": model_nbytes(reference),
        "bytes_quantized": model_nbytes(quantized),
        "size_ratio": model_nbytes(quantized) / model_nbytes(reference),
        "weight_relative_error_mean": sum(layer_err.values()) / max(len(layer_err), 1),
        "weight_relative_error_max": max(layer_err.values(), default=0.0),
        "tokens_evaluated": count,
        "perplexity_reference": p_ref.perplexity,
        "perplexity_quantized": p_q.perplexity,
        "perplexity_change_percent": 100.0 * (p_q.perplexity / p_ref.perplexity - 1.0),
        "kl_mean_nats": kl_sum / count,
        "top1_agreement": top1 / count,
        "max_abs_logit_diff": max_diff,
        "group_size": quantized.quantization_info["group_size"]
        if hasattr(quantized, "quantization_info")
        else None,
        "quantized_layers": len(layer_err),
        "log2_vocab": math.log2(reference.config.vocab_size),
    }
