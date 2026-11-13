"""Compare architecture variants (measured numbers vs. analytic estimates)."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch

from ..config import OglodonConfig
from ..models.gpt import Oglodon


@dataclass
class ArchitectureResult:
    """One row of a comparison.

    MEASURED on this machine: ``num_params``, ``forward_ms``, ``generate_ms_per_token``,
    ``peak_memory_mb`` (None unless CUDA), ``measured_kv_cache_bytes``.
    ESTIMATED analytically (not timed): ``estimated_kv_cache_bytes``.
    """

    name: str
    attention: str
    num_params: int
    forward_ms: float
    generate_ms_per_token: float
    peak_memory_mb: float | None
    measured_kv_cache_bytes: int
    estimated_kv_cache_bytes: int


def estimate_kv_cache_bytes(cfg: OglodonConfig, batch_size: int, seq_len: int, bytes_per_el: int = 4) -> int:
    """Theory: ``2 (K and V) * layers * batch * kv_heads * seq * head_dim * bytes``."""
    return 2 * cfg.num_layers * batch_size * cfg.num_kv_heads * seq_len * cfg.head_dim * bytes_per_el


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median_ms(fn, repeats: int, device: torch.device) -> float:
    times = []
    for _ in range(repeats):
        _sync(device)
        t = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


@torch.no_grad()
def compare_architectures(
    configs: list[OglodonConfig],
    batch_size: int = 4,
    sequence_length: int = 128,
    new_tokens: int = 32,
    names: list[str] | None = None,
    repeats: int = 5,
    device: str | torch.device | None = None,
    seed: int = 0,
) -> list[ArchitectureResult]:
    """Time each config on random data (weights are random: only speed/memory matter).

    ``sequence_length`` is the prompt length for the forward pass; generation uses a prompt of
    the same length and decodes ``new_tokens`` more with the KV cache. Each config's
    ``max_seq_len`` must be at least ``sequence_length + new_tokens``.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    names = names or [f"{c.attention_type.lower()}-{i}" for i, c in enumerate(configs)]
    if len(names) != len(configs):
        raise ValueError("names and configs must have the same length")
    results = []
    for name, cfg in zip(names, configs, strict=True):
        if cfg.max_seq_len < sequence_length + new_tokens:
            raise ValueError(f"{name}: max_seq_len {cfg.max_seq_len} < {sequence_length + new_tokens}")
        torch.manual_seed(seed)
        results.append(_bench_one(name, cfg, batch_size, sequence_length, new_tokens, repeats, device))
    return results


@torch.no_grad()
def _bench_one(name, cfg, batch_size, sequence_length, new_tokens, repeats, device) -> ArchitectureResult:
    model = Oglodon(cfg).to(device).eval()
    x = torch.randint(0, cfg.vocab_size, (batch_size, sequence_length), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model(x)  # warm-up

    def run_forward():
        model(x)

    def run_generate():
        model.generate(x, max_new_tokens=new_tokens)

    fwd = _median_ms(run_forward, repeats, device)
    gen = _median_ms(run_generate, max(1, repeats // 2), device)
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    cache = model(x, use_cache=True).past_key_values
    return ArchitectureResult(
        name=name,
        attention=cfg.attention_type,
        num_params=model.num_parameters(),
        forward_ms=fwd,
        generate_ms_per_token=gen / new_tokens,
        peak_memory_mb=peak,
        measured_kv_cache_bytes=cache.nbytes(),
        estimated_kv_cache_bytes=estimate_kv_cache_bytes(cfg, batch_size, sequence_length),
    )


def format_results(results: list[ArchitectureResult]) -> str:
    """Plain-text table. Columns marked (est) are analytic, the rest were measured."""
    head = (
        f"{'name':<12}{'attn':<6}{'params':>10}{'fwd ms':>10}{'gen ms/tok':>12}"
        f"{'peak MB':>10}{'kv bytes':>12}{'kv (est)':>12}"
    )
    rows = [head, "-" * len(head)]
    for r in results:
        peak = "n/a" if r.peak_memory_mb is None else f"{r.peak_memory_mb:.1f}"
        rows.append(
            f"{r.name:<12}{r.attention:<6}{r.num_params:>10,}{r.forward_ms:>10.2f}"
            f"{r.generate_ms_per_token:>12.3f}{peak:>10}{r.measured_kv_cache_bytes:>12,}"
            f"{r.estimated_kv_cache_bytes:>12,}"
        )
    return "\n".join(rows)
