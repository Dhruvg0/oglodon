"""Sanity checks on what a model generates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ._util import model_device


@dataclass
class GenerationReport:
    prompts: int
    new_tokens_per_prompt: float
    distinct_1: float  # unique tokens / tokens (diversity; 1.0 = no token repeated)
    distinct_2: float  # unique bigrams / bigrams
    repeated_3gram_fraction: float  # share of 3-grams that already occurred earlier in the same generation
    greedy_deterministic: bool  # two greedy runs give identical tokens
    cache_matches_full: (
        bool | None
    )  # greedy with KV cache == greedy without (None: model has no cache switch)
    samples: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _ngrams(seq: list[int], n: int) -> list[tuple[int, ...]]:
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


@torch.no_grad()
def check_generation(
    model: Any,
    prompts: list[list[int]],
    *,
    max_new_tokens: int = 32,
    device: str | torch.device | None = None,
    keep_samples: int = 3,
) -> GenerationReport:
    """Greedy-generate from each prompt and report diversity, repetition and consistency.

    ``prompts`` are token-id lists (equal length is not required: each is run on its own). The two consistency
    flags are exact comparisons of greedy outputs. Diversity numbers describe behaviour (a degenerate model
    loops and scores low); they are not a quality score.
    """
    if not prompts:
        raise ValueError("no prompts")
    dev = model_device(model, device)
    model.eval()
    gens: list[list[int]] = []
    det = True
    cache_ok: bool | None = None

    def run(p: list[int], **kw: Any) -> list[int]:
        x = torch.tensor([p], device=dev)
        out = model.generate(x, max_new_tokens=max_new_tokens, **kw)
        return out[0, len(p) :].tolist()

    for p in prompts:
        a = run(p)
        det = det and a == run(p)
        gens.append(a)
        try:
            b = run(p, use_cache=False)
        except TypeError:
            b = None
        if b is not None:
            cache_ok = (cache_ok is not False) and a == b
    toks = [t for g in gens for t in g]
    bi = [bg for g in gens for bg in _ngrams(g, 2)]
    rep = tot = 0
    for g in gens:
        seen: set[tuple[int, ...]] = set()
        for tg in _ngrams(g, 3):
            rep += tg in seen
            tot += 1
            seen.add(tg)
    return GenerationReport(
        prompts=len(prompts),
        new_tokens_per_prompt=len(toks) / len(prompts),
        distinct_1=len(set(toks)) / max(len(toks), 1),
        distinct_2=len(set(bi)) / max(len(bi), 1),
        repeated_3gram_fraction=rep / max(tot, 1),
        greedy_deterministic=det,
        cache_matches_full=cache_ok,
        samples=gens[:keep_samples],
    )
