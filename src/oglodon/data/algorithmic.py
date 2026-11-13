"""Synthetic algorithmic tasks from the original MiniGPT notebook.

Every example is ``prompt + answer + <eos>`` and the loss covers the answer
(and the ``<eos>``) only.

* ``add``:  ``A321+654=`` -> ``9750``  (operands and sum written least-significant
  digit first, sum zero-padded to ``n+1`` digits: 123 + 456 = 579 -> "9750")
* ``sort``: ``S3142>``    -> ``1234``
* ``reverse``: ``R1234>`` -> ``4321``
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from ..tokenization.tokenizer import CharTokenizer

TASKS = ("add", "sort", "reverse")
DEFAULT_LENGTHS = {"add": (1, 4), "sort": (2, 8), "reverse": (2, 8)}


def algorithmic_tokenizer() -> CharTokenizer:
    """Vocabulary used by all tasks (``<pad>=0``, ``<eos>=1``, then characters)."""
    return CharTokenizer("0123456789+=>ASR")


def make_example(task: str, n: int, rng: random.Random) -> tuple[str, str]:
    """Return ``(prompt, answer)`` for ``n`` digits of ``task``."""
    digits = "0123456789"
    if task == "add":
        a = "".join(rng.choice(digits) for _ in range(n))
        b = "".join(rng.choice(digits) for _ in range(n))
        return f"A{a[::-1]}+{b[::-1]}=", str(int(a) + int(b)).zfill(n + 1)[::-1]
    s = "".join(rng.choice(digits) for _ in range(n))
    if task == "sort":
        return f"S{s}>", "".join(sorted(s))
    if task == "reverse":
        return f"R{s}>", s[::-1]
    raise ValueError(f"unknown task {task!r}; choose from {TASKS}")


class AlgorithmicDataset(Dataset):
    """Deterministic dataset: example ``i`` depends only on ``(seed, i)``.

    ``task_weights`` (e.g. ``{"add": 2}``) changes how often each task is sampled (default: uniform).
    """

    def __init__(
        self,
        size: int,
        tokenizer: CharTokenizer | None = None,
        tasks=TASKS,
        lengths: dict[str, tuple[int, int]] | None = None,
        seed: int = 0,
        task_weights: dict[str, float] | None = None,
    ) -> None:
        self.size, self.seed, self.tasks = size, seed, tuple(tasks)
        self.weights = [(task_weights or {}).get(t, 1.0) for t in self.tasks]
        self.tok = tokenizer or algorithmic_tokenizer()
        self.lengths = {**DEFAULT_LENGTHS, **(lengths or {})}

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        rng = random.Random(self.seed * 1_000_003 + i)
        task = rng.choices(self.tasks, weights=self.weights)[0]
        lo, hi = self.lengths[task]
        prompt, answer = make_example(task, rng.randint(lo, hi), rng)
        p, a = self.tok.encode(prompt), self.tok.encode(answer) + [self.tok.eos_token_id]
        ids = torch.tensor(p + a)
        labels = torch.tensor([-100] * len(p) + a)
        return {"input_ids": ids, "labels": labels}


def collate_pad(batch: list[dict[str, torch.Tensor]], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """Right-pad ``input_ids`` with ``pad_id`` and ``labels`` with ``-100``."""
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    mask = torch.zeros(len(batch), L, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n], labels[i, :n], mask[i, :n] = b["input_ids"], b["labels"], 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


@torch.no_grad()
def exact_match(
    model, task: str, n: int, num_examples: int = 100, seed: int = 123, tokenizer: CharTokenizer | None = None
) -> float:
    """Fraction of ``num_examples`` random ``n``-digit problems solved exactly (greedy)."""
    tok = tokenizer or algorithmic_tokenizer()
    rng = random.Random(seed)
    pairs = [make_example(task, n, rng) for _ in range(num_examples)]
    device = next(model.parameters()).device
    prompts = torch.tensor([tok.encode(p) for p, _ in pairs], device=device)  # equal length
    out = model.generate(
        prompts, max_new_tokens=n + 2, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
    )
    ok = 0
    for row, (_, answer) in zip(out[:, prompts.shape[1] :].tolist(), pairs, strict=True):
        end = row.index(tok.eos_token_id) if tok.eos_token_id in row else len(row)
        ok += tok.decode(row[:end]) == answer
    return ok / num_examples
