"""Datasets for language modelling on a token stream."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class TokenBlockDataset(Dataset):
    """Sliding windows of ``block_size`` tokens over a 1-D id sequence.

    Items are ``{"input_ids": ids, "labels": ids}``; the next-token shift is done
    inside the model's loss. ``stride`` defaults to ``block_size`` (no overlap).
    """

    def __init__(self, token_ids: list[int] | torch.Tensor, block_size: int, stride: int | None = None):
        self.ids = torch.as_tensor(token_ids, dtype=torch.long)
        if self.ids.dim() != 1 or len(self.ids) < block_size:
            raise ValueError("token_ids must be 1-D and at least block_size long")
        self.block_size, self.stride = block_size, stride or block_size

    def __len__(self) -> int:
        return (len(self.ids) - self.block_size) // self.stride + 1

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        chunk = self.ids[i * self.stride : i * self.stride + self.block_size]
        return {"input_ids": chunk, "labels": chunk.clone()}


def make_corpus(n_sentences: int = 1500, seed: int = 0) -> str:
    """A small synthetic 'world' with a fixed grammar (no external files needed).

    Sentences are random combinations, so a held-out slice contains *new* sentences that follow the
    same rules: validation loss measures whether the model learned the grammar.
    """
    import random

    rng = random.Random(seed)
    adj = ["small", "red", "quiet", "old", "bright", "clever"]
    noun = ["cat", "robot", "river", "farmer", "dragon", "student"]
    verb = ["sees", "follows", "paints", "builds", "finds", "greets"]
    return " ".join(
        f"the {rng.choice(adj)} {rng.choice(noun)} {rng.choice(verb)} the {rng.choice(adj)} {rng.choice(noun)}."
        for _ in range(n_sentences)
    )
