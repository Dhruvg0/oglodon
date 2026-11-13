"""Datasets used by the examples and tests."""

from .algorithmic import (
    AlgorithmicDataset,
    algorithmic_tokenizer,
    collate_pad,
    exact_match,
    make_example,
)
from .corpus import TokenFile, WindowDataset, collect_text_files, read_texts, tokenize_to_bin
from .text import TokenBlockDataset, make_corpus

__all__ = [
    "AlgorithmicDataset",
    "TokenBlockDataset",
    "TokenFile",
    "WindowDataset",
    "algorithmic_tokenizer",
    "collate_pad",
    "collect_text_files",
    "exact_match",
    "make_corpus",
    "make_example",
    "read_texts",
    "tokenize_to_bin",
]
