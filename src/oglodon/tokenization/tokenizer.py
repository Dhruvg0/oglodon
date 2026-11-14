"""Tokenizers.

Oglodon models only see integer ids, so any object with ``encode``/``decode``/
``vocab_size`` works. ``CharTokenizer`` is the built-in option (the one from the
original MiniGPT notebook); ``HFTokenizerAdapter`` wraps Hugging Face tokenizers
without importing them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal interface Oglodon expects from a tokenizer."""

    vocab_size: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Iterable[int]) -> str: ...


class CharTokenizer:
    """Character-level tokenizer with explicit special tokens.

    Ids ``0..len(specials)-1`` are the special tokens (default ``<pad>=0``,
    ``<eos>=1``); characters follow in the given order.
    """

    def __init__(self, chars: Iterable[str], special_tokens: Iterable[str] = ("<pad>", "<eos>")):
        self.special_tokens = list(special_tokens)
        self.chars = list(dict.fromkeys(chars))  # unique, order-preserving
        if any(len(c) != 1 for c in self.chars):
            raise ValueError("chars must be single characters")
        self.itos = self.special_tokens + self.chars
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        if len(self.stoi) != len(self.itos):
            raise ValueError("duplicate token between special_tokens and chars")

    @classmethod
    def from_text(cls, text: str, **kw: Any) -> CharTokenizer:
        return cls(sorted(set(text)), **kw)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_token_id(self) -> int | None:
        return self.stoi.get("<pad>")

    @property
    def eos_token_id(self) -> int | None:
        return self.stoi.get("<eos>")

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[c] for c in text]
        except KeyError as e:
            raise ValueError(f"character {e.args[0]!r} is not in the vocabulary") from None

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        out = []
        for i in ids:
            tok = self.itos[int(i)]
            if skip_special and tok in self.special_tokens:
                continue
            out.append(tok)
        return "".join(out)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"type": "char", "chars": self.chars, "special_tokens": self.special_tokens}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "char":
            raise ValueError("not a CharTokenizer file")
        return cls(data["chars"], data["special_tokens"])


class HFTokenizerAdapter:
    """Wrap a Hugging Face tokenizer (``transformers`` or ``tokenizers``).

    ``hf_tokenizer`` only needs ``encode`` (returning a list of ids or an object
    with ``.ids``), ``decode`` and a size (``len(tok)`` or ``get_vocab_size()``).
    Install with ``pip install "oglodon-transformer[hf]"``. Use
    ``HFTokenizerAdapter.from_pretrained("gpt2")`` to download one.
    """

    def __init__(self, hf_tokenizer: Any) -> None:
        self.tokenizer = hf_tokenizer
        if hasattr(hf_tokenizer, "get_vocab_size"):
            self.vocab_size = int(hf_tokenizer.get_vocab_size())
        else:
            self.vocab_size = len(hf_tokenizer)
        self.eos_token_id = getattr(hf_tokenizer, "eos_token_id", None)
        self.pad_token_id = getattr(hf_tokenizer, "pad_token_id", None)

    @classmethod
    def from_pretrained(cls, name: str) -> HFTokenizerAdapter:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise ImportError('Install the extra: pip install "oglodon-transformer[hf]"') from e
        return cls(AutoTokenizer.from_pretrained(name))

    def encode(self, text: str) -> list[int]:
        out = self.tokenizer.encode(text)
        return list(getattr(out, "ids", out))

    def decode(self, ids: Iterable[int]) -> str:
        return self.tokenizer.decode([int(i) for i in ids])


def load_tokenizer(path: str | Path) -> Any:
    """Load ``tokenizer.json`` from a checkpoint directory (char or BPE tokenizers); ``None`` if absent."""
    f = Path(path) / "tokenizer.json"
    if not f.exists():
        return None
    kind = json.loads(f.read_text(encoding="utf-8")).get("type")
    if kind == "bpe":
        from .bpe import BPETokenizer

        return BPETokenizer.load(f)
    return CharTokenizer.load(f)
