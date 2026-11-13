"""A small, readable byte-level BPE tokenizer you can train on your own text.

*Lossless*: text is first split into pre-tokens (words, numbers, punctuation runs, whitespace), each pre-token
is encoded as UTF-8 bytes, and merges are learned over byte symbols. Any string, in any language, encodes to
ids and decodes back to exactly the same string, so there is no "unknown token".

Training is the classic algorithm (count pair frequencies over the distinct pre-tokens, merge the most
frequent pair, update the counts incrementally). It is plain Python: fine for tens of megabytes of text, not
for terabytes (use ``tokenizers`` / ``HFTokenizerAdapter`` for those).
"""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

# Letters (with an optional leading space), digits, other symbols, then whitespace runs. Python's ``re`` has no
# \p{L}; ``[^\W\d_]`` is "a letter in any script".
# The last alternatives make the pattern exhaustive (``_`` is a "word" character but not a letter or digit):
# every character of the input lands in exactly one pre-token, which is what makes encoding lossless.
_PRETOKEN = re.compile(r" ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+|.", re.UNICODE | re.DOTALL)


class BPETokenizer:
    """Byte-level BPE with explicit special tokens (default ``<pad>`` = 0 and ``<eos>`` = 1).

    ids: ``[specials..., 256 byte symbols..., merged tokens...]``.
    """

    def __init__(
        self,
        merges: list[tuple[int, int]] | None = None,
        special_tokens: Iterable[str] = ("<pad>", "<eos>"),
    ) -> None:
        self.special_tokens = list(special_tokens)
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special tokens must be unique")
        self.n_special = len(self.special_tokens)
        self.merges: list[tuple[int, int]] = [tuple(m) for m in (merges or [])]  # type: ignore[misc]
        self._build()

    # ------------------------------------------------------------------ construction
    def _build(self) -> None:
        base = self.n_special
        self.vocab: dict[int, bytes] = {base + b: bytes([b]) for b in range(256)}
        self.rank: dict[tuple[int, int], int] = {}
        for i, (a, b) in enumerate(self.merges):
            if a not in self.vocab or b not in self.vocab:
                raise ValueError(f"merge {i} refers to an id that does not exist yet")
            new = base + 256 + i
            self.vocab[new] = self.vocab[a] + self.vocab[b]
            self.rank[(a, b)] = i
        self.special_ids = {t: i for i, t in enumerate(self.special_tokens)}
        self._special_re = (
            re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return self.n_special + 256 + len(self.merges)

    @property
    def pad_token_id(self) -> int | None:
        return self.special_ids.get("<pad>")

    @property
    def eos_token_id(self) -> int | None:
        return self.special_ids.get("<eos>")

    @classmethod
    def train(
        cls,
        texts: str | Iterable[str],
        vocab_size: int,
        special_tokens: Iterable[str] = ("<pad>", "<eos>"),
        min_frequency: int = 2,
    ) -> BPETokenizer:
        """Learn ``vocab_size - n_special - 256`` merges from ``texts`` (a string or an iterable of strings).

        Stops early when no pair occurs at least ``min_frequency`` times, so the result can have fewer ids
        than requested on small corpora (check ``tok.vocab_size``).
        """
        specials = list(special_tokens)
        base = len(specials)
        if vocab_size < base + 256:
            raise ValueError(f"vocab_size must be at least {base + 256} (specials + 256 byte symbols)")
        if isinstance(texts, str):
            texts = [texts]
        words: Counter[tuple[int, ...]] = Counter()
        for text in texts:
            for piece in _PRETOKEN.findall(text):
                words[tuple(base + b for b in piece.encode("utf-8"))] += 1
        word_list = [list(w) for w in words]
        freq = list(words.values())

        pair_counts: Counter[tuple[int, int]] = Counter()
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, w in enumerate(word_list):
            for pr in zip(w, w[1:], strict=False):
                pair_counts[pr] += freq[wi]
                where[pr].add(wi)

        # Max-heap (lazy deletion) over (-count, pair): picking the best pair is O(log P) instead of a scan
        # over every pair, which is what makes large vocabularies (tens of thousands of merges) practical.
        # Ties go to the numerically smallest pair, so training is deterministic.
        heap = [(-n, pr) for pr, n in pair_counts.items()]
        heapq.heapify(heap)
        merges: list[tuple[int, int]] = []
        while base + 256 + len(merges) < vocab_size and heap:
            neg, best = heapq.heappop(heap)
            n = pair_counts.get(best, 0)
            if n != -neg:  # stale entry: the count changed since it was pushed
                if n > 0:
                    heapq.heappush(heap, (-n, best))
                continue
            if n < min_frequency:
                break
            new_id = base + 256 + len(merges)
            merges.append(best)
            touched: set[tuple[int, int]] = set()
            for wi in list(where[best]):
                w, f = word_list[wi], freq[wi]
                for pr in zip(w, w[1:], strict=False):  # remove this word's old pairs
                    pair_counts[pr] -= f
                    touched.add(pr)
                out, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                word_list[wi] = out
                for pr in zip(out, out[1:], strict=False):  # add the new ones
                    pair_counts[pr] += f
                    where[pr].add(wi)
                    touched.add(pr)
            pair_counts.pop(best, None)
            where.pop(best, None)
            touched.discard(best)
            for pr in touched:
                c = pair_counts.get(pr, 0)
                if c <= 0:
                    pair_counts.pop(pr, None)
                else:
                    heapq.heappush(heap, (-c, pr))
        return cls(merges, specials)

    # ------------------------------------------------------------------ encode / decode
    def _encode_piece(self, piece: str) -> list[int]:
        hit = self._cache.get(piece)
        if hit is not None:
            return hit
        ids = [self.n_special + b for b in piece.encode("utf-8")]
        while len(ids) > 1:
            ranked = [(self.rank.get(pr, 1 << 60), i) for i, pr in enumerate(zip(ids, ids[1:], strict=False))]
            r, i = min(ranked)
            if r == 1 << 60:
                break
            ids[i : i + 2] = [self.n_special + 256 + r]
        if len(self._cache) < 100_000:
            self._cache[piece] = ids
        return ids

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """Text -> ids. Special-token strings in ``text`` (e.g. ``"<eos>"``) map to their ids."""
        out: list[int] = []
        parts = self._special_re.split(text) if self._special_re else [text]
        for part in parts:
            if part in self.special_ids:
                out.append(self.special_ids[part])
            elif part:
                for piece in _PRETOKEN.findall(part):
                    out.extend(self._encode_piece(piece))
        if add_eos and self.eos_token_id is not None:
            out.append(self.eos_token_id)
        return out

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        chunks: list[bytes] = []
        for i in ids:
            i = int(i)
            if i < self.n_special:
                if not skip_special:
                    chunks.append(self.special_tokens[i].encode("utf-8"))
            elif i in self.vocab:
                chunks.append(self.vocab[i])
            else:
                raise ValueError(f"id {i} is outside the vocabulary (size {self.vocab_size})")
        return b"".join(chunks).decode("utf-8", errors="replace")

    def token_str(self, i: int) -> str:
        """Human-readable form of one token (for debugging)."""
        if i < self.n_special:
            return self.special_tokens[i]
        return self.vocab[i].decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"type": "bpe", "special_tokens": self.special_tokens, "merges": self.merges})
        )

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        data = json.loads(Path(path).read_text())
        if data.get("type") != "bpe":
            raise ValueError("not a BPETokenizer file")
        return cls([tuple(m) for m in data["merges"]], data["special_tokens"])
