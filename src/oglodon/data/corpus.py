"""From a folder of text files to a memory-mapped token file and training windows.

    files = collect_text_files("my_corpus/")                 # a file, a directory or a glob, sorted
    tok = BPETokenizer.train(read_texts(files), vocab_size=2048)
    tf = tokenize_to_bin(files, tok, "corpus.bin")           # streams file by file, writes corpus.bin + corpus.json
    train, val = tf.dataset(block_size=256, split="train"), tf.dataset(block_size=256, split="val")

The token file is a flat ``uint16`` (or ``uint32`` for vocabularies above 65,535) array, opened with
``numpy.memmap``, so corpora larger than RAM work. The train/validation split is a single cut position
(the last ``val_fraction`` of the token stream), so the two never overlap.
"""

from __future__ import annotations

import glob as _glob
import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

TEXT_SUFFIXES = (
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".js",
    ".ts",
    ".c",
    ".h",
    ".cpp",
    ".java",
    ".go",
    ".rs",
    ".json",
    ".csv",
    ".tex",
)


def collect_text_files(
    source: str | Path | Iterable[str | Path], suffixes: Iterable[str] = TEXT_SUFFIXES
) -> list[Path]:
    """Resolve a file, directory (recursive, filtered by ``suffixes``), glob pattern or list of those to a sorted list."""
    suffixes = tuple(s.lower() for s in suffixes)
    items = [source] if isinstance(source, str | Path) else list(source)
    found: set[Path] = set()
    for item in items:
        p = Path(item)
        if p.is_dir():
            found.update(
                q
                for q in p.rglob("*")
                if q.is_file() and q.suffix.lower() in suffixes and ".git" not in q.parts
            )
        elif p.is_file():
            found.add(p)
        else:
            found.update(Path(q) for q in _glob.glob(str(item), recursive=True) if Path(q).is_file())
    if not found:
        raise FileNotFoundError(f"no text files found for {source!r}")
    return sorted(found)


def read_texts(files: Iterable[str | Path]) -> Iterator[str]:
    """Yield each file's text (UTF-8, undecodable bytes replaced) one at a time."""
    for f in files:
        yield Path(f).read_text(encoding="utf-8", errors="replace")


@dataclass
class TokenFile:
    """A tokenised corpus on disk (``<name>.bin`` + ``<name>.json``)."""

    path: Path
    meta: dict[str, Any]

    @classmethod
    def open(cls, path: str | Path) -> TokenFile:
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(path, meta)

    @property
    def vocab_size(self) -> int:
        return int(self.meta["vocab_size"])

    def __len__(self) -> int:
        return int(self.meta["n_tokens"])

    def array(self) -> np.memmap:
        return np.memmap(self.path, dtype=np.dtype(self.meta["dtype"]), mode="r", shape=(len(self),))

    def split_point(self, val_fraction: float = 0.1) -> int:
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        return int(len(self) * (1.0 - val_fraction))

    def dataset(
        self, block_size: int, split: str = "all", val_fraction: float = 0.1, stride: int | None = None
    ) -> WindowDataset:
        """Windows of ``block_size`` tokens from the ``"train"``, ``"val"`` or ``"all"`` part of the stream."""
        cut = self.split_point(val_fraction)
        lo, hi = {"all": (0, len(self)), "train": (0, cut), "val": (cut, len(self))}[split]
        return WindowDataset(self.array(), lo, hi, block_size, stride)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]


class WindowDataset(Dataset):
    """Fixed windows over ``tokens[lo:hi]`` read lazily from a memmap; items are ``{"input_ids", "labels"}``."""

    def __init__(
        self, tokens: np.ndarray, lo: int, hi: int, block_size: int, stride: int | None = None
    ) -> None:
        if hi - lo < block_size:
            raise ValueError(f"split has {hi - lo} tokens, fewer than block_size={block_size}")
        self.tokens, self.lo, self.hi, self.block_size = tokens, lo, hi, block_size
        self.stride = stride or block_size

    def __len__(self) -> int:
        return (self.hi - self.lo - self.block_size) // self.stride + 1

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        if not 0 <= i < len(self):
            raise IndexError(i)
        s = self.lo + i * self.stride
        chunk = torch.from_numpy(np.asarray(self.tokens[s : s + self.block_size], dtype=np.int64))
        return {"input_ids": chunk, "labels": chunk.clone()}


_WORKER_TOKENIZER: Any = None


def _init_worker(tokenizer: Any) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = tokenizer  # sent once per worker process, not once per file


def _encode_file(path: str) -> tuple[list[int], int, str]:
    """Worker: read and encode one file -> (ids, byte count, sha256). Top-level so it can be pickled."""
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8", errors="replace")
    return _WORKER_TOKENIZER.encode(text), len(raw), hashlib.sha256(raw).hexdigest()


def tokenize_to_bin(
    files: Iterable[str | Path],
    tokenizer: Any,
    out_path: str | Path,
    eos_between_files: bool = True,
    workers: int = 1,
) -> TokenFile:
    """Tokenise ``files`` (one at a time, so memory stays flat) into ``out_path`` and a ``.json`` sidecar.

    An end-of-text token is inserted between files when the tokenizer has one. The sidecar records the
    vocabulary size, token count, dtype and the file list with SHA-256 hashes, so a run can be reproduced.

    ``workers > 1`` encodes files in parallel processes (the tokenizer must be picklable; ``BPETokenizer`` and
    ``CharTokenizer`` are). The output is byte-identical to ``workers=1``: parallelism is across files only,
    never inside one file, because cutting a file could change how a word at the cut is tokenised.
    Worker processes are *spawned*, so call this under ``if __name__ == "__main__":`` in scripts (standard
    Python multiprocessing rule); the default ``workers=1`` needs no guard.
    """
    files = [Path(f) for f in files]
    vocab = int(tokenizer.vocab_size)
    dtype = np.uint16 if vocab <= 65536 else np.uint32
    out_path = Path(out_path)
    eos = getattr(tokenizer, "eos_token_id", None)
    n, sources = 0, []
    if workers < 1:
        raise ValueError("workers must be >= 1")
    jobs = [str(f) for f in files]
    pool = None
    encoded: Iterable[tuple[list[int], int, str]]
    if workers == 1 or len(files) < 2:
        _init_worker(tokenizer)
        encoded = map(_encode_file, jobs)  # lazily: one file in memory at a time
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # "spawn", not fork: forking a multi-threaded process (torch) can deadlock and is deprecated in 3.12+.
        pool = ProcessPoolExecutor(
            max_workers=min(workers, len(files)),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(tokenizer,),
        )
        encoded = pool.map(_encode_file, jobs, chunksize=max(1, len(jobs) // (workers * 4)))
    try:
        with out_path.open("wb") as fh:
            for k, (f, (ids, nbytes, sha)) in enumerate(zip(files, encoded, strict=True)):
                if eos_between_files and eos is not None and k < len(files) - 1:
                    ids = [*ids, eos]
                arr = np.asarray(ids, dtype=np.int64)
                if arr.size and (arr.min() < 0 or arr.max() >= vocab):
                    raise ValueError(f"{f}: tokenizer produced ids outside [0, {vocab})")
                arr.astype(dtype).tofile(fh)
                n += arr.size
                sources.append({"path": str(f), "bytes": nbytes, "sha256": sha, "tokens": int(arr.size)})
    finally:
        if pool is not None:
            pool.shutdown()
    meta = {
        "vocab_size": vocab,
        "n_tokens": n,
        "dtype": np.dtype(dtype).name,
        "files": sources,
        "eos_between_files": eos_between_files,
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return TokenFile(out_path, meta)
