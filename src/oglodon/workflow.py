"""End-to-end workflows behind ``oglodon train`` and ``oglodon generate``.

``train_from_files`` takes text files, trains a tokenizer, tokenises to a memory-mapped file, trains a model and
writes a checkpoint directory (weights, config, tokenizer, trainer state and a JSON report). Under ``torchrun`` it
trains data-parallel automatically.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch

from .config import OglodonConfig
from .data.corpus import TokenFile, collect_text_files, read_texts, tokenize_to_bin
from .models.gpt import Oglodon
from .tokenization import BPETokenizer, CharTokenizer, load_tokenizer

SETTINGS_FILE = "train_settings.json"
REPORT_FILE = "training_report.json"


@dataclass
class TrainSettings:
    """Everything that defines a training run (saved next to the checkpoint so a run can be resumed or repeated)."""

    # tokenizer and data
    tokenizer: str = "bpe"  # "bpe" or "char"
    vocab_size: int = 1024  # BPE only; the final size can be smaller on a small corpus
    tokenizer_sample_mb: float = 64.0  # at most this much training text is used to learn the BPE merges
    val_fraction: float = 0.05
    workers: int = 1  # parallel tokenisation processes
    # model
    block_size: int = 128  # context length (the model's max_seq_len)
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 8
    num_kv_heads: int | None = None  # default: num_heads // 4 (grouped-query attention) when divisible
    model: dict[str, Any] = field(
        default_factory=dict
    )  # any other OglodonConfig field, e.g. {"use_alibi": ...}
    # optimisation
    steps: int = 1000
    batch_size: int = 16  # per process
    grad_accum_steps: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    warmup_steps: int | None = None  # default: 5% of steps
    lr_schedule: str = "cosine"
    max_grad_norm: float | None = 1.0
    seed: int = 0
    precision: str = "fp32"
    gradient_checkpointing: bool = False
    compile: bool = False
    eval_interval: int | None = None  # default: ten evaluations per run
    save_interval: int | None = None  # default: only at the end

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainSettings:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"unknown training setting(s): {', '.join(unknown)}; valid: {', '.join(sorted(known))}"
            )
        return cls(**data)

    def merged(self, overrides: dict[str, Any]) -> TrainSettings:
        """A copy with every non-``None`` entry of ``overrides`` applied."""
        d = asdict(self)
        d.update({k: v for k, v in overrides.items() if v is not None})
        return TrainSettings.from_dict(d)


@dataclass
class TrainResult:
    out_dir: Path
    config: OglodonConfig
    steps: int
    train_loss: float
    val_loss: float | None
    val_perplexity: float | None
    train_tokens: int  # tokens in the training split
    tokens_seen: int
    tokens_per_second: float
    seconds: float
    resumed_from_step: int
    report_path: Path
    history: list[dict[str, float]]
    parameters: int = 0


def _split_for_tokenizer(texts: list[str], val_fraction: float, max_chars: int) -> Iterator[str]:
    """The texts the tokenizer may learn from: the first ``1 - val_fraction`` of the characters (validation text
    must not influence the tokenizer), capped at ``max_chars``."""
    total = sum(len(t) for t in texts)
    budget = min(int(total * (1.0 - val_fraction)), max_chars)
    used = 0
    for t in texts:
        if used >= budget:
            break
        piece = t[: budget - used]
        used += len(piece)
        yield piece


def _share_error(world_size: int, err: str | None) -> str | None:
    """Rank 0's error message (or ``None``), delivered to every rank so that nobody waits forever."""
    if world_size == 1:
        return err
    import torch.distributed as dist

    box: list[str | None] = [err]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def _prepare(data: Any, out: Path, s: TrainSettings, log: Callable[[str], None]) -> None:
    """Rank 0 only: files -> tokenizer -> ``out/data/corpus.bin``; saves ``tokenizer.json`` into ``out``."""
    files = collect_text_files(data)
    log(f"{len(files)} text file(s), {sum(f.stat().st_size for f in files) / 1e6:.2f} MB")
    texts = list(read_texts(files))
    if s.tokenizer == "bpe":
        sample = _split_for_tokenizer(texts, s.val_fraction, int(s.tokenizer_sample_mb * 1e6))
        t0 = time.perf_counter()
        tok: Any = BPETokenizer.train(sample, s.vocab_size)
        log(f"trained BPE tokenizer: {tok.vocab_size} tokens in {time.perf_counter() - t0:.1f}s")
    elif s.tokenizer == "char":
        tok = CharTokenizer.from_text("".join(texts))
        log(f"character tokenizer: {tok.vocab_size} symbols")
    else:
        raise ValueError(f"tokenizer must be 'bpe' or 'char', got {s.tokenizer!r}")
    del texts
    (out / "data").mkdir(parents=True, exist_ok=True)
    tok.save(out / "tokenizer.json")
    t0 = time.perf_counter()
    tf = tokenize_to_bin(files, tok, out / "data" / "corpus.bin", workers=s.workers)
    log(f"tokenised {len(tf):,} tokens in {time.perf_counter() - t0:.1f}s")


def train_from_files(
    data: str | Path | Sequence[str | Path],
    out: str | Path,
    settings: TrainSettings | None = None,
    *,
    device: str | None = None,
    resume: bool = False,
    overwrite: bool = False,
    logger: Callable[[str], None] | None = print,
) -> TrainResult:
    """Train a model on text files and write a complete checkpoint directory to ``out``.

    ``out`` receives ``model.safetensors`` + ``config.json`` + ``tokenizer.json`` (load with
    ``Oglodon.from_pretrained``), ``trainer_state.pt`` (for ``resume=True``), ``train_settings.json``,
    ``training_report.json`` and ``data/corpus.bin`` (+ sidecar with file hashes). With ``resume=True`` the saved
    settings, tokenizer and tokenised data are reused, and training continues exactly where it stopped.
    Raises ``FileNotFoundError`` for missing data and ``ValueError`` for bad settings or a too-small corpus.
    """
    from .training.distributed import distributed_loader, setup_distributed
    from .training.trainer import Trainer
    from .utils.reproducibility import seed_everything

    ctx = setup_distributed()
    out = Path(out)
    log: Callable[[str], None] = (logger or (lambda _m: None)) if ctx.is_main else (lambda _m: None)
    s = settings or TrainSettings()
    has_model = (out / "model.safetensors").exists()
    state_file = out / "trainer_state.pt"

    err: str | None = None
    if ctx.is_main:
        try:
            if resume:
                if not (has_model and state_file.exists() and (out / SETTINGS_FILE).exists()):
                    raise ValueError(f"nothing to resume in {out} (need model, trainer state and settings)")
                saved = TrainSettings.from_dict(json.loads((out / SETTINGS_FILE).read_text()))
                changed = {k: v for k, v in asdict(s).items() if v != asdict(TrainSettings())[k]}
                s = saved.merged(changed)
                if "steps" in changed and changed["steps"] != saved.steps:
                    log(
                        f"note: steps changed {saved.steps} -> {s.steps}; the learning-rate schedule stretches"
                    )
            elif has_model and not overwrite:
                raise ValueError(
                    f"{out} already contains a model; use resume=True / --resume, --overwrite, or another folder"
                )
            out.mkdir(parents=True, exist_ok=True)
            if not resume:
                _prepare(data, out, s, log)
            (out / SETTINGS_FILE).write_text(json.dumps(asdict(s), indent=2) + "\n")
        except (ValueError, FileNotFoundError, OSError) as e:
            err = f"{type(e).__name__}: {e}"
    err = _share_error(ctx.world_size, err)
    if err is not None:
        etype, _, msg = err.partition(": ")
        raise {"FileNotFoundError": FileNotFoundError, "OSError": OSError}.get(etype, ValueError)(msg)
    if not ctx.is_main:  # settings were written by rank 0 (possibly merged with saved ones on resume)
        s = TrainSettings.from_dict(json.loads((out / SETTINGS_FILE).read_text()))

    tok = load_tokenizer(out)
    tf = TokenFile.open(out / "data" / "corpus.bin")
    seed_everything(s.seed)
    if resume:
        model = Oglodon.from_pretrained(out)
        cfg = model.config
        block = cfg.max_seq_len
    else:
        block = s.block_size
        heads = s.num_heads
        kv = s.num_kv_heads or (heads // 4 if heads % 4 == 0 else heads)
        cfg = OglodonConfig(
            vocab_size=tok.vocab_size, hidden_size=s.hidden_size, num_layers=s.num_layers, num_heads=heads,
            num_kv_heads=kv, max_seq_len=block, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
            **s.model,
        )  # fmt: skip
        model = Oglodon(cfg)
    try:
        train_ds = tf.dataset(block, "train", s.val_fraction)
        val_ds = tf.dataset(block, "val", s.val_fraction)
    except ValueError as e:
        raise ValueError(
            f"the corpus is too small for block_size={block} ({len(tf):,} tokens, validation fraction "
            f"{s.val_fraction}): use more text, a smaller block_size or a smaller val_fraction ({e})"
        ) from e
    batch = s.batch_size
    if len(train_ds) < batch * ctx.world_size:
        batch = max(1, len(train_ds) // ctx.world_size)
        log(f"note: only {len(train_ds)} training windows; batch size lowered {s.batch_size} -> {batch}")
    gen = torch.Generator().manual_seed(s.seed)
    from torch.utils.data import DataLoader

    if ctx.world_size > 1:
        train_loader = distributed_loader(train_ds, batch, shuffle=True, seed=s.seed)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch, shuffle=True, generator=gen, drop_last=len(train_ds) >= batch
        )
    val_loader = DataLoader(val_ds, batch_size=max(1, min(batch, len(val_ds))))
    warmup = s.warmup_steps if s.warmup_steps is not None else max(1, s.steps // 20)
    trainer = Trainer(
        model, train_loader, val_loader, learning_rate=s.learning_rate, weight_decay=s.weight_decay,
        max_steps=s.steps, warmup_steps=warmup, lr_schedule=s.lr_schedule, grad_accum_steps=s.grad_accum_steps,
        max_grad_norm=s.max_grad_norm, eval_interval=s.eval_interval or max(1, s.steps // 10), eval_batches=50,
        log_interval=max(1, s.steps // 20), precision=s.precision, gradient_checkpointing=s.gradient_checkpointing,
        compile=s.compile, distributed=ctx.world_size > 1, device=ctx.device if device is None else device,
        checkpoint_dir=out, save_interval=s.save_interval or 0, logger=log,
    )  # fmt: skip
    start_step = 0
    if resume:
        trainer.load_checkpoint(out)
        start_step = trainer.step
        log(f"resuming from step {start_step}")
    n_params = sum(p.numel() for p in model.parameters())
    per_step = batch * s.grad_accum_steps * ctx.world_size * block
    log(
        f"{len(tf):,} tokens ({len(train_ds)} train / {len(val_ds)} val windows of {block}); vocab {cfg.vocab_size}; "
        f"{n_params / 1e6:.2f}M parameters; device {trainer.device}"
        f"{f' x {ctx.world_size} processes' if ctx.world_size > 1 else ''}; {per_step:,} tokens per step"
    )
    t0 = time.perf_counter()
    history = trainer.train()
    seconds = time.perf_counter() - t0 - trainer.eval_seconds
    steps_done = trainer.step - start_step
    trainer.save_checkpoint(out, tokenizer=tok)
    evals = [h for h in history if "val_loss" in h]
    trains = [h for h in history if "train_loss" in h]
    result = TrainResult(
        out_dir=out, config=cfg, steps=trainer.step,
        train_loss=trains[-1]["train_loss"] if trains else float("nan"),
        val_loss=evals[-1]["val_loss"] if evals else None,
        val_perplexity=evals[-1]["val_ppl"] if evals else None,
        train_tokens=len(train_ds) * block, tokens_seen=steps_done * per_step,
        tokens_per_second=steps_done * per_step / max(seconds, 1e-9), seconds=seconds,
        resumed_from_step=start_step, report_path=out / REPORT_FILE, history=history, parameters=n_params,
    )  # fmt: skip
    if ctx.is_main:
        _write_report(out, s, result, tf, tok, ctx.world_size, str(trainer.device))
        if result.val_loss is not None:
            log(
                f"done: validation loss {result.val_loss:.3f}, perplexity {result.val_perplexity:.1f} "
                f"(a model that guesses uniformly scores {cfg.vocab_size}); {result.tokens_per_second:,.0f} tokens/s"
            )
        log(f"checkpoint written to {out}")
    return result


def _write_report(
    out: Path, s: TrainSettings, r: TrainResult, tf: TokenFile, tok: Any, world: int, device: str
) -> None:
    import platform

    from . import __version__
    from .experiments.compare import _environment_metadata

    report = {
        "oglodon_version": __version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        **_environment_metadata(),
        "device": device,
        "processes": world,
        "settings": asdict(s),
        "config": r.config.to_dict(),
        "parameters": r.parameters,
        "data": {"tokens": len(tf), "fingerprint": tf.fingerprint(), "files": tf.meta.get("files", []),
                 "tokenizer": type(tok).__name__, "vocab_size": tok.vocab_size},
        "result": {
            "steps": r.steps, "resumed_from_step": r.resumed_from_step, "train_loss": r.train_loss,
            "val_loss": r.val_loss, "val_perplexity": r.val_perplexity,
            "uniform_baseline_perplexity": r.config.vocab_size, "tokens_seen": r.tokens_seen,
            "tokens_per_second": r.tokens_per_second, "train_seconds": r.seconds,
        },
        "history": r.history,
    }  # fmt: skip
    r.report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")


def complete(
    model: Oglodon,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = None,
    min_p: float | None = None,
    repetition_penalty: float = 1.0,
    stop: Sequence[str] = (),
    seed: int | None = None,
    device: str | torch.device | None = None,
) -> Iterator[str]:
    """Stream the continuation of ``prompt`` as text pieces (what ``oglodon generate`` prints).

    An empty prompt starts from the end-of-text token (or id 0). A prompt too long for the context window keeps its
    last tokens. ``stop`` strings are tokenised and matched as token sequences; the model's end-of-text token also
    stops generation. Raises ``ValueError`` for impossible settings.
    """
    from .generation.sampler import generate_stream, iter_text

    cfg = model.config
    if max_new_tokens < 1 or max_new_tokens >= cfg.max_seq_len:
        raise ValueError(
            f"max_new_tokens must be in [1, {cfg.max_seq_len - 1}] for a context of {cfg.max_seq_len}"
        )
    ids = list(tokenizer.encode(prompt)) if prompt else [tokenizer.eos_token_id or 0]
    ids = ids[-(cfg.max_seq_len - max_new_tokens) :]
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(dev).eval()
    stops = [list(tokenizer.encode(x)) for x in stop if x]
    gen = torch.Generator(device=dev).manual_seed(seed) if seed is not None else None
    stream = generate_stream(
        model, torch.tensor([ids], device=dev), max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k, top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty,
        stop_sequences=stops or None, generator=gen,
    )  # fmt: skip
    yield from iter_text(tokenizer, stream)
