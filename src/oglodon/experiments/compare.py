"""Controlled architecture comparison.

Every variant is trained on the *same* token windows in the *same* order (per seed), with the same
tokenizer, optimiser settings and number of training tokens, then evaluated on the same held-out
split. Alongside the numbers the report states which comparisons are not fair and why.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..config import OglodonConfig
from ..data.text import TokenBlockDataset
from ..models.gpt import Oglodon
from ..tokenization.tokenizer import CharTokenizer
from ..training.trainer import Trainer
from ..utils.reproducibility import get_device, seed_everything
from ..validation import validate_model
from ..validation.checks_model import expected_num_parameters
from ..validation.report import collect_metadata


@dataclass
class ExperimentData:
    """One tokenised corpus with a fixed train/validation split shared by every variant."""

    train_ids: list[int]
    val_ids: list[int]
    vocab_size: int
    block_size: int
    tokenizer: Any = None

    @classmethod
    def from_text(
        cls,
        text: str,
        block_size: int = 64,
        val_fraction: float = 0.1,
        tokenizer: Any = None,
        bpe_vocab_size: int | None = None,
    ) -> ExperimentData:
        """Tokenise ``text``; the last ``val_fraction`` becomes the validation split.

        * default: character-level (the vocabulary is built from the whole text; it only fixes the id map).
        * ``bpe_vocab_size=N``: train a byte-level BPE on the **training part only** (so no validation text
          influences the tokenizer), then encode both parts.
        * ``tokenizer=...``: use an existing tokenizer (anything with ``encode`` / ``vocab_size``).
        """
        if bpe_vocab_size is not None:
            if tokenizer is not None:
                raise ValueError("give either tokenizer or bpe_vocab_size, not both")
            from ..tokenization import BPETokenizer

            cut_chars = int(len(text) * (1 - val_fraction))
            bpe = BPETokenizer.train(text[:cut_chars], bpe_vocab_size)
            train_ids, val_ids = bpe.encode(text[:cut_chars]), bpe.encode(text[cut_chars:])
            if len(train_ids) < block_size * 2 or len(val_ids) < block_size:
                raise ValueError("corpus too small for this block_size / val_fraction")
            return cls(train_ids, val_ids, bpe.vocab_size, block_size, bpe)
        tok = tokenizer or CharTokenizer.from_text(text)
        ids = tok.encode(text)
        cut = int(len(ids) * (1 - val_fraction))
        if cut < block_size * 2 or len(ids) - cut < block_size:
            raise ValueError("corpus too small for this block_size / val_fraction")
        return cls(ids[:cut], ids[cut:], tok.vocab_size, block_size, tok)

    def fingerprint(self) -> str:
        h = hashlib.sha256(json.dumps([self.train_ids, self.val_ids]).encode())
        return h.hexdigest()[:16]


def match_intermediate_size(cfg: OglodonConfig, target_params: int) -> OglodonConfig:
    """Resize the feed-forward width so that ``cfg`` has (about) ``target_params`` parameters.

    Parameters are linear in ``intermediate_size`` (``mult * d * h`` per layer, mult = 3 for SwiGLU,
    2 for GELU), so the width is solved directly and rounded to the nearest integer.
    """
    mult = 3 if cfg.use_swiglu else 2
    per_unit = cfg.num_layers * mult * cfg.hidden_size
    h = max(1, cfg.intermediate_size + round((target_params - expected_num_parameters(cfg)) / per_unit))
    return OglodonConfig.from_dict({**cfg.to_dict(), "intermediate_size": h})


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median_ms(fn, repeats: int, device: torch.device) -> float:
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        _sync(device)
        t = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def _mean_std(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if math.isfinite(x)]
    if not xs:
        return float("nan"), float("nan")
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


@dataclass
class RunResult:
    variant: str
    seed: int
    val_loss: float
    val_ppl: float
    train_loss_last: float
    train_tokens: int
    train_seconds: float
    tokens_per_second: float
    forward_ms: float
    generate_ms_per_token: float
    peak_memory_mb: float | None
    kv_cache_bytes: int
    curve: list[dict[str, float]] = field(default_factory=list)  # train/val loss over steps
    eval_seconds: float = 0.0


@dataclass
class ComparisonReport:
    config: dict[str, Any]
    variants: dict[str, dict[str, Any]]  # name -> config dict + param-matching info
    runs: list[RunResult]
    summary: dict[str, dict[str, Any]]  # per variant aggregates over seeds
    paired: dict[str, dict[str, Any]]  # per non-baseline variant vs baseline
    validation: dict[str, dict[str, Any] | None]
    fairness_notes: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    config_diff: dict[str, dict[str, list[Any]]] = field(
        default_factory=dict
    )  # variant -> field -> [baseline, variant]
    curves: dict[str, list[dict[str, float]]] = field(
        default_factory=dict
    )  # variant -> mean curve over seeds

    def to_dict(self) -> dict[str, Any]:
        from ..validation.report import _json_safe

        return _json_safe(
            {
                "metadata": self.metadata,
                "config": self.config,
                "config_diff": self.config_diff,
                "curves": self.curves,
                "variants": self.variants,
                "summary": self.summary,
                "paired_vs_baseline": self.paired,
                "validation": self.validation,
                "fairness_notes": self.fairness_notes,
                "runs": [r.__dict__ for r in self.runs],
            }
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def to_markdown(self) -> str:
        c = self.config
        out = [
            "# Oglodon controlled comparison",
            "",
            f"Baseline: **{c['baseline']}**. Seeds: {c['seeds']}. Steps: {c['steps']}, batch {c['batch_size']}, "
            f"block {c['block_size']} = **{c['train_tokens']:,} training tokens per run**. "
            f"Device: {self.metadata.get('device')}. Data fingerprint: `{c['data_fingerprint']}`.",
            "",
            "| variant | attn | params | val loss | val ppl | tokens/s | fwd ms | gen ms/tok | KV bytes | peak MB | validation |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, s in self.summary.items():
            v = self.validation.get(name)
            vtxt = "not run" if v is None else f"{v['counts']['pass']} pass / {v['counts']['fail']} fail"
            peak = "n/a" if s["peak_memory_mb"] is None else f"{s['peak_memory_mb']:.0f}"
            out.append(
                f"| {name} | {self.variants[name]['attention']} | {s['params']:,} | "
                f"{s['val_loss_mean']:.4f} ± {s['val_loss_std']:.4f} | {s['val_ppl_mean']:.3f} | "
                f"{s['tokens_per_second_mean']:.0f} ± {s['tokens_per_second_std']:.0f} | {s['forward_ms_mean']:.2f} | "
                f"{s['generate_ms_per_token_mean']:.3f} | {s['kv_cache_bytes']:,} | {peak} | {vtxt} |"
            )
        out += ["", "## Paired difference in validation loss (variant minus baseline, same seed)", ""]
        for name, p in self.paired.items():
            out.append(
                f"* **{name}**: {p['mean_diff']:+.4f} ± {p['std_diff']:.4f} over {p['n']} seeds (t = {p['t_stat']:.2f}). {p['verdict']}"
            )
        out += ["", "## Configuration differences vs baseline", ""]
        if not any(self.config_diff.values()):
            out.append("* none")
        for name, d in self.config_diff.items():
            if d:
                out.append(f"* **{name}**: " + ", ".join(f"`{k}` {a} -> {b}" for k, (a, b) in d.items()))
        out += ["", "## Validation loss during training (mean over seeds)", ""]
        steps_axis = [p["step"] for p in next(iter(self.curves.values()), []) if "val_loss" in p]
        if steps_axis:
            out.append("| step | " + " | ".join(self.curves) + " |")
            out.append("|---|" + "---|" * len(self.curves))
            for i, st in enumerate(steps_axis):
                row = [
                    f"{[p for p in c if 'val_loss' in p][i]['val_loss']:.4f}" for c in self.curves.values()
                ]
                out.append(f"| {int(st)} | " + " | ".join(row) + " |")
        m = self.metadata
        out += ["", "## Reproducibility", ""]
        for k in (
            "oglodon_version",
            "torch_version",
            "python_version",
            "numpy_version",
            "safetensors_version",
            "transformers_version",
            "platform",
            "device",
            "cuda_device_name",
            "cuda_version",
            "torch_num_threads",
            "timestamp_utc",
            "seeds",
            "tokenizer",
            "train_tokens_in_corpus",
            "val_tokens_in_corpus",
            "source_config_sha256",
        ):
            if m.get(k) is not None:
                out.append(f"* {k}: `{m[k]}`")
        for n, h in (m.get("variant_config_sha256") or {}).items():
            out.append(f"* config sha256 ({n}): `{h}`")
        out += ["", "## Fairness notes", ""] + [f"* {n}" for n in self.fairness_notes]
        return "\n".join(out) + "\n"

    def plot(self, path: str | Path) -> Path:
        """Write validation-loss curves (mean over seeds) as an image. Needs matplotlib (``[viz]`` extra)."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as e:  # pragma: no cover
            raise ImportError('plotting needs: pip install "oglodon-transformer[viz]"') from e
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, curve in self.curves.items():
            pts = [(p["step"], p["val_loss"]) for p in curve if "val_loss" in p]
            ax.plot(*zip(*pts, strict=True), marker="o", label=name)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("validation loss")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return Path(path)


def run_comparison(
    base: OglodonConfig,
    variants: dict[str, dict[str, Any]],
    data: ExperimentData,
    *,
    baseline: str | None = None,
    steps: int = 200,
    batch_size: int = 16,
    learning_rate: float = 3e-3,
    weight_decay: float = 0.0,
    seeds: tuple[int, ...] = (0, 1, 2),
    match_params: bool = True,
    validate: bool = True,
    validation_categories: tuple[str, ...] = ("attention", "kv_cache", "model"),
    device: str | torch.device | None = None,
    latency_repeats: int = 5,
    generate_tokens: int = 16,
    logger=print,
) -> ComparisonReport:
    """Train and compare architecture variants under controlled conditions.

    Args:
        base: shared config (``vocab_size`` is overwritten from ``data``; ``max_seq_len`` is raised
            to fit ``block_size + generate_tokens`` if needed).
        variants: ``{name: overrides}`` applied on top of ``base`` (e.g. ``{"mha": {"num_kv_heads": 8},
            "gqa": {"num_kv_heads": 2}}``).
        baseline: name of the reference variant (default: the first).
        match_params: resize the feed-forward width of the other variants so their parameter count
            matches the baseline (the report records the resulting gap).
        validate: run :func:`oglodon.validation.validate_model` on one trained model of each variant.
    """
    log = logger or (lambda _m: None)
    names = list(variants)
    baseline = baseline or names[0]
    if baseline not in variants:
        raise ValueError(f"baseline {baseline!r} is not one of {names}")
    dev = get_device(device)
    block = data.block_size
    seeds = tuple(seeds)

    # ---- build the configs
    shared = {
        **base.to_dict(),
        "vocab_size": data.vocab_size,
        "max_seq_len": max(base.max_seq_len, block + generate_tokens),
    }
    shared["intermediate_size"] = base.intermediate_size
    cfgs = {n: OglodonConfig.from_dict({**shared, **variants[n]}) for n in names}
    target = expected_num_parameters(cfgs[baseline])
    info: dict[str, dict[str, Any]] = {}
    for n in names:
        if match_params and n != baseline:
            cfgs[n] = match_intermediate_size(cfgs[n], target)
        info[n] = {
            **cfgs[n].to_dict(),
            "attention": cfgs[n].attention_type,
            "params": expected_num_parameters(cfgs[n]),
            "param_gap_vs_baseline": expected_num_parameters(cfgs[n]) / target - 1,
        }

    curve_every = max(1, steps // 10)
    train_ds = TokenBlockDataset(data.train_ids, block, stride=block // 2)
    val_ds = TokenBlockDataset(data.val_ids, block, stride=block)

    runs: list[RunResult] = []
    validation: dict[str, dict[str, Any] | None] = dict.fromkeys(names)
    for n in names:
        for seed in seeds:
            seed_everything(seed)
            model = Oglodon(cfgs[n]).to(dev)
            gen = torch.Generator().manual_seed(seed)  # same batch order for every variant
            loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size)
            tr = Trainer(
                model, loader, val_loader, learning_rate=learning_rate, weight_decay=weight_decay,
                max_steps=steps, warmup_steps=max(1, steps // 10), log_interval=curve_every, eval_interval=curve_every,
                device=dev, logger=None,
            )  # fmt: skip
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            _sync(dev)
            t0 = time.perf_counter()
            hist = tr.train()
            _sync(dev)
            seconds = time.perf_counter() - t0 - tr.eval_seconds  # throughput excludes periodic evaluation
            peak = torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else None
            val_loss = tr.evaluate()
            model.eval()
            x = torch.randint(0, data.vocab_size, (batch_size, block), device=dev)
            with torch.no_grad():
                fwd = _median_ms(lambda m=model, x=x: m(x), latency_repeats, dev)
                prompt = x[:, : block // 2]
                gen_ms = _median_ms(
                    lambda m=model, p=prompt: m.generate(
                        p, max_new_tokens=generate_tokens, eos_token_id=None
                    ),
                    max(1, latency_repeats // 2),
                    dev,
                )
                kv = model(x, use_cache=True).past_key_values.nbytes()
            tokens = steps * batch_size * block
            last_train = next((h["train_loss"] for h in reversed(hist) if "train_loss" in h), float("nan"))
            runs.append(
                RunResult(n, seed, val_loss, math.exp(val_loss), last_train, tokens, seconds, tokens / seconds,
                          fwd, gen_ms / generate_tokens, peak, kv, curve=[dict(h) for h in hist],
                          eval_seconds=tr.eval_seconds)
            )  # fmt: skip
            log(f"{n:<10} seed {seed}: val loss {val_loss:.4f}, {tokens / seconds:,.0f} tok/s")
            if validate and validation[n] is None:
                rep = validate_model(model, cfgs[n], device=dev, seed=seed, categories=validation_categories)
                validation[n] = {**rep.to_dict(), "checks": [c.to_dict() for c in rep.checks if not c.passed]}
                log(f"{n:<10} validation: {rep.counts()}")

    # ---- aggregate
    summary: dict[str, dict[str, Any]] = {}
    for n in names:
        rs = [r for r in runs if r.variant == n]
        agg: dict[str, Any] = {"params": info[n]["params"], "n_seeds": len(rs), "kv_cache_bytes": rs[0].kv_cache_bytes,
                               "train_tokens": rs[0].train_tokens}  # fmt: skip
        for key, attr in (("val_loss", "val_loss"), ("val_ppl", "val_ppl"), ("tokens_per_second", "tokens_per_second"),
                          ("forward_ms", "forward_ms"), ("generate_ms_per_token", "generate_ms_per_token"),
                          ("train_loss_last", "train_loss_last")):  # fmt: skip
            m, s = _mean_std([getattr(r, attr) for r in rs])
            agg[f"{key}_mean"], agg[f"{key}_std"] = m, s
        agg["peak_memory_mb"] = (
            None if rs[0].peak_memory_mb is None else max(r.peak_memory_mb or 0.0 for r in rs)
        )
        summary[n] = agg

    paired: dict[str, dict[str, Any]] = {}
    base_loss = {r.seed: r.val_loss for r in runs if r.variant == baseline}
    for n in names:
        if n == baseline:
            continue
        diffs = [r.val_loss - base_loss[r.seed] for r in runs if r.variant == n]
        m, s = _mean_std(diffs)
        t = m / (s / math.sqrt(len(diffs))) if len(diffs) > 1 and s > 0 else float("nan")
        if len(diffs) < 5:
            verdict = f"Only {len(diffs)} seeds: treat as indicative, not as evidence of a real difference."
        elif abs(t) < 2:
            verdict = "The difference is within seed-to-seed noise."
        else:
            verdict = "The difference is larger than the seed noise (|t| >= 2) for this setup."
        paired[n] = {"mean_diff": m, "std_diff": s, "n": len(diffs), "t_stat": t, "verdict": verdict}

    notes = _fairness_notes(cfgs, info, baseline, seeds, steps, dev, validation, match_params, summary)
    meta = collect_metadata(base.to_dict(), dev, torch.float32, seeds[0])
    meta["seeds"] = list(seeds)
    config = {
        "baseline": baseline, "seeds": list(seeds), "steps": steps, "batch_size": batch_size, "block_size": block,
        "learning_rate": learning_rate, "weight_decay": weight_decay, "match_params": match_params,
        "train_tokens": steps * batch_size * block, "val_tokens": len(data.val_ids),
        "data_fingerprint": data.fingerprint(), "vocab_size": data.vocab_size,
    }  # fmt: skip
    meta.update(_environment_metadata())
    meta["train_tokens_in_corpus"], meta["val_tokens_in_corpus"] = len(data.train_ids), len(data.val_ids)
    meta["tokenizer"] = type(data.tokenizer).__name__ if data.tokenizer is not None else None
    meta["variant_config_sha256"] = {
        n: hashlib.sha256(json.dumps(cfgs[n].to_dict(), sort_keys=True).encode()).hexdigest() for n in names
    }
    diff = config_diff({n: cfgs[n].to_dict() for n in names}, baseline)
    curves = _mean_curves(runs, names)
    return ComparisonReport(config, info, runs, summary, paired, validation, notes, meta, diff, curves)


def config_diff(configs: dict[str, dict[str, Any]], baseline: str) -> dict[str, dict[str, list[Any]]]:
    """``{variant: {field: [baseline value, variant value]}}`` for every field that differs from ``baseline``."""
    base = configs[baseline]
    return {
        n: {k: [base.get(k), c.get(k)] for k in sorted(set(base) | set(c)) if base.get(k) != c.get(k)}
        for n, c in configs.items()
        if n != baseline
    }


def _environment_metadata() -> dict[str, Any]:
    """Versions of the libraries that influence results (only those installed)."""
    from importlib import metadata as md

    out: dict[str, Any] = {}
    for pkg in ("numpy", "safetensors", "transformers"):
        with contextlib.suppress(md.PackageNotFoundError):
            out[f"{pkg}_version"] = md.version(pkg)
    return out


def _mean_curves(runs: list[RunResult], names: list[str]) -> dict[str, list[dict[str, float]]]:
    """Average each variant's logged curve over seeds (curves share the same steps)."""
    out: dict[str, list[dict[str, float]]] = {}
    for n in names:
        rs = [r.curve for r in runs if r.variant == n and r.curve]
        if not rs:
            out[n] = []
            continue
        merged: list[dict[str, float]] = []
        for pts in zip(*rs, strict=True):
            rec = {"step": pts[0]["step"]}
            for key in ("train_loss", "val_loss", "val_ppl"):
                vals = [p[key] for p in pts if key in p]
                if vals:
                    rec[key] = statistics.fmean(vals)
            merged.append(rec)
        out[n] = merged
    return out


def _fairness_notes(cfgs, info, baseline, seeds, steps, dev, validation, match_params, summary) -> list[str]:
    notes = [
        "Same corpus, split, tokenizer, batch order (per seed), optimiser settings and number of training "
        "tokens for every variant. Initial weights differ because the layers have different shapes, so the "
        "variants cannot share an initialisation.",
    ]
    base = cfgs[baseline]
    for n, c in cfgs.items():
        if n == baseline:
            continue
        gap = info[n]["param_gap_vs_baseline"]
        diffs = [
            k
            for k in (
                "hidden_size",
                "num_layers",
                "num_heads",
                "use_rope",
                "use_alibi",
                "use_rmsnorm",
                "use_swiglu",
                "ffn_activation",
                "sliding_window",
                "dropout",
            )
            if getattr(c, k) != getattr(base, k)
        ]
        if diffs:
            notes.append(
                f"{n} differs from {baseline} in more than the attention layout ({', '.join(diffs)}): the comparison mixes several changes."
            )
        if abs(gap) > 0.01:
            notes.append(
                f"{n} has {gap:+.1%} parameters relative to {baseline}; a loss difference may reflect capacity, not the architecture."
            )
        elif match_params and c.intermediate_size != base.intermediate_size:
            notes.append(
                f"{n} is parameter-matched ({gap:+.2%}) by changing the feed-forward width from {base.intermediate_size} to {c.intermediate_size}, so it is not identical to {baseline} apart from the attention layout."
            )
        if not match_params and c.num_kv_heads != base.num_kv_heads:
            notes.append(
                f"Parameters were not matched: fewer KV heads means fewer parameters ({gap:+.1%} for {n}), which favours the baseline in capacity."
            )
    if len(seeds) < 5:
        notes.append(
            f"{len(seeds)} seeds: standard deviations are rough and differences between variants are indicative only."
        )
    if dev.type == "cpu":
        notes.append(
            "Peak memory is not measured on CPU (torch exposes no per-run CPU peak); only CUDA runs report it. Throughput and latency are wall-clock on a possibly shared CPU and are noisy."
        )
    else:
        notes.append(
            "Throughput and latency include a short warm-up but no clock locking; other GPU work would distort them."
        )
    notes.append(
        f"Training is only {steps} steps on a small corpus: conclusions apply to this regime and do not transfer to large-scale training."
    )
    notes.append(
        "GQA/MQA mainly reduce KV-cache memory and decoding cost; a training-loss comparison at this scale does not show that benefit, which is why cache bytes and generation latency are reported separately."
    )
    for n, v in validation.items():
        if v is not None and v["counts"]["fail"]:
            notes.append(
                f"{n} FAILED validation ({v['counts']['fail']} checks): its numbers should not be trusted until that is fixed."
            )
        elif v is None:
            notes.append(f"{n}: numerical validation was not run.")
    return notes
