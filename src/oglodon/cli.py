"""Command line interface: ``oglodon validate CONFIG`` and ``oglodon compare CONFIG``.

Config files are YAML (needs ``pip install "oglodon-transformer[yaml]"``) or JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .config import OglodonConfig


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Read a YAML or JSON mapping from ``path``."""
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as e:
        raise SystemExit(f"error: cannot read config {path}: {e.strerror or e}") from e
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise SystemExit(f"error: {path} is not valid JSON: {e}") from e
    else:
        try:
            import yaml
        except ImportError as e:
            raise SystemExit(
                'YAML configs need PyYAML: pip install "oglodon-transformer[yaml]" (or use a .json config)'
            ) from e
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a mapping at the top level")
    return data


_VERIFY_CATEGORIES = ("structure", "causality", "batching", "cache", "numerics", "gradients")
_NON_MODEL_KEYS = {"model", "checkpoint", "validation", "variants", "baseline", "data", "train"}


def model_config_from(data: dict[str, Any]) -> OglodonConfig:
    """``model:`` section, or a flat file whose keys are OglodonConfig fields."""
    raw = data["model"] if "model" in data else {k: v for k, v in data.items() if k not in _NON_MODEL_KEYS}
    try:
        return OglodonConfig.from_dict(raw)
    except (TypeError, ValueError) as e:
        raise SystemExit(f"invalid model config: {e}") from e


def _validate(args: argparse.Namespace) -> int:
    from .models.gpt import Oglodon
    from .tokenization import load_tokenizer
    from .validation import validate_model

    data = load_config_file(args.config)
    cfg = model_config_from(data)
    opts: dict[str, Any] = dict(data.get("validation") or {})
    for key in ("device", "dtype", "seed"):
        if getattr(args, key) is not None:
            opts[key] = getattr(args, key)
    if args.categories:
        opts["categories"] = args.categories
    ckpt = args.checkpoint or data.get("checkpoint")
    tokenizer = None
    torch.manual_seed(int(opts.get("seed", 0)))
    if ckpt:
        model = Oglodon.from_pretrained(ckpt)
        tokenizer = load_tokenizer(ckpt)
    else:
        model = Oglodon(cfg)
    try:
        report = validate_model(
            model,
            cfg,
            device=opts.get("device"),
            dtype=opts.get("dtype"),
            seed=int(opts.get("seed", 0)),
            categories=opts.get("categories"),
            exclude=tuple(opts.get("exclude", ())),
            tokenizer=tokenizer,
            tolerances=opts.get("tolerances"),
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(report)
    if args.out:
        report.save(args.out)
        print(f"report written to {args.out}")
    if report.failures:
        return 1
    if args.strict and not report.complete:
        print("--strict: some checks were skipped or unsupported", file=sys.stderr)
        return 1
    return 0


def _compare(args: argparse.Namespace) -> int:
    import hashlib

    from .data import make_corpus
    from .experiments import ExperimentData, run_comparison

    paths = list(args.config)
    data = load_config_file(paths[0])
    base = model_config_from(data)
    source_sha = hashlib.sha256(b"".join(Path(p).read_bytes() for p in paths)).hexdigest()
    if len(paths) == 1:
        variants = data.get("variants")
        if not variants:
            print(
                "error: the config needs a 'variants:' mapping (or pass several config files)",
                file=sys.stderr,
            )
            return 2
    else:
        # One file per variant: its model section is a variant; data/train must be shared (taken from the first).
        base_dict, variants = base.to_dict(), {}
        for p in paths:
            other = load_config_file(p)
            for key in ("data", "train"):
                if other.get(key) is not None and other[key] != data.get(key):
                    print(
                        f"error: {p} has a different '{key}:' section than {paths[0]}; "
                        "a controlled comparison needs the same data and training settings",
                        file=sys.stderr,
                    )
                    return 2
            name = Path(p).stem
            if name in variants:
                print(f"error: two config files are called {name!r}; rename one", file=sys.stderr)
                return 2
            cfg = model_config_from(other).to_dict()
            variants[name] = {k: v for k, v in cfg.items() if k != "vocab_size" and base_dict.get(k) != v}
        data = {**data, "baseline": Path(paths[0]).stem}
    d, t = data.get("data") or {}, data.get("train") or {}
    block = int(d.get("block_size", 64))
    if d.get("path"):
        from .data import collect_text_files, read_texts

        try:
            text = "\n".join(read_texts(collect_text_files(d["path"])))
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    else:
        text = make_corpus(int(d.get("sentences", 1500)))
    tok_kind = d.get("tokenizer", "char")
    if tok_kind not in ("char", "bpe"):
        print("error: data.tokenizer must be 'char' or 'bpe'", file=sys.stderr)
        return 2
    exp = ExperimentData.from_text(
        text,
        block,
        float(d.get("val_fraction", 0.1)),
        bpe_vocab_size=int(d.get("vocab_size", 512)) if tok_kind == "bpe" else None,
    )
    try:
        report = run_comparison(
            base,
            variants,
            exp,
            baseline=data.get("baseline"),
            steps=int(t.get("steps", 200)),
            batch_size=int(t.get("batch_size", 16)),
            learning_rate=float(t.get("learning_rate", 3e-3)),
            seeds=tuple(t.get("seeds", (0, 1, 2))),
            match_params=bool(t.get("match_params", True)),
            validate=bool(t.get("validate", True)),
            device=args.device,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    report.metadata["source_config_sha256"] = source_sha
    report.metadata["source_configs"] = [str(p) for p in paths]
    print(report.to_markdown())
    if args.out:
        report.save(args.out)
        Path(args.out).with_suffix(".md").write_text(report.to_markdown())
        print(f"report written to {args.out} and {Path(args.out).with_suffix('.md')}")
        if args.plot:
            print(f"loss curves written to {report.plot(Path(args.out).with_suffix('.png'))}")
    elif args.plot:
        print("error: --plot needs --out (the image is written next to it)", file=sys.stderr)
        return 2
    failed = [n for n, v in report.validation.items() if v and v["counts"]["fail"]]
    return 1 if failed else 0


def _train(args: argparse.Namespace) -> int:
    from .workflow import TrainSettings, train_from_files

    overrides: dict[str, Any] = {
        "tokenizer": args.tokenizer, "vocab_size": args.vocab_size, "val_fraction": args.val_fraction,
        "workers": args.workers, "block_size": args.block_size, "hidden_size": args.hidden_size,
        "num_layers": args.layers, "num_heads": args.heads, "num_kv_heads": args.kv_heads,
        "steps": args.steps, "batch_size": args.batch_size, "grad_accum_steps": args.grad_accum,
        "learning_rate": args.lr, "weight_decay": args.weight_decay, "warmup_steps": args.warmup,
        "seed": args.seed, "precision": args.precision, "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
        "gradient_checkpointing": True if args.grad_checkpointing else None,
        "compile": True if args.compile else None,
    }  # fmt: skip
    try:
        base = TrainSettings()
        if args.config:
            cfg = load_config_file(args.config)
            flat = {**cfg.get("model", {}), **cfg.get("data", {}), **cfg.get("train", {})}
            base = TrainSettings.from_dict({k: v for k, v in flat.items() if v is not None})
        settings = base.merged(overrides)
        train_from_files(
            args.data, args.out, settings, device=args.device, resume=args.resume, overwrite=args.overwrite
        )
    except (ValueError, FileNotFoundError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


def _generate(args: argparse.Namespace) -> int:
    from .models.gpt import Oglodon
    from .tokenization import load_tokenizer
    from .workflow import complete

    try:
        model = Oglodon.from_pretrained(args.checkpoint)
        tokenizer = load_tokenizer(args.checkpoint)
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        print(f"error: cannot load checkpoint {args.checkpoint}: {e}", file=sys.stderr)
        return 2
    if tokenizer is None:
        print("error: the checkpoint has no tokenizer.json", file=sys.stderr)
        return 2
    prompt = args.prompt if args.prompt is not None else (sys.stdin.read() if not sys.stdin.isatty() else "")
    try:
        stream = complete(
            model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k or None, top_p=args.top_p, min_p=args.min_p,
            repetition_penalty=args.repetition_penalty, stop=args.stop or (), seed=args.seed, device=args.device,
        )  # fmt: skip
        if not args.only_new:
            print(prompt, end="", flush=True)
        pieces = []
        for piece in stream:
            pieces.append(piece)
            if not args.no_stream:
                print(piece, end="", flush=True)
        if args.no_stream:
            print("".join(pieces), end="")
        print()
    except ValueError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 2
    return 0


def _quantize(args: argparse.Namespace) -> int:
    from .models.gpt import Oglodon
    from .quantization import is_quantized, model_nbytes, quantization_report, quantize_int8
    from .tokenization import load_tokenizer

    try:
        model = Oglodon.from_pretrained(args.checkpoint)
        tokenizer = load_tokenizer(args.checkpoint)
        if is_quantized(model):
            raise ValueError("this checkpoint is already quantized")
        q = quantize_int8(model, group_size=args.group_size, include_lm_head=args.include_lm_head)
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    report: dict[str, Any] = {"source": str(args.checkpoint), "scheme": q.quantization_info["scheme"]}
    try:
        if args.text:
            if tokenizer is None:
                raise ValueError("--text needs a tokenizer.json in the checkpoint")
            ids = tokenizer.encode(Path(args.text).read_text())
            report.update(quantization_report(model, q, ids, block_size=args.block_size))
        else:
            report.update(bytes_reference=model_nbytes(model), bytes_quantized=model_nbytes(q))
            report["size_ratio"] = report["bytes_quantized"] / report["bytes_reference"]
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    q.save_pretrained(out, tokenizer=tokenizer)
    weights_before = (Path(args.checkpoint) / "model.safetensors").stat().st_size
    weights_after = (out / "model.safetensors").stat().st_size
    report.update(file_bytes_reference=weights_before, file_bytes_quantized=weights_after)
    (out / "quantization_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"quantized {len(q.quantization_info['modules'])} linear layers to int8 (group size "
        f"{q.quantization_info['group_size'] or 'per channel'})"
    )
    print(
        f"model.safetensors: {weights_before / 1e6:.2f} MB -> {weights_after / 1e6:.2f} MB "
        f"({weights_after / weights_before:.0%})"
    )
    if args.text:
        print(
            f"perplexity {report['perplexity_reference']:.3f} -> {report['perplexity_quantized']:.3f} "
            f"({report['perplexity_change_percent']:+.2f}%); top-1 unchanged on {report['top1_agreement']:.1%} "
            f"of positions; mean KL {report['kl_mean_nats']:.5f} nats"
        )
    else:
        print("no --text given: quality drift was NOT measured (pass --text FILE with text you care about)")
    print(f"written to {out}; load it with Oglodon.from_pretrained, `oglodon generate`, `oglodon eval`")
    return 0


def _eval(args: argparse.Namespace) -> int:
    from .evaluation import check_generation, evaluate_multiple_choice, load_jsonl, perplexity
    from .models.gpt import Oglodon
    from .tokenization import load_tokenizer

    if not (args.text or args.choices or args.prompt):
        print("error: give at least one of --text, --choices, --prompt", file=sys.stderr)
        return 2
    try:
        model = Oglodon.from_pretrained(args.checkpoint)
        tokenizer = load_tokenizer(args.checkpoint)
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        print(f"error: cannot load checkpoint {args.checkpoint}: {e}", file=sys.stderr)
        return 2
    if tokenizer is None:
        print(
            "error: the checkpoint has no tokenizer.json (save it with save_pretrained(tokenizer=...))",
            file=sys.stderr,
        )
        return 2
    model.to(args.device or "cpu")
    result: dict[str, Any] = {"checkpoint": str(args.checkpoint), "device": args.device or "cpu"}
    try:
        if args.text:
            ids = tokenizer.encode(Path(args.text).read_text())
            r = perplexity(
                model, ids, block_size=args.block_size, stride=args.stride, batch_size=args.batch_size
            )
            result["perplexity"] = {
                **r.to_dict(),
                "text": str(args.text),
                "uniform_baseline": model.config.vocab_size,
            }
            print(
                f"perplexity {r.perplexity:.3f} (loss {r.loss:.4f} nats, {r.tokens_scored:,} tokens, "
                f"block {r.block_size}, stride {r.stride}; a uniform model scores {model.config.vocab_size})"
            )
        if args.choices:
            mc = evaluate_multiple_choice(model, load_jsonl(args.choices), tokenizer, keep_predictions=False)
            result["multiple_choice"] = {**mc.to_dict(), "file": str(args.choices)}
            print(
                f"multiple choice: acc {mc.acc:.3f} +- {mc.stderr:.3f}, acc_norm {mc.acc_norm:.3f} over {mc.n} examples"
            )
        if args.prompt:
            rep = check_generation(
                model, [list(tokenizer.encode(p)) for p in args.prompt], max_new_tokens=args.max_new_tokens
            )
            samples = [tokenizer.decode(s) for s in rep.samples]
            result["generation"] = {**rep.to_dict(), "decoded_samples": samples}
            print(
                f"generation: distinct-1 {rep.distinct_1:.2f}, distinct-2 {rep.distinct_2:.2f}, repeated 3-grams "
                f"{rep.repeated_3gram_fraction:.2f}, deterministic {rep.greedy_deterministic}, "
                f"cache==full {rep.cache_matches_full}"
            )
            for s in samples:
                print(f"  sample: {s!r}")
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(f"report written to {args.out}")
    return 0


def _load_verify_target(args: argparse.Namespace):
    """Return ``(model, source_description)`` for ``oglodon verify``."""
    import importlib

    given = [n for n in ("hf", "hf_config", "factory", "checkpoint", "config") if getattr(args, n)]
    if len(given) != 1:
        raise SystemExit("error: give exactly one of --hf, --hf-config, --factory, --checkpoint, --config")
    if args.hf or args.hf_config:
        try:
            from transformers import AutoConfig, AutoModelForCausalLM
        except ImportError as e:
            raise SystemExit('error: Hugging Face models need: pip install "oglodon-transformer[hf]"') from e
        if args.hf:
            return AutoModelForCausalLM.from_pretrained(args.hf), f"hf:{args.hf}"
        path = Path(args.hf_config)
        cfg = (
            AutoConfig.for_model(**json.loads(path.read_text()))
            if path.is_file()
            else AutoConfig.from_pretrained(str(path))
        )
        return AutoModelForCausalLM.from_config(cfg), f"hf-config:{args.hf_config} (random weights)"
    if args.factory:
        mod, _, fn = args.factory.partition(":")
        if not fn:
            raise SystemExit("error: --factory must look like package.module:function")
        sys.path.insert(0, str(Path.cwd()))
        try:
            model = getattr(importlib.import_module(mod), fn)()
        except (ImportError, AttributeError) as e:
            raise SystemExit(f"error: cannot load factory {args.factory}: {e}") from e
        return model, f"factory:{args.factory}"
    from .models.gpt import Oglodon

    if args.checkpoint:
        return Oglodon.from_pretrained(args.checkpoint), f"checkpoint:{args.checkpoint}"
    return Oglodon(model_config_from(load_config_file(args.config))), f"config:{args.config}"


def _verify(args: argparse.Namespace) -> int:
    from .verify import verify_causal_lm

    model, source = _load_verify_target(args)
    try:
        report = verify_causal_lm(
            model,
            vocab_size=args.vocab_size,
            device=args.device,
            dtype=args.dtype,
            seed=args.seed,
            categories=args.categories,
            exclude=tuple(args.exclude or ()),
            stress=not args.no_stress,
            max_fp64_params=None if args.max_fp64_params < 0 else args.max_fp64_params,
        )
    except (ValueError, TypeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    report.metadata["source"] = source
    if args.cross_check:
        from .convert import from_hf_llama
        from .verify import verify_conversion

        xmodel = model
        if args.dtype or args.device:  # the cross-check must run at the precision/device that was asked for
            import copy

            import torch

            xmodel = copy.deepcopy(model)
            if args.dtype:
                xmodel = xmodel.to(getattr(torch, args.dtype))
            if args.device:
                xmodel = xmodel.to(args.device)
        eq = verify_conversion(xmodel, from_hf_llama, seed=args.seed)
        report.checks.extend(eq.checks)
        report.metadata["cross_check"] = "oglodon.convert.from_hf_llama"
    print(f"source: {source}")
    print(report)
    if args.out:
        report.save(args.out)
        print(f"report written to {args.out}")
    if report.failures:
        return 1
    if args.strict and not report.complete:
        print("--strict: some checks were skipped or unsupported", file=sys.stderr)
        return 1
    return 0


def _info(args: argparse.Namespace) -> int:
    """Print the environment (for bug reports and for labelling GPU runs)."""
    import platform

    import torch

    from . import __version__

    cuda = torch.cuda.is_available()
    info = {
        "oglodon": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else [],
        "bf16_supported": bool(cuda and torch.cuda.is_bf16_supported()),
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = None
    print(json.dumps(info, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oglodon", description="Oglodon: a behaviour-verified Transformer lab")
    p.add_argument("--version", action="version", version=f"oglodon {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="run behavioural validation on the model described by a config")
    v.add_argument("config")
    v.add_argument("--checkpoint", help="load weights from a save_pretrained() directory")
    v.add_argument("--device")
    v.add_argument("--dtype", choices=["float32", "float64", "bfloat16", "float16"])
    v.add_argument("--seed", type=int)
    v.add_argument("--categories", nargs="+", choices=["attention", "kv_cache", "model", "training"])
    v.add_argument("--out", help="write the JSON report here")
    v.add_argument(
        "--strict", action="store_true", help="exit 1 also if any check was skipped or unsupported"
    )
    v.set_defaults(func=_validate)

    c = sub.add_parser("compare", help="train and compare architecture variants under controlled conditions")
    c.add_argument(
        "config",
        nargs="+",
        help="one config with a 'variants:' mapping, or several configs (one variant each; the first is the baseline)",
    )
    c.add_argument("--device")
    c.add_argument("--out", help="write JSON (and a .md next to it) here")
    c.add_argument(
        "--plot", action="store_true", help="also write validation-loss curves as a .png (needs [viz])"
    )
    c.set_defaults(func=_compare)

    t = sub.add_parser("train", help="train a model on your own text files and write a checkpoint")
    t.add_argument("data", nargs="+", help="text files, folders (searched recursively) or glob patterns")
    t.add_argument("--out", required=True, help="checkpoint folder (model, tokenizer, trainer state, report)")
    t.add_argument(
        "--config", help="YAML/JSON with optional model:/data:/train: sections (flags override it)"
    )
    t.add_argument("--tokenizer", choices=["bpe", "char"], help="default: bpe")
    t.add_argument("--vocab-size", type=int, help="BPE vocabulary size (default 1024)")
    t.add_argument(
        "--val-fraction", type=float, help="share of the tokens held out for validation (default 0.05)"
    )
    t.add_argument("--workers", type=int, help="tokenisation processes (default 1)")
    t.add_argument("--block-size", type=int, help="context length (default 128)")
    t.add_argument("--hidden-size", type=int, help="default 128")
    t.add_argument("--layers", type=int, help="default 4")
    t.add_argument("--heads", type=int, help="default 8")
    t.add_argument(
        "--kv-heads", type=int, help="key/value heads (default heads // 4: grouped-query attention)"
    )
    t.add_argument("--steps", type=int, help="optimizer steps (default 1000)")
    t.add_argument("--batch-size", type=int, help="per process (default 16)")
    t.add_argument("--grad-accum", type=int, help="micro-batches per step (default 1)")
    t.add_argument("--lr", type=float, help="peak learning rate (default 1e-3)")
    t.add_argument("--weight-decay", type=float)
    t.add_argument("--warmup", type=int, help="warm-up steps (default 5%% of steps)")
    t.add_argument("--seed", type=int, help="default 0")
    t.add_argument("--precision", choices=["fp32", "bf16", "fp16"], help="fp16 needs CUDA")
    t.add_argument("--eval-interval", type=int, help="default: ten evaluations per run")
    t.add_argument(
        "--save-interval", type=int, help="also checkpoint every N steps (default: only at the end)"
    )
    t.add_argument("--grad-checkpointing", action="store_true", help="recompute activations to save memory")
    t.add_argument("--compile", action="store_true", help="torch.compile (slow first step)")
    t.add_argument("--device", help="cpu, cuda, cuda:1 ... (default: cuda if available)")
    t.add_argument("--resume", action="store_true", help="continue the run in --out exactly where it stopped")
    t.add_argument("--overwrite", action="store_true", help="allow replacing a model already in --out")
    t.set_defaults(func=_train)

    g = sub.add_parser("generate", help="generate text from a saved checkpoint (streams to the terminal)")
    g.add_argument("checkpoint", help="a folder written by `oglodon train` or save_pretrained(tokenizer=...)")
    g.add_argument("--prompt", "-p", help="text to continue (default: read stdin; empty: unconditional)")
    g.add_argument("--max-new-tokens", type=int, default=100)
    g.add_argument("--temperature", type=float, default=0.8, help="0 = greedy (default 0.8)")
    g.add_argument("--top-k", type=int, default=40, help="0 disables (default 40)")
    g.add_argument("--top-p", type=float)
    g.add_argument("--min-p", type=float)
    g.add_argument("--repetition-penalty", type=float, default=1.0)
    g.add_argument("--stop", action="append", help="stop after this text (repeatable)")
    g.add_argument("--seed", type=int, help="make sampling reproducible")
    g.add_argument("--only-new", action="store_true", help="do not echo the prompt")
    g.add_argument("--no-stream", action="store_true", help="print once at the end")
    g.add_argument("--device")
    g.set_defaults(func=_generate)

    qz = sub.add_parser(
        "quantize", help="make an int8 weight-only copy of a checkpoint and measure the drift"
    )
    qz.add_argument("checkpoint", help="a float checkpoint folder")
    qz.add_argument("--out", required=True, help="folder for the int8 checkpoint")
    qz.add_argument(
        "--group-size", type=int, help="scale per group of N input channels (default: per output channel)"
    )
    qz.add_argument(
        "--include-lm-head", action="store_true", help="also quantize an untied output projection"
    )
    qz.add_argument("--text", help="text file to measure perplexity/KL/top-1 drift on (strongly recommended)")
    qz.add_argument("--block-size", type=int)
    qz.set_defaults(func=_quantize)

    e = sub.add_parser("eval", help="perplexity / multiple-choice / generation checks for a saved checkpoint")
    e.add_argument("checkpoint", help="a save_pretrained() directory that contains tokenizer.json")
    e.add_argument("--text", help="text file: report perplexity of its tokens")
    e.add_argument(
        "--block-size", type=int, help="window length (default: the model's maximum, at most 1024)"
    )
    e.add_argument(
        "--stride", type=int, help="window step; smaller than --block-size gives more context per token"
    )
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument(
        "--choices", help='JSONL of {"context", "choices", "label"}: log-likelihood multiple choice'
    )
    e.add_argument("--prompt", nargs="+", help="prompts for a greedy generation sanity check")
    e.add_argument("--max-new-tokens", type=int, default=32)
    e.add_argument("--device")
    e.add_argument("--out", help="write the JSON report here")
    e.set_defaults(func=_eval)

    w = sub.add_parser(
        "verify",
        help="model-agnostic behavioural checks for ANY causal LM (Hugging Face, custom PyTorch, Oglodon)",
    )
    w.add_argument("--hf", help="Hugging Face model id or local directory (AutoModelForCausalLM)")
    w.add_argument(
        "--hf-config", help="HF config directory or JSON with model_type: random weights, works offline"
    )
    w.add_argument(
        "--factory", help="package.module:function returning an nn.Module (run from your project dir)"
    )
    w.add_argument("--checkpoint", help="Oglodon save_pretrained() directory")
    w.add_argument("--config", help="Oglodon config file (random weights)")
    w.add_argument("--vocab-size", type=int)
    w.add_argument("--device")
    w.add_argument("--dtype", choices=["float32", "float64", "bfloat16", "float16"])
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--categories", nargs="+", choices=list(_VERIFY_CATEGORIES))
    w.add_argument("--exclude", nargs="+", help="check names or prefixes to skip")
    w.add_argument(
        "--no-stress",
        action="store_true",
        help="skip the large-weight copy; check the weights as given (saves memory)",
    )
    w.add_argument(
        "--max-fp64-params",
        type=int,
        default=250_000_000,
        help="float64-based checks report 'unsupported' above this many parameters (default 250M; -1 = no limit)",
    )
    w.add_argument(
        "--cross-check",
        action="store_true",
        help="also convert a Hugging Face Llama/Mistral model to Oglodon and compare the two implementations",
    )
    w.add_argument("--out", help="write the JSON report here")
    w.add_argument(
        "--strict", action="store_true", help="exit 1 also if any check was skipped or unsupported"
    )
    w.set_defaults(func=_verify)

    i = sub.add_parser(
        "info", help="print versions and CUDA availability (attach to bug reports and GPU runs)"
    )
    i.set_defaults(func=_info)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except SystemExit as e:
        if isinstance(
            e.code, str
        ):  # a usage/config error raised with a message: exit 2, not 1 (= check failed)
            print(e.code, file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
