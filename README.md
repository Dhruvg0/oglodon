# Oglodon

**Build. Train. Understand. Verify.**

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](https://github.com/Dhruvg0/oglodon/blob/main/LICENSE)
[![Tests](https://github.com/Dhruvg0/oglodon/actions/workflows/tests.yml/badge.svg)](https://github.com/Dhruvg0/oglodon/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/oglodon-transformer)](https://pypi.org/project/oglodon-transformer/)

Oglodon is a small PyTorch library for building, training and checking decoder-only Transformers (GPT-style, with
RoPE, RMSNorm, SwiGLU, grouped-query attention and a KV cache).

Its most useful part is **`oglodon verify`**: it tests *any* causal language model (Hugging Face, your own PyTorch
module, or Oglodon) for bugs that still train but quietly make the model worse, such as a token seeing the future,
padding changing the answer, or a KV cache that disagrees with a full forward pass.

> **Status: alpha, for learning and research at small scale.** Not production-ready and not a replacement for large-scale
> frameworks. See [what has and has not been tested](#what-has-been-tested).

## Install

```bash
pip install oglodon-transformer
```

Needs Python 3.11+ and PyTorch. Optional extras: `[hf]` Hugging Face import, export and verification, `[viz]` plots,
`[yaml]` YAML configs, `[dev]` tests and linters. Working from a clone: `pip install -e ".[dev]"`.

## Try it in two minutes

Put a few `.txt` or `.md` files in a folder, then:

```bash
oglodon train my_texts/ --out run1 --steps 500
oglodon generate run1 --prompt "Once upon a time" --seed 1
oglodon eval run1 --text my_texts/some_file.txt
```

`train` builds a tokenizer, trains a small model and saves a checkpoint; `eval` prints perplexity next to the score of
a model that guesses at random, so you can see whether it learned anything. A model trained for minutes learns your
text's vocabulary and local patterns, not fluent language. No Python code is needed.

## How it fits together

```text
 text files ──► BPE tokenizer ──► token file ──► train ──► CHECKPOINT
                                                              │
        ┌─────────────┬──────────────┬──────────────┬─────────┴────────┐
        ▼             ▼              ▼              ▼                  ▼
    generate         eval          verify        quantize         export to
  (stream text)  (perplexity)   (behaviour)    (int8 copy)      Hugging Face
                                     ▲
 any Hugging Face / PyTorch model ───┘     verify accepts any causal LM
```

Inside the model:

```text
tokens ─► Embedding ─► [ Norm ─► Attention (RoPE, KV cache) ─► + ─► Norm ─► FFN ─► + ] × N ─► Norm ─► LM head ─► logits
                                MHA / GQA / MQA                          SwiGLU / GELU
```

## What you can do

| Command | What it does |
|---|---|
| `oglodon train` | Text files to tokenizer, trained model and report. Exact `--resume`, data-parallel under `torchrun` |
| `oglodon generate` | Streams text: temperature, top-k, top-p, min-p, repetition penalty, stop strings, KV cache |
| `oglodon eval` | Perplexity, multiple-choice accuracy, generation sanity checks, saved as JSON |
| `oglodon verify` | Behaviour checks on any causal LM; `--cross-check` compares against a Hugging Face model |
| `oglodon validate` | Compares an Oglodon model with an independent float64 reference implementation |
| `oglodon compare` | Trains variants on identical data and seeds and writes reproducible comparison reports |
| `oglodon quantize` | Int8 weight-only copy of a checkpoint with a measured accuracy drift |
| `oglodon info` | Versions and whether CUDA is available |

Architecture options: MHA, GQA or MQA attention, RoPE, learned or ALiBi positions, RMSNorm or LayerNorm, SwiGLU,
GeGLU, GELU or ReLU² feed-forward, sliding window, qkv bias. Export to Llama, Mistral or Qwen2 format for
Hugging Face with `export_hf(model, "hf_dir")`.

## Verify any model

```bash
oglodon verify --checkpoint run1                      # an Oglodon checkpoint
oglodon verify --hf HuggingFaceTB/SmolLM2-135M --cross-check --dtype float32
```

| Check | What it catches |
|---|---|
| Causality | A token's prediction changes when later tokens change |
| Padding and batching | Padded or batched inputs giving different answers than single ones |
| KV cache | Cached decoding disagreeing with a full forward pass |
| Numerics | float32 vs float64 drift, non-finite values |
| Gradients | Autograd disagreeing with finite differences, dead parameters |
| Cross-check | An Oglodon model and its Hugging Face twin producing different logits |

Skipped or unsupported checks are reported as such and are never counted as passed.

## What has been tested

| Verified | Not yet |
|---|---|
| Test suite on CPU: Python 3.11, 3.12 and 3.13 (Linux) and the suite on Windows | macOS |
| Test suite, `validate`, `verify`, bf16 and fp16 steps on one Tesla T4 (Colab) | Multi-GPU, FlashAttention kernels, other GPUs |
| `verify --cross-check` on SmolLM2-135M in float32: logits agree to about 1e-4, greedy output identical | Other pretrained models |
| Int8 drift measured on a small trained model (perplexity change about 0.001%) | Int8 on large models or on a GPU |
| Hugging Face export matches logits to about 1e-6 | Gemma, Phi, Mixtral converters, FSDP, 4-bit quantization |

Details and exact numbers: [Limitations](https://github.com/Dhruvg0/oglodon/blob/main/docs/limitations.md) and [Performance and scaling](https://github.com/Dhruvg0/oglodon/blob/main/docs/performance-and-scaling.md).

## Documentation

* [Getting started](https://github.com/Dhruvg0/oglodon/blob/main/docs/getting-started.md) and [command reference](https://github.com/Dhruvg0/oglodon/blob/main/docs/cli.md)
* [Training](https://github.com/Dhruvg0/oglodon/blob/main/docs/training.md), [data and tokenizers](https://github.com/Dhruvg0/oglodon/blob/main/docs/data-and-tokenization.md), [generation](https://github.com/Dhruvg0/oglodon/blob/main/docs/generation.md),
  [evaluation](https://github.com/Dhruvg0/oglodon/blob/main/docs/evaluation.md)
* [Verify](https://github.com/Dhruvg0/oglodon/blob/main/docs/verify.md), [validation](https://github.com/Dhruvg0/oglodon/blob/main/docs/validation.md), [experiments](https://github.com/Dhruvg0/oglodon/blob/main/docs/experiments.md),
  [quantization](https://github.com/Dhruvg0/oglodon/blob/main/docs/quantization.md), [Hugging Face interop](https://github.com/Dhruvg0/oglodon/blob/main/docs/interoperability.md)
* [Python and feature reference](https://github.com/Dhruvg0/oglodon/blob/main/docs/reference.md), [architecture](https://github.com/Dhruvg0/oglodon/blob/main/docs/architecture.md),
  [configuration](https://github.com/Dhruvg0/oglodon/blob/main/docs/configuration.md), [troubleshooting](https://github.com/Dhruvg0/oglodon/blob/main/docs/troubleshooting.md)
* Runnable scripts in [`examples/`](https://github.com/Dhruvg0/oglodon/tree/main/examples)

## Contributing and licence

Contributions are welcome; see [docs/contributing.md](https://github.com/Dhruvg0/oglodon/blob/main/docs/contributing.md). Released under the [MIT licence](https://github.com/Dhruvg0/oglodon/blob/main/LICENSE).

Oglodon is an independent project. It is not affiliated with, endorsed by or derived from NVIDIA or any hardware vendor.
