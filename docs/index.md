# Oglodon documentation

Oglodon is a PyTorch toolkit for **building, training, understanding and verifying** decoder-only Transformers, with
tools that check a model's behaviour instead of trusting it. Its core: a readable Transformer (RoPE, RMSNorm, SwiGLU,
GQA, KV cache), a trainer, a text-to-checkpoint workflow, and a verification layer that also works on models that are
not Oglodon.

## Start here

| I want to... | Read |
|---|---|
| install it and train a first model on my own text in minutes | [Getting started](getting-started.md) |
| look up a command or flag | [Command-line reference](cli.md) |
| train, resume, and scale training | [Training](training.md) |
| prepare data and tokenizers | [Data and tokenization](data-and-tokenization.md) |
| generate text, stream it, control sampling | [Generation](generation.md) |
| measure a model (perplexity, multiple choice) | [Evaluation](evaluation.md) |
| shrink a model with int8 weights | [Quantization](quantization.md) |
| compare architectures fairly (MHA/GQA/MQA, ALiBi, ...) | [Experiments](experiments.md) |
| check that any causal LM behaves correctly | [verify](verify.md) and [validation](validation.md) |
| move models to and from Hugging Face | [Interoperability](interoperability.md) |
| know what runs on GPU / many processes | [Performance and scaling](performance-and-scaling.md) |
| see every model option | [Model configuration](configuration.md) |
| fix an error | [Troubleshooting](troubleshooting.md) |
| contribute or release | [Contributing](contributing.md), [Publishing](publishing.md) |

## What it is, and is not

* It **is** a small, readable, tested codebase for learning, research and verification at small scale, plus a
  model-agnostic way to test causal language models.
* It is **not** a framework for training large models: single machine, data parallel only (no FSDP, tensor or pipeline
  parallelism), no pretrained weights, no serving engine. See [Limitations](limitations.md).

## Verified versus unverified

Everything in these docs that says "tested" was run on CPU (Linux, Python 3.11 and 3.13, and the test suite on Windows).
A single Tesla T4 on Colab ran the test suite, `validate`, `verify`, bf16 and fp16 once. FlashAttention kernels and
multi-GPU training have **not** been executed; the docs mark those paths as unverified, and `scripts/gpu_check.sh` is
how you verify them on your hardware.
