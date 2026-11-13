# Performance and scaling

This page separates what has been **run** from what only has code behind it.

| feature | how | status |
|---|---|---|
| SDPA causal fast path | `attn_impl="sdpa"` | run on CPU, equal to the masked path; the FlashAttention kernels it enables on supported GPUs are **not run** |
| gradient checkpointing | `--grad-checkpointing`, `Trainer(gradient_checkpointing=True)` | run on CPU: identical gradients (with dropout too), measurably fewer stored activations |
| `torch.compile` | `--compile`, `Trainer(compile=True)` | run on CPU: matches eager at dropout 0; dropout masks differ from eager because compiled code uses another random stream; not combinable with `distributed` |
| bf16 autocast | `--precision bf16` | run on CPU only |
| fp16 + loss scaling | `--precision fp16` | CUDA only, **not run** |
| data-parallel training | `torchrun` + `distributed=True` | run with 2 CPU processes (gloo): equals one process on the combined batch |
| multi-GPU (NCCL) | same | **not run** |
| int8 weight-only quantization | `oglodon quantize` | run on CPU: smaller files and memory, small measured drift on a tiny trained model; no speed gain; GPU **not run** |
| FSDP, tensor/pipeline parallelism, 4-bit | | not implemented |

## Data-parallel training

```bash
torchrun --nproc_per_node=4 -m oglodon train corpus/ --out run1 --batch-size 8 --steps 5000
```

`oglodon train` detects `torchrun` and wraps the model in `DistributedDataParallel`. Rank 0 prepares the data and
writes everything; the other ranks wait. `--batch-size` is per process, so the effective batch is
`batch_size x processes x grad_accum`. Losses are weighted by the **global** number of supervised tokens, so N
processes take the same optimizer step as one process with the combined batch (tested, including batches with very
different numbers of supervised tokens). Only rank 0 prints and saves.

From Python:

```python
from oglodon.training import Trainer, setup_distributed, distributed_loader

ctx = setup_distributed()  # reads torchrun's environment
loader = distributed_loader(dataset, batch_size=8, shuffle=True, seed=0)
Trainer(model, loader, distributed=True, device=ctx.device, max_steps=5000).train()
```

Call `loader.sampler.set_epoch(epoch)` yourself if you iterate epochs manually. Every process holds a full model copy.

## Checking your GPU

```bash
bash scripts/gpu_check.sh [optional-hf-model-id-or-path]
```

Runs the test suite, `oglodon validate` and `oglodon verify` on CUDA (and optionally verifies and cross-checks a real
Hugging Face model), writes every result to `reports/*.json`, and refuses to run (exit 2) when there is no CUDA device,
so it can never produce a fake CUDA result. `.github/workflows/gpu.yml` runs the same on a self-hosted runner labelled
`gpu`; it has not been executed by the author.

## Measuring speed honestly

`oglodon compare` reports tokens/s with evaluation time excluded; `oglodon.benchmarks.compare_architectures` times
forward and generation. Both measure *your* machine; CPU timings are noisy, and the report says so.
