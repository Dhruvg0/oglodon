# Changelog

## 0.4.1

### Fixed
- Text files (corpus, evaluation text, JSON configs and reports) are now always read and written as UTF-8. On Windows
  `oglodon eval --text` failed on ordinary UTF-8 text (for example curly quotes) because Python used the system code page.

## 0.4.0 — research toolkit

### Added
- **Comparison reports** (`oglodon compare`): validation-loss curves (mean over seeds, optional `--plot` PNG), a
  per-variant configuration diff, reproducibility metadata (library versions, seeds, corpus size, tokenizer, SHA-256
  of every variant config and of the source config files). Throughput now excludes periodic evaluation time.
  `oglodon compare a.yaml b.yaml` takes one config file per variant (data/training sections must match).
- **Position and feed-forward variants**: `use_alibi` (ALiBi, no position parameters) and `ffn_activation`
  (`silu` / `gelu` / `relu2`, gated or plain). Defaults are unchanged. The float64 reference, parameter-count check and
  `verify` understand them; `configs/compare_positions_ffn.yaml` compares RoPE, learned, ALiBi, GeGLU and ReLU².
- **Generation**: `generate_stream` / `Oglodon.generate_stream` (token by token), `iter_text` (incremental text that
  never emits half a UTF-8 character), `stop_token_ids`, `stop_sequences`, `repetition_penalty`, `min_p`.
  `generate` is now exactly the stream concatenated onto the prompt; outputs with default arguments are unchanged.
- **Evaluation** (`oglodon.evaluation`, `oglodon eval`): sliding-window perplexity, log-likelihood multiple choice
  (acc / acc_norm / acc_token_norm, standard error) and generation sanity checks; all verified against naive references.
- **Hugging Face export**: `to_hf_llama` / `export_hf` write Llama, Mistral (sliding window) or Qwen2 (q/k/v bias)
  checkpoints that `AutoModelForCausalLM` loads; logits, loss and greedy generation match to ~1e-6. Architectures
  Llama cannot express (ALiBi, learned positions, LayerNorm, non-SwiGLU) raise `NotImplementedError`.
- **Training**: gradient checkpointing (`Trainer(gradient_checkpointing=True)`, identical gradients), optional
  `torch.compile` (`compile=True`), and multi-process data parallelism (`distributed=True`, `setup_distributed`,
  `distributed_loader`): losses are weighted by the global token count, and two CPU processes reproduce the
  single-process update (tested with gloo). A GPU workflow file is included but has not been run; `scripts/gpu_check.sh` was run once on a Tesla T4.

- **No-code workflow**: `oglodon train DATA --out DIR` (files or folders to BPE tokenizer, token file, training,
  checkpoint and `training_report.json`; `--config`, exact `--resume`, refuses to overwrite a model, automatic
  data-parallel under `torchrun`) and `oglodon generate DIR` (streaming output, sampling flags, stop strings, stdin
  prompts). The same code is available as `oglodon.workflow.train_from_files` / `complete`; `python -m oglodon` works.
- **Int8 weight-only quantization** (`oglodon quantize`, `quantize_int8`, `quantization_report`): pure PyTorch,
  per-channel or per-group scales, block linears only (tied output matrix, embeddings, norms, biases stay float),
  saved as a normal checkpoint plus `quantization.json`. A drift report (perplexity, KL, top-1 agreement) is measured
  on your own text. Inference only; no speed gain on CPU; not run on GPU. `validate` refuses int8 models, `verify` works.
- **Documentation**: a `docs/` folder (getting started, CLI and configuration reference, training, data, generation,
  evaluation, experiments, interoperability, performance and scaling, troubleshooting, limitations, contributing,
  publishing) with `mkdocs.yml` (`pip install -e ".[docs]"`), and tests that fail if a documented flag does not exist,
  a Python snippet does not parse, or a documented config does not load.
- **Release workflow** (`.github/workflows/release.yml`): tag `v*` builds, checks the tag equals the version and
  publishes through PyPI trusted publishing; a manual TestPyPI dry run is included. Not executed by the author.

### Changed
- With `attn_impl="sdpa"`, plain causal attention (no padding mask, window or ALiBi) uses SDPA's causal fast path, which
  is what allows FlashAttention kernels on supported GPUs. Results are the same as the masked path (tested).

### Fixed
- Sampling with a CPU-seeded `torch.Generator` on a CUDA model crashed (found by running `oglodon validate` on a Tesla T4).
- `verify --cross-check` ignored `--dtype` and `--device`: the comparison with Hugging Face always ran at the source
  model's own precision (bfloat16 for SmolLM2), although the report header said float32. It now runs at the requested
  precision and device. Found by reading the SmolLM2-135M log; the float32 re-run then passed all 23 checks.
- `verify --cross-check`: the greedy-generation comparison no longer fails on harmless near-ties. It compares the two
  implementations on the tokens the source generated wherever the top-1 lead exceeds twice the measured logit
  difference. A real difference still fails, and if no position is decisive the check is reported as skipped, never passed.
- README: checkpoints do store `BPETokenizer` (the text claimed only `CharTokenizer`); the pure-Python BPE speed is stated.

### Not done (on purpose)
- Gemma / Phi / Mixtral converters need model features Oglodon lacks (tanh-GELU, unit-offset norm, embedding scaling,
  partial rotary, MoE). FSDP, 4-bit quantization and CUDA runs are not
  implemented or not executed here.

## Renamed: DGX is now Oglodon
- Package `oglodon-transformer`, import `oglodon`, command `oglodon`; classes `Oglodon`, `OglodonConfig`, `OglodonOutput`.
- Checkpoints are unchanged (same `config.json` keys and `model.safetensors` names), so models saved under the old name load with `Oglodon.from_pretrained`. Python code needs the new import names; no `dgx` alias is shipped.
- Earlier entries below were written under the old name.

## 0.3.0 — verify any model, train on your own data

### Added (scale and reach)
- `qkv_bias` config option (default off, fully backward compatible) and a Qwen2 converter: Oglodon equals Hugging Face
  Qwen2 to ~3e-7 including random q/k/v biases; the independent reference and the parameter-count check understand biases.
- Stress weights now also randomise biases (they start at zero almost everywhere, which hid bias bugs); a wrong bias
  re-ordering in a converter is caught.
- `oglodon verify --no-stress`, `--max-fp64-params`, `oglodon info`, and `scripts/gpu_check.sh` (CUDA runbook that refuses to
  run without CUDA). Local Hugging Face checkpoints (`--hf <dir>`) are tested.

### Added (cross-implementation verification)
- `oglodon.convert.from_hf_llama` (Llama, Mistral incl. sliding window, GQA/MHA/MQA, tied/untied) and
  `oglodon.verify.verify_conversion` / `oglodon verify --cross-check`: compares two independent implementations with identical
  weights (logits, padded logits, loss, cached greedy generation). Oglodon equals Hugging Face to ~3e-6; planted RoPE and
  softmax-scale bugs are caught; unsupported architectures are skipped, never passed.
- Trained-weights tests: Llama and Mistral are trained for a few hundred steps inside the test, then verified and cross-checked.

### Changed
- BPE training uses a lazy-deletion heap: 3-4x faster with identical merges (verified against a slow reference in the tests).
- `tokenize_to_bin(..., workers=N)`: parallel, byte-identical output; workers are spawned (use `if __name__ == "__main__":`).

### Added
- `oglodon.verify` / `oglodon verify`: model-agnostic behavioural verification of any causal LM (Hugging Face, custom
  PyTorch, Oglodon) by metamorphic invariants: causality, padding, batching, KV cache, numerics, finite-difference
  gradients, loss convention. 18 checks, run on given and stress weights; skipped/unsupported never count as passed.
  Sources: `--hf`, `--hf-config` (offline random weights), `--factory module:fn`, `--checkpoint`, `--config`.
- `BPETokenizer`: trainable, lossless, deterministic byte-level BPE; stored in and loaded from checkpoints.
- `oglodon.data`: `collect_text_files`, `tokenize_to_bin`, `TokenFile`, `WindowDataset` (memmap, non-overlapping train/val).
- `ExperimentData.from_text(bpe_vocab_size=...)` fits BPE on the training split only; `oglodon compare` accepts
  `data: {path, tokenizer: bpe, vocab_size}` for folders/globs.
- Examples 08 (BPE + files + training) and 09 (verify a Hugging Face model); docs/verify.md.

### Verified on more architectures (random weights, CPU)
- `oglodon verify` passes all 18 checks on Hugging Face Mistral (sliding window, GQA), Qwen2, Gemma (MQA), Phi, GPT-NeoX, OPT and Falcon; Bloom (ALiBi, no `position_ids`) passes with the padding check honestly skipped; Mixtral (MoE) has no failures.
- mypy is now clean (0 errors, was 78) and enforced in CI. Real fixes: `KVCache.update` no longer relies on re-reading Optional entries; `Trainer`/inspector are typed as `Oglodon`.

### Fixed
- BPE pre-tokenisation initially dropped `_` (found by a round-trip test before release); the pattern is now exhaustive and fuzz-tested.
- CLI usage/config errors exit 2 (check failures remain exit 1).

## 0.2.0 — behaviour-verified lab

### Fixed (found by audit; each has a regression test)
- Exact resume: restores data position, RNG and AMP scaler; RNG restored at the right draw (previously dropout diverged when a checkpoint fell on an epoch boundary). New `Trainer.train(until=)`.
- Micro-batches / eval batches with no supervised tokens no longer produce NaN loss; training skips them, eval ignores them.
- `Trainer.evaluate` is token-weighted (was an unweighted mean of batch means).
- New `Trainer.compute_gradients`: token-weighted accumulation, equal to one larger batch.
- `generate(eos_token_id=None)` now disables EOS (unset still uses the config).
- `position_ids` shape/range validated; `KVCache.reset()` added; cache batch/shape mismatch raises.
- Float64 models no longer silently computed in float32 (RMSNorm, eager softmax, RoPE tables, loss). Float32 behaviour is unchanged; bf16/fp16 softmax and RMSNorm statistics stay in float32. RoPE tables are now built in float64 (rounded to the model dtype as before).
- CLI: unreadable/invalid config gives a clean error instead of a traceback.

### Added
- `oglodon.validation`: 26 checks (attention, kv_cache, model, training) against an independent float64 reference; pass/fail/warning/skipped/unsupported statuses; JSON reports with reproducibility metadata. `oglodon.testing` remains as an alias.
- `oglodon` CLI: `oglodon validate`, `oglodon compare` (YAML needs the `yaml` extra; JSON works without it).
- `oglodon.experiments`: controlled MHA/GQA/MQA comparison (same data/split/batches/tokens, seeds, parameter matching, paired stats, fairness notes).
- `attn_impl="eager"|"sdpa"` and `sliding_window` config options (defaults preserve old behaviour).
- `oglodon.data.make_corpus`; `configs/`; `docs/validation.md`; `examples/07_validate_and_compare.py`.
- Tests: validation (incl. 13 planted bugs), audit regressions, experiments, CLI.

### Known limitations
- CPU only was available: no CUDA, no fp16 execution. 
