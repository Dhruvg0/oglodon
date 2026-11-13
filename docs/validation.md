# Behaviour validation

`oglodon.validation` checks a *model's behaviour*, not just that its code runs. Passing unit tests does not make a
Transformer correct, so every check compares Oglodon against something independent.

## Method

* **Independent reference** (`oglodon/validation/reference.py`): float64, written differently from the library.
  Masks come from a Python-scalar predicate (`j <= i + offset`, window, key padding), RoPE uses complex
  multiplication, GQA maps head `h` to KV head `h // group` by explicit loop, softmax and cross-entropy are
  hand-written. It shares no code with `oglodon.layers`.
* **Stress weights**: the model under test is deep-copied and re-initialised with std 0.3 in addition to the weights
  you gave it. Tiny default init (0.02) makes attention nearly uniform and hides masking and head-mapping bugs.
* **Scale-aware tolerances**: `tol * max(1, scale)`; tighter in float64, looser in bf16. Override with
  `tolerances={"ref": ..., "equiv": ...}`.
* **The validator never modifies your model** (deep copies, restores train/eval mode).
* **Proof that it can fail**: `tests/test_validation.py` monkeypatches in realistic bugs and asserts the relevant
  check reports FAIL. Add one whenever you add a check.

## Statuses

| status | meaning | counts as pass? |
|---|---|---|
| `pass` | ran and agreed within tolerance | yes |
| `fail` | ran and disagreed, or crashed | no (`report.failures`) |
| `warning` | ran, nothing wrong proven, but suspicious (e.g. all-zero weights, tokenizer pad id differs) | no |
| `skipped` | not run: excluded, or a precondition is missing (no tokenizer given) | no |
| `unsupported` | this environment cannot run it (fp16 without CUDA, no SDPA) | no |

`report.passed` = no failures and at least one pass. `report.complete` = nothing skipped or unsupported.
An empty run is not a pass.

## Selecting checks

`validate_model(model, config, categories=["attention"], select=["kv_cache."], exclude=["training.mixed"])`.
`select`/`exclude` match by name prefix. `device="cuda"` raises if CUDA is unavailable rather than silently using CPU.

## Report

`report.save(path)` writes strict JSON (NaN encoded as null): per-check status, detail, `max_abs_diff`, `tolerance`,
device, dtype, duration; plus metadata (Oglodon/torch/Python versions, platform, device, dtype, seed, UTC time, config and
its SHA-256, thread count, tolerances).

## Audit findings that shaped the checks

Found by probing and mutation testing the original library; all fixed, each with a regression test in
`tests/test_audit_fixes.py`:

1. Resume was not exact: data position, RNG and AMP scaler were not restored. Also, restoring RNG after the replay
   was off by one DataLoader-iterator draw whenever a checkpoint fell on an epoch boundary (dropout masks diverged).
2. Zero-supervised-token micro-batches (all labels `-100`) produced NaN loss in training and evaluation.
3. Eval loss was an unweighted mean of batch means; now token-weighted like training.
4. `generate(eos_token_id=None)` could not disable the config's EOS.
5. `position_ids` of the wrong shape, negative or beyond `max_seq_len` were not validated (silent wrong RoPE/embedding lookup).
6. `KVCache` accepted a batch-size/shape mismatch and had no `reset()`.
7. Float64 models were silently computed in float32 in RMSNorm, eager softmax, RoPE tables and the loss.
8. bf16/fp16 softmax and RMSNorm statistics now computed in float32.
9. Hazards documented rather than changed (to avoid altering behaviour): `model(x, attention_mask=...)` does not
   derive positions from the mask; SDPA gradients differ on fully masked pad rows; sliding-window does not trim the cache.

## Limits

CPU only was available during development. CUDA and fp16 paths have not been executed; the validator reports them
`unsupported` instead of passing them. Checks cover the single-device decoder in this repository only.
