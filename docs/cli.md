# Command-line reference

`oglodon <command> [options]` (also `python -m oglodon ...`). Run `oglodon <command> --help` for the authoritative
list; this page explains what each command does.

**Exit codes**: `0` success; `1` a check failed (`validate`, `verify`; with `--strict` also when something was
skipped or unsupported); `2` usage or configuration error (bad flag, missing file, unusable settings).
Errors print one `error: ...` line, not a traceback.

## `oglodon train DATA... --out DIR`

Train on text files. `DATA` is any mix of files, folders (searched recursively for text-like extensions such as
`.txt .md .py .json .csv .tex`) and glob patterns.

| option | default | meaning |
|---|---|---|
| `--out DIR` | required | checkpoint folder |
| `--config FILE` | | YAML/JSON with `model:`, `data:`, `train:` sections; flags override it |
| `--tokenizer {bpe,char}` | `bpe` | byte-level BPE trained on the training split, or one symbol per character |
| `--vocab-size N` | 1024 | BPE vocabulary (the result can be smaller on a small corpus) |
| `--val-fraction F` | 0.05 | held-out share of the tokens |
| `--workers N` | 1 | parallel tokenization processes |
| `--block-size N` | 128 | context length (`max_seq_len`) |
| `--hidden-size N`, `--layers N`, `--heads N` | 128, 4, 8 | model size |
| `--kv-heads N` | heads // 4 | key/value heads (GQA); `--kv-heads` equal to `--heads` is MHA |
| `--steps N` | 1000 | optimizer steps |
| `--batch-size N` | 16 | per process |
| `--grad-accum N` | 1 | micro-batches per step |
| `--lr F`, `--weight-decay F`, `--warmup N` | 1e-3, 0.1, 5% of steps | optimization |
| `--seed N` | 0 | seeds data order and initialization |
| `--precision {fp32,bf16,fp16}` | `fp32` | autocast; `fp16` needs CUDA |
| `--eval-interval N`, `--save-interval N` | 10 evals/run, end only | evaluation and checkpoint cadence |
| `--grad-checkpointing` | off | recompute activations: less memory, more compute |
| `--compile` | off | `torch.compile` (first step is slow) |
| `--device D` | cuda if available | `cpu`, `cuda`, `cuda:1`, ... |
| `--resume` | off | continue the run in `--out` exactly where it stopped |
| `--overwrite` | off | allow replacing a model already in `--out` |

Without `--resume` or `--overwrite`, `train` refuses to touch a folder that already holds a model. Under
`torchrun` it trains data-parallel automatically: see [Performance and scaling](performance-and-scaling.md).
Details: [Training](training.md).

## `oglodon generate CHECKPOINT`

Stream text from a checkpoint made by `oglodon train` (or `save_pretrained(..., tokenizer=...)`).

| option | default | meaning |
|---|---|---|
| `--prompt`, `-p TEXT` | stdin | text to continue; empty means unconditional |
| `--max-new-tokens N` | 100 | must be below the model's context length |
| `--temperature F` | 0.8 | `0` is greedy |
| `--top-k N`, `--top-p F`, `--min-p F` | 40, off, off | sampling filters (`--top-k 0` disables) |
| `--repetition-penalty F` | 1.0 | above 1 discourages repeats |
| `--stop TEXT` | | stop after this text; repeatable |
| `--seed N` | | reproducible sampling |
| `--only-new` | off | do not echo the prompt |
| `--no-stream` | off | print once at the end |

Details: [Generation](generation.md).

## `oglodon eval CHECKPOINT`

At least one of: `--text FILE` (perplexity; `--block-size`, `--stride`, `--batch-size`), `--choices FILE.jsonl`
(log-likelihood multiple choice), `--prompt P [P ...]` (greedy generation sanity check, `--max-new-tokens`).
`--out FILE` writes a JSON report. Details: [Evaluation](evaluation.md).

## `oglodon quantize CHECKPOINT --out DIR`

Writes an int8 weight-only copy of a checkpoint (about 4x smaller linear weights; see [Quantization](quantization.md)).

| flag | default | meaning |
|---|---|---|
| `--out DIR` | required | where the int8 checkpoint goes |
| `--group-size N` | per channel | one scale per N input channels; must divide every quantized layer's input width |
| `--include-lm-head` | off | also quantize the output matrix (only when embeddings are not tied) |
| `--text FILE` | none | measure perplexity, KL and top-1 agreement on this text; without it drift is **not** measured |
| `--block-size N` | model maximum | window length for that measurement |

The result loads with `Oglodon.from_pretrained`, `oglodon generate` and `oglodon eval`.

## `oglodon compare CONFIG [CONFIG ...]`

Train architecture variants under identical conditions and report. One config with a `variants:` map, or several
config files (one variant each; the first is the baseline). `--out FILE.json` (also writes `.md`), `--plot`
(validation-loss curves as PNG, needs the `[viz]` extra), `--device`. Details: [Experiments](experiments.md).

## `oglodon verify`

Behavioural checks on any causal LM. Choose the model with exactly one of `--hf ID_OR_DIR`, `--hf-config FILE_OR_DIR`
(random weights, offline), `--factory package.module:function`, `--checkpoint DIR`, `--config FILE`. Other options:
`--cross-check`, `--categories`, `--exclude`, `--no-stress`, `--max-fp64-params`, `--device`, `--dtype`, `--seed`,
`--vocab-size`, `--strict`, `--out`. Details: [verify](verify.md).

## `oglodon validate CONFIG`

Compare an Oglodon model against the independent float64 reference. `--checkpoint DIR`, `--categories`, `--device`,
`--dtype`, `--seed`, `--strict`, `--out`. Details: [validation](validation.md).

## `oglodon info`

Prints versions, CUDA availability and bf16 support as JSON.
