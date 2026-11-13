# Training

Two ways in: the `oglodon train` command (text files to checkpoint) or the `Trainer` class (your own datasets).

## `oglodon train` in detail

```bash
oglodon train corpus/ --out run1 --steps 2000 --hidden-size 256 --layers 6 --heads 8 --block-size 256
```

Pipeline: collect files, train the tokenizer on the **training part of the text only**, tokenize to a memory-mapped
file, split train/validation at one cut point (no overlap), train, evaluate ten times per run, write the checkpoint and
`training_report.json`.

### Settings

Every setting can come from a config file (`--config`), a flag, or the default; flags beat the file, the file beats
defaults. The file has optional `model:`, `data:`, `train:` sections (they are merged; the names are the fields of
`oglodon.workflow.TrainSettings`):

```yaml
model:
  hidden_size: 256
  num_layers: 6
  num_heads: 8
  num_kv_heads: 2
  block_size: 256
data:
  tokenizer: bpe
  vocab_size: 4096
  val_fraction: 0.05
train:
  steps: 5000
  batch_size: 32
  learning_rate: 0.001
  precision: bf16
  model: {use_alibi: true, use_rope: false}   # any other OglodonConfig field
```

| setting | default | note |
|---|---|---|
| `tokenizer`, `vocab_size` | `bpe`, 1024 | `tokenizer_sample_mb` (64) caps the text the BPE merges are learned from |
| `val_fraction` | 0.05 | |
| `block_size` | 128 | context length of the model |
| `hidden_size`, `num_layers`, `num_heads` | 128, 4, 8 | |
| `num_kv_heads` | `num_heads // 4` | GQA; set equal to `num_heads` for MHA |
| `model` | `{}` | extra `OglodonConfig` fields, see [configuration](configuration.md) |
| `steps`, `batch_size`, `grad_accum_steps` | 1000, 16, 1 | batch size is per process |
| `learning_rate`, `weight_decay`, `warmup_steps`, `lr_schedule` | 1e-3, 0.1, 5% of steps, `cosine` | `linear` and `constant` also exist |
| `max_grad_norm`, `seed`, `precision` | 1.0, 0, `fp32` | |
| `gradient_checkpointing`, `compile` | off, off | see [performance](performance-and-scaling.md) |
| `eval_interval`, `save_interval` | `steps // 10`, end only | |

An unknown key in the config is an error that lists the valid names.

### Reading the result

The last line compares validation perplexity with the **uniform baseline** (the vocabulary size). The report
(`training_report.json`) holds the same plus tokens/s, parameter count, library versions, the file list with SHA-256
hashes, a fingerprint of the tokenized corpus and the whole loss history. Treat tokens/s as a property of *your
machine*; nothing in this repository claims a speed.

If the corpus is smaller than a few blocks you get a clear error (use more text, a smaller `--block-size`, or a smaller
`--val-fraction`); if there are fewer training windows than the batch size, the batch is lowered with a note.

### Resuming

```bash
oglodon train corpus/ --out run1 --steps 2000 --save-interval 200     # killed at step 1300
oglodon train corpus/ --out run1 --resume                              # continues from step 1200
```

`--resume` reuses the saved settings, tokenizer and tokenized data, restores model, optimizer, schedule, random state
and the position in the data, and continues **as if it had never stopped**: the test suite checks that a crashed and
resumed run ends with the same weights as an uninterrupted one. Passing a different `--steps` stretches the
learning-rate schedule (the run is then no longer identical to an uninterrupted one, and a note says so).

## `Trainer` (your own data)

```python
from torch.utils.data import DataLoader
from oglodon import Oglodon, OglodonConfig, Trainer

model = Oglodon(OglodonConfig(vocab_size=1000, hidden_size=128, num_layers=4, num_heads=8, max_seq_len=128))
trainer = Trainer(
    model,
    DataLoader(train_ds, batch_size=16, shuffle=True),
    DataLoader(val_ds, batch_size=16),
    max_steps=2000,
    learning_rate=1e-3,
    warmup_steps=100,
    eval_interval=200,
    checkpoint_dir="run2",
    save_interval=500,
)
trainer.train()
trainer.save_checkpoint("run2", tokenizer=my_tokenizer)
```

Batches are dicts (`input_ids`, optional `labels` and `attention_mask`) or `(input_ids, labels)` tuples; labels use
`-100` to ignore positions. AdamW without decay on norms and embeddings, warm-up then cosine/linear/constant decay,
gradient clipping, and gradient accumulation weighted by the number of supervised tokens (so accumulation equals a
larger batch). `trainer.train(until=N)` stops early without changing the schedule. Options for memory and speed:
`gradient_checkpointing=True`, `compile=True`, `precision="bf16"`, `distributed=True`.

## What was and was not verified

Tested on CPU: loss weighting, accumulation equivalence, exact resume, gradient checkpointing (identical gradients),
`torch.compile` (matches eager at dropout 0), two-process data parallel equal to one process on the combined batch.
Not run: bf16/fp16 on a GPU, multi-GPU. See [Performance and scaling](performance-and-scaling.md).
