# Controlled experiments (`oglodon compare`)

Question: "is variant B better than variant A?" Answer it by training both under identical conditions and reading a
report that says how much of the difference could be noise.

```bash
oglodon compare configs/mha.yaml configs/gqa.yaml --out comparison.json --plot     # one config file per variant
oglodon compare configs/compare_positions_ffn.yaml                                 # or one file with a variants: map
```

## What is held constant

Same tokenized data and train/validation split, same batch order per seed, same number of training tokens, same
hardware, several seeds (default 3). Parameter counts are matched by adjusting the feed-forward width (`match_params`).

## One config with variants

```yaml
model: {vocab_size: 1, hidden_size: 64, num_layers: 2, num_heads: 8, num_kv_heads: 8, max_seq_len: 96}
variants:
  mha: {}
  gqa: {num_kv_heads: 2}
  mqa: {num_kv_heads: 1}
  alibi: {use_rope: false, use_alibi: true}
  geglu: {ffn_activation: gelu}
baseline: mha
data: {block_size: 32, val_fraction: 0.1}      # or {path: folder/, tokenizer: bpe, vocab_size: 512}
train: {steps: 200, batch_size: 16, learning_rate: 0.003, seeds: [0, 1, 2], match_params: true, validate: true}
```

A variant is any set of `OglodonConfig` overrides: attention layout (MHA/GQA/MQA), position method (RoPE, learned, ALiBi),
RMSNorm vs LayerNorm, feed-forward (SwiGLU, GeGLU, GELU, ReLU²), sliding window, and so on. With several files, each
file's `model:` section is a variant named after the file; `data:` and `train:` must match across files (otherwise the
comparison is not controlled and the command refuses).

## The report (JSON plus Markdown)

| section | content |
|---|---|
| summary | validation loss and perplexity (mean ± std over seeds), parameters, training tokens/s (evaluation time excluded), forward and per-token generation latency, KV-cache bytes, peak memory (CUDA only, otherwise `null`) |
| paired differences | per-seed difference against the baseline, which removes seed-to-seed variation |
| validation loss curves | mean over seeds at each evaluation step (`--plot` draws them) |
| configuration differences | exactly which fields differ from the baseline |
| validation | each variant's result against the float64 reference |
| reproducibility | library versions, seeds, corpus size, tokenizer, SHA-256 of every variant config and of the source files |
| fairness notes | automatic warnings: parameter gap, noisy CPU timing, few seeds, differences within noise |

Read the fairness notes before the numbers. Differences smaller than the seed-to-seed spread are noise. The bundled
configs are demonstrations; no benchmark numbers ship with the package, because speed and quality depend on your
hardware and data.

## From Python

```python
from oglodon import OglodonConfig
from oglodon.experiments import ExperimentData, run_comparison

data = ExperimentData.from_text(text, block_size=32)
base = OglodonConfig(vocab_size=1, hidden_size=64, num_layers=2, num_heads=8, max_seq_len=96)
report = run_comparison(
    base, {"mha": {"num_kv_heads": 8}, "gqa": {"num_kv_heads": 2}}, data, steps=200, seeds=(0, 1, 2)
)
print(report.to_markdown())
report.save("comparison.json")
report.plot("curves.png")
```
