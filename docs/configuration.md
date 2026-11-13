# Model configuration (`OglodonConfig`)

```python
from oglodon import OglodonConfig

cfg = OglodonConfig(
    vocab_size=4096, hidden_size=256, num_layers=6, num_heads=8, num_kv_heads=2, max_seq_len=512
)
cfg.save("config.json")
cfg = OglodonConfig.load("config.json")  # also written inside every checkpoint
```

The constructor validates everything and raises `ValueError` with a specific message (divisibility of `hidden_size` by
`num_heads` and of `num_heads` by `num_kv_heads`, positive sizes, an even head dimension for RoPE, dropout in `[0, 1)`,
special-token ids inside the vocabulary, incompatible combinations such as ALiBi together with RoPE).

| field | default | meaning |
|---|---|---|
| `vocab_size` | required | number of token ids |
| `hidden_size` | 256 | model width |
| `num_layers` | 6 | Transformer blocks |
| `num_heads` | 8 | query heads |
| `num_kv_heads` | `num_heads` | key/value heads: equal to `num_heads` is MHA, `1` is MQA, in between is GQA |
| `max_seq_len` | 512 | longest sequence (prompt + generated); a hard limit |
| `intermediate_size` | ~8d/3 (SwiGLU) or 4d | feed-forward width |
| `dropout` | 0.0 | attention weights, residual branches, feed-forward hidden layer |
| `use_rope` | True | rotary positions; if False, learned absolute positions (unless `use_alibi`) |
| `use_alibi` | False | ALiBi distance biases instead of any position embedding; needs `use_rope=False`; no parameters |
| `rope_base` | 10000 | RoPE frequency base |
| `use_rmsnorm` | True | RMSNorm, else LayerNorm |
| `norm_eps` | 1e-6 | |
| `use_swiglu` | True | gated feed-forward, else plain MLP |
| `ffn_activation` | `None` | `silu`, `gelu` or `relu2`; `None` means SiLU for the gated layer and GELU for the plain MLP |
| `qkv_bias` | False | learnable q/k/v biases (Qwen2 style) |
| `tie_embeddings` | True | share input embedding and output projection |
| `init_std` | 0.02 | weight initialization |
| `attn_impl` | `eager` | `eager` (explicit softmax, supports attention maps) or `sdpa` (PyTorch fused attention; same maths) |
| `sliding_window` | None | each token sees only the last N tokens; changes the mask, not the cache size |
| `pad_token_id`, `eos_token_id` | None | used by generation |

`forward(input_ids, attention_mask=None, position_ids=None, labels=None, past_key_values=None, use_cache=False,
output_attentions=False)` returns a logits tensor, or an `OglodonOutput(logits, loss, past_key_values, attentions)`
when you ask for labels, cache or attentions. Labels are **unshifted** (the model shifts them); `-100` is ignored.

Checkpoints written before a field existed still load: missing fields take their defaults.
