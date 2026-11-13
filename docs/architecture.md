# Architecture

```
tokens (B,T)
   |  embed_tokens                      (+ learned positions if use_rope=False and not use_alibi)
   v
x (B,T,d) --+--------------------------------------------------------------+
            |   TransformerBlock x num_layers                              |
            |     h = x + Attn(Norm1(x))      q,k RoPE (or ALiBi bias)     |
            |     y = h + FFN(Norm2(h))       FFN = gated or plain MLP     |
            +--------------------------------------------------------------+
   |  final_norm
   v
lm_head (tied with embed_tokens by default)  ->  logits (B,T,V)
```

## Attention layouts

`num_heads` query heads share `num_kv_heads` key/value heads; query head `h` reads K/V head
`h // (num_heads // num_kv_heads)`.

| layout | `num_kv_heads` | KV-cache size vs MHA |
|---|---|---|
| MHA | `num_heads` | 1x |
| GQA | `1 < k < num_heads` | `k / num_heads` |
| MQA | `1` | `1 / num_heads` |

## Position methods and feed-forward variants

| option | effect |
|---|---|
| `use_rope=True` (default) | rotary positions applied to q and k |
| `use_rope=False` | learned absolute position embeddings |
| `use_rope=False, use_alibi=True` | no position parameters; a linear distance penalty (per-head slope) is added to the attention scores |
| `use_swiglu=True` with `ffn_activation` `silu` / `gelu` / `relu2` | gated feed-forward: SwiGLU / GeGLU / ReLU²-gated |
| `use_swiglu=False` with `ffn_activation` `gelu` / `silu` / `relu2` | plain two-matrix MLP |

With `attn_impl="sdpa"`, plain causal attention (no padding mask, window or ALiBi) calls PyTorch's fused attention with
`is_causal=True`; every other case passes an explicit mask. Both give the same result (tested).

## Package map

| package | role |
|---|---|
| `oglodon.layers`, `oglodon.models` | attention, RoPE, norms, feed-forward, KV cache, the model |
| `oglodon.generation` | sampling, streaming, stop criteria |
| `oglodon.training` | `Trainer`, checkpoints, distributed helpers |
| `oglodon.workflow` | text files to checkpoint (`train_from_files`), `complete` |
| `oglodon.data`, `oglodon.tokenization` | token files, datasets, BPE/char tokenizers |
| `oglodon.evaluation` | perplexity, multiple choice, generation checks |
| `oglodon.validation` | float64 reference checks for Oglodon models |
| `oglodon.verify` | model-agnostic behavioural checks, cross-implementation equivalence |
| `oglodon.convert` | Hugging Face import and export |
| `oglodon.experiments` | controlled comparisons and reports |

## Where each notebook component went

| MiniGPT notebook | Oglodon |
|---|---|
| `RMSNorm` | `oglodon.layers.normalization.RMSNorm` (+ `nn.LayerNorm` option) |
| `rope_tables`, `apply_rope` | `oglodon.layers.rope.RotaryEmbedding`, `apply_rope` (same interleaved pairing) |
| `Attention` (GQA, concat cache) | `oglodon.layers.attention.CausalSelfAttention` + `oglodon.layers.cache.KVCache` |
| `SwiGLU` | `oglodon.layers.feedforward.SwiGLU` (+ `GELUMLP`) |
| `Block`, `MiniGPT` | `oglodon.models.transformer.TransformerBlock`, `oglodon.models.gpt.Oglodon` |
| char vocab, `enc`/`dec` | `oglodon.tokenization.CharTokenizer` |
| `make`, `batch`, `acc` | `oglodon.data.algorithmic` (`make_example`, `AlgorithmicDataset`, `exact_match`) |
| training loop | `oglodon.training.Trainer` |
| causality / cache checks | `oglodon.validation.validate_model` and `tests/` |

## Changes relative to the notebook (deliberate)

* **Masking uses the dtype's most negative finite value instead of `-inf`**, so a fully masked row
  (a padding query) gives a uniform softmax instead of NaN. This enabled padding masks.
* **Padding masks and `position_ids`** (the notebook only handled equal-length batches).
* **Dropout, optional learned positions, LayerNorm and GELU options**, all configurable.
* **GPT-2-style initialisation** (`N(0, 0.02)`, residual projections scaled by `1/sqrt(2L)`).
* **EOS handling** in generation (the notebook generated a fixed number of tokens), plus
  temperature / top-k / top-p sampling.
* **Trainer** averages loss over supervised tokens and weights gradient-accumulation micro-batches by
  token count, so accumulation is equivalent to a larger batch.
* The notebook's answer terminator `;` became a real `<eos>` token.

## Design rules

* Attention knows nothing about positions: RoPE is applied to `q`/`k` by the caller-supplied
  `rope` tuple, so swapping position schemes does not touch attention code.
* Attention maps are only returned when `output_attentions=True`; they are never stored on modules.
* The Trainer only calls `model(**batch)` and reads `.loss`; any module with that contract trains.
