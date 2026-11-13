# Generation

## Command line

```bash
oglodon generate run1 --prompt "The history of" --max-new-tokens 120 --temperature 0.7 --top-p 0.9 --seed 1
echo "A story about" | oglodon generate run1 --only-new
oglodon generate run1 --prompt "Q: " --stop $'\n\n' --temperature 0   # bash: a blank line stops it
```

Text streams as it is produced. A prompt longer than the context window keeps its **last** tokens;
`--max-new-tokens` must be smaller than the context length. `--temperature 0` is greedy. Stop strings are tokenized and
matched as token sequences, and the model's end-of-text token also stops generation.

## Python

```python
from oglodon.generation import generate, generate_stream, iter_text

out = model.generate(ids, max_new_tokens=50, temperature=0.8, top_k=40)  # (B, T + new)
for step in model.generate_stream(ids, max_new_tokens=50):  # (B,) tensor per step
    ...
for piece in iter_text(tok, model.generate_stream(ids, max_new_tokens=50)):  # text pieces for row 0
    print(piece, end="", flush=True)
```

Options (same for `generate` and `generate_stream`):

| option | effect |
|---|---|
| `temperature`, `top_k`, `top_p`, `min_p` | sampling (`temperature=0` is greedy); `min_p` keeps tokens at least `min_p` times as likely as the best |
| `repetition_penalty` | above 1 makes tokens already seen less likely (also affects greedy decoding) |
| `eos_token_id` | defaults to the config's; `None` disables EOS handling |
| `stop_token_ids`, `stop_sequences` | extra stop conditions; the matched tokens stay in the output |
| `attention_mask` | left-padded batches; position ids are derived from it |
| `use_cache` | KV cache on (default) or off; results are identical |
| `generator` | a seeded `torch.Generator` for reproducible sampling |

Finished rows are filled with the padding id while the rest of the batch continues, and generation ends early when
every row is done. `generate` is exactly the stream concatenated onto the prompt. `iter_text` never emits half of a
multi-byte character (byte-level BPE can split one across tokens): it holds the fragment back until it is complete.

## The KV cache

With `use_cache=True` each step feeds only the newest token; keys and values of earlier tokens (after RoPE) are kept
per layer. Grouped-query and multi-query attention store only `num_kv_heads` heads, shrinking the cache by
`num_heads / num_kv_heads`. The cache grows by concatenation (simple, not memory-optimal), and a sliding window changes
the mask but does not trim the cache. Cached and uncached generation are tested to be identical.
