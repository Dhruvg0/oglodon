# Troubleshooting

**`error: ... already contains a model`** — `oglodon train` will not overwrite a finished run. Use `--resume` to
continue it, `--overwrite` to replace it, or another `--out`.

**`error: nothing to resume`** — `--resume` needs `model.safetensors`, `trainer_state.pt` and `train_settings.json` in
`--out`; the last one is written at the start, the first two by `--save-interval` checkpoints or at the end.

**`no text files found`** — the path matched nothing. Folders are searched recursively for text-like extensions
(`.txt .md .py .json .csv .tex` and similar); pass a file or a glob (`'data/**/*.log'`) for other extensions.

**`the corpus is too small for block_size=...`** — fewer tokens than one context window. Add text, lower
`--block-size`, or lower `--val-fraction`.

**`YAML configs need PyYAML`** — `pip install "oglodon-transformer[yaml]"`, or write the config as JSON.

**`unknown training setting(s)`** — a typo in a config file. The message lists the valid names.

**`max_new_tokens must be in [1, N]`** — generation needs room in the context window. Lower it or train with a larger
`--block-size`.

**Loss does not fall / perplexity stays near the vocabulary size** — too few steps or too little data for the model;
check the learning rate (`--lr`), warm-up, and that the validation perplexity in the report is far below the uniform
baseline. Tiny corpora can be memorized: watch the gap between training and validation loss.

**Generated text is gibberish** — expected for models trained for minutes on small corpora. The report tells you the
perplexity; improve data, steps and size.

**`fp16 needs CUDA`** — use `--precision bf16` or `fp32` on CPU.

**First step with `--compile` takes a long time** — `torch.compile` compiles on the first call (tens of seconds on CPU)
and needs a C++ compiler there. Compiled dropout uses a different random stream than eager mode, so results differ
when `dropout > 0`.

**Multi-process tokenization hangs or errors in a script** — workers are spawned processes; guard the entry point with
`if __name__ == "__main__":`. The command line has no such problem.

**`torchrun` runs but only one process talks** — intended: only rank 0 prints and writes.

**Hugging Face `generate` disagrees with Oglodon on a prompt that contains the pad token id** — with `pad_token_id` set
and no attention mask, Hugging Face treats prompt tokens equal to the pad id as padding. Pass an explicit all-ones
`attention_mask`.

**`NotImplementedError: cannot export to a Hugging Face Llama-family model`** — your config uses something Llama
cannot express (ALiBi, learned positions, LayerNorm, a non-SwiGLU or non-SiLU feed-forward, or a sliding window with
q/k/v biases). The message names the reason; see [Interoperability](interoperability.md).

**`oglodon verify` reports `unsupported` for float64 checks on a big model** — the float64 copy is skipped above
`--max-fp64-params` (default 250M) to protect memory; add `--no-stress` and raise the limit if you have the memory.
`unsupported` and `skipped` are never counted as passed.

**Left-padded input gives different logits from a plain `model(x, attention_mask=...)` call** — a plain call does not
derive position ids from the mask; pass `position_ids` yourself. `generate(attention_mask=...)` does it for you.

**With `attn_impl="sdpa"`, gradients at padded positions differ from eager** — fully masked query rows are unspecified
there; keep padded positions out of the loss (labels `-100`).

**Old checkpoint will not load** — the format is pre-1.0; checkpoints from the project's earlier name (DGX) load
unchanged with the new import names.

Still stuck? Run `oglodon info` and open an issue with its output and the full `error:` line.
