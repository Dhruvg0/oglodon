# Limitations

Stated plainly so you can decide whether Oglodon fits.

* **Scale**: one machine, data-parallel only. No FSDP, tensor or pipeline parallelism, no pretrained weights, no serving
  engine; quantization is int8 weight-only (no 4-bit, no speed gain on CPU).
* **Not run on GPU**: CUDA, fp16, bf16 on GPU, FlashAttention kernels and multi-GPU training have code paths and CPU
  tests but no GPU execution by the author. Use `scripts/gpu_check.sh`.
* **Architectures**: one family (decoder-only, RoPE/learned/ALiBi positions, MHA/GQA/MQA, SwiGLU/GELU-style MLPs). No MoE,
  state-space models, encoder-decoder or multimodal models. Converters cover Llama, Mistral and Qwen2.
* **Verification is not proof**: `verify` and `validate` catch classes of bugs (leaks, off-by-ones, cache errors,
  numerical drift). A model can pass and still be a poor model. `skipped` and `unsupported` results are never passes.
* **Context**: `max_seq_len` is hard (no extension tricks); a sliding window does not shrink the KV cache, which grows by
  concatenation.
* **Tokenizer**: the built-in BPE is pure Python (about 4-5 MB/s per core).
* **Small-model behaviour**: models trained here show the usual length-generalization failure on algorithmic tasks, and
  small corpora give models that learn local structure, not fluent language.
* **Stability**: the version is pre-1.0; the checkpoint format may change.
