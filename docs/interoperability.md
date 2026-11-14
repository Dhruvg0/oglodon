# Interoperability with Hugging Face

Needs `pip install "oglodon-transformer[hf]"`.

## Import: Hugging Face to Oglodon

```python
from oglodon.convert import from_hf_llama

model = from_hf_llama(hf_model)  # LlamaForCausalLM, MistralForCausalLM or Qwen2ForCausalLM
```

Hugging Face pairs RoPE channels as (i, i + d/2) and Oglodon as (2i, 2i + 1); the converter re-orders the query/key rows
(and biases) per head. Unsupported variants raise `NotImplementedError` instead of converting something different:
attention-output or MLP biases, RoPE scaling, activations other than SiLU, Qwen2 sliding-window layers.

## Export: Oglodon to Hugging Face

```python
from oglodon.convert import export_hf, to_hf_llama

hf_model = to_hf_llama(model)  # in memory
export_hf(model, "hf_out")  # writes a folder AutoModelForCausalLM.from_pretrained("hf_out") loads
```

The exported class is chosen from the config: Llama; Mistral if `sliding_window` is set; Qwen2 if `qkv_bias` is set.
Logits, loss, left-padded logits and greedy generation match Hugging Face to about 1e-6 in the tests, including a disk
round trip. Tokenizers are not exported (save yours next to the folder).

| Oglodon feature | exportable? |
|---|---|
| RoPE, RMSNorm, SwiGLU (SiLU), MHA/GQA/MQA, tied/untied embeddings | yes (Llama) |
| `sliding_window` | yes (Mistral) |
| `qkv_bias` | yes (Qwen2) |
| `sliding_window` together with `qkv_bias` | no (no matching architecture) |
| ALiBi, learned positions, LayerNorm, GeGLU/ReLU² or a plain MLP | no, `NotImplementedError` naming the reason |

## Cross-implementation verification

```bash
oglodon verify --hf ./my-llama --cross-check
```

Converts the Hugging Face model to Oglodon and compares the two independent implementations with identical weights
(logits, padded logits, loss, cached greedy generation). On random and briefly trained weights they agree to about
3e-6. Run it on your real model to get evidence about *that* model. Details: [verify](verify.md).

Use `--dtype float32` for a meaningful comparison: the cross-check runs at the precision you ask for, and bfloat16
logits differ by about 3% of their scale, so only a loose tolerance can be applied there.

**One real model so far.** `HuggingFaceTB/SmolLM2-135M` (Llama architecture) was cross-checked once on a CPU. All
structure, causality, cache, numerics, gradient and logit-agreement checks passed. That run happened at bfloat16 (the
model's own precision) because the cross-check ignored `--dtype` at the time (since fixed); its greedy-generation
comparison compared no positions. A float32 re-run with the fixed code has **not** been done yet.

## Running standard benchmarks

Export the model, then use a harness that reads Hugging Face folders, for example
`lm_eval --model hf --model_args pretrained=hf_out --tasks hellaswag` (you also need a Hugging Face tokenizer saved in
that folder). This route is documented but has not been run by the author.

## Not supported

Int8 checkpoints cannot be exported to Hugging Face (export the float checkpoint instead).

Converters for Gemma, Phi, Mixtral (MoE) and other architectures: they need model features Oglodon does not have
(tanh-GELU, unit-offset norm, embedding scaling, partial rotary, mixture of experts). `oglodon verify` still checks
such models behaviourally; it just cannot cross-check them.
