"""Verify a model that is NOT from Oglodon: a Hugging Face model (needs `pip install transformers`) and a custom one.

    python examples/09_verify_any_model.py
The same checks work on any causal LM: they test invariants (causality, padding, batching, KV cache, numerics,
gradients) instead of comparing against a reference implementation.
"""

from oglodon.verify import verify_causal_lm

try:
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    report = verify_causal_lm(LlamaForCausalLM(cfg))  # random weights: nothing is downloaded
    print(report)
    report.save("llama_verification.json")
except ImportError:
    print("transformers is not installed; skipping the Hugging Face example")

# Your own model: any nn.Module mapping input_ids (B, T) to logits (B, T, V) works.
# For unusual signatures, wrap it:  CausalLMAdapter(model, vocab_size=V, forward=lambda m, ids, mask, pos: m(ids))
