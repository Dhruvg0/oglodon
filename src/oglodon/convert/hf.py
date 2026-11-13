"""Convert Hugging Face Llama-family models (Llama, Mistral, Qwen2) into :class:`oglodon.Oglodon`.

Why this exists: a model can satisfy every invariant and still compute the wrong function. The strongest
independent check is another implementation of the same architecture with the same weights. Hugging Face's
Llama is the most widely used one, so ``from_hf_llama`` lets the test-suite (and you) assert
``HF logits == Oglodon logits`` to float precision.

The only non-trivial step is RoPE: Hugging Face rotates the pairs ``(i, i + d/2)`` of each head, Oglodon rotates
the interleaved pairs ``(2i, 2i + 1)``. Re-ordering the output rows of ``q_proj`` / ``k_proj`` per head
(``new[2i] = old[i]``, ``new[2i+1] = old[i + d/2]``) turns one into the other; ``q . k`` is unchanged because
both are permuted identically, and ``v`` / ``o_proj`` are untouched.

Unsupported variants (biases, rope scaling, explicit ``head_dim``) raise ``NotImplementedError`` instead of
silently converting to something different.
"""

from __future__ import annotations

from typing import Any

import torch

from ..config import OglodonConfig
from ..models.gpt import Oglodon


def _rope_theta(cfg: Any) -> float:
    params = getattr(cfg, "rope_parameters", None)
    if isinstance(params, dict):
        kind = params.get("rope_type", "default")
        if kind not in ("default", None):
            raise NotImplementedError(f"rope scaling '{kind}' is not supported by Oglodon")
        if "rope_theta" in params:
            return float(params["rope_theta"])
    scaling = getattr(cfg, "rope_scaling", None)
    if scaling and (scaling.get("rope_type") or scaling.get("type")) not in (None, "default"):
        raise NotImplementedError(f"rope scaling {scaling} is not supported by Oglodon")
    return float(getattr(cfg, "rope_theta", 10000.0))


def hf_llama_config_to_oglodon(cfg: Any) -> OglodonConfig:
    """Map a Hugging Face ``LlamaConfig`` / ``MistralConfig`` / ``Qwen2Config`` to a :class:`OglodonConfig` (or raise)."""
    kind = getattr(cfg, "model_type", None)
    if kind not in ("llama", "mistral", "qwen2"):
        raise NotImplementedError(f"only llama, mistral and qwen2 are supported, got {kind!r}")
    if getattr(cfg, "attention_bias", False) or getattr(cfg, "mlp_bias", False):
        raise NotImplementedError(
            "attention-output / MLP biases are not supported by Oglodon (only q/k/v biases, as in Qwen2)"
        )
    if kind == "qwen2" and getattr(cfg, "use_sliding_window", False):
        raise NotImplementedError("qwen2 with use_sliding_window=True (per-layer windows) is not supported")
    if getattr(cfg, "hidden_act", "silu") != "silu":
        raise NotImplementedError(f"hidden_act={cfg.hidden_act!r}: Oglodon implements SwiGLU (silu) only")
    heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // heads
    if head_dim * heads != cfg.hidden_size:
        raise NotImplementedError("head_dim * num_heads != hidden_size is not supported by Oglodon")
    window = getattr(cfg, "sliding_window", None) if kind != "qwen2" else None
    return OglodonConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_hidden_layers,
        num_heads=heads,
        num_kv_heads=getattr(cfg, "num_key_value_heads", None) or heads,
        max_seq_len=cfg.max_position_embeddings,
        intermediate_size=cfg.intermediate_size,
        dropout=0.0,
        rope_base=_rope_theta(cfg),
        norm_eps=cfg.rms_norm_eps,
        tie_embeddings=bool(getattr(cfg, "tie_word_embeddings", False)),
        sliding_window=window if isinstance(window, int) and window > 0 else None,
        qkv_bias=kind == "qwen2",
        eos_token_id=None,
        pad_token_id=None,
    )


def _rope_rows(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Re-order the rows of a (n_heads * d, in) projection, or of a (n_heads * d,) bias, from half-split to
    interleaved RoPE pairing."""
    is_vec = w.dim() == 1
    d = w.shape[0] // n_heads
    w = w.view(n_heads, d, -1)
    out = torch.empty_like(w)
    out[:, 0::2] = w[:, : d // 2]
    out[:, 1::2] = w[:, d // 2 :]
    out = out.reshape(n_heads * d, -1)
    return out.squeeze(-1) if is_vec else out


@torch.no_grad()
def from_hf_llama(hf_model: Any) -> Oglodon:
    """Return a :class:`Oglodon` computing the same function as a Hugging Face Llama/Mistral/Qwen2 causal LM."""
    cfg = hf_llama_config_to_oglodon(hf_model.config)
    oglodon = Oglodon(cfg)
    sd = {k: v.detach() for k, v in hf_model.state_dict().items()}
    new: dict[str, torch.Tensor] = {"embed_tokens.weight": sd["model.embed_tokens.weight"]}
    for i in range(cfg.num_layers):
        p, q = f"model.layers.{i}.", f"blocks.{i}."
        new[q + "attn_norm.weight"] = sd[p + "input_layernorm.weight"]
        new[q + "attn.q_proj.weight"] = _rope_rows(sd[p + "self_attn.q_proj.weight"], cfg.num_heads)
        new[q + "attn.k_proj.weight"] = _rope_rows(sd[p + "self_attn.k_proj.weight"], cfg.num_kv_heads)
        new[q + "attn.v_proj.weight"] = sd[p + "self_attn.v_proj.weight"]
        if cfg.qkv_bias:
            new[q + "attn.q_proj.bias"] = _rope_rows(sd[p + "self_attn.q_proj.bias"], cfg.num_heads)
            new[q + "attn.k_proj.bias"] = _rope_rows(sd[p + "self_attn.k_proj.bias"], cfg.num_kv_heads)
            new[q + "attn.v_proj.bias"] = sd[p + "self_attn.v_proj.bias"]
        new[q + "attn.o_proj.weight"] = sd[p + "self_attn.o_proj.weight"]
        new[q + "ffn_norm.weight"] = sd[p + "post_attention_layernorm.weight"]
        new[q + "ffn.w1.weight"] = sd[p + "mlp.gate_proj.weight"]
        new[q + "ffn.w3.weight"] = sd[p + "mlp.up_proj.weight"]
        new[q + "ffn.w2.weight"] = sd[p + "mlp.down_proj.weight"]
    new["final_norm.weight"] = sd["model.norm.weight"]
    if not cfg.tie_embeddings:
        new["lm_head.weight"] = sd["lm_head.weight"]
    missing, unexpected = oglodon.load_state_dict({k: v.clone() for k, v in new.items()}, strict=False)
    allowed_missing = {"lm_head.weight"} if cfg.tie_embeddings else set()
    if unexpected or set(missing) - allowed_missing:
        raise RuntimeError(f"weight mapping incomplete: missing={missing}, unexpected={unexpected}")
    return oglodon.to(next(hf_model.parameters()).dtype).eval()


def _rope_rows_inverse(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Inverse of :func:`_rope_rows`: interleaved RoPE pairing back to Hugging Face's half-split pairing."""
    is_vec = w.dim() == 1
    d = w.shape[0] // n_heads
    w = w.reshape(n_heads, d, -1)
    out = torch.empty_like(w)
    out[:, : d // 2] = w[:, 0::2]
    out[:, d // 2 :] = w[:, 1::2]
    out = out.reshape(n_heads * d, -1)
    return out.squeeze(-1) if is_vec else out


def oglodon_config_to_hf(cfg: OglodonConfig) -> Any:
    """Map a :class:`OglodonConfig` to a Hugging Face ``LlamaConfig`` / ``MistralConfig`` / ``Qwen2Config``.

    Raises ``NotImplementedError`` for anything those architectures cannot express (ALiBi or learned
    positions, LayerNorm, a non-SwiGLU or non-SiLU feed-forward, sliding window together with q/k/v biases).
    """
    try:
        import transformers as tr
    except ImportError as e:  # pragma: no cover
        raise ImportError('exporting needs: pip install "oglodon-transformer[hf]"') from e
    problems = []
    if not cfg.use_rope:
        problems.append("learned/ALiBi positions (Hugging Face Llama uses RoPE)")
    if not cfg.use_rmsnorm:
        problems.append("LayerNorm (Llama uses RMSNorm)")
    if not cfg.use_swiglu or cfg.ffn_act != "silu":
        problems.append(
            f"feed-forward other than SwiGLU/silu (use_swiglu={cfg.use_swiglu}, activation={cfg.ffn_act})"
        )
    if cfg.sliding_window is not None and cfg.qkv_bias:
        problems.append("sliding window together with q/k/v biases (no matching Hugging Face architecture)")
    if problems:
        raise NotImplementedError(
            "cannot export to a Hugging Face Llama-family model: " + "; ".join(problems)
        )
    common: dict[str, Any] = dict(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_kv_heads,
        max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=cfg.norm_eps,
        tie_word_embeddings=cfg.tie_embeddings,
        pad_token_id=cfg.pad_token_id,
        eos_token_id=cfg.eos_token_id,
        rope_theta=cfg.rope_base,  # transformers 4.x; 5.x converts it (checked below)
    )
    hf_cfg: Any
    if cfg.qkv_bias:
        hf_cfg = tr.Qwen2Config(**common)
    elif cfg.sliding_window is not None:
        hf_cfg = tr.MistralConfig(**common, sliding_window=cfg.sliding_window)
    else:
        hf_cfg = tr.LlamaConfig(**common)
    if abs(_rope_theta(hf_cfg) - cfg.rope_base) > 1e-6:  # a transformers version that ignored rope_theta
        raise RuntimeError("the installed transformers did not keep rope_theta; cannot export faithfully")
    return hf_cfg


@torch.no_grad()
def to_hf_llama(model: Oglodon) -> Any:
    """Return a Hugging Face ``LlamaForCausalLM`` / ``MistralForCausalLM`` / ``Qwen2ForCausalLM`` with the same weights.

    The inverse of :func:`from_hf_llama`; check it with ``oglodon.verify.verify_conversion`` or
    :func:`oglodon.convert.check_roundtrip`.
    """
    import transformers as tr

    cfg = model.config
    hf_cfg = oglodon_config_to_hf(cfg)
    cls = {"qwen2": tr.Qwen2ForCausalLM, "mistral": tr.MistralForCausalLM, "llama": tr.LlamaForCausalLM}[
        hf_cfg.model_type
    ]
    hf: Any = cls(hf_cfg)
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    new: dict[str, torch.Tensor] = {"model.embed_tokens.weight": sd["embed_tokens.weight"]}
    for i in range(cfg.num_layers):
        p, q = f"blocks.{i}.", f"model.layers.{i}."
        new[q + "input_layernorm.weight"] = sd[p + "attn_norm.weight"]
        new[q + "self_attn.q_proj.weight"] = _rope_rows_inverse(sd[p + "attn.q_proj.weight"], cfg.num_heads)
        new[q + "self_attn.k_proj.weight"] = _rope_rows_inverse(
            sd[p + "attn.k_proj.weight"], cfg.num_kv_heads
        )
        new[q + "self_attn.v_proj.weight"] = sd[p + "attn.v_proj.weight"]
        if cfg.qkv_bias:
            new[q + "self_attn.q_proj.bias"] = _rope_rows_inverse(sd[p + "attn.q_proj.bias"], cfg.num_heads)
            new[q + "self_attn.k_proj.bias"] = _rope_rows_inverse(
                sd[p + "attn.k_proj.bias"], cfg.num_kv_heads
            )
            new[q + "self_attn.v_proj.bias"] = sd[p + "attn.v_proj.bias"]
        new[q + "self_attn.o_proj.weight"] = sd[p + "attn.o_proj.weight"]
        new[q + "post_attention_layernorm.weight"] = sd[p + "ffn_norm.weight"]
        new[q + "mlp.gate_proj.weight"] = sd[p + "ffn.w1.weight"]
        new[q + "mlp.up_proj.weight"] = sd[p + "ffn.w3.weight"]
        new[q + "mlp.down_proj.weight"] = sd[p + "ffn.w2.weight"]
    new["model.norm.weight"] = sd["final_norm.weight"]
    if not cfg.tie_embeddings:
        new["lm_head.weight"] = sd["lm_head.weight"]
    missing, unexpected = hf.load_state_dict({k: v.clone() for k, v in new.items()}, strict=False)
    allowed_missing = {"lm_head.weight"} if cfg.tie_embeddings else set()
    ignorable = {k for k in missing if k.endswith("rotary_emb.inv_freq")}  # buffers, not weights
    if unexpected or set(missing) - allowed_missing - ignorable:
        raise RuntimeError(f"weight mapping incomplete: missing={missing}, unexpected={unexpected}")
    if cfg.tie_embeddings:
        hf.tie_weights()
    return hf.to(next(model.parameters()).dtype).eval()


def export_hf(model: Oglodon, path: Any) -> Any:
    """Write ``model`` as a Hugging Face checkpoint directory (``AutoModelForCausalLM.from_pretrained(path)``).

    Returns the exported Hugging Face model. Tokenizers are not exported; save your tokenizer next to it.
    """
    hf = to_hf_llama(model)
    hf.save_pretrained(path)
    return hf
