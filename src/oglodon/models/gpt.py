"""The Oglodon decoder-only language model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..config import OglodonConfig
from ..layers.cache import KVCache
from ..layers.embeddings import LearnedPositionalEmbedding
from ..layers.normalization import build_norm
from ..layers.rope import RotaryEmbedding
from .transformer import TransformerBlock


@dataclass
class OglodonOutput:
    """Rich forward output, returned when ``labels``, ``use_cache`` or
    ``output_attentions`` is requested.

    ``attentions`` is a tuple with one ``(B, num_heads, T, kv_len)`` tensor per layer.
    """

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: KVCache | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


class Oglodon(nn.Module):
    """Decoder-only Transformer language model.

    ``forward`` returns the logits tensor ``(B, T, vocab_size)`` for a plain call,
    and a :class:`OglodonOutput` as soon as ``labels``, ``use_cache`` or
    ``output_attentions`` is given.

    Labels are *unshifted* (usually ``labels = input_ids``); the one-token shift
    happens inside the loss. Positions equal to ``-100`` are ignored.
    """

    def __init__(self, config: OglodonConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        if c.use_rope:
            self.rope: RotaryEmbedding | None = RotaryEmbedding(c.head_dim, c.max_seq_len, c.rope_base)
            self.embed_positions: LearnedPositionalEmbedding | None = None
        elif c.use_alibi:
            self.rope = None
            self.embed_positions = None  # positions enter through the attention bias
        else:
            self.rope = None
            self.embed_positions = LearnedPositionalEmbedding(c.max_seq_len, c.hidden_size)
        self.embed_drop = nn.Dropout(c.dropout)
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList(TransformerBlock(c, i) for i in range(c.num_layers))
        self.final_norm = build_norm(c.hidden_size, c.use_rmsnorm, c.norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)

        self.apply(self._init_weights)
        # GPT-2 style: shrink the residual-branch output projections by 1/sqrt(2L)
        # so activations do not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith(("attn.o_proj.weight", "ffn.w2.weight", "ffn.fc2.weight")):
                nn.init.normal_(p, std=c.init_std / math.sqrt(2 * c.num_layers))
        if c.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> torch.Tensor | OglodonOutput:
        """
        Args:
            input_ids: ``(B, T)`` token ids (only the *new* tokens when a cache is given).
            attention_mask: optional ``(B, past + T)`` tensor, 1 = real token, 0 = padding.
            position_ids: optional ``(B, T)`` positions; defaults to ``past .. past+T-1``.
            labels: optional ``(B, T)`` targets for the language-modelling loss.
            past_key_values: a :class:`KVCache` from an earlier call.
            use_cache: create/extend a cache and return it in the output.
            output_attentions: also return per-layer attention weights (opt-in; costs memory).
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be (batch, seq), got shape {tuple(input_ids.shape)}")
        B, T = input_ids.shape
        past = 0 if past_key_values is None else past_key_values.seq_len
        if past + T > self.config.max_seq_len:
            raise ValueError(f"sequence length {past + T} exceeds max_seq_len={self.config.max_seq_len}")
        if attention_mask is not None and attention_mask.shape != (B, past + T):
            raise ValueError(
                f"attention_mask must have shape {(B, past + T)}, got {tuple(attention_mask.shape)}"
            )
        if position_ids is None:
            position_ids = torch.arange(past, past + T, device=input_ids.device).expand(B, T)
        else:
            if position_ids.shape != (B, T):
                raise ValueError(f"position_ids must have shape {(B, T)}, got {tuple(position_ids.shape)}")
            if int(position_ids.min()) < 0 or int(position_ids.max()) >= self.config.max_seq_len:
                raise ValueError(f"position_ids must lie in [0, max_seq_len={self.config.max_seq_len})")
        if use_cache and past_key_values is None:
            past_key_values = KVCache(self.config.num_layers)

        x = self.embed_tokens(input_ids)
        rope = None
        if self.rope is not None:
            rope = self.rope(position_ids)
        elif self.embed_positions is not None:
            x = x + self.embed_positions(position_ids)
        x = self.embed_drop(x)

        cache = past_key_values if use_cache else None
        attentions: list[torch.Tensor] = []
        recompute = (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and cache is None
            and not output_attentions
        )
        for block in self.blocks:
            if recompute:
                # activations inside the block are recomputed during backward instead of being stored
                x, w = checkpoint(block, x, rope, attention_mask, None, False, use_reentrant=False)
            else:
                x, w = block(x, rope, attention_mask, cache, output_attentions)
            if w is not None:
                attentions.append(w)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1]
                .to(torch.promote_types(logits.dtype, torch.float32))
                .reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        if labels is None and not use_cache and not output_attentions:
            return logits
        return OglodonOutput(
            logits=logits,
            loss=loss,
            past_key_values=cache,
            attentions=tuple(attentions) if output_attentions else None,
        )

    # --------------------------------------------------------------- utilities
    def num_parameters(self, trainable_only: bool = False) -> int:
        """Parameter count (a tied embedding/output matrix is counted once)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def set_attn_impl(self, impl: str) -> None:
        """Switch between ``"eager"`` and ``"sdpa"`` attention in place (same weights, same maths)."""
        if impl not in ("eager", "sdpa"):
            raise ValueError(f"attn_impl must be 'eager' or 'sdpa', got {impl!r}")
        self.config.attn_impl = impl
        for block in self.blocks:
            assert isinstance(block, TransformerBlock)
            block.attn.attn_impl = impl

    def gradient_checkpointing_enable(self) -> None:
        """Recompute each block's activations in the backward pass: less memory, about one extra forward.

        Only active in training mode with gradients enabled and without a KV cache; results are identical.
        """
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Autoregressive generation. See :func:`oglodon.generation.generate`."""
        from ..generation.sampler import generate

        return generate(self, input_ids, **kwargs)

    def generate_stream(self, input_ids: torch.Tensor, **kwargs):
        """Yield one ``(B,)`` tensor of new tokens per step. See :func:`oglodon.generation.generate_stream`."""
        from ..generation.sampler import generate_stream

        return generate_stream(self, input_ids, **kwargs)

    # ------------------------------------------------------------ checkpoints
    def save_pretrained(self, path: str | Path, tokenizer=None) -> None:
        """Write ``config.json`` + ``model.safetensors`` (+ tokenizer) to ``path``."""
        from ..training.checkpoint import save_pretrained

        save_pretrained(self, path, tokenizer)

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str | torch.device = "cpu") -> Oglodon:
        """Load a model written by :meth:`save_pretrained`."""
        from ..training.checkpoint import load_pretrained

        return load_pretrained(cls, path, device)
