"""A tiny adapter that lets the black-box checks drive *any* causal language model.

A causal LM, for Oglodon's purposes, is anything that maps integer ``input_ids`` of shape ``(B, T)`` to
next-token logits ``(B, T, V)``. The adapter discovers (from the ``forward`` signature) whether the model
also understands an attention mask, explicit position ids, labels (loss) and a KV cache, so a check can
run when it is applicable and report *skipped* when it is not, instead of guessing.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import torch
from torch import nn


def extract_logits(out: Any) -> torch.Tensor:
    """Logits from a tensor, an object with ``.logits``, a mapping with ``"logits"`` or a tuple."""
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits
    if isinstance(out, dict) and "logits" in out:
        return out["logits"]
    if isinstance(out, tuple | list) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(
        f"cannot find logits in a model output of type {type(out).__name__}; "
        "pass a CausalLMAdapter with a custom `forward=` that returns a tensor"
    )


class CausalLMAdapter:
    """Uniform access to a causal LM.

    Args:
        model: an ``nn.Module`` (or any callable ``input_ids -> logits``).
        vocab_size: size of the output vocabulary; inferred from ``model.config`` or by probing if omitted.
        max_positions: longest sequence the model accepts; inferred from its config if omitted.
        forward: optional ``f(model, input_ids, attention_mask=None, position_ids=None) -> logits`` for models
            whose ``forward`` does not follow the usual conventions. Capabilities are then taken from
            ``supports_mask`` / ``supports_positions`` (default: whichever of those keywords ``forward`` accepts).
        supports_cache / supports_labels: override the signature-based detection.
    """

    _MAX_POS_KEYS = (
        "max_position_embeddings",
        "n_positions",
        "max_seq_len",
        "max_sequence_length",
        "seq_length",
    )

    def __init__(
        self,
        model: Any,
        vocab_size: int | None = None,
        max_positions: int | None = None,
        forward: Callable[..., torch.Tensor] | None = None,
        supports_mask: bool | None = None,
        supports_positions: bool | None = None,
        supports_cache: bool | None = None,
        supports_labels: bool | None = None,
    ) -> None:
        self.model = model
        self._forward = forward
        fn = getattr(model, "forward", model) if forward is None else forward
        try:
            params = set(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            params = set()
        self.supports_mask = ("attention_mask" in params) if supports_mask is None else supports_mask
        self.supports_positions = (
            ("position_ids" in params) if supports_positions is None else supports_positions
        )
        if forward is not None and supports_mask is None:
            self.supports_mask = "attention_mask" in params
        self.supports_cache = (
            ("use_cache" in params and "past_key_values" in params)
            if supports_cache is None
            else supports_cache
        )
        self.supports_labels = ("labels" in params) if supports_labels is None else supports_labels
        self.config = getattr(model, "config", None)
        self.vocab_size = vocab_size or self._config_value(("vocab_size",))
        self.max_positions = max_positions or self._config_value(self._MAX_POS_KEYS)

    def _config_value(self, keys: tuple[str, ...]) -> int | None:
        for key in keys:
            v = getattr(self.config, key, None)
            if isinstance(v, int) and v > 0:
                return v
        return None

    # --------------------------------------------------------------------------- calls
    def logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._forward is not None:
            return extract_logits(self._forward(self.model, input_ids, attention_mask, position_ids))
        kw: dict[str, Any] = {}
        if attention_mask is not None:
            kw["attention_mask"] = attention_mask
        if position_ids is not None:
            kw["position_ids"] = position_ids
        if self.supports_cache:
            kw["use_cache"] = False
        return extract_logits(self.model(input_ids, **kw))

    def loss(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor | None:
        """The model's own loss for ``labels`` (``None`` if the model does not compute one)."""
        if not self.supports_labels or self._forward is not None:
            return None
        out = self.model(input_ids, labels=labels)
        loss = getattr(out, "loss", None)
        if loss is None and isinstance(out, dict):
            loss = out.get("loss")
        return loss

    def cached_logits(self, input_ids: torch.Tensor, prefill: int, chunk: int = 1) -> torch.Tensor:
        """Logits for all positions computed as prefill + ``chunk``-sized steps with the model's KV cache."""
        if not self.supports_cache or self._forward is not None:
            raise NotImplementedError("model has no use_cache / past_key_values interface")
        out = self.model(input_ids[:, :prefill], use_cache=True)
        pieces = [extract_logits(out)]
        past = getattr(out, "past_key_values", None)
        if past is None:
            raise NotImplementedError("model did not return past_key_values with use_cache=True")
        for s in range(prefill, input_ids.shape[1], chunk):
            out = self.model(input_ids[:, s : s + chunk], past_key_values=past, use_cache=True)
            pieces.append(extract_logits(out))
            past = out.past_key_values
        return torch.cat(pieces, dim=1)

    # --------------------------------------------------------------------------- derived
    def infer_vocab(self, device: torch.device) -> int:
        """Vocabulary size from the config, else by probing the output width."""
        if self.vocab_size:
            return self.vocab_size
        with torch.no_grad():
            return int(self.logits(torch.zeros(1, 2, dtype=torch.long, device=device)).shape[-1])

    def with_model(self, model: Any) -> CausalLMAdapter:
        """The same adapter around another module of the same architecture (a copy)."""
        a = object.__new__(CausalLMAdapter)
        a.__dict__.update(self.__dict__)
        a.model = model
        return a


def adapt(model: Any, **kw: Any) -> CausalLMAdapter:
    return model if isinstance(model, CausalLMAdapter) else CausalLMAdapter(model, **kw)


def count_parameters(model: Any) -> int | None:
    return sum(p.numel() for p in model.parameters()) if isinstance(model, nn.Module) else None
