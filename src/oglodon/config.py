"""Model configuration with validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class OglodonConfig:
    """Hyper-parameters of a Oglodon decoder-only Transformer.

    Attention layout
        ``num_heads`` query heads share ``num_kv_heads`` key/value heads.
        ``num_kv_heads == num_heads`` is multi-head attention (MHA),
        ``1 < num_kv_heads < num_heads`` is grouped-query attention (GQA) and
        ``num_kv_heads == 1`` is multi-query attention (MQA).

    Args:
        vocab_size: number of token ids.
        hidden_size: model width ``d``.
        num_layers: number of Transformer blocks.
        num_heads: number of query heads (``hidden_size`` must divide evenly).
        num_kv_heads: number of key/value heads. ``None`` means ``num_heads`` (MHA).
        max_seq_len: longest sequence (prompt + generated) the model accepts.
        intermediate_size: feed-forward width. ``None`` picks ``~8d/3`` (rounded
            up to a multiple of 8) for SwiGLU and ``4d`` for GELU.
        dropout: dropout probability for attention weights, residual branches
            and the feed-forward hidden layer.
        use_rope: rotary positions if True, learned absolute embeddings if False (unless ``use_alibi``).
        use_alibi: ALiBi linear distance biases instead of any position embedding (needs ``use_rope=False``;
            adds no parameters). Distances are measured in sequence slots, so left padding is harmless.
        ffn_activation: ``"silu"``, ``"gelu"`` or ``"relu2"`` (ReLU squared). ``None`` keeps the classic
            choice: SiLU for the gated (SwiGLU) layer, GELU for the plain MLP. With ``use_swiglu`` the layer is
            gated (SwiGLU/GeGLU/ReGLU-squared), otherwise plain.
        use_rmsnorm: RMSNorm if True, LayerNorm if False.
        use_swiglu: SwiGLU feed-forward if True, GELU MLP if False.
        rope_base: RoPE frequency base (only used when ``use_rope``).
        norm_eps: epsilon of the normalisation layers.
        tie_embeddings: share the input embedding with the output projection.
        init_std: standard deviation of the normal weight initialisation.
        attn_impl: ``"eager"`` (explicit softmax, readable, supports attention maps) or ``"sdpa"``
            (``torch.nn.functional.scaled_dot_product_attention`` with the same mask). Same maths.
        sliding_window: if set, a token attends only to the last ``sliding_window`` tokens
            (itself included). The KV cache is *not* trimmed, so this changes the mask, not memory.
        qkv_bias: learnable biases on the query/key/value projections (Qwen2 style); the output projection
            and feed-forward never have biases.
        pad_token_id / eos_token_id: optional special ids used by ``generate``.
    """

    vocab_size: int
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = None  # type: ignore[assignment]  # None means num_heads; resolved in __post_init__
    max_seq_len: int = 512
    intermediate_size: int = None  # type: ignore[assignment]  # None means ~8d/3 or 4d; resolved in __post_init__
    dropout: float = 0.0
    use_rope: bool = True
    use_rmsnorm: bool = True
    use_swiglu: bool = True
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    init_std: float = 0.02
    attn_impl: str = "eager"
    sliding_window: int | None = None
    qkv_bias: bool = False
    use_alibi: bool = False
    ffn_activation: str | None = None
    pad_token_id: int | None = None
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        self.validate()
        if self.intermediate_size is None:
            if self.use_swiglu:
                self.intermediate_size = -(-(8 * self.hidden_size // 3) // 8) * 8
            else:
                self.intermediate_size = 4 * self.hidden_size

    # ------------------------------------------------------------------ checks
    def validate(self) -> None:
        """Raise ``ValueError`` with a precise message for any invalid value."""

        def positive_int(name: str) -> None:
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive integer, got {v!r}")

        for name in ("vocab_size", "hidden_size", "num_layers", "num_heads", "max_seq_len"):
            positive_int(name)
        positive_int("num_kv_heads")
        if self.intermediate_size is not None:
            positive_int("intermediate_size")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            )
        if self.use_rope and (self.hidden_size // self.num_heads) % 2 != 0:
            raise ValueError(
                "RoPE needs an even head dimension, got "
                f"{self.hidden_size // self.num_heads} (hidden_size / num_heads)"
            )
        if self.attn_impl not in ("eager", "sdpa"):
            raise ValueError(f"attn_impl must be 'eager' or 'sdpa', got {self.attn_impl!r}")
        if self.sliding_window is not None and (
            isinstance(self.sliding_window, bool)
            or not isinstance(self.sliding_window, int)
            or self.sliding_window <= 0
        ):
            raise ValueError(
                f"sliding_window must be a positive integer or None, got {self.sliding_window!r}"
            )
        if not isinstance(self.use_alibi, bool):
            raise ValueError(f"use_alibi must be a bool, got {self.use_alibi!r}")
        if self.use_alibi and self.use_rope:
            raise ValueError("use_alibi replaces RoPE: set use_rope=False when use_alibi=True")
        if self.ffn_activation not in (None, "silu", "gelu", "relu2"):
            raise ValueError(
                f"ffn_activation must be None, 'silu', 'gelu' or 'relu2', got {self.ffn_activation!r}"
            )
        if not isinstance(self.qkv_bias, bool):
            raise ValueError(f"qkv_bias must be a bool, got {self.qkv_bias!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.norm_eps <= 0 or self.rope_base <= 0 or self.init_std <= 0:
            raise ValueError("norm_eps, rope_base and init_std must be > 0")
        for name in ("pad_token_id", "eos_token_id"):
            v = getattr(self, name)
            if v is not None and not 0 <= v < self.vocab_size:
                raise ValueError(f"{name} ({v}) must be in [0, vocab_size={self.vocab_size})")

    # ------------------------------------------------------------- properties
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def ffn_act(self) -> str:
        """The feed-forward activation actually used."""
        return self.ffn_activation or ("silu" if self.use_swiglu else "gelu")

    @property
    def attention_type(self) -> str:
        if self.num_kv_heads == self.num_heads:
            return "MHA"
        return "MQA" if self.num_kv_heads == 1 else "GQA"

    # ---------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OglodonConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> OglodonConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))
