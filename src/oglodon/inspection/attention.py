"""Opt-in attention inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..models.gpt import Oglodon


@dataclass
class AttentionResult:
    """Attention weights of one layer.

    ``weights`` has shape ``(B, H, T, T)`` (``H`` = selected heads) and
    ``heads`` lists the original head indices.
    """

    weights: torch.Tensor
    layer: int
    heads: list[int]


class AttentionInspector:
    """Extract and plot attention maps without touching normal inference.

    Weights are only materialised when :meth:`inspect` is called (it runs the model
    with ``output_attentions=True`` under ``no_grad``); a regular ``model(x)`` or
    ``generate`` never keeps them.
    """

    def __init__(self, model: Oglodon) -> None:
        self.model = model

    @torch.no_grad()
    def inspect(
        self,
        input_ids: torch.Tensor,
        layer: int = 0,
        head: int | list[int] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> AttentionResult:
        cfg = self.model.config
        if input_ids.dim() == 1:
            input_ids = input_ids[None]
        if not -cfg.num_layers <= layer < cfg.num_layers:
            raise ValueError(f"layer {layer} out of range for {cfg.num_layers} layers")
        layer = layer % cfg.num_layers
        heads = (
            list(range(cfg.num_heads)) if head is None else ([head] if isinstance(head, int) else list(head))
        )
        bad = [h for h in heads if not 0 <= h < cfg.num_heads]
        if bad:
            raise ValueError(f"head index {bad} out of range for {cfg.num_heads} heads")
        was_training = self.model.training
        self.model.eval()
        try:
            out = self.model(
                input_ids.to(next(self.model.parameters()).device),
                attention_mask=attention_mask,
                output_attentions=True,
            )
        finally:
            self.model.train(was_training)
        weights = out.attentions[layer][:, heads].cpu()
        return AttentionResult(weights=weights, layer=layer, heads=heads)

    def plot(
        self,
        result: AttentionResult,
        tokens: list[str] | None = None,
        batch_index: int = 0,
        cols: int = 4,
        cmap: str = "viridis",
        show: bool = False,
    ):
        """Heatmap grid (rows = query token, columns = attended-to token).

        Requires ``matplotlib`` (``pip install "oglodon-transformer[viz]"``). Returns the figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise ImportError('Install the extra: pip install "oglodon-transformer[viz]"') from e
        w = result.weights[batch_index]
        n, T = w.shape[0], w.shape[-1]
        if tokens is not None and len(tokens) != T:
            raise ValueError(f"expected {T} token labels, got {len(tokens)}")
        cols = min(cols, n)
        rows = -(-n // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for i, h in enumerate(result.heads):
            ax = axes[i // cols][i % cols]
            ax.axis("on")
            ax.imshow(w[i].numpy(), cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"layer {result.layer} head {h}", fontsize=9)
            if tokens is not None:
                ax.set_xticks(range(T), tokens, rotation=90, fontsize=7)
                ax.set_yticks(range(T), tokens, fontsize=7)
        fig.tight_layout()
        if show:
            plt.show()
        return fig
