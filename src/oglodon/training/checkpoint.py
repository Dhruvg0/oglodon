"""Checkpoint I/O.

Model weights are stored as **safetensors** (a plain tensor container that cannot
execute code on load), the config and tokenizer as JSON. Optimizer/trainer state
is a ``torch.save`` file that is *only ever loaded with* ``weights_only=True``,
which refuses to unpickle arbitrary Python objects. Never ``torch.load`` a
checkpoint from an untrusted source without ``weights_only=True``: a classic
pickle checkpoint can run arbitrary code when opened.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import load_file, save_file

from ..config import OglodonConfig

if TYPE_CHECKING:
    from ..models.gpt import Oglodon

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
TOKENIZER_FILE = "tokenizer.json"
TRAINER_STATE_FILE = "trainer_state.pt"


def save_pretrained(model: Oglodon, path: str | Path, tokenizer: Any = None) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if model.config.tie_embeddings:
        state.pop("lm_head.weight", None)  # shared with embed_tokens.weight
    save_file(state, str(path / WEIGHTS_FILE), metadata={"format": "pt"})
    model.config.save(path / CONFIG_FILE)
    from ..quantization import write_quantization_info

    write_quantization_info(model, path)
    if tokenizer is not None:
        if not hasattr(tokenizer, "save"):
            raise TypeError("tokenizer must implement .save(path) to be stored in a checkpoint")
        tokenizer.save(path / TOKENIZER_FILE)


def load_pretrained(cls: type[Oglodon], path: str | Path, device: str | torch.device = "cpu") -> Oglodon:
    path = Path(path)
    config = OglodonConfig.load(path / CONFIG_FILE)
    model = cls(config)
    from ..quantization import apply_quantized_structure, read_quantization_info

    info = read_quantization_info(path)
    if info is not None:  # an int8 checkpoint: build the same int8 structure, then load its buffers
        apply_quantized_structure(model, info)
    state = load_file(str(path / WEIGHTS_FILE), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected_missing = {"lm_head.weight"} if config.tie_embeddings else set()
    if unexpected or set(missing) != expected_missing:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device).eval()


def save_training_state(path: str | Path, state: dict[str, Any]) -> None:
    """Save optimizer/scheduler/step state (tensors and plain containers only)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(state, path / TRAINER_STATE_FILE)


def load_training_state(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path) / TRAINER_STATE_FILE, map_location=device, weights_only=True)
