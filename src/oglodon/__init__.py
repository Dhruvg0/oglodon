"""Oglodon: Build. Train. Understand. Verify.

A modular PyTorch toolkit for building, training, inspecting and behaviour-verifying
Transformers (your own, Hugging Face, or Oglodon's) from scratch.
"""

from .config import OglodonConfig
from .models.gpt import Oglodon, OglodonOutput
from .training.trainer import Trainer

__version__ = "0.4.0"

__all__ = ["Oglodon", "OglodonConfig", "OglodonOutput", "Trainer", "__version__"]
