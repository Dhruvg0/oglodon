"""Model-agnostic behavioural verification of causal language models (Oglodon, Hugging Face, custom PyTorch)."""

from .adapter import CausalLMAdapter, adapt, extract_logits
from .checks import CATEGORIES, VREGISTRY
from .equivalence import verify_conversion
from .runner import verify_causal_lm

__all__ = [
    "CATEGORIES",
    "VREGISTRY",
    "CausalLMAdapter",
    "adapt",
    "extract_logits",
    "verify_causal_lm",
    "verify_conversion",
]
