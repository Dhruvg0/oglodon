"""Behavioural validation: does the model do what a Transformer is supposed to do?

``validate_model(model)`` runs independent-reference and equivalence checks for attention, the
KV cache, the model and the training utilities, and returns a :class:`ValidationReport` that says
exactly what passed, failed, was skipped or is unsupported.
"""

from . import reference
from .registry import CATEGORIES, REGISTRY
from .report import CheckResult, Status, ValidationReport
from .runner import validate_model

__all__ = [
    "CATEGORIES",
    "REGISTRY",
    "CheckResult",
    "Status",
    "ValidationReport",
    "reference",
    "validate_model",
]
