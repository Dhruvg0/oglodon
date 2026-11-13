"""Backward-compatible alias for :mod:`oglodon.validation` (``from oglodon.testing import validate_model`` still works)."""

from .validation import CheckResult, Status, ValidationReport, validate_model

__all__ = ["CheckResult", "Status", "ValidationReport", "validate_model"]
