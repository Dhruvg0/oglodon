"""Deprecated location: the validator moved to :mod:`oglodon.validation` (richer checks, JSON reports)."""

from ..validation import CheckResult, Status, ValidationReport, validate_model

__all__ = ["CheckResult", "Status", "ValidationReport", "validate_model"]
