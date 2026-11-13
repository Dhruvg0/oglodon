"""Controlled experiments (same data, same tokens, same seeds) with fairness notes."""

from .compare import (
    ComparisonReport,
    ExperimentData,
    RunResult,
    config_diff,
    match_intermediate_size,
    run_comparison,
)

__all__ = [
    "ComparisonReport",
    "ExperimentData",
    "RunResult",
    "config_diff",
    "match_intermediate_size",
    "run_comparison",
]
