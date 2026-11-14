"""Result types for behavioural validation."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch


class Status(StrEnum):
    """Outcome of one check. Only ``PASS`` counts as passed; nothing else is ever promoted to it."""

    PASS = "pass"  # the check ran and the behaviour is correct
    FAIL = "fail"  # the check ran and found a correctness problem (or crashed)
    WARNING = "warning"  # the check ran; something is suspicious but not provably wrong
    SKIPPED = "skipped"  # the check was not run (excluded by the caller or a precondition is missing)
    UNSUPPORTED = "unsupported"  # this model / device / torch build cannot run the check


@dataclass
class CheckResult:
    """One validation check.

    ``max_abs_diff`` is the largest absolute difference between the implementation and its reference
    (or between two supposedly equivalent paths); ``tolerance`` is the *effective* threshold it was
    compared against. Both are ``None`` for checks that are not numeric comparisons.
    """

    name: str
    category: str
    status: Status
    detail: str = ""
    max_abs_diff: float | None = None
    tolerance: float | None = None
    device: str = "cpu"
    dtype: str = "float32"
    duration_s: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status is Status.PASS

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


def _json_safe(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)  # JSON has no NaN/Infinity
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def collect_metadata(
    config: dict[str, Any], device: torch.device, dtype: torch.dtype, seed: int
) -> dict[str, Any]:
    """Everything needed to reproduce a validation run."""
    from .. import __version__

    cfg_json = json.dumps(config, sort_keys=True)
    meta: dict[str, Any] = {
        "oglodon_version": __version__,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "seed": seed,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config,
        "config_sha256": hashlib.sha256(cfg_json.encode()).hexdigest(),
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        meta["cuda_device_name"] = torch.cuda.get_device_name(0)
        meta["cuda_version"] = torch.version.cuda
    return meta


@dataclass
class ValidationReport:
    """All check results plus reproducibility metadata.

    ``passed`` means: no check failed **and** at least one check passed. It says nothing about checks
    that were skipped or unsupported; those are never counted as passed. Use ``complete`` to ask whether
    every requested check actually ran.
    """

    checks: list[CheckResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- queries
    def _with(self, status: Status) -> list[CheckResult]:
        return [c for c in self.checks if c.status is status]

    @property
    def failures(self) -> list[CheckResult]:
        return self._with(Status.FAIL)

    @property
    def warnings(self) -> list[CheckResult]:
        return self._with(Status.WARNING)

    @property
    def skipped(self) -> list[CheckResult]:
        return self._with(Status.SKIPPED)

    @property
    def unsupported(self) -> list[CheckResult]:
        return self._with(Status.UNSUPPORTED)

    @property
    def passes(self) -> list[CheckResult]:
        return self._with(Status.PASS)

    @property
    def passed(self) -> bool:
        return not self.failures and bool(self.passes)

    @property
    def complete(self) -> bool:
        """True iff no check was skipped or unsupported."""
        return not self.skipped and not self.unsupported

    @property
    def results(self) -> list[CheckResult]:  # alias kept for the pre-0.2 ``oglodon.testing`` API
        return self.checks

    def counts(self) -> dict[str, int]:
        return {s.value: len(self._with(s)) for s in Status}

    def get(self, name: str) -> CheckResult:
        for c in self.checks:
            if c.name == name:
                return c
        raise KeyError(name)

    def raise_if_failed(self) -> None:
        if self.failures:
            raise AssertionError("; ".join(f"{c.name}: {c.detail}" for c in self.failures))

    # ----------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "passed": self.passed,
                "complete": self.complete,
                "counts": self.counts(),
                "metadata": self.metadata,
                "checks": [c.to_dict() for c in self.checks],
            }
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> ValidationReport:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        checks = []
        for c in data["checks"]:
            c = dict(c)
            c["status"] = Status(c["status"])
            for k in ("max_abs_diff", "tolerance"):
                if isinstance(c.get(k), str):
                    c[k] = float(c[k])
            checks.append(CheckResult(**c))
        return cls(checks=checks, metadata=data.get("metadata", {}))

    def __str__(self) -> str:
        m = self.metadata
        lines = [f"Oglodon validation | device={m.get('device')} dtype={m.get('dtype')} seed={m.get('seed')}"]
        for c in self.checks:
            diff = "" if c.max_abs_diff is None else f" diff={c.max_abs_diff:.2e}"
            tol = "" if c.tolerance is None else f" tol={c.tolerance:.1e}"
            lines.append(f"[{c.status.value.upper():<11}] {c.name:<40}{diff}{tol}  {c.detail}")
        n = self.counts()
        lines.append(
            "summary: "
            + ", ".join(f"{v} {k}" for k, v in n.items())
            + f" | passed={self.passed} complete={self.complete}"
        )
        return "\n".join(lines)
