"""Common result returned by every Facet prompt execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from facet_runtime.adapters.base import ExecutionMetrics

BackendName = Literal["cpu", "gpu", "npu", "auto"]
ActualBackendName = Literal["cpu", "gpu", "npu"]


@dataclass(frozen=True, slots=True)
class RunResult:
    text: str
    requested_backend: BackendName
    actual_backend: ActualBackendName
    runtime: str
    model: str
    device: str
    elapsed_ms: float
    fallback: bool
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
