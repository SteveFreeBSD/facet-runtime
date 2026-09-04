"""Common result returned by every Facet prompt execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
