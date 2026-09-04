"""Internal adapter protocol used by the runtime coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

AdapterBackend = Literal["cpu", "gpu", "npu"]


@dataclass(frozen=True, slots=True)
class AdapterOutput:
    text: str
    runtime: str
    model: str
    device: str


class BackendAdapter(Protocol):
    backend: AdapterBackend

    def is_available(self) -> bool: ...

    def run(self, prompt: str) -> AdapterOutput: ...
