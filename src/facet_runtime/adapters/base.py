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


@dataclass(frozen=True, slots=True)
class ImageRuntimeMetadata:
    protocol: str
    response_format: str
    strict_json_schema: bool


@dataclass(frozen=True, slots=True)
class ImageAdapterOutput:
    backend: AdapterBackend
    transcription: str
    uncertainties: tuple[str, ...]
    runtime: str
    model: str
    device: str
    runtime_metadata: ImageRuntimeMetadata
    accelerator_verified: bool


class BackendAdapter(Protocol):
    backend: AdapterBackend

    def is_available(self) -> bool: ...

    def run(self, prompt: str) -> AdapterOutput: ...


class ImageBackendAdapter(Protocol):
    backend: AdapterBackend

    def inspect_image(self, image_path: str) -> ImageAdapterOutput: ...
