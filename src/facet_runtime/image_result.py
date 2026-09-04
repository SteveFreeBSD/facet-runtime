"""Structured result for a heterogeneous image inspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from facet_runtime.adapters.base import ImageRuntimeMetadata


@dataclass(frozen=True, slots=True)
class ImagePassResult:
    backend: Literal["npu", "gpu"]
    transcription: str
    uncertainties: tuple[str, ...]
    model: str
    runtime: str
    device: str
    runtime_metadata: ImageRuntimeMetadata
    elapsed_ms: float
    accelerator_verified: bool


@dataclass(frozen=True, slots=True)
class TranscriptionDifference:
    npu_line: int | None
    gpu_line: int | None
    npu: str | None
    gpu: str | None


@dataclass(frozen=True, slots=True)
class ImageInspectionResult:
    image: str
    image_sha256: str
    npu: ImagePassResult
    gpu: ImagePassResult
    agreement: bool
    disagreement: tuple[TranscriptionDifference, ...]
    total_elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
