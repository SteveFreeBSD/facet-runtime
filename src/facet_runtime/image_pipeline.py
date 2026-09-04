"""Narrow two-accelerator image transcription pipeline."""

from __future__ import annotations

import hashlib
import time
from difflib import SequenceMatcher
from pathlib import Path

from facet_runtime.adapters import FastFlowAdapter, OllamaAdapter
from facet_runtime.adapters.base import ImageBackendAdapter
from facet_runtime.errors import BackendMismatchError
from facet_runtime.image_result import (
    ImageInspectionResult,
    ImagePassResult,
    TranscriptionDifference,
)


def _normalized_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.replace("\r\n", "\n").strip().split("\n")]


def _differences(npu_text: str, gpu_text: str) -> tuple[TranscriptionDifference, ...]:
    npu_lines = _normalized_lines(npu_text)
    gpu_lines = _normalized_lines(gpu_text)
    differences = []
    matcher = SequenceMatcher(a=npu_lines, b=gpu_lines, autojunk=False)
    for tag, npu_start, npu_end, gpu_start, gpu_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        width = max(npu_end - npu_start, gpu_end - gpu_start)
        for offset in range(width):
            npu_index = npu_start + offset
            gpu_index = gpu_start + offset
            has_npu = npu_index < npu_end
            has_gpu = gpu_index < gpu_end
            differences.append(
                TranscriptionDifference(
                    npu_line=npu_index + 1 if has_npu else None,
                    gpu_line=gpu_index + 1 if has_gpu else None,
                    npu=npu_lines[npu_index] if has_npu else None,
                    gpu=gpu_lines[gpu_index] if has_gpu else None,
                )
            )
    return tuple(differences)


def _run_pass(
    expected_backend: str, adapter: ImageBackendAdapter, image_path: Path
) -> ImagePassResult:
    started = time.perf_counter()
    output = adapter.inspect_image(str(image_path))
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    if output.backend != expected_backend:
        raise BackendMismatchError(
            f"image adapter reported {output.backend}, expected {expected_backend}"
        )
    if not output.accelerator_verified:
        raise BackendMismatchError(
            f"{expected_backend} adapter did not verify accelerator execution"
        )
    return ImagePassResult(
        backend=output.backend,
        transcription=output.transcription,
        uncertainties=output.uncertainties,
        model=output.model,
        runtime=output.runtime,
        device=output.device,
        runtime_metadata=output.runtime_metadata,
        elapsed_ms=elapsed_ms,
        accelerator_verified=True,
        metrics=output.metrics,
        evidence=dict(output.evidence),
    )


def inspect_image(
    image: str,
    *,
    npu_adapter: ImageBackendAdapter | None = None,
    gpu_adapter: ImageBackendAdapter | None = None,
) -> ImageInspectionResult:
    path = Path(image).expanduser().resolve(strict=True)
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("image must be a PNG or JPEG file")

    npu = npu_adapter or FastFlowAdapter()
    gpu = gpu_adapter or OllamaAdapter("gpu")
    total_started = time.perf_counter()
    npu_result = _run_pass("npu", npu, path)
    gpu_result = _run_pass("gpu", gpu, path)
    total_elapsed_ms = round((time.perf_counter() - total_started) * 1000, 3)
    differences = _differences(npu_result.transcription, gpu_result.transcription)
    return ImageInspectionResult(
        image=str(path),
        image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        npu=npu_result,
        gpu=gpu_result,
        agreement=not differences,
        disagreement=differences,
        total_elapsed_ms=total_elapsed_ms,
    )
