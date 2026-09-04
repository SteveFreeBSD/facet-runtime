"""Internal adapter protocol used by the runtime coordinator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

AdapterBackend = Literal["cpu", "gpu", "npu"]


@dataclass(frozen=True, slots=True)
class ExecutionMetrics:
    """Token counts and throughput reported by the runtime that executed."""

    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    prefill_tps: float | None = None
    decode_tps: float | None = None


def metrics_from_ollama_api(response: dict[str, Any]) -> ExecutionMetrics:
    """Read the Ollama-compatible timing block that Ollama and FastFlowLM emit."""

    def positive(key: str) -> int:
        value = response.get(key)
        return value if isinstance(value, int) and value > 0 else 0

    prompt_tokens = positive("prompt_eval_count")
    prompt_ns = positive("prompt_eval_duration")
    generated_tokens = positive("eval_count")
    generated_ns = positive("eval_duration")
    return ExecutionMetrics(
        prompt_tokens=prompt_tokens or None,
        generated_tokens=generated_tokens or None,
        prefill_tps=(
            round(prompt_tokens / (prompt_ns / 1e9), 2)
            if prompt_tokens and prompt_ns
            else None
        ),
        decode_tps=(
            round(generated_tokens / (generated_ns / 1e9), 2)
            if generated_tokens and generated_ns
            else None
        ),
    )


def metrics_from_openai_usage(response: dict[str, Any]) -> ExecutionMetrics:
    """Read the OpenAI-shaped usage block that FastFlowLM returns."""
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return ExecutionMetrics()

    def number(key: str) -> float | None:
        value = usage.get(key)
        return float(value) if isinstance(value, (int, float)) and value > 0 else None

    prompt_tokens = usage.get("prompt_tokens")
    generated_tokens = usage.get("completion_tokens")
    prefill = number("prefill_speed_tps")
    decode = number("decoding_speed_tps")
    return ExecutionMetrics(
        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
        generated_tokens=(
            generated_tokens if isinstance(generated_tokens, int) else None
        ),
        prefill_tps=round(prefill, 2) if prefill else None,
        decode_tps=round(decode, 2) if decode else None,
    )


@dataclass(frozen=True, slots=True)
class AdapterOutput:
    text: str
    runtime: str
    model: str
    device: str
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    evidence: dict[str, Any] = field(default_factory=dict)


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
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    evidence: dict[str, Any] = field(default_factory=dict)


class BackendAdapter(Protocol):
    backend: AdapterBackend

    def is_available(self) -> bool: ...

    def run(self, prompt: str) -> AdapterOutput: ...


class ImageBackendAdapter(Protocol):
    backend: AdapterBackend

    def inspect_image(self, image_path: str) -> ImageAdapterOutput: ...
