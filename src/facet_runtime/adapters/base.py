"""Internal adapter protocol used by the runtime coordinator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from facet_runtime.errors import FacetRuntimeError

AdapterBackend = Literal["cpu", "gpu", "npu"]

#: What a runtime calls a generation that stopped on its output cap rather than
#: because the model had finished. Ollama says `length`; the OpenAI-shaped
#: endpoint FastFlowLM serves says the same.
TRUNCATED_STOP = "length"


@dataclass(frozen=True, slots=True)
class ExecutionMetrics:
    """Token counts and throughput reported by the runtime that executed.

    `stop_reason` and `output_token_limit` travel with the counts because
    `generated_tokens` alone cannot say whether a run finished or was cut off,
    and a run that was cut off has produced something different from a run that
    ended when the model was done.
    """

    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    prefill_tps: float | None = None
    decode_tps: float | None = None
    #: Why generation stopped, in the executing runtime's own vocabulary.
    stop_reason: str | None = None
    #: The output cap this run was given, so a count can be read against it.
    output_token_limit: int | None = None

    def truncated(self) -> bool:
        """Whether the runtime stopped this run on its output cap."""
        return self.stop_reason == TRUNCATED_STOP


def empty_completion_error(
    runtime: str,
    model: str,
    *,
    metrics: ExecutionMetrics,
    reasoning_chars: int = 0,
) -> FacetRuntimeError:
    """Say why a completion came back empty, not merely that it did.

    An empty answer has two causes that need opposite fixes, and reporting
    both the same way is what makes the failure hard to act on. A model
    stopped on its output cap did not decline the question: it spent the whole
    budget on internal reasoning and never reached an answer, which is a
    budget or reasoning-effort problem in this repository's own model
    assignment. Anything else is the model genuinely returning nothing, which
    is a problem with the question. Neither is recoverable here, so this
    reports the difference rather than acting on it.
    """
    if metrics.truncated():
        spent = metrics.generated_tokens or metrics.output_token_limit
        reasoning = (
            f", all of it internal reasoning ({reasoning_chars} characters of it)"
            if reasoning_chars
            else ""
        )
        return FacetRuntimeError(
            f"{runtime} stopped {model} on its output cap after {spent} of "
            f"{metrics.output_token_limit} tokens{reasoning}, so it never wrote "
            "an answer; raise max_output_tokens or lower reasoning_effort for "
            "this assignment"
        )
    return FacetRuntimeError(
        f"{runtime} returned no response text for {model} "
        f"(stop reason {metrics.stop_reason or 'unreported'})"
    )


def metrics_from_ollama_api(
    response: dict[str, Any], *, output_token_limit: int | None = None
) -> ExecutionMetrics:
    """Read the Ollama-compatible timing block that Ollama and FastFlowLM emit."""

    def positive(key: str) -> int:
        value = response.get(key)
        return value if isinstance(value, int) and value > 0 else 0

    prompt_tokens = positive("prompt_eval_count")
    prompt_ns = positive("prompt_eval_duration")
    generated_tokens = positive("eval_count")
    generated_ns = positive("eval_duration")
    stop_reason = response.get("done_reason")
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
        stop_reason=stop_reason if isinstance(stop_reason, str) else None,
        output_token_limit=output_token_limit,
    )


def _finish_reason(response: dict[str, Any]) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    reason = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
    return reason if isinstance(reason, str) else None


def metrics_from_openai_usage(
    response: dict[str, Any], *, output_token_limit: int | None = None
) -> ExecutionMetrics:
    """Read the OpenAI-shaped usage block that FastFlowLM returns."""
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return ExecutionMetrics(
            stop_reason=_finish_reason(response),
            output_token_limit=output_token_limit,
        )

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
        stop_reason=_finish_reason(response),
        output_token_limit=output_token_limit,
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
