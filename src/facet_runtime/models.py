"""Explicit model assignment for each Facet compute backend.

Every model Facet runs is declared here once, with the runtime that serves it,
the measured weight footprint, and the reason that model was chosen for that
device. Adapters never name a model themselves, so the mapping from backend to
model is auditable in one place and cannot drift between the text and image
paths.

Each assignment can be overridden through its environment variable for
experiments. An override changes the model but never the backend: Facet still
runs on the requested device or fails.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

Backend = Literal["cpu", "gpu", "npu"]
Role = Literal["text", "vision"]
Runtime = Literal["ollama", "fastflowlm"]


@dataclass(frozen=True, slots=True)
class ModelAssignment:
    """One backend/role pairing and the evidence behind it."""

    backend: Backend
    role: Role
    runtime: Runtime
    model: str
    parameters: str
    quantization: str
    # On-disk size of the model as its runtime stores it.
    disk_gib: float
    context_tokens: int
    max_output_tokens: int
    #: How much internal reasoning the model may spend before it answers, or
    #: `None` to ask it not to reason at all. A harmony reasoning model such as
    #: gpt-oss ignores that request and reasons unconditionally, so it is given
    #: an explicit effort instead. Those tokens are spent out of
    #: `max_output_tokens`, ahead of the answer, so an effort and a budget only
    #: mean anything together.
    reasoning_effort: str | None
    rationale: str
    env_override: str

    def resolved_model(self) -> str:
        """The model actually used, honouring an explicit environment override."""
        override = os.environ.get(self.env_override, "").strip()
        return override or self.model

    def overridden(self) -> bool:
        return self.resolved_model() != self.model

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "role": self.role,
            "runtime": self.runtime,
            "model": self.resolved_model(),
            "default_model": self.model,
            "overridden": self.overridden(),
            "env_override": self.env_override,
            "parameters": self.parameters,
            "quantization": self.quantization,
            "disk_gib": self.disk_gib,
            "context_tokens": self.context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "reasoning_effort": self.reasoning_effort,
            "rationale": self.rationale,
        }


ASSIGNMENTS: tuple[ModelAssignment, ...] = (
    ModelAssignment(
        backend="cpu",
        role="text",
        runtime="ollama",
        model="qwen3.5:2b",
        parameters="2.3B",
        quantization="Q8_0",
        disk_gib=2.55,
        context_tokens=16384,
        max_output_tokens=768,
        reasoning_effort=None,
        rationale=(
            "The CPU path has no accelerator to hide behind: prefill on twelve "
            "Zen 5 cores falls off far faster with model size than decode does, "
            "so the CPU takes the smallest model that is still a capable "
            "instruct model rather than the largest one that technically runs."
        ),
        env_override="FACET_CPU_TEXT_MODEL",
    ),
    ModelAssignment(
        backend="gpu",
        role="text",
        runtime="ollama",
        model="gpt-oss:20b",
        parameters="20B MoE, 3.6B active",
        quantization="MXFP4",
        disk_gib=12.1,
        context_tokens=16384,
        max_output_tokens=2048,
        reasoning_effort="low",
        rationale=(
            "The 890M reaches the whole 14.8 GiB GTT aperture, and a "
            "mixture-of-experts model reads only its active experts per token, "
            "so this is both the largest and the fastest thing the GPU can "
            "hold: 504 prefill and 21.2 decode tokens per second against 319 "
            "and 14.4 for a dense 9B. Ollama reports 12.75 GB of 12.75 GB in "
            "device memory at a 16k context, so nothing spills to the CPU. "
            "This model reasons unconditionally and ignores a request not to, "
            "so the effort is stated rather than refused: a measured quadratic "
            "regression spent all 1024 tokens of the previous budget reasoning "
            "and returned an empty answer, and the same question at low effort "
            "finished in 889 including its answer. The budget is 2048 so that "
            "a harder question has somewhere to go before it hits the cap."
        ),
        env_override="FACET_GPU_TEXT_MODEL",
    ),
    ModelAssignment(
        backend="npu",
        role="text",
        runtime="fastflowlm",
        model="gpt-oss:20b",
        parameters="20B MoE, 3.6B active",
        quantization="NPU2",
        disk_gib=14.0,
        context_tokens=16384,
        max_output_tokens=2048,
        reasoning_effort=None,
        rationale=(
            "This remains the measured preferred NPU text worker: its 18.7 "
            "decode tokens per second beat 9.3 for a dense 9B and come within "
            "12% of the 890M on the identical model. One hard reasoning request "
            "returned an empty completion and therefore a Facet error. That "
            "failure has since been reproduced and measured on the GPU path "
            "with this same model: it spends its output budget reasoning and "
            "stops on the cap before writing an answer. FastFlowLM reasons "
            "unconditionally here too and counts those tokens against "
            "max_output_tokens, so it takes the same enlarged budget. No "
            "reasoning effort is stated because FastFlowLM exposes no such "
            "control; the budget and the truncation report are the whole "
            "defence on this path. Set FACET_NPU_TEXT_MODEL to qwen3.5:9b for "
            "a 7.7 GiB, faster-loading, vision-capable alternative at roughly "
            "half the decode rate."
        ),
        env_override="FACET_NPU_TEXT_MODEL",
    ),
    ModelAssignment(
        backend="gpu",
        role="vision",
        runtime="ollama",
        model="qwen3.5:9b",
        parameters="9B",
        quantization="Q4_K_M",
        disk_gib=6.2,
        context_tokens=8192,
        max_output_tokens=512,
        reasoning_effort=None,
        rationale=(
            "Image inspection runs both accelerators over the same picture, so "
            "the two passes stay in the same model and size class and a "
            "difference between them means a device difference. At 9B both "
            "devices recover the heading and the exact glyphs that 4B dropped."
        ),
        env_override="FACET_GPU_VISION_MODEL",
    ),
    ModelAssignment(
        backend="npu",
        role="vision",
        runtime="fastflowlm",
        model="qwen3.5:9b",
        parameters="9B",
        quantization="Q4_1 (NPU2)",
        disk_gib=7.7,
        context_tokens=8192,
        max_output_tokens=512,
        reasoning_effort=None,
        rationale=(
            "The NPU half of the image pair matches the GPU half exactly. The "
            "two passes run one after the other, so only one 9B vision model "
            "is resident at a time."
        ),
        env_override="FACET_NPU_VISION_MODEL",
    ),
)


def _index() -> dict[tuple[str, str], ModelAssignment]:
    return {(a.backend, a.role): a for a in ASSIGNMENTS}


def assignment(backend: Backend, role: Role = "text") -> ModelAssignment:
    try:
        return _index()[(backend, role)]
    except KeyError:
        raise ValueError(
            f"no {role} model is assigned to the {backend} backend"
        ) from None


def model_for(backend: Backend, role: Role = "text") -> str:
    return assignment(backend, role).resolved_model()


def report() -> list[dict[str, Any]]:
    return [a.to_dict() for a in ASSIGNMENTS]
