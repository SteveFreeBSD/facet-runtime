"""Reproducible measurement of the assigned model on each compute backend.

The benchmark drives the same adapters that `facet run` uses, so every reported
number comes from an execution that already passed Facet's device checks: full
GPU residency for the Radeon path, a zero-VRAM CPU path, and a confirmed NPU
lock for FastFlowLM. A measurement that cannot prove its device fails instead of
being reported.
"""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from facet_runtime import models
from facet_runtime.adapters.base import BackendAdapter, ExecutionMetrics
from facet_runtime.errors import FacetRuntimeError
from facet_runtime.result import ActualBackendName
from facet_runtime.runtime import BACKENDS, default_adapters

_CONTEXT_PARAGRAPH = (
    "The Ryzen AI 9 HX 370 combines twelve Zen 5 and Zen 5c cores with a Radeon "
    "890M RDNA 3.5 integrated GPU and a second-generation XDNA neural processing "
    "unit. All three engines read the same LPDDR5x memory, so sustained decoding "
    "throughput is bounded by memory bandwidth rather than by arithmetic "
    "capability. The NPU is built for sustained low-power matrix work, the GPU "
    "for wide parallel throughput, and the CPU for latency-sensitive control "
    "flow. "
)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    prompt: str
    measures: str


CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        name="latency",
        prompt=(
            "Explain in three sentences why memory bandwidth limits local "
            "language model decoding speed."
        ),
        measures="short-prompt turnaround and decode throughput",
    ),
    BenchmarkCase(
        name="context",
        prompt=(
            "Read the following technical background carefully.\n\n"
            + _CONTEXT_PARAGRAPH * 22
            + "\n\nSummarize the passage above in exactly two sentences."
        ),
        measures="prefill throughput over roughly 2.3k prompt tokens",
    ),
)


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    backend: ActualBackendName
    case: str
    iteration: int
    model: str
    runtime: str
    device: str
    elapsed_ms: float
    metrics: ExecutionMetrics
    evidence: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BenchmarkFailure:
    backend: ActualBackendName
    case: str
    iteration: int
    error: str


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    host: str
    kernel: str
    started_at: str
    repeat: int
    assignments: list[dict[str, Any]]
    runs: list[BenchmarkRun] = field(default_factory=list)
    failures: list[BenchmarkFailure] = field(default_factory=list)

    def summary(self) -> list[dict[str, Any]]:
        """Median of each metric, grouped by backend and case."""
        grouped: dict[tuple[str, str], list[BenchmarkRun]] = {}
        for run in self.runs:
            grouped.setdefault((run.backend, run.case), []).append(run)
        rows = []
        for (backend, case), runs in grouped.items():

            def median(values: list[float | None]) -> float | None:
                present = [value for value in values if value is not None]
                return round(statistics.median(present), 2) if present else None

            rows.append(
                {
                    "backend": backend,
                    "case": case,
                    "model": runs[0].model,
                    "runtime": runs[0].runtime,
                    "device": runs[0].device,
                    "iterations": len(runs),
                    "median_elapsed_ms": median([r.elapsed_ms for r in runs]),
                    "median_prefill_tps": median([r.metrics.prefill_tps for r in runs]),
                    "median_decode_tps": median([r.metrics.decode_tps for r in runs]),
                    "prompt_tokens": runs[-1].metrics.prompt_tokens,
                    "generated_tokens": runs[-1].metrics.generated_tokens,
                }
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "kernel": self.kernel,
            "started_at": self.started_at,
            "repeat": self.repeat,
            "assignments": self.assignments,
            "summary": self.summary(),
            "runs": [asdict(run) for run in self.runs],
            "failures": [asdict(failure) for failure in self.failures],
        }


def run_benchmark(
    backends: Sequence[str] = BACKENDS,
    *,
    cases: Sequence[BenchmarkCase] = CASES,
    repeat: int = 2,
    adapters: Mapping[str, BackendAdapter] | None = None,
) -> BenchmarkReport:
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    unknown = [name for name in backends if name not in BACKENDS]
    if unknown:
        raise ValueError(f"unsupported backend: {', '.join(unknown)}")

    adapter_map = dict(adapters or default_adapters())
    report = BenchmarkReport(
        host=platform.node(),
        kernel=platform.release(),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        repeat=repeat,
        assignments=models.report(),
    )
    for backend in backends:
        adapter = adapter_map[backend]
        for case in cases:
            for iteration in range(1, repeat + 1):
                started = time.perf_counter()
                try:
                    output = adapter.run(case.prompt)
                except (FacetRuntimeError, ValueError) as error:
                    report.failures.append(
                        BenchmarkFailure(
                            backend=backend,
                            case=case.name,
                            iteration=iteration,
                            error=f"{type(error).__name__}: {error}",
                        )
                    )
                    continue
                report.runs.append(
                    BenchmarkRun(
                        backend=backend,
                        case=case.name,
                        iteration=iteration,
                        model=output.model,
                        runtime=output.runtime,
                        device=output.device,
                        elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                        metrics=output.metrics,
                        evidence=output.evidence,
                    )
                )
    return report
