from __future__ import annotations

from dataclasses import dataclass

import pytest

from facet_runtime.adapters.base import AdapterOutput, ExecutionMetrics
from facet_runtime.benchmark import CASES, run_benchmark
from facet_runtime.errors import BackendUnavailableError


@dataclass
class FakeAdapter:
    backend: str
    decode_tps: float = 10.0
    error: Exception | None = None
    prompts: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.prompts = []

    def is_available(self) -> bool:
        return self.error is None

    def run(self, prompt: str) -> AdapterOutput:
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return AdapterOutput(
            text="ok",
            runtime="fake-runtime",
            model="fake-model",
            device=f"fake-{self.backend}",
            metrics=ExecutionMetrics(
                prompt_tokens=10,
                generated_tokens=20,
                prefill_tps=100.0,
                decode_tps=self.decode_tps,
            ),
            evidence={"source": "fake"},
        )


def _adapters(**overrides) -> dict[str, FakeAdapter]:
    adapters = {name: FakeAdapter(name) for name in ("cpu", "gpu", "npu")}
    adapters.update(overrides)
    return adapters


def test_benchmark_measures_every_backend_and_case() -> None:
    adapters = _adapters()
    report = run_benchmark(("cpu", "gpu", "npu"), repeat=2, adapters=adapters)
    assert len(report.runs) == 3 * len(CASES) * 2
    assert report.failures == []
    assert {row["backend"] for row in report.summary()} == {"cpu", "gpu", "npu"}
    assert all(adapter.prompts for adapter in adapters.values())


def test_summary_reports_the_median_of_repeated_runs() -> None:
    adapters = _adapters(gpu=FakeAdapter("gpu", decode_tps=42.0))
    report = run_benchmark(("gpu",), cases=CASES[:1], repeat=3, adapters=adapters)
    row = report.summary()[0]
    assert row["median_decode_tps"] == 42.0
    assert row["iterations"] == 3
    assert row["prompt_tokens"] == 10


def test_a_backend_that_cannot_prove_its_device_is_recorded_as_a_failure() -> None:
    adapters = _adapters(
        npu=FakeAdapter("npu", error=BackendUnavailableError("no NPU lock"))
    )
    report = run_benchmark(("gpu", "npu"), cases=CASES[:1], repeat=1, adapters=adapters)
    assert [failure.backend for failure in report.failures] == ["npu"]
    assert "no NPU lock" in report.failures[0].error
    assert [run.backend for run in report.runs] == ["gpu"]
    assert [row["backend"] for row in report.summary()] == ["gpu"]


def test_context_case_prompt_is_long_enough_to_measure_prefill() -> None:
    context = next(case for case in CASES if case.name == "context")
    assert len(context.prompt) > 8000


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported backend"):
        run_benchmark(("tpu",), adapters=_adapters())
