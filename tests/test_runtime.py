from __future__ import annotations

from dataclasses import dataclass

import pytest

from facet_runtime.adapters.base import AdapterOutput
from facet_runtime.errors import BackendMismatchError, FacetRuntimeError
from facet_runtime.runtime import run_prompt


@dataclass
class FakeAdapter:
    backend: str
    available: bool = True
    error: Exception | None = None
    calls: int = 0

    def is_available(self) -> bool:
        return self.available

    def run(self, prompt: str) -> AdapterOutput:
        self.calls += 1
        if self.error:
            raise self.error
        return AdapterOutput(
            prompt.upper(), "fake-runtime", "fake-model", "fake-device"
        )


def test_common_result_shape() -> None:
    adapters = {name: FakeAdapter(name) for name in ("cpu", "gpu", "npu")}
    result = run_prompt("hello", "cpu", adapters=adapters)
    assert result.to_dict().keys() == {
        "text",
        "requested_backend",
        "actual_backend",
        "runtime",
        "model",
        "device",
        "elapsed_ms",
        "fallback",
    }
    assert result.requested_backend == result.actual_backend == "cpu"
    assert result.fallback is False


def test_auto_preference_is_npu_then_gpu_then_cpu() -> None:
    adapters = {
        "cpu": FakeAdapter("cpu"),
        "gpu": FakeAdapter("gpu"),
        "npu": FakeAdapter("npu"),
    }
    result = run_prompt("hello", "auto", adapters=adapters)
    assert result.requested_backend == "auto"
    assert result.actual_backend == "npu"
    assert adapters["npu"].calls == 1
    assert adapters["gpu"].calls == adapters["cpu"].calls == 0


def test_auto_does_not_retry_after_selected_backend_fails() -> None:
    adapters = {
        "cpu": FakeAdapter("cpu"),
        "gpu": FakeAdapter("gpu"),
        "npu": FakeAdapter("npu", error=FacetRuntimeError("npu failed")),
    }
    with pytest.raises(FacetRuntimeError, match="npu failed"):
        run_prompt("hello", "auto", adapters=adapters)
    assert adapters["npu"].calls == 1
    assert adapters["gpu"].calls == adapters["cpu"].calls == 0


def test_adapter_cannot_report_a_different_backend() -> None:
    adapters = {
        "cpu": FakeAdapter("gpu"),
        "gpu": FakeAdapter("gpu"),
        "npu": FakeAdapter("npu"),
    }
    with pytest.raises(BackendMismatchError):
        run_prompt("hello", "cpu", adapters=adapters)
