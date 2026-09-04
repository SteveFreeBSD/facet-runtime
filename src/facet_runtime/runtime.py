"""Backend selection and the common Facet runtime contract."""

from __future__ import annotations

import time
from collections.abc import Mapping

from facet_runtime.adapters import FastFlowAdapter, OllamaAdapter
from facet_runtime.adapters.base import BackendAdapter
from facet_runtime.errors import BackendMismatchError, BackendUnavailableError
from facet_runtime.result import BackendName, RunResult

AUTO_PREFERENCE = ("npu", "gpu", "cpu")


def default_adapters() -> dict[str, BackendAdapter]:
    return {
        "cpu": OllamaAdapter("cpu"),
        "gpu": OllamaAdapter("gpu"),
        "npu": FastFlowAdapter(),
    }


def run_prompt(
    prompt: str,
    requested_backend: BackendName = "auto",
    *,
    adapters: Mapping[str, BackendAdapter] | None = None,
) -> RunResult:
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if requested_backend not in {"cpu", "gpu", "npu", "auto"}:
        raise ValueError(f"unsupported backend: {requested_backend}")

    adapter_map = dict(adapters or default_adapters())
    started = time.perf_counter()
    if requested_backend == "auto":
        selected = next(
            (name for name in AUTO_PREFERENCE if adapter_map[name].is_available()),
            None,
        )
        if selected is None:
            raise BackendUnavailableError("no Facet backend is available")
    else:
        selected = requested_backend

    adapter = adapter_map[selected]
    output = adapter.run(prompt)
    if adapter.backend != selected:
        raise BackendMismatchError(
            f"adapter reported {adapter.backend}, but Facet selected {selected}"
        )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return RunResult(
        text=output.text,
        requested_backend=requested_backend,
        actual_backend=adapter.backend,
        runtime=output.runtime,
        model=output.model,
        device=output.device,
        elapsed_ms=elapsed_ms,
        fallback=False,
    )
