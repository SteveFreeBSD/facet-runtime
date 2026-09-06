from __future__ import annotations

import time

import pytest

from facet_runtime import models
from facet_runtime.adapters import ollama
from facet_runtime.errors import BackendMismatchError, FacetRuntimeError

LOADED_BYTES = 4096


def _fake_requests(
    monkeypatch: pytest.MonkeyPatch,
    *,
    size_vram: int,
    model_name: str | None = None,
    size: int = LOADED_BYTES,
    generate: dict | None = None,
) -> list[dict]:
    payloads: list[dict] = []
    reported = model_name or models.model_for("cpu")

    def request(path: str, payload: dict | None = None) -> dict:
        if path == "/api/version":
            return {"version": "test"}
        if path == "/api/ps":
            return {
                "models": [{"model": reported, "size": size, "size_vram": size_vram}]
            }
        payloads.append(payload or {})
        return {
            "response": "ok",
            "done_reason": "stop",
            "prompt_eval_count": 20,
            "prompt_eval_duration": 100_000_000,
            "eval_count": 40,
            "eval_duration": 1_000_000_000,
            **(generate or {}),
        }

    monkeypatch.setattr(ollama, "_request_json", request)
    monkeypatch.setattr(
        ollama, "_gpu_device", lambda: "AMD Radeon 890M Graphics (RADV STRIX1)"
    )
    monkeypatch.setattr(ollama, "_cpu_model", lambda: "AMD Ryzen AI 9 HX 370")
    return payloads


def test_cpu_forces_zero_gpu_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = _fake_requests(monkeypatch, size_vram=0)
    output = ollama.OllamaAdapter("cpu").run("hello")
    assert output.text == "ok"
    assert output.model == models.model_for("cpu")
    assert payloads[0]["options"]["num_gpu"] == 0
    assert payloads[0]["model"] == models.model_for("cpu")


def test_run_reports_measured_throughput(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_requests(monkeypatch, size_vram=0)
    metrics = ollama.OllamaAdapter("cpu").run("hello").metrics
    assert metrics.prompt_tokens == 20
    assert metrics.generated_tokens == 40
    assert metrics.prefill_tps == 200.0
    assert metrics.decode_tps == 40.0


def test_a_run_reports_why_it_stopped_and_what_it_was_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token count says nothing on its own about whether a run finished."""
    _fake_requests(monkeypatch, size_vram=0)
    metrics = ollama.OllamaAdapter("cpu").run("hello").metrics

    assert metrics.stop_reason == "stop"
    assert metrics.output_token_limit == models.assignment("cpu").max_output_tokens
    assert metrics.truncated() is False


def test_an_empty_completion_is_a_failure_not_an_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _fake_requests(
        monkeypatch, size_vram=0, generate={"response": "   \n ", "done_reason": "stop"}
    )

    with pytest.raises(FacetRuntimeError, match="no response text"):
        ollama.OllamaAdapter("cpu").run("hello")
    assert payloads


def test_a_spent_output_budget_is_reported_as_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live failure: reasoning consumed the whole budget, so nothing was answered.

    Recorded from a real run of the quadratic regression specialist against
    `gpt-oss:20b` on this machine, which stopped on the cap with 4942
    characters of internal reasoning and an empty `response`. Reported as "no
    response text" it looks like a model that declined; reported as a spent
    budget it names the two settings that actually govern it.
    """
    budget = models.assignment("gpu").max_output_tokens
    _fake_requests(
        monkeypatch,
        size_vram=LOADED_BYTES,
        model_name=models.model_for("gpu"),
        generate={
            "response": "",
            "thinking": "x" * 4942,
            "done_reason": "length",
            "eval_count": budget,
        },
    )

    with pytest.raises(FacetRuntimeError) as failure:
        ollama.OllamaAdapter("gpu").run("hello")

    message = str(failure.value)
    assert "stopped" in message and "output cap" in message
    assert f"{budget} of {budget} tokens" in message
    assert "4942 characters" in message
    # The two settings a reader can actually change, named in the failure.
    assert "max_output_tokens" in message and "reasoning_effort" in message


def test_the_declared_reasoning_effort_is_what_is_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that reasons regardless is given an effort, not a refusal."""
    payloads = _fake_requests(
        monkeypatch, size_vram=LOADED_BYTES, model_name=models.model_for("gpu")
    )
    ollama.OllamaAdapter("gpu").run("hello")

    assignment = models.assignment("gpu")
    assert payloads[0]["think"] == assignment.reasoning_effort
    assert payloads[0]["options"]["num_predict"] == assignment.max_output_tokens


def test_a_model_that_does_not_reason_is_asked_not_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _fake_requests(monkeypatch, size_vram=0)
    ollama.OllamaAdapter("cpu").run("hello")

    assert models.assignment("cpu").reasoning_effort is None
    assert payloads[0]["think"] is False


def test_gpu_requires_every_loaded_byte_in_device_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _fake_requests(
        monkeypatch, size_vram=LOADED_BYTES, model_name=models.model_for("gpu")
    )
    output = ollama.OllamaAdapter("gpu").run("hello")
    assert payloads[0]["options"]["num_gpu"] == 999
    assert output.evidence["device_resident_fraction"] == 1.0
    assert output.evidence["device_memory_bytes"] == LOADED_BYTES


def test_gpu_refuses_silent_cpu_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_requests(monkeypatch, size_vram=0, model_name=models.model_for("gpu"))
    with pytest.raises(BackendMismatchError, match="VRAM"):
        ollama.OllamaAdapter("gpu").run("hello")


def test_gpu_refuses_a_partial_offload(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_requests(
        monkeypatch, size_vram=LOADED_BYTES // 4, model_name=models.model_for("gpu")
    )
    with pytest.raises(BackendMismatchError, match="not fully resident"):
        ollama.OllamaAdapter("gpu").run("hello")


def test_gpu_refuses_an_offload_that_rounds_to_fully_resident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_requests(
        monkeypatch,
        size_vram=LOADED_BYTES - 1,
        model_name=models.model_for("gpu"),
    )
    with pytest.raises(BackendMismatchError, match=r"4095 of 4096 bytes"):
        ollama.OllamaAdapter("gpu").run("hello")


def test_cpu_refuses_any_device_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_requests(monkeypatch, size_vram=512)
    with pytest.raises(BackendMismatchError, match="GPU memory"):
        ollama.OllamaAdapter("cpu").run("hello")


def test_gpu_image_request_uses_vision_model_and_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    image = tmp_path / "fixture.png"
    image.write_bytes(b"small test image")
    payloads: list[dict] = []
    vision_model = models.model_for("gpu", "vision")

    def request(path: str, payload: dict | None = None) -> dict:
        if path == "/api/version":
            return {"version": "test"}
        if path == "/api/ps":
            return {
                "models": [
                    {
                        "model": vision_model,
                        "size": LOADED_BYTES,
                        "size_vram": LOADED_BYTES,
                    }
                ]
            }
        payloads.append(payload or {})
        return {"response": '{"transcription":"hello","uncertainties":[]}'}

    monkeypatch.setattr(ollama, "_request_json", request)
    monkeypatch.setattr(
        ollama, "_gpu_device", lambda: "AMD Radeon 890M Graphics (RADV STRIX1)"
    )
    output = ollama.OllamaAdapter("gpu").inspect_image(str(image))
    assert output.backend == "gpu"
    assert output.accelerator_verified is True
    assert payloads[0]["model"] == vision_model
    assert payloads[0]["images"]
    assert payloads[0]["format"]["required"] == ["transcription", "uncertainties"]
    assert payloads[0]["options"]["num_gpu"] == 999
    assert output.runtime_metadata.strict_json_schema is True


class _FakeOllama:
    """An Ollama whose `/api/ps` answer differs during and after a generation.

    The live failure needs exactly that shape: the model answers, and by the
    time the adapter asks where it ran, a competing load has evicted it.
    """

    def __init__(
        self,
        *,
        during: list[dict] | None,
        after: list[dict] | None,
        appear_after: int = 0,
        generate_s: float = 0.4,
    ) -> None:
        self.during = during or []
        self.after = after or []
        self.appear_after = appear_after
        self.generate_s = generate_s
        self.generating = False
        self.polls = 0
        self.payloads: list[dict] = []

    def request(self, path: str, payload: dict | None = None) -> dict:
        if path == "/api/version":
            return {"version": "test"}
        if path == "/api/ps":
            if not self.generating:
                return {"models": self.after}
            self.polls += 1
            seen = [] if self.polls <= self.appear_after else self.during
            return {"models": seen}
        self.payloads.append(payload or {})
        if "prompt" not in (payload or {}):
            return {"done_reason": "unload"}  # the adapter's _unload
        self.generating = True
        time.sleep(self.generate_s)
        self.generating = False
        return {
            "response": "ok",
            "done_reason": "stop",
            "prompt_eval_count": 20,
            "prompt_eval_duration": 100_000_000,
            "eval_count": 40,
            "eval_duration": 1_000_000_000,
        }


def _resident(model: str, vram: int = LOADED_BYTES) -> list[dict]:
    return [{"model": model, "size": LOADED_BYTES, "size_vram": vram}]


def _serve(monkeypatch: pytest.MonkeyPatch, server: _FakeOllama) -> _FakeOllama:
    monkeypatch.setattr(ollama, "RESIDENCY_POLL_S", 0.02, raising=False)
    monkeypatch.setattr(ollama, "_request_json", server.request)
    monkeypatch.setattr(
        ollama, "_gpu_device", lambda: "AMD Radeon 890M Graphics (RADV STRIX1)"
    )
    monkeypatch.setattr(ollama, "_cpu_model", lambda: "AMD Ryzen AI 9 HX 370")
    return server


def test_a_model_evicted_after_it_answers_is_still_proven_by_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live failure: the answer was computed on the GPU, then evicted.

    Ollama serialises loads against a single GPU, so a second client asking for
    a model that will not fit alongside this one makes the scheduler evict the
    runner that just answered. Sampled after the fact, `/api/ps` reports the
    replacement; measured on this machine that read blocks ~40ms against ~30us
    for an uncontended one, which is how every occurrence was identified. The
    weights were resident for the whole generation, so the run stands.
    """
    gpu_model = models.model_for("gpu")
    server = _serve(
        monkeypatch,
        _FakeOllama(during=_resident(gpu_model), after=[]),
    )

    output = ollama.OllamaAdapter("gpu").run("hello")

    assert output.text == "ok"
    assert output.evidence["observed"] == "during generation"
    assert output.evidence["device_memory_bytes"] == LOADED_BYTES
    assert output.evidence["device_resident_fraction"] == 1.0
    # Provenance is unchanged: still this model, on this device.
    assert output.model == gpu_model
    assert output.device == "AMD Radeon 890M Graphics (RADV STRIX1)"
    assert server.payloads[0]["options"]["num_gpu"] == 999


def test_a_cold_load_is_proven_from_the_moment_the_weights_land(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cold start: nothing is resident until the load finishes mid-generation."""
    gpu_model = models.model_for("gpu")
    server = _serve(
        monkeypatch,
        _FakeOllama(during=_resident(gpu_model), after=[], appear_after=3),
    )

    output = ollama.OllamaAdapter("gpu").run("hello")

    assert output.evidence["observed"] == "during generation"
    assert output.evidence["loaded_bytes"] == LOADED_BYTES
    assert server.polls > 3  # it really did wait out the load


def test_a_still_resident_model_is_judged_on_the_live_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The warm path is untouched: the post-generation sample still decides."""
    gpu_model = models.model_for("gpu")
    _serve(
        monkeypatch,
        _FakeOllama(during=_resident(gpu_model), after=_resident(gpu_model)),
    )

    output = ollama.OllamaAdapter("gpu").run("hello")

    assert output.evidence["observed"] == "after generation"
    assert output.evidence["device_resident_fraction"] == 1.0


def test_a_model_that_never_loads_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never resident, during or after: still a hard failure, same message."""
    _serve(monkeypatch, _FakeOllama(during=[], after=[]))

    with pytest.raises(BackendMismatchError, match="did not report the generated"):
        ollama.OllamaAdapter("gpu").run("hello")


def test_an_in_flight_sighting_is_held_to_the_same_device_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silent CPU fallback is still caught when the evidence came in flight.

    The fallback widens *when* the reading may be taken, never what it has to
    show, so a GPU run that never reached VRAM fails exactly as before.
    """
    gpu_model = models.model_for("gpu")
    _serve(
        monkeypatch,
        _FakeOllama(during=_resident(gpu_model, vram=0), after=[]),
    )

    with pytest.raises(BackendMismatchError, match="VRAM"):
        ollama.OllamaAdapter("gpu").run("hello")


def test_a_partial_offload_seen_in_flight_is_still_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_model = models.model_for("gpu")
    _serve(
        monkeypatch,
        _FakeOllama(during=_resident(gpu_model, vram=LOADED_BYTES // 4), after=[]),
    )

    with pytest.raises(BackendMismatchError, match="not fully resident"):
        ollama.OllamaAdapter("gpu").run("hello")


def test_a_cpu_run_that_touched_the_gpu_in_flight_is_still_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_model = models.model_for("cpu")
    _serve(
        monkeypatch,
        _FakeOllama(during=_resident(cpu_model, vram=512), after=[]),
    )

    with pytest.raises(BackendMismatchError, match="GPU memory"):
        ollama.OllamaAdapter("cpu").run("hello")
