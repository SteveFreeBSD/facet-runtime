from __future__ import annotations

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
            "prompt_eval_count": 20,
            "prompt_eval_duration": 100_000_000,
            "eval_count": 40,
            "eval_duration": 1_000_000_000,
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


def test_an_empty_completion_is_a_failure_not_an_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _fake_requests(monkeypatch, size_vram=0)

    original = ollama._request_json

    def request(path: str, payload: dict | None = None) -> dict:
        result = original(path, payload)
        if path == "/api/generate" and payload and "prompt" in payload:
            return {**result, "response": "   \n "}
        return result

    monkeypatch.setattr(ollama, "_request_json", request)
    with pytest.raises(FacetRuntimeError, match="no response text"):
        ollama.OllamaAdapter("cpu").run("hello")
    assert payloads


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
    with pytest.raises(BackendMismatchError, match="offloaded only 25"):
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
