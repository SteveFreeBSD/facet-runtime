from __future__ import annotations

import pytest

from facet_runtime.adapters import ollama
from facet_runtime.errors import BackendMismatchError


def _fake_requests(monkeypatch: pytest.MonkeyPatch, *, size_vram: int) -> list[dict]:
    payloads: list[dict] = []

    def request(path: str, payload: dict | None = None) -> dict:
        if path == "/api/version":
            return {"version": "test"}
        if path == "/api/ps":
            return {"models": [{"model": ollama.OLLAMA_MODEL, "size_vram": size_vram}]}
        payloads.append(payload or {})
        return {"response": "ok"}

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
    assert payloads[0]["options"]["num_gpu"] == 0


def test_gpu_forces_layers_and_requires_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = _fake_requests(monkeypatch, size_vram=1024)
    ollama.OllamaAdapter("gpu").run("hello")
    assert payloads[0]["options"]["num_gpu"] == 999


def test_gpu_refuses_silent_cpu_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_requests(monkeypatch, size_vram=0)
    with pytest.raises(BackendMismatchError, match="VRAM"):
        ollama.OllamaAdapter("gpu").run("hello")


def test_gpu_image_request_uses_vision_model_and_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    image = tmp_path / "fixture.png"
    image.write_bytes(b"small test image")
    payloads: list[dict] = []

    def request(path: str, payload: dict | None = None) -> dict:
        if path == "/api/version":
            return {"version": "test"}
        if path == "/api/ps":
            return {
                "models": [{"model": ollama.OLLAMA_VISION_MODEL, "size_vram": 1024}]
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
    assert payloads[0]["model"] == "qwen3.5:4b"
    assert payloads[0]["images"]
    assert payloads[0]["format"]["required"] == ["transcription", "uncertainties"]
    assert payloads[0]["options"]["num_gpu"] == 999
    assert output.runtime_metadata.strict_json_schema is True
