"""Ollama adapters with explicit CPU-only and Vulkan GPU execution."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from typing import Any, Literal

from facet_runtime import models
from facet_runtime.adapters.base import (
    AdapterOutput,
    ImageAdapterOutput,
    ImageRuntimeMetadata,
    metrics_from_ollama_api,
)
from facet_runtime.adapters.image_contract import (
    TRANSCRIPTION_PROMPT,
    TRANSCRIPTION_SCHEMA,
    encode_image,
    parse_transcription,
)
from facet_runtime.errors import (
    BackendMismatchError,
    BackendUnavailableError,
    FacetRuntimeError,
)

OLLAMA_URL = os.environ.get("FACET_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")

# A model is only accepted as GPU-resident when Ollama reports every byte it
# loaded sitting in device memory. A partial offload is a silent CPU fallback.
FULL_RESIDENCY = 1.0


def _request_json(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{OLLAMA_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=600.0) as response:
            return json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise FacetRuntimeError(f"Ollama request failed: {error}") from error


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo.read(), re.MULTILINE)
    except OSError:
        match = None
    return match.group(1).strip() if match else "native CPU"


def _gpu_device() -> str | None:
    executable = shutil.which("vulkaninfo")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            (executable, "--summary"),
            check=False,
            capture_output=True,
            text=True,
            timeout=15.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = result.stdout + result.stderr
    name = re.search(r"^\s*deviceName\s*=\s*(.+)$", output, re.MULTILINE)
    driver = re.search(r"^\s*driverName\s*=\s*(.+)$", output, re.MULTILINE)
    if result.returncode != 0 or name is None or driver is None:
        return None
    if "Radeon 890M" not in name.group(1) or driver.group(1).strip().lower() != "radv":
        return None
    return name.group(1).strip()


def _loaded_model(model_name: str) -> dict[str, Any] | None:
    for model in _request_json("/api/ps").get("models", []):
        if model.get("model") == model_name or model.get("name") == model_name:
            return model
    return None


def _verify_loaded_backend(
    model_name: str, backend: Literal["cpu", "gpu"]
) -> dict[str, Any]:
    """Prove where Ollama put the weights, and return that proof as evidence."""
    loaded = _loaded_model(model_name)
    if loaded is None:
        raise BackendMismatchError(
            "Ollama did not report the generated model as loaded"
        )
    size = int(loaded.get("size", 0))
    vram = int(loaded.get("size_vram", 0))
    resident = round(vram / size, 4) if size else 0.0
    if backend == "cpu" and vram != 0:
        raise BackendMismatchError(
            f"CPU-only execution used {vram} bytes of GPU memory"
        )
    if backend == "gpu":
        if vram <= 0:
            raise BackendMismatchError(
                "GPU execution did not load model data into VRAM"
            )
        if resident < FULL_RESIDENCY:
            raise BackendMismatchError(
                f"GPU execution offloaded only {resident:.1%} of {model_name} "
                f"({vram} of {size} bytes); the remainder would run on the CPU"
            )
    return {
        "source": "ollama /api/ps",
        "loaded_bytes": size,
        "device_memory_bytes": vram,
        "device_resident_fraction": resident,
        "context_length": loaded.get("context_length"),
        "quantization": (loaded.get("details") or {}).get("quantization_level"),
    }


def _unload(model_name: str) -> None:
    try:
        _request_json("/api/generate", {"model": model_name, "keep_alive": 0})
    except FacetRuntimeError:
        pass


class OllamaAdapter:
    """Execute the assigned Ollama model on exactly one requested processor."""

    def __init__(self, backend: Literal["cpu", "gpu"]) -> None:
        self.backend = backend

    @property
    def assignment(self) -> models.ModelAssignment:
        return models.assignment(self.backend, "text")

    @property
    def model(self) -> str:
        return self.assignment.resolved_model()

    def is_available(self) -> bool:
        try:
            _request_json("/api/version")
        except FacetRuntimeError:
            return False
        return self.backend == "cpu" or _gpu_device() is not None

    def _device(self) -> str:
        device = _cpu_model() if self.backend == "cpu" else _gpu_device()
        if device is None:
            raise BackendUnavailableError("Radeon 890M RADV device is unavailable")
        return device

    def run(self, prompt: str) -> AdapterOutput:
        if not self.is_available():
            raise BackendUnavailableError(
                f"Ollama {self.backend} backend is unavailable"
            )
        device = self._device()
        assignment = self.assignment
        model_name = assignment.resolved_model()
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": "30s",
            "options": {
                "temperature": 0,
                "num_ctx": assignment.context_tokens,
                "num_predict": assignment.max_output_tokens,
                "num_gpu": 0 if self.backend == "cpu" else 999,
            },
        }
        try:
            response = _request_json("/api/generate", payload)
            evidence = _verify_loaded_backend(model_name, self.backend)
            text = response.get("response")
            if not isinstance(text, str) or not text.strip():
                raise FacetRuntimeError("Ollama returned no response text")
            version = _request_json("/api/version").get("version", "unknown")
            return AdapterOutput(
                text=text,
                runtime=f"Ollama {version}",
                model=model_name,
                device=device,
                metrics=metrics_from_ollama_api(response),
                evidence=evidence,
            )
        finally:
            _unload(model_name)

    def inspect_image(self, image_path: str) -> ImageAdapterOutput:
        if self.backend != "gpu":
            raise BackendMismatchError(
                "Ollama image inspection requires the GPU adapter"
            )
        if not self.is_available():
            raise BackendUnavailableError("Ollama GPU backend is unavailable")
        device = self._device()
        assignment = models.assignment("gpu", "vision")
        model_name = assignment.resolved_model()

        encoded_image, _ = encode_image(image_path)
        payload = {
            "model": model_name,
            "prompt": TRANSCRIPTION_PROMPT,
            "images": [encoded_image],
            "stream": False,
            "think": False,
            "keep_alive": "30s",
            "format": TRANSCRIPTION_SCHEMA,
            "options": {
                "temperature": 0,
                "num_ctx": assignment.context_tokens,
                "num_predict": assignment.max_output_tokens,
                "num_gpu": 999,
            },
        }
        try:
            response = _request_json("/api/generate", payload)
            evidence = _verify_loaded_backend(model_name, "gpu")
            raw = response.get("response")
            if not isinstance(raw, str):
                raise FacetRuntimeError("Ollama returned no image transcription")
            transcription, uncertainties = parse_transcription(raw)
            version = _request_json("/api/version").get("version", "unknown")
            return ImageAdapterOutput(
                backend="gpu",
                transcription=transcription,
                uncertainties=uncertainties,
                runtime=f"Ollama {version}",
                model=model_name,
                device=device,
                runtime_metadata=ImageRuntimeMetadata(
                    protocol="ollama_generate_images",
                    response_format="json_schema",
                    strict_json_schema=True,
                ),
                accelerator_verified=True,
                metrics=metrics_from_ollama_api(response),
                evidence=evidence,
            )
        finally:
            _unload(model_name)
