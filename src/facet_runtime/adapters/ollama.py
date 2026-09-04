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

from facet_runtime.adapters.base import AdapterOutput
from facet_runtime.errors import (
    BackendMismatchError,
    BackendUnavailableError,
    FacetRuntimeError,
)

OLLAMA_MODEL = "qwen3:0.6b"
OLLAMA_URL = os.environ.get("FACET_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


def _request_json(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{OLLAMA_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=180.0) as response:
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


def _loaded_model() -> dict[str, Any] | None:
    for model in _request_json("/api/ps").get("models", []):
        if model.get("model") == OLLAMA_MODEL or model.get("name") == OLLAMA_MODEL:
            return model
    return None


class OllamaAdapter:
    """Execute Qwen3 through Ollama on exactly one requested processor."""

    def __init__(self, backend: Literal["cpu", "gpu"]) -> None:
        self.backend = backend

    def is_available(self) -> bool:
        try:
            _request_json("/api/version")
        except FacetRuntimeError:
            return False
        return self.backend == "cpu" or _gpu_device() is not None

    def run(self, prompt: str) -> AdapterOutput:
        if not self.is_available():
            raise BackendUnavailableError(
                f"Ollama {self.backend} backend is unavailable"
            )

        device = _cpu_model() if self.backend == "cpu" else _gpu_device()
        if device is None:
            raise BackendUnavailableError("Radeon 890M RADV device is unavailable")

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": "30s",
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
                "num_predict": 256,
                "num_gpu": 0 if self.backend == "cpu" else 999,
            },
        }
        try:
            response = _request_json("/api/generate", payload)
            loaded = _loaded_model()
            if loaded is None:
                raise BackendMismatchError(
                    "Ollama did not report the generated model as loaded"
                )
            vram = int(loaded.get("size_vram", 0))
            if self.backend == "cpu" and vram != 0:
                raise BackendMismatchError(
                    f"CPU-only execution used {vram} bytes of GPU memory"
                )
            if self.backend == "gpu" and vram <= 0:
                raise BackendMismatchError(
                    "GPU execution did not load model data into VRAM"
                )
            text = response.get("response")
            if not isinstance(text, str):
                raise FacetRuntimeError("Ollama returned no response text")
            version = _request_json("/api/version").get("version", "unknown")
            return AdapterOutput(
                text=text,
                runtime=f"Ollama {version}",
                model=OLLAMA_MODEL,
                device=device,
            )
        finally:
            try:
                _request_json("/api/generate", {"model": OLLAMA_MODEL, "keep_alive": 0})
            except FacetRuntimeError:
                pass
