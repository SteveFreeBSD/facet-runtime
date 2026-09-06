"""Ollama adapters with explicit CPU-only and Vulkan GPU execution."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from typing import Any, Literal, Self

from facet_runtime import models
from facet_runtime.adapters.base import (
    AdapterOutput,
    ImageAdapterOutput,
    ImageRuntimeMetadata,
    empty_completion_error,
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


#: How often the residency watch samples `/api/ps` while a generation is in
#: flight. The watch returns at its first sighting, so this bounds how soon
#: after the weights land the evidence is taken, not how long anything waits.
RESIDENCY_POLL_S = 0.25


class _ResidencyWatch:
    """Record where the weights sat *while* the tokens were being computed.

    `/api/ps` describes the present. Sampled after `/api/generate` returns it
    describes a runner that is already idle and evictable, so a second client
    loading a model that does not fit alongside it makes Ollama evict the
    runner that just answered, and the sample reports the replacement instead.
    Measured here: that check normally answers in ~30us, but when it contends
    with a scheduler eviction it blocks ~40ms and returns the post-eviction
    state -- the signature of every observed occurrence of this failure.

    Ollama lists a model only once it is fully resident, so a sighting taken
    during the generation is the same evidence about the same weights, observed
    while our own request holds the runner and nothing can have replaced it.
    """

    def __init__(self, model_name: str) -> None:
        self._model = model_name
        self._stop = threading.Event()
        self._sighting: dict[str, Any] | None = None
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self) -> None:
        while not self._stop.is_set():
            try:
                loaded = _loaded_model(self._model)
            except FacetRuntimeError:
                loaded = None  # A blip here costs the fallback, not the run.
            if loaded is not None:
                self._sighting = loaded
                return
            self._stop.wait(RESIDENCY_POLL_S)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=RESIDENCY_POLL_S * 2)

    @property
    def sighting(self) -> dict[str, Any] | None:
        """The model as `/api/ps` reported it mid-generation, if it was seen."""
        return self._sighting


def _verify_loaded_backend(
    model_name: str,
    backend: Literal["cpu", "gpu"],
    observed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove where Ollama put the weights, and return that proof as evidence."""
    loaded = _loaded_model(model_name)
    # The live reading is preferred, so a run whose model is still resident is
    # judged exactly as before, on exactly the same numbers. `observed` answers
    # only the case where the runner was evicted between the answer and this
    # check, and it is held to every one of the same tests below.
    taken = "after generation"
    if loaded is None and observed is not None:
        loaded, taken = observed, "during generation"
    if loaded is None:
        raise BackendMismatchError(
            "Ollama did not report the generated model as loaded"
        )
    size = int(loaded.get("size", 0))
    vram = int(loaded.get("size_vram", 0))
    resident = vram / size if size else 0.0
    if backend == "cpu" and vram != 0:
        raise BackendMismatchError(
            f"CPU-only execution used {vram} bytes of GPU memory"
        )
    if backend == "gpu":
        if vram <= 0:
            raise BackendMismatchError(
                "GPU execution did not load model data into VRAM"
            )
        if vram != size:
            raise BackendMismatchError(
                f"GPU execution was not fully resident for {model_name} "
                f"({vram} of {size} bytes in device memory; fraction "
                f"{resident:.9f}); the remainder would run on the CPU"
            )
    return {
        "source": "ollama /api/ps",
        "observed": taken,
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
            # A model that reasons unconditionally ignores `false` and reasons
            # anyway, so the assignment states an effort instead of a refusal
            # the model would not honour. Those tokens come out of num_predict
            # before the answer does.
            "think": assignment.reasoning_effort or False,
            "keep_alive": "30s",
            "options": {
                "temperature": 0,
                "num_ctx": assignment.context_tokens,
                "num_predict": assignment.max_output_tokens,
                "num_gpu": 0 if self.backend == "cpu" else 999,
            },
        }
        try:
            with _ResidencyWatch(model_name) as watch:
                response = _request_json("/api/generate", payload)
            evidence = _verify_loaded_backend(model_name, self.backend, watch.sighting)
            metrics = metrics_from_ollama_api(
                response, output_token_limit=assignment.max_output_tokens
            )
            text = response.get("response")
            if not isinstance(text, str) or not text.strip():
                # An empty answer from a reasoning model is usually a spent
                # budget rather than a refusal, and the two need opposite
                # fixes, so the reason is reported rather than flattened.
                raise empty_completion_error(
                    "Ollama",
                    model_name,
                    metrics=metrics,
                    reasoning_chars=len(response.get("thinking") or ""),
                )
            version = _request_json("/api/version").get("version", "unknown")
            return AdapterOutput(
                text=text,
                runtime=f"Ollama {version}",
                model=model_name,
                device=device,
                metrics=metrics,
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
            "think": assignment.reasoning_effort or False,
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
            with _ResidencyWatch(model_name) as watch:
                response = _request_json("/api/generate", payload)
            evidence = _verify_loaded_backend(model_name, "gpu", watch.sighting)
            metrics = metrics_from_ollama_api(
                response, output_token_limit=assignment.max_output_tokens
            )
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
                metrics=metrics,
                evidence=evidence,
            )
        finally:
            _unload(model_name)
