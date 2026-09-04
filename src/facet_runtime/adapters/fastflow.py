"""FastFlowLM adapter for explicit XDNA2 NPU execution."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from facet_runtime.adapters.base import AdapterOutput
from facet_runtime.errors import (
    BackendMismatchError,
    BackendUnavailableError,
    FacetRuntimeError,
)

FASTFLOW_MODEL = "qwen3:0.6b"


def _command_json(*command: str) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        raise BackendUnavailableError(f"{command[0]} is not installed")
    try:
        result = subprocess.run(
            (executable, *command[1:]),
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BackendUnavailableError(f"{command[0]} could not run: {error}") from error
    if result.returncode != 0:
        raise BackendUnavailableError(result.stderr.strip() or f"{command[0]} failed")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise FacetRuntimeError(f"{command[0]} returned invalid JSON") from error


def _http_json(
    url: str, payload: dict[str, Any] | None = None, timeout: float = 5.0
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise FacetRuntimeError(f"FastFlowLM request failed: {error}") from error


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class FastFlowAdapter:
    """Run Qwen3 on XDNA2 using a short-lived local FastFlowLM server."""

    backend = "npu"

    def _validation(self) -> dict[str, Any]:
        return _command_json("flm", "validate", "--json")

    def _model_installed(self) -> bool:
        models = _command_json("flm", "list", "--filter", "installed", "--json").get(
            "models", []
        )
        return any(
            model.get("model") == FASTFLOW_MODEL and model.get("installed")
            for model in models
        )

    def is_available(self) -> bool:
        try:
            return bool(self._validation().get("ready")) and self._model_installed()
        except FacetRuntimeError:
            return False

    def run(self, prompt: str) -> AdapterOutput:
        validation = self._validation()
        if not validation.get("ready"):
            raise BackendUnavailableError(
                "FastFlowLM reports that the NPU stack is not ready"
            )
        if not self._model_installed():
            raise BackendUnavailableError(
                f"FastFlowLM model {FASTFLOW_MODEL} is not installed"
            )

        devices = validation.get("devices") or []
        accel_device = (
            devices[0].get("device", "/dev/accel/accel0")
            if devices
            else "/dev/accel/accel0"
        )
        executable = shutil.which("flm")
        if executable is None:
            raise BackendUnavailableError("flm is not installed")

        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        log_text = ""
        with tempfile.TemporaryFile() as log_file:
            process = subprocess.Popen(
                (
                    executable,
                    "serve",
                    FASTFLOW_MODEL,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--pmode",
                    "performance",
                ),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 60.0
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        break
                    try:
                        _http_json(f"{base_url}/api/tags", timeout=0.5)
                        break
                    except FacetRuntimeError:
                        time.sleep(0.1)
                else:
                    raise BackendUnavailableError(
                        "FastFlowLM server did not become ready"
                    )
                if process.poll() is not None:
                    raise BackendUnavailableError(
                        "FastFlowLM server exited before becoming ready"
                    )

                response = _http_json(
                    f"{base_url}/api/generate",
                    {
                        "model": FASTFLOW_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "think": False,
                        "options": {"temperature": 0, "num_predict": 256},
                    },
                    timeout=180.0,
                )
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5.0)
                log_file.seek(0)
                log_text = log_file.read().decode("utf-8", errors="replace")

        if "NPU Locked!" not in log_text or "NPU Lock Released!" not in log_text:
            raise BackendMismatchError("FastFlowLM did not confirm NPU execution")
        text = response.get("response")
        if not isinstance(text, str):
            raise FacetRuntimeError("FastFlowLM returned no response text")
        version = _command_json("flm", "version", "--json").get("version", "unknown")
        return AdapterOutput(
            text=text,
            runtime=f"FastFlowLM {version}",
            model=FASTFLOW_MODEL,
            device=f"AMD XDNA2 NPU ({Path(accel_device)})",
        )
