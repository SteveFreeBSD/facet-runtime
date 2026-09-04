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

from facet_runtime.adapters.base import (
    AdapterOutput,
    ImageAdapterOutput,
    ImageRuntimeMetadata,
)
from facet_runtime.adapters.image_contract import (
    TRANSCRIPTION_JSON_PROMPT,
    encode_image,
    parse_transcription,
)
from facet_runtime.errors import (
    BackendMismatchError,
    BackendUnavailableError,
    FacetRuntimeError,
)

FASTFLOW_MODEL = "qwen3:0.6b"
FASTFLOW_VISION_MODEL = "qwen3.5:4b"


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
    """Run text and image Qwen models through short-lived NPU servers."""

    backend = "npu"

    def _validation(self) -> dict[str, Any]:
        return _command_json("flm", "validate", "--json")

    def _model_installed(self, model_name: str) -> bool:
        models = _command_json("flm", "list", "--filter", "installed", "--json").get(
            "models", []
        )
        return any(
            model.get("model") == model_name and model.get("installed")
            for model in models
        )

    def is_available(self) -> bool:
        try:
            return bool(self._validation().get("ready")) and self._model_installed(
                FASTFLOW_MODEL
            )
        except FacetRuntimeError:
            return False

    def _prepare(self, model_name: str) -> tuple[str, str]:
        validation = self._validation()
        if not validation.get("ready"):
            raise BackendUnavailableError(
                "FastFlowLM reports that the NPU stack is not ready"
            )
        if not self._model_installed(model_name):
            raise BackendUnavailableError(
                f"FastFlowLM model {model_name} is not installed"
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
        return executable, f"AMD XDNA2 NPU ({Path(accel_device)})"

    def _serve_request(
        self,
        executable: str,
        model_name: str,
        endpoint: str,
        payload: dict[str, Any],
        *,
        image: bool = False,
    ) -> tuple[dict[str, Any], str]:
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        with tempfile.TemporaryFile() as log_file:
            command = [
                executable,
                "serve",
                model_name,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--pmode",
                "performance",
            ]
            if image:
                command.extend(("--img-pre-resize", "0"))
            process = subprocess.Popen(
                command,
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
                    f"{base_url}{endpoint}",
                    payload,
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
        return response, log_text

    @staticmethod
    def _verify_npu(log_text: str, *, image: bool = False) -> None:
        if "NPU Locked!" not in log_text or "NPU Lock Released!" not in log_text:
            raise BackendMismatchError("FastFlowLM did not confirm NPU execution")
        if image and "Total images: 1" not in log_text:
            raise BackendMismatchError("FastFlowLM did not confirm image ingestion")

    def run(self, prompt: str) -> AdapterOutput:
        executable, device = self._prepare(FASTFLOW_MODEL)
        response, log_text = self._serve_request(
            executable,
            FASTFLOW_MODEL,
            "/api/generate",
            {
                "model": FASTFLOW_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 256},
            },
        )
        self._verify_npu(log_text)
        text = response.get("response")
        if not isinstance(text, str):
            raise FacetRuntimeError("FastFlowLM returned no response text")
        version = _command_json("flm", "version", "--json").get("version", "unknown")
        return AdapterOutput(
            text=text,
            runtime=f"FastFlowLM {version}",
            model=FASTFLOW_MODEL,
            device=device,
        )

    def inspect_image(self, image_path: str) -> ImageAdapterOutput:
        executable, device = self._prepare(FASTFLOW_VISION_MODEL)
        encoded_image, media_type = encode_image(image_path)
        response, log_text = self._serve_request(
            executable,
            FASTFLOW_VISION_MODEL,
            "/v1/chat/completions",
            {
                "model": FASTFLOW_VISION_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": TRANSCRIPTION_JSON_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{encoded_image}"
                                },
                            },
                        ],
                    }
                ],
                "stream": False,
                "temperature": 0,
                "max_tokens": 256,
            },
            image=True,
        )
        self._verify_npu(log_text, image=True)
        try:
            raw = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise FacetRuntimeError(
                "FastFlowLM returned no image transcription"
            ) from error
        if not isinstance(raw, str):
            raise FacetRuntimeError("FastFlowLM returned no image transcription")
        transcription, uncertainties = parse_transcription(raw)
        version = _command_json("flm", "version", "--json").get("version", "unknown")
        return ImageAdapterOutput(
            backend="npu",
            transcription=transcription,
            uncertainties=uncertainties,
            runtime=f"FastFlowLM {version}",
            model=FASTFLOW_VISION_MODEL,
            device=device,
            runtime_metadata=ImageRuntimeMetadata(
                protocol="openai_chat_completions_image_url",
                response_format="prompted_json_normalized",
                strict_json_schema=False,
            ),
            accelerator_verified=True,
        )
