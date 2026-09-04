"""Discover the CPU, Radeon GPU, and XDNA2 NPU runtime paths."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _run(*command: str, timeout: float = 15.0) -> tuple[int, str]:
    """Run a discovery command without raising when an optional tool is absent."""
    executable = shutil.which(command[0])
    if executable is None:
        return 127, ""
    try:
        result = subprocess.run(
            (executable, *command[1:]),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""
    return result.returncode, (result.stdout + result.stderr).strip()


def _match(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else None


def _cpu_model() -> str:
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return platform.processor() or "unknown"
    return (
        _match(cpuinfo, r"^model name\s*:\s*(.+)$") or platform.processor() or "unknown"
    )


def discover_cpu() -> dict[str, Any]:
    return {
        "available": True,
        "backend": "native CPU",
        "model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
    }


def _ollama_status() -> dict[str, Any]:
    status: dict[str, Any] = {"installed": shutil.which("ollama") is not None}
    code, output = _run("ollama", "--version")
    status["version"] = (
        _match(output, r"(?:version is|version)\s+([0-9][^\s]*)") if code == 0 else None
    )
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:11434/api/version", timeout=1.0
        ) as response:
            status["service"] = "running"
            status["service_version"] = json.load(response).get("version")
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        status["service"] = "not reachable"
        status["service_version"] = None
    return status


def discover_gpu() -> dict[str, Any]:
    code, output = _run("vulkaninfo", "--summary")
    render_nodes = sorted(str(path) for path in Path("/dev/dri").glob("renderD*"))
    name = _match(output, r"^\s*deviceName\s*=\s*(.+)$")
    driver = _match(output, r"^\s*driverName\s*=\s*(.+)$")
    return {
        "available": code == 0 and bool(render_nodes) and bool(name),
        "backend": "Vulkan",
        "device": name,
        "driver": driver,
        "driver_info": _match(output, r"^\s*driverInfo\s*=\s*(.+)$"),
        "api_version": _match(output, r"^\s*apiVersion\s*=\s*(.+)$"),
        "render_nodes": render_nodes,
        "ollama": _ollama_status(),
    }


def _fastflow_status() -> dict[str, Any]:
    code, output = _run("flm", "version", "--json")
    version = None
    if code == 0:
        try:
            version = json.loads(output).get("version")
        except json.JSONDecodeError:
            version = _match(output, r"v([0-9][^\s]*)")

    validate_code, validate_output = _run("flm", "validate", "--json")
    validation: dict[str, Any] | None = None
    if validate_code == 0:
        try:
            validation = json.loads(validate_output)
        except json.JSONDecodeError:
            pass
    return {
        "installed": shutil.which("flm") is not None,
        "version": version,
        "validation": validation,
    }


def discover_npu() -> dict[str, Any]:
    code, output = _run("xrt-smi", "examine")
    accel_nodes = sorted(str(path) for path in Path("/dev/accel").glob("accel*"))
    fastflow = _fastflow_status()
    validation = fastflow.get("validation") or {}
    return {
        "available": code == 0 and bool(accel_nodes) and bool(validation.get("ready")),
        "backend": "XDNA2 via XRT/FastFlowLM",
        "device": _match(output, r"^\|\[[^]]+\]\s*\|\s*([^|]+)\|"),
        "driver": "amdxdna" if Path("/sys/module/amdxdna").exists() else None,
        "xrt_version": _match(output, r"^\s*Version\s*:\s*(.+)$"),
        "firmware_version": _match(output, r"^\s*NPU Firmware Version\s*:\s*(.+)$"),
        "accel_nodes": accel_nodes,
        "fastflowlm": fastflow,
    }


def discover() -> dict[str, Any]:
    return {
        "host": platform.node(),
        "kernel": platform.release(),
        "backends": {
            "cpu": discover_cpu(),
            "gpu": discover_gpu(),
            "npu": discover_npu(),
        },
    }


def _print_human(report: dict[str, Any]) -> None:
    print(f"Facet runtime discovery on {report['host']} ({report['kernel']})")
    for label, details in report["backends"].items():
        state = "READY" if details["available"] else "NOT READY"
        device = details.get("model") or details.get("device") or details["backend"]
        print(f"{label.upper():>3}  {state:<9} {device}")
        if label == "gpu":
            print(
                f"     driver={details.get('driver_info') or details.get('driver')} ollama={details['ollama']['service']}"
            )
        elif label == "npu":
            print(
                "     "
                f"driver={details.get('driver')} xrt={details.get('xrt_version')} "
                f"firmware={details.get('firmware_version')} flm={details['fastflowlm'].get('version')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    args = parser.parse_args()
    report = discover()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human(report)


if __name__ == "__main__":
    main()
