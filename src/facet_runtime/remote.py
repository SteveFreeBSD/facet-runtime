"""Facet remote protocol v1: one bounded intelligence request over stdin.

A remote consumer -- Ethnos today, others later -- writes one JSON request to
this helper's standard input and reads one JSON response from its standard
output. That is the entire remote surface. A consumer may name only an
operation from a closed set and supply the text to execute; it cannot pass a
shell command, a path, a URL, an environment, a runtime, a model, or a device.

The division of labour is deliberate. A consumer states what it *needs* as a
constraint -- "this must run on an accelerator", "do not fall back" -- and
Facet alone decides which runtime and which processor satisfies it, then
reports what it actually did along with the metrics and the evidence for that
claim. Nothing here knows what the caller is for, and nothing here knows which
machine it is running on: a constraint is the only thing that crosses the
boundary in the request direction, and provenance is the only thing that
crosses it coming back.

A request either executes under its constraints or returns a structured
failure. There is no partial success and no quiet substitution.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Any, BinaryIO, Literal, TextIO

from facet_runtime.adapters.base import BackendAdapter
from facet_runtime.errors import FacetRuntimeError
from facet_runtime.result import BackendName
from facet_runtime.runtime import AUTO_PREFERENCE, default_adapters, run_prompt

PROTOCOL_VERSION = 1

#: The closed set of operations a remote consumer may name. Growing this set is
#: a protocol change; nothing outside it is reachable from a request.
SUPPORTED_OPERATIONS: tuple[str, ...] = ("generate_text",)

#: Backends that count as an accelerator for `accelerator_required`. This is
#: Facet's own hardware knowledge, not the consumer's: a consumer asks for an
#: accelerator, never for a named device.
ACCELERATORS: frozenset[str] = frozenset({"gpu", "npu"})

MAX_REQUEST_BYTES = 16 * 1024
MAX_PROMPT_BYTES = 12 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_MESSAGE_CHARS = 400

REQUEST_FIELDS = ("facet_protocol_version", "operation", "request_id", "prompt")
OPTIONAL_REQUEST_FIELDS = ("constraints",)
CONSTRAINT_FIELDS = ("accelerator_required", "allow_fallback")

# A request id is echoed back so a consumer can correlate a reply, including a
# reply to a request that failed to parse. Bounding its shape keeps an
# untrusted string from becoming an unbounded one in someone else's log.
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,64}\Z")

ErrorKind = Literal[
    "invalid_request",
    "unsupported_version",
    "unsupported_operation",
    "constraint_unsatisfied",
    "execution_failed",
    "internal_error",
]


class RemoteProtocolError(Exception):
    """A refusal that carries the kind of refusal it is."""

    def __init__(self, kind: ErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind: ErrorKind = kind


@dataclass(frozen=True, slots=True)
class RemoteRequest:
    """One validated remote request. Every field here survived validation."""

    operation: str
    request_id: str
    prompt: str
    accelerator_required: bool = False
    allow_fallback: bool = False


def _reject(kind: ErrorKind, message: str) -> RemoteProtocolError:
    return RemoteProtocolError(kind, message)


def _decode(raw: bytes) -> Any:
    if len(raw) > MAX_REQUEST_BYTES:
        raise _reject("invalid_request", "request exceeds the protocol size limit")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _reject(
            "invalid_request", f"request was not valid JSON: {error}"
        ) from error


def _checked_version(payload: dict[str, Any]) -> None:
    version = payload.get("facet_protocol_version")
    # `True == 1` in Python, so the type is checked before the value.
    if not isinstance(version, int) or isinstance(version, bool):
        raise _reject("invalid_request", "facet_protocol_version must be an integer")
    if version != PROTOCOL_VERSION:
        raise _reject(
            "unsupported_version",
            f"this helper speaks Facet remote protocol {PROTOCOL_VERSION}, "
            f"not {version}",
        )


def _checked_fields(payload: dict[str, Any]) -> None:
    missing = [name for name in REQUEST_FIELDS if name not in payload]
    unknown = sorted(set(payload) - {*REQUEST_FIELDS, *OPTIONAL_REQUEST_FIELDS})
    if not missing and not unknown:
        return
    detail = ", ".join(
        part
        for part in (
            f"missing {', '.join(missing)}" if missing else "",
            f"unknown {', '.join(unknown)}" if unknown else "",
        )
        if part
    )
    raise _reject("invalid_request", f"request fields are wrong: {detail}")


def _checked_operation(payload: dict[str, Any]) -> str:
    operation = payload["operation"]
    if not isinstance(operation, str) or operation not in SUPPORTED_OPERATIONS:
        raise _reject(
            "unsupported_operation",
            f"supported operations: {', '.join(SUPPORTED_OPERATIONS)}",
        )
    return operation


def _checked_request_id(payload: dict[str, Any]) -> str:
    request_id = payload["request_id"]
    if not isinstance(request_id, str) or not REQUEST_ID_PATTERN.match(request_id):
        raise _reject(
            "invalid_request",
            "request_id must be 1 to 64 characters of A-Z a-z 0-9 . _ : -",
        )
    return request_id


def _checked_prompt(payload: dict[str, Any]) -> str:
    prompt = payload["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise _reject("invalid_request", "prompt must be a non-empty string")
    if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise _reject("invalid_request", "prompt exceeds the protocol size limit")
    return prompt


def _checked_constraints(payload: dict[str, Any]) -> dict[str, bool]:
    constraints = payload.get("constraints", {})
    if not isinstance(constraints, dict):
        raise _reject("invalid_request", "constraints must be an object")
    unknown = sorted(set(constraints) - set(CONSTRAINT_FIELDS))
    if unknown:
        # An unrecognised constraint is a constraint Facet is not honouring.
        # Ignoring it would answer a question that was never asked.
        raise _reject("invalid_request", f"unknown constraint: {', '.join(unknown)}")
    for name in CONSTRAINT_FIELDS:
        if name in constraints and not isinstance(constraints[name], bool):
            raise _reject("invalid_request", f"constraint {name} must be true or false")
    return {name: bool(constraints.get(name, False)) for name in CONSTRAINT_FIELDS}


def parse_request(raw: bytes) -> RemoteRequest:
    """Validate one request completely, or refuse it with a reason."""
    payload = _decode(raw)
    if not isinstance(payload, dict):
        raise _reject("invalid_request", "request was not a JSON object")
    # Version first: a future protocol must be told the version is wrong rather
    # than that its new fields are unknown.
    _checked_version(payload)
    _checked_fields(payload)
    constraints = _checked_constraints(payload)
    return RemoteRequest(
        operation=_checked_operation(payload),
        request_id=_checked_request_id(payload),
        prompt=_checked_prompt(payload),
        **constraints,
    )


def _backend_for(
    request: RemoteRequest, adapters: dict[str, BackendAdapter]
) -> BackendName:
    """Turn the consumer's constraint into Facet's own backend decision.

    This is the seam a real router will one day own. Today it is the smallest
    honest thing: an unconstrained request takes Facet's ordinary preference,
    and a request that needs an accelerator takes the first available one in
    that same order.
    """
    if not request.accelerator_required:
        return "auto"
    for name in AUTO_PREFERENCE:
        adapter = adapters.get(name)
        if name in ACCELERATORS and adapter is not None and adapter.is_available():
            return name  # type: ignore[return-value]
    raise _reject("constraint_unsatisfied", "no Facet accelerator is available")


def execute(
    request: RemoteRequest,
    *,
    adapters: dict[str, BackendAdapter] | None = None,
    run=run_prompt,
) -> dict[str, Any]:
    """Run one validated request and return the full runtime result."""
    adapter_map = dict(adapters or default_adapters())
    backend = _backend_for(request, adapter_map)
    try:
        result = run(request.prompt, backend, adapters=adapter_map)
    except FacetRuntimeError as error:
        raise _reject("execution_failed", str(error)) from error
    except ValueError as error:
        raise _reject("invalid_request", str(error)) from error
    # The constraint is checked again against what actually happened. Facet
    # states where the work ran, so it must also be the thing that refuses when
    # that is not where the caller required it to run.
    if request.accelerator_required and result.actual_backend not in ACCELERATORS:
        raise _reject(
            "constraint_unsatisfied",
            f"execution ran on {result.actual_backend}, which is not an accelerator",
        )
    if result.fallback and not request.allow_fallback:
        raise _reject("constraint_unsatisfied", "execution fell back to another path")
    return result.to_dict()


def success_envelope(
    request_id: str, operation: str, result: dict[str, Any]
) -> dict[str, Any]:
    return {
        "facet_protocol_version": PROTOCOL_VERSION,
        "status": "ok",
        "request_id": request_id,
        "operation": operation,
        "result": result,
    }


def error_envelope(request_id: str, kind: ErrorKind, message: str) -> dict[str, Any]:
    return {
        "facet_protocol_version": PROTOCOL_VERSION,
        "status": "error",
        "request_id": request_id,
        "error": {"kind": kind, "message": message[:MAX_MESSAGE_CHARS]},
    }


def _peek_request_id(raw: bytes) -> str:
    """Recover a well-formed request id from a request that did not validate."""
    if len(raw) > MAX_REQUEST_BYTES:
        return ""
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ""
    candidate = payload.get("request_id") if isinstance(payload, dict) else None
    if isinstance(candidate, str) and REQUEST_ID_PATTERN.match(candidate):
        return candidate
    return ""


def handle(
    raw: bytes,
    *,
    adapters: dict[str, BackendAdapter] | None = None,
    run=run_prompt,
) -> tuple[dict[str, Any], int]:
    """Answer one request with an envelope and the exit status to leave with."""
    try:
        request = parse_request(raw)
        result = execute(request, adapters=adapters, run=run)
    except RemoteProtocolError as error:
        return error_envelope(_peek_request_id(raw), error.kind, str(error)), 1
    except Exception as error:  # noqa: BLE001 - a helper must always answer
        return (
            error_envelope(
                _peek_request_id(raw),
                "internal_error",
                f"{type(error).__name__}: {error}",
            ),
            1,
        )
    return success_envelope(request.request_id, request.operation, result), 0


def run_remote(stdin: BinaryIO, stdout: TextIO) -> int:
    envelope, code = handle(stdin.read(MAX_REQUEST_BYTES + 1))
    body = json.dumps(envelope, ensure_ascii=False)
    if len(body.encode("utf-8")) > MAX_RESPONSE_BYTES:
        envelope = error_envelope(
            envelope["request_id"],
            "execution_failed",
            "response exceeds the protocol size limit",
        )
        body, code = json.dumps(envelope, ensure_ascii=False), 1
    stdout.write(f"{body}\n")
    stdout.flush()
    return code


def main() -> None:
    raise SystemExit(run_remote(sys.stdin.buffer, sys.stdout))


if __name__ == "__main__":
    main()
