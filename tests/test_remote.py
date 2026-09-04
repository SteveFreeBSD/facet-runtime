"""The Facet remote protocol: what a remote consumer may ask, and what it gets."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, replace

import pytest

from facet_runtime.adapters.base import AdapterOutput, ExecutionMetrics
from facet_runtime.errors import BackendUnavailableError, FacetRuntimeError
from facet_runtime.remote import (
    PROTOCOL_VERSION,
    SUPPORTED_OPERATIONS,
    RemoteProtocolError,
    handle,
    parse_request,
    run_remote,
)
from facet_runtime.runtime import run_prompt

EVIDENCE = {
    "source": "ollama /api/ps",
    "loaded_bytes": 13780173824,
    "device_memory_bytes": 13780173824,
    "device_resident_fraction": 1.0,
}


@dataclass
class FakeAdapter:
    backend: str
    available: bool = True
    error: Exception | None = None

    def is_available(self) -> bool:
        return self.available

    def run(self, prompt: str) -> AdapterOutput:
        if self.error:
            raise self.error
        return AdapterOutput(
            text=f"answer to {prompt}",
            runtime="fake-runtime",
            model="fake-model",
            device=f"fake {self.backend} device",
            metrics=ExecutionMetrics(
                prompt_tokens=11, generated_tokens=5, prefill_tps=504.0, decode_tps=21.2
            ),
            evidence=dict(EVIDENCE),
        )


def adapters(**overrides) -> dict[str, FakeAdapter]:
    made = {name: FakeAdapter(name) for name in ("cpu", "gpu", "npu")}
    made.update(overrides)
    return made


def request_bytes(**changes) -> bytes:
    payload = {
        "facet_protocol_version": PROTOCOL_VERSION,
        "operation": "generate_text",
        "request_id": "consumer-1",
        "prompt": "Reply with one short sentence.",
    }
    payload.update(changes)
    for name, value in list(payload.items()):
        if value is None:
            del payload[name]
    return json.dumps(payload).encode("utf-8")


def test_a_valid_exchange_returns_the_whole_runtime_result() -> None:
    envelope, code = handle(request_bytes(), adapters=adapters())

    assert code == 0
    assert envelope["facet_protocol_version"] == PROTOCOL_VERSION
    assert envelope["status"] == "ok"
    assert envelope["request_id"] == "consumer-1"
    assert envelope["operation"] == "generate_text"
    assert envelope["result"].keys() == {
        "text",
        "requested_backend",
        "actual_backend",
        "runtime",
        "model",
        "device",
        "elapsed_ms",
        "fallback",
        "metrics",
        "evidence",
    }


def test_metrics_and_evidence_reach_the_consumer_unaltered() -> None:
    envelope, _code = handle(request_bytes(), adapters=adapters())

    result = envelope["result"]
    assert result["metrics"] == {
        "prompt_tokens": 11,
        "generated_tokens": 5,
        "prefill_tps": 504.0,
        "decode_tps": 21.2,
    }
    assert result["evidence"] == EVIDENCE
    # The envelope must survive the wire, not just the function call.
    assert json.loads(json.dumps(envelope))["result"]["evidence"] == EVIDENCE


@pytest.mark.parametrize("version", [0, 2, "1", 1.0, True, None])
def test_a_request_of_another_protocol_version_is_refused(version) -> None:
    envelope, code = handle(
        request_bytes(facet_protocol_version=version), adapters=adapters()
    )

    assert code == 1
    assert envelope["status"] == "error"
    assert envelope["error"]["kind"] in {"unsupported_version", "invalid_request"}
    assert "result" not in envelope


@pytest.mark.parametrize(
    "raw",
    [b"", b"not json", b"[]", b'"text"', b"\xff\xfe", b"{" + b"x" * 20_000 + b"}"],
)
def test_a_malformed_request_is_refused_as_invalid(raw: bytes) -> None:
    envelope, code = handle(raw, adapters=adapters())

    assert code == 1
    assert envelope["error"]["kind"] == "invalid_request"


@pytest.mark.parametrize(
    "operation", ["", "run", "generate_text ", "inspect_image", "eval"]
)
def test_an_operation_outside_the_closed_set_is_refused(operation: str) -> None:
    envelope, code = handle(request_bytes(operation=operation), adapters=adapters())

    assert code == 1
    assert envelope["error"]["kind"] == "unsupported_operation"
    assert "generate_text" in envelope["error"]["message"]


def test_the_closed_set_is_one_operation() -> None:
    assert SUPPORTED_OPERATIONS == ("generate_text",)


@pytest.mark.parametrize(
    "field",
    ["backend", "model", "device", "command", "cwd", "env", "url", "path", "host"],
)
def test_execution_configuration_cannot_be_named_in_a_request(field: str) -> None:
    envelope, code = handle(
        request_bytes(**{field: "caller-chosen"}), adapters=adapters()
    )

    assert code == 1
    assert envelope["error"]["kind"] == "invalid_request"
    assert f"unknown {field}" in envelope["error"]["message"]


@pytest.mark.parametrize("field", ["operation", "request_id", "prompt"])
def test_every_required_field_is_required(field: str) -> None:
    envelope, code = handle(request_bytes(**{field: None}), adapters=adapters())

    assert code == 1
    assert envelope["error"]["kind"] == "invalid_request"
    assert f"missing {field}" in envelope["error"]["message"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operation", 1),
        ("request_id", 7),
        ("request_id", ""),
        ("request_id", "x" * 65),
        ("request_id", "spaces are not ids"),
        ("prompt", ""),
        ("prompt", "   "),
        ("prompt", 42),
        ("prompt", ["a"]),
        ("prompt", "x" * 12_500),
    ],
)
def test_every_field_is_type_and_bound_checked(field: str, value) -> None:
    envelope, code = handle(request_bytes(**{field: value}), adapters=adapters())

    assert code == 1
    assert envelope["status"] == "error"
    assert "result" not in envelope


def test_a_request_id_is_echoed_back_even_when_the_request_fails() -> None:
    envelope, _code = handle(
        request_bytes(request_id="hawkes-42", operation="nope"), adapters=adapters()
    )

    assert envelope["request_id"] == "hawkes-42"


def test_an_unparseable_request_gets_an_empty_request_id() -> None:
    envelope, _code = handle(b"{", adapters=adapters())

    assert envelope["request_id"] == ""


@pytest.mark.parametrize(
    "constraints",
    [
        {"gpu": True},
        {"accelerator_required": "yes"},
        {"allow_fallback": 1},
        {"accelerator_required": True, "model": "qwen3.5:2b"},
        "accelerator_required",
    ],
)
def test_an_unknown_or_mistyped_constraint_is_refused(constraints) -> None:
    envelope, code = handle(request_bytes(constraints=constraints), adapters=adapters())

    assert code == 1
    assert envelope["error"]["kind"] == "invalid_request"


def test_an_unconstrained_request_takes_facets_ordinary_preference() -> None:
    envelope, _code = handle(request_bytes(), adapters=adapters())

    assert envelope["result"]["requested_backend"] == "auto"
    assert envelope["result"]["actual_backend"] == "gpu"


def test_an_accelerator_constraint_names_a_need_not_a_device() -> None:
    envelope, code = handle(
        request_bytes(constraints={"accelerator_required": True}),
        adapters=adapters(gpu=FakeAdapter("gpu", available=False)),
    )

    assert code == 0
    # The consumer never said "npu". Facet chose it because the GPU was gone.
    assert envelope["result"]["actual_backend"] == "npu"


def test_an_accelerator_request_never_silently_lands_on_the_cpu() -> None:
    envelope, code = handle(
        request_bytes(constraints={"accelerator_required": True}),
        adapters=adapters(
            gpu=FakeAdapter("gpu", available=False),
            npu=FakeAdapter("npu", available=False),
        ),
    )

    assert code == 1
    assert envelope["error"]["kind"] == "constraint_unsatisfied"
    assert "accelerator" in envelope["error"]["message"]


def test_a_result_that_broke_its_constraint_is_refused_not_returned() -> None:
    def run_on_the_wrong_device(prompt, backend, *, adapters):
        return run_prompt(prompt, "cpu", adapters=adapters)

    envelope, code = handle(
        request_bytes(constraints={"accelerator_required": True}),
        adapters=adapters(),
        run=run_on_the_wrong_device,
    )

    assert code == 1
    assert envelope["error"]["kind"] == "constraint_unsatisfied"
    assert "not an accelerator" in envelope["error"]["message"]


def test_a_fallback_is_a_failure_unless_the_consumer_allowed_one() -> None:
    def run_with_a_fallback(prompt, backend, *, adapters):
        return replace(run_prompt(prompt, backend, adapters=adapters), fallback=True)

    refused, code = handle(
        request_bytes(), adapters=adapters(), run=run_with_a_fallback
    )
    assert code == 1
    assert refused["error"]["kind"] == "constraint_unsatisfied"
    assert "fell back" in refused["error"]["message"]

    allowed, code = handle(
        request_bytes(constraints={"allow_fallback": True}),
        adapters=adapters(),
        run=run_with_a_fallback,
    )
    assert code == 0
    assert allowed["result"]["fallback"] is True


@pytest.mark.parametrize(
    "error",
    [
        FacetRuntimeError("Ollama returned no response text"),
        BackendUnavailableError("Radeon 890M RADV device is unavailable"),
    ],
)
def test_an_execution_failure_is_structured_not_an_empty_answer(error) -> None:
    envelope, code = handle(
        request_bytes(), adapters=adapters(gpu=FakeAdapter("gpu", error=error))
    )

    assert code == 1
    assert envelope["error"]["kind"] == "execution_failed"
    assert str(error) in envelope["error"]["message"]
    assert "result" not in envelope


def test_an_unexpected_failure_still_answers_with_an_envelope() -> None:
    def explode(prompt, backend, *, adapters):
        raise KeyError("something nobody predicted")

    envelope, code = handle(request_bytes(), adapters=adapters(), run=explode)

    assert code == 1
    assert envelope["error"]["kind"] == "internal_error"
    assert "KeyError" in envelope["error"]["message"]


def test_an_error_message_is_bounded() -> None:
    envelope, _code = handle(
        request_bytes(),
        adapters=adapters(gpu=FakeAdapter("gpu", error=FacetRuntimeError("x" * 5_000))),
    )

    assert len(envelope["error"]["message"]) <= 400


def test_the_prompt_is_never_echoed_into_a_failure() -> None:
    secret = "Coursework the consumer would rather not see in a log."
    envelope, _code = handle(
        request_bytes(prompt=secret, operation="nope"), adapters=adapters()
    )

    assert secret not in json.dumps(envelope)


def test_parse_request_refuses_before_anything_executes() -> None:
    with pytest.raises(RemoteProtocolError) as refusal:
        parse_request(request_bytes(operation="run"))

    assert refusal.value.kind == "unsupported_operation"


def test_the_helper_writes_one_json_line_and_an_exit_status(monkeypatch) -> None:
    monkeypatch.setattr("facet_runtime.remote.default_adapters", lambda: adapters())
    stdout = io.StringIO()

    code = run_remote(io.BytesIO(request_bytes()), stdout)

    written = stdout.getvalue()
    assert code == 0
    assert written.endswith("\n")
    assert json.loads(written)["status"] == "ok"


def test_the_helper_answers_a_refused_request_on_stdout_too(monkeypatch) -> None:
    monkeypatch.setattr("facet_runtime.remote.default_adapters", lambda: adapters())
    stdout = io.StringIO()

    code = run_remote(io.BytesIO(b"garbage"), stdout)

    assert code == 1
    assert json.loads(stdout.getvalue())["error"]["kind"] == "invalid_request"


def test_an_oversized_request_is_refused_without_being_read_whole() -> None:
    # The helper reads one byte past the limit and stops; a consumer cannot make
    # it buffer an unbounded request.
    envelope, code = handle(request_bytes(prompt="x" * 40_000))

    assert code == 1
    assert envelope["error"]["kind"] == "invalid_request"
    assert "size limit" in envelope["error"]["message"]
