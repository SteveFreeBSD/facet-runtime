from __future__ import annotations

import pytest

from facet_runtime import models
from facet_runtime.adapters import fastflow
from facet_runtime.errors import BackendMismatchError, FacetRuntimeError

NPU_LOG = "NPU Locked!\nTotal images: 1\nNPU Lock Released!"


def _stub(monkeypatch: pytest.MonkeyPatch, adapter, response, log=NPU_LOG) -> dict:
    captured: dict = {}
    monkeypatch.setattr(
        adapter, "_prepare", lambda model: ("/usr/bin/flm", "AMD XDNA2 NPU")
    )
    monkeypatch.setattr(fastflow, "_command_json", lambda *command: {"version": "test"})

    def serve_request(
        executable, model, endpoint, payload, *, image=False, context_tokens=None
    ):
        captured.update(
            executable=executable,
            model=model,
            endpoint=endpoint,
            payload=payload,
            image=image,
            context_tokens=context_tokens,
        )
        return response, log, 1.5

    monkeypatch.setattr(adapter, "_serve_request", serve_request)
    return captured


def _completion(content: str) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 34,
            "prefill_speed_tps": 250.5,
            "decoding_speed_tps": 18.25,
        },
    }


def test_text_run_caps_output_through_the_endpoint_that_honours_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = fastflow.FastFlowAdapter()
    captured = _stub(monkeypatch, adapter, _completion("npu answer"))
    assignment = models.assignment("npu", "text")

    output = adapter.run("hello")

    # FastFlowLM silently ignores num_predict on /api/generate, so the text path
    # must use the OpenAI endpoint for the stated cap to be real.
    assert captured["endpoint"] == "/v1/chat/completions"
    assert captured["payload"]["max_tokens"] == assignment.max_output_tokens
    assert captured["context_tokens"] == assignment.context_tokens
    assert captured["model"] == assignment.resolved_model()
    assert output.text == "npu answer"
    assert output.metrics.prefill_tps == 250.5
    assert output.metrics.decode_tps == 18.25
    assert output.metrics.generated_tokens == 34
    assert output.evidence["npu_locked"] is True
    assert output.evidence["server_ready_s"] == 1.5


def test_text_run_refuses_a_server_that_never_locked_the_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = fastflow.FastFlowAdapter()
    _stub(monkeypatch, adapter, _completion("npu answer"), log="starting up")
    with pytest.raises(BackendMismatchError, match="did not confirm NPU execution"):
        adapter.run("hello")


def test_an_empty_completion_is_a_failure_not_an_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A reasoning model can spend its whole token budget on internal analysis
    # and return nothing; that is a failed run, not an answer.
    adapter = fastflow.FastFlowAdapter()
    _stub(monkeypatch, adapter, _completion(""))
    with pytest.raises(FacetRuntimeError, match="no response text"):
        adapter.run("hello")


def test_image_run_uses_openai_content_array(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = fastflow.FastFlowAdapter()
    captured = _stub(
        monkeypatch,
        adapter,
        _completion('```json\n{"transcription":"pixels read","uncertainties":[]}\n```'),
    )
    monkeypatch.setattr(
        fastflow, "encode_image", lambda path: ("encoded-pixels", "image/png")
    )

    output = adapter.inspect_image("fixture.png")

    content = captured["payload"]["messages"][0]["content"]
    assert captured["model"] == models.model_for("npu", "vision")
    assert captured["endpoint"] == "/v1/chat/completions"
    assert captured["image"] is True
    assert content[0]["type"] == "text"
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,encoded-pixels"},
    }
    assert output.transcription == "pixels read"
    assert output.runtime_metadata.response_format == "prompted_json_normalized"
    assert output.runtime_metadata.strict_json_schema is False


def test_image_run_refuses_a_pass_without_confirmed_image_ingestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = fastflow.FastFlowAdapter()
    _stub(
        monkeypatch,
        adapter,
        _completion('{"transcription":"x","uncertainties":[]}'),
        log="NPU Locked!\nNPU Lock Released!",
    )
    monkeypatch.setattr(
        fastflow, "encode_image", lambda path: ("encoded-pixels", "image/png")
    )
    with pytest.raises(BackendMismatchError, match="image ingestion"):
        adapter.inspect_image("fixture.png")
