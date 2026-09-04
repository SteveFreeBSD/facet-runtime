from __future__ import annotations

from facet_runtime.adapters import fastflow


def test_fastflow_image_uses_openai_content_array(monkeypatch) -> None:
    captured: dict = {}
    adapter = fastflow.FastFlowAdapter()
    monkeypatch.setattr(
        adapter, "_prepare", lambda model: ("/usr/bin/flm", "AMD XDNA2 NPU")
    )
    monkeypatch.setattr(
        fastflow, "encode_image", lambda path: ("encoded-pixels", "image/png")
    )
    monkeypatch.setattr(
        fastflow,
        "_command_json",
        lambda *command: {"version": "test"},
    )

    def serve_request(executable, model, endpoint, payload, *, image=False):
        captured.update(
            executable=executable,
            model=model,
            endpoint=endpoint,
            payload=payload,
            image=image,
        )
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": '```json\n{"transcription":"pixels read","uncertainties":[]}\n```'
                        }
                    }
                ]
            },
            "NPU Locked!\nTotal images: 1\nNPU Lock Released!",
        )

    monkeypatch.setattr(adapter, "_serve_request", serve_request)
    output = adapter.inspect_image("fixture.png")
    content = captured["payload"]["messages"][0]["content"]
    assert captured["model"] == "qwen3.5:4b"
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
