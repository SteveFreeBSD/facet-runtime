# Facet

Facet is a clean local-AI development project for exploring explicit workload
execution and routing across the AMD Ryzen AI 9 HX 370's three compute paths:

- native CPU execution;
- Radeon 890M GPU execution through Vulkan/RADV and Ollama; and
- XDNA2 NPU execution through `amdxdna`, XRT, and FastFlowLM.

This first pass deliberately contains no agent architecture. It establishes a
small Python package, reproducible `uv` environment, hardware discovery, an
explicit model assignment per device, and known-good runtime checks that later
work can build on.

## Quick start

```bash
uv sync
facet run "Reply with one short sentence." --backend cpu
facet run "Reply with one short sentence." --backend gpu
facet run "Reply with one short sentence." --backend npu
facet run "Reply with one short sentence." --backend auto
facet models
facet bench --repeat 2
facet inspect-image tests/fixtures/inspect_image_fixture.png
uv run facet-discover
uv run facet-discover --json
```

`facet run` always emits one JSON result with `text`, `requested_backend`,
`actual_backend`, `runtime`, `model`, `device`, `elapsed_ms`, `fallback`,
`metrics`, and `evidence`. `metrics` carries the token counts and the prefill
and decode rates the runtime itself reported. `evidence` carries the proof that
the work ran where Facet says it ran: Ollama's loaded and device-resident byte
counts for CPU and GPU, and FastFlowLM's NPU lock plus accelerator node for the
NPU.

`auto` selects the first available backend in a fixed GPU, NPU, CPU order and
then commits to it. That order reflects measured decode throughput on this
machine, not a workload router: later routing will be capability- and
workload-based. `auto` does not retry another backend if the selected one
fails. Explicit backend requests either run on that exact backend or fail; they
never fall back silently, and a runtime that returns an empty completion is a
failed run rather than an answer.

## Remote protocol

`facet-remote` is the only way another machine reaches Facet. It reads one JSON
request on standard input and writes one JSON response on standard output. A
consumer may name an operation from a closed set and supply the text to
execute; it cannot pass a shell command, a path, a URL, an environment, a
runtime, a model, or a device. Facet chooses where the work runs and reports
what it actually did.

```bash
echo '{"facet_protocol_version": 1, "operation": "generate_text",
       "request_id": "demo-1", "prompt": "Reply with one short sentence.",
       "constraints": {"accelerator_required": true}}' | facet-remote
```

A consumer states a *need*, not a device. `accelerator_required` says the work
must not land on the CPU; Facet picks the accelerator. `allow_fallback`
defaults to false, and a result that fell back is a failure rather than an
answer. Both constraints are checked again against what actually happened, so
Facet refuses a result that broke the constraint it accepted.

| Request field            | Required | Meaning                                  |
| ------------------------ | -------- | ---------------------------------------- |
| `facet_protocol_version` | yes      | Exactly `1`.                             |
| `operation`              | yes      | One of `generate_text`.                  |
| `request_id`             | yes      | 1-64 of `A-Z a-z 0-9 . _ : -`; echoed back. |
| `prompt`                 | yes      | Non-empty, at most 12 KiB.               |
| `constraints`            | no       | `accelerator_required`, `allow_fallback`; booleans. |

A request is validated strictly: an unknown field, an unknown constraint, a
wrong type, or more than 16 KiB is refused before anything executes. A success
carries `status: "ok"` and the whole `RunResult` -- including `metrics` and
`evidence` -- under `result`. A failure carries `status: "error"` and an
`error` object whose `kind` is one of `invalid_request`,
`unsupported_version`, `unsupported_operation`, `constraint_unsatisfied`,
`execution_failed`, or `internal_error`. The helper exits 0 for a success and 1
for a structured failure, and never writes an answer alongside an error.

Adding a field to a response is a compatible change; consumers are expected to
ignore fields they do not know. Adding or changing a request field, an
operation, or a constraint is a protocol version change.

Routing today is the fixed device preference described above. When Facet later
gains a real router, it takes over `_backend_for` in `remote.py`; the wire
contract does not move, because a consumer already asks for a constraint rather
than a device.

## Model assignment

Every model Facet runs is declared once in `src/facet_runtime/models.py`, with
the runtime that serves it, its footprint, and the reason it belongs on that
device. `facet models` prints that table. Each entry can be replaced for an
experiment through its environment variable — `FACET_GPU_TEXT_MODEL` and
friends — which changes the model but never the device.

| Backend | Role   | Runtime    | Model         | Why |
| ------- | ------ | ---------- | ------------- | --- |
| CPU     | text   | Ollama     | `qwen3.5:2b`  | The installed 2.3B Q8_0 artifact occupies 2.55 GiB. CPU prefill degrades with model size far faster than decode does, so this is the smallest genuinely capable measured worker: 288 prefill and 29 decode tokens per second, against 118 and 16 for a 4B. |
| GPU     | text   | Ollama     | `gpt-oss:20b` | A mixture-of-experts model reads only its active experts per token, so this is both the largest and the fastest thing the 14.8 GiB aperture can hold: 504 prefill and 21.2 decode, against 319 and 14.4 for a dense 9B. Fully device-resident at a 16k context. |
| NPU     | text   | FastFlowLM | `gpt-oss:20b` | The measured preferred NPU text worker: 18.7 decode against 9.3 for a dense 9B, and within 12% of the 890M on the identical model. One hard reasoning request returned an empty completion/Facet error; its cause was not captured. |
| GPU     | vision | Ollama     | `qwen3.5:9b`  | Both halves of the image pair run the same model and size class, so a difference between passes means a device difference. At 9B both devices recover a heading and the exact glyphs that 4B dropped. |
| NPU     | vision | FastFlowLM | `qwen3.5:9b`  | Matches the GPU half exactly. The two passes run one after the other, so only one vision model is resident at a time. |

`gpt-oss:20b` has no vision, which is why the image pipeline keeps `qwen3.5:9b`.
FastFlowLM's build of `gpt-oss:20b` reasons unconditionally and counts those
tokens against `max_output_tokens`, which creates an output-budget risk. One
hard reasoning request returned an empty completion and Facet correctly raised
an error, but the underlying runtime cause was not captured; the budget behavior
is not claimed as the cause. Set `FACET_NPU_TEXT_MODEL=qwen3.5:9b` for an NPU
text model that is 7.7 GiB instead of 14 GiB, loads in about half the time, and
does not reason unconditionally, at roughly half the decode rate.

The GPU path requires every loaded byte to sit in device memory. A partial
offload is a silent CPU fallback, so Facet fails instead of reporting it as GPU
execution. The CPU path requires zero device memory. The NPU path requires
FastFlowLM to log both an NPU lock and its release.

## Measuring

`facet bench` drives the same adapters `facet run` uses, so every number comes
from an execution that already passed those device checks. It reports a short
latency case and a roughly 2.3k-token prefill case per backend, with medians
across repeats and any backend that failed to prove its device.

```bash
facet bench --repeat 3
facet bench --backend gpu,npu --case context
```

## Foundation checks

```bash
# GPU driver and Ollama service
vulkaninfo --summary
curl http://127.0.0.1:11434/api/version
ollama ps

# NPU driver, XRT, and FastFlowLM
xrt-smi examine
flm validate --json
flm list --filter installed
```

Ollama is installed with CachyOS's Vulkan runner, and its system service is
configured to allow the integrated Radeon GPU. The package-managed FastFlowLM
remains the default `flm` command. An isolated upstream build may be kept under
`tooling/fastflowlm/<version>/`; downloaded runtime binaries and model data are
ignored by git.

Note that FastFlowLM ignores `num_predict` on its Ollama-compatible
`/api/generate` endpoint but honours `max_tokens` on `/v1/chat/completions`, so
Facet's NPU text path uses the latter. Without that the stated output cap would
not be the cap the NPU applies.

## Project layout

```text
src/facet_runtime/       Python package, model assignment, discovery, benchmark
tests/                   Lightweight foundation tests
tooling/fastflowlm/      Optional, isolated upstream runtime builds
```

## Scope boundary

Facet currently detects and proves the local compute foundation. Backend
abstractions, model lifecycle management, routing policy, tools, memory, and
agent behavior belong to later milestones.
