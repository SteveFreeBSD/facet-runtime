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

## Model assignment

Every model Facet runs is declared once in `src/facet_runtime/models.py`, with
the runtime that serves it, its footprint, and the reason it belongs on that
device. `facet models` prints that table. Each entry can be replaced for an
experiment through its environment variable — `FACET_GPU_TEXT_MODEL` and
friends — which changes the model but never the device.

| Backend | Role   | Runtime    | Model         | Why |
| ------- | ------ | ---------- | ------------- | --- |
| CPU     | text   | Ollama     | `qwen3.5:2b`  | CPU prefill degrades with model size far faster than decode does, so the CPU takes the smallest genuinely capable instruct model: 288 prefill and 29 decode tokens per second, against 118 and 16 for a 4B. |
| GPU     | text   | Ollama     | `gpt-oss:20b` | A mixture-of-experts model reads only its active experts per token, so this is both the largest and the fastest thing the 14.8 GiB aperture can hold: 504 prefill and 21.2 decode, against 319 and 14.4 for a dense 9B. Fully device-resident at a 16k context. |
| NPU     | text   | FastFlowLM | `gpt-oss:20b` | The same architecture suits XDNA2 for the same reason: 18.7 decode against 9.3 for a dense 9B, and within 12% of the 890M on the identical model. |
| GPU     | vision | Ollama     | `qwen3.5:9b`  | Both halves of the image pair run the same model and size class, so a difference between passes means a device difference. At 9B both devices recover a heading and the exact glyphs that 4B dropped. |
| NPU     | vision | FastFlowLM | `qwen3.5:9b`  | Matches the GPU half exactly. The two passes run one after the other, so only one vision model is resident at a time. |

`gpt-oss:20b` has no vision, which is why the image pipeline keeps `qwen3.5:9b`.
FastFlowLM's build of `gpt-oss:20b` reasons unconditionally and counts those
tokens against `max_output_tokens`, so a hard multi-step prompt can spend the
whole budget and return nothing; Facet raises that as a failed run rather than
reporting an empty answer. Set `FACET_NPU_TEXT_MODEL=qwen3.5:9b` for an NPU
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
