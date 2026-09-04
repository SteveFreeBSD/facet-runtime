# Facet

Facet is a clean local-AI development project for exploring explicit workload
execution and routing across the AMD Ryzen AI 9 HX 370's three compute paths:

- native CPU execution;
- Radeon 890M GPU execution through Vulkan/RADV and Ollama; and
- XDNA2 NPU execution through `amdxdna`, XRT, and FastFlowLM.

This first pass deliberately contains no agent architecture. It establishes a
small Python package, reproducible `uv` environment, hardware discovery, and
known-good runtime checks that later work can build on.

## Quick start

```bash
uv sync
uv run facet-discover
uv run facet-discover --json
```

The discovery command reports a backend as ready only when its device and the
corresponding runtime path are both visible. It does not download models or run
inference.

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

## Project layout

```text
src/facet_runtime/       Python package and discovery command
tests/                   Lightweight foundation tests
tooling/fastflowlm/      Optional, isolated upstream runtime builds
```

## Scope boundary

Facet currently detects and proves the local compute foundation. Backend
abstractions, model lifecycle management, routing policy, tools, memory, and
agent behavior belong to later milestones.

