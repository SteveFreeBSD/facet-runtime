# Facet documentation

Facet is small enough that [the README](../README.md) is the authority for
everything current: the wire protocol, solver routing, the answer contract, the
model assignment and its evidence, measuring, checking the prompts, and
deployment.

| Question | Where |
|---|---|
| What crosses the wire, and what is refused | [README — Remote protocol](../README.md#remote-protocol) |
| How a question is answered, and by which route | [README — Solver routing](../README.md#solver-routing) |
| Which model runs on which device, and why | [README — Model assignment](../README.md#model-assignment), and `src/facet_runtime/models.py` |
| What to reinstall after a change | [README — Deploying a change](../README.md#deploying-a-change) |
| Whether Facet's own prompts still answer | [README — Checking the prompts](../README.md#checking-the-prompts), and `facet prompts` |
| What the consumer can enter, once Facet has answered | [`facet-hawkes/docs/ANSWER_CAPABILITIES.md`](../../facet-hawkes/docs/ANSWER_CAPABILITIES.md) |

Read that last one before adding an exact solver family. Facet owns *what the
answer is*; the consumer owns *how it is entered*, and a family that Facet
answers correctly but nothing can enter is not finished.

## Historical

[`history/`](history/PROMPT_REBASE_PREP.md) holds evidence that is no longer
instruction. Each document says so in its own first lines, and none of it
describes the current state.

| Document | What it records |
|---|---|
| [Prompt rebaseline evidence pack](history/PROMPT_REBASE_PREP.md) | The September 2026 prompt inventory, the empty-completion investigation, and the measurements behind the reasoning-effort and output-budget settings |
