# Facet

**Facet answers questions.** A consumer hands it a question -- an instruction in
words, the exact expressions or measurements the question is about, and how many
separate values the answer takes -- and Facet decides how it gets answered,
answers it, and reports which route it took. Deterministic exact mathematics
runs first and settles most of it with no model, no accelerator and no network;
only what genuinely falls past that reaches a reasoning model.

It runs that work explicitly across the AMD Ryzen AI 9 HX 370's three compute
paths, and proves where each run actually happened:

- native CPU execution;
- Radeon 890M GPU execution through Vulkan/RADV and Ollama; and
- XDNA2 NPU execution through `amdxdna`, XRT, and FastFlowLM.

Its one consumer today is **Facet Hawkes Assistant**, in the sibling
`facet-hawkes` repository, which owns a browser and a page and hands Facet
nothing about either. That boundary is deliberate and is described from the
other side in [`facet-hawkes/docs/FACET_BRIDGE.md`](../facet-hawkes/docs/FACET_BRIDGE.md).
Facet never learns which document, window, frame, field or editor a question
came from, or that there is a browser at all.

Start with [Remote protocol](#remote-protocol) for the wire, [Solver
routing](#solver-routing) for how a question is answered, and [Model
assignment](#model-assignment) for what runs where.

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

`facet-remote` is the only way anything reaches Facet. It is normally a **local
subprocess** -- the consumer runs the installed helper and writes to its
standard input -- and the same helper serves a genuinely remote consumer over
`ssh` unchanged, because the protocol never depended on a network. "Remote"
here means *out of process*, not *off this machine*: the helper is the runtime,
process and protocol boundary, and a subprocess draws that boundary as well as
a connection does.

It reads one JSON request on standard input and writes one JSON response on
standard output. A
consumer may name an operation from a closed set and supply the text to
execute; it cannot pass a shell command, a path, a URL, an environment, a
runtime, a model, or a device. Facet chooses where the work runs and reports
what it actually did.

```bash
echo '{"facet_protocol_version": 2, "operation": "generate_text",
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
| `facet_protocol_version` | yes      | Exactly `2`.                             |
| `operation`              | yes      | `generate_text` or `solve_math`.         |
| `request_id`             | yes      | 1-64 of `A-Z a-z 0-9 . _ : -`; echoed back. |
| `prompt`                 | `generate_text` | Non-empty, at most 12 KiB.        |
| `problem`                | `solve_math` | The question; see below.             |
| `constraints`            | no       | `accelerator_required`, `allow_fallback`; booleans. |

Each operation takes its own payload and no other: a `solve_math` request
carrying a `prompt` is not a request Facet half-understands, it is a request
for something else, and is refused.

A request is validated strictly: an unknown field, an unknown constraint, a
wrong type, or more than 16 KiB is refused before anything executes. A success
carries `status: "ok"` and the whole `RunResult` -- including `metrics` and
`evidence` -- under `result`. A failure carries `status: "error"` and an
`error` object whose `kind` is one of `invalid_request`,
`unsupported_version`, `unsupported_operation`, `constraint_unsatisfied`,
`execution_failed`, `unusable_result`, or `internal_error`. The helper exits 0
for a success and 1 for a structured failure, and never writes an answer
alongside an error.

Adding a field to a response is a compatible change; consumers are expected to
ignore fields they do not know. Adding or changing a request field, an
operation, or a constraint is a protocol version change.

*Device* routing today is the fixed preference described above. When Facet
later gains a real device router, it takes over `_backend_for` in `remote.py`;
the wire contract does not move, because a consumer already asks for a
constraint rather than a device.

## Solver routing

`solve_math` is the operation for a consumer that has a *question* rather than
a prompt. It hands over the instruction in words, the exact expressions the
question is about, and how many separate values the answer takes — and Facet
decides how it gets answered.

```bash
echo '{"facet_protocol_version": 2, "operation": "solve_math",
       "request_id": "demo-2",
       "problem": {"instruction": "Simplify. Express your answer using rational exponents.",
                   "expressions": ["y^{3/4} \\cdot y^{2/5}"], "answer_parts": 1}}' | facet-remote
```

That decision is Facet's, and it is made in `src/facet_runtime/solve.py`. The
deterministic solvers in `src/facet_runtime/exact/` run first: they read the
question's own verb, run the exact SymPy operation that matches, and check the
answer against its own input. Anything they settle is settled in a millisecond,
with no model, no accelerator and no network, and the answer is checkable
rather than merely plausible. Only what genuinely falls past them reaches a
reasoning model, and the reason it fell past is carried out with the result.

Not every question is an expression to rewrite. Some state a *property* and ask
for the object that has it, and those are answered from the properties rather
than from a verb: a parabola's vertex, points on it away from its named
landmarks, and — in `src/facet_runtime/exact/linear.py` — a linear function
given as a slope and a point, a slope and an intercept, or two points. Two
facts determine a line and one does not, so that solver derives `y = mx + b`
over exact rationals, puts every stated property back into the result, and
declines rather than answering when one of them does not hold, when the
properties disagree, or when they leave the line undetermined.

The same exact route answers one normalized Cartesian point when the question
asks for the coordinates of a labeled point. The consumer proves the label-to-
point association and graph geometry before the point crosses; Facet returns
an `ordered-pair` with its x and y components preserved separately. More than
one candidate or any answer contract other than two components is a claimed
refusal and never reaches a model.

An inequality is answered the same way. `src/facet_runtime/exact/inequality.py`
solves one-variable linear inequalities -- simple, chained like `a < bx + c <= d`,
or joined by "and" -- over exact rationals, turning each comparison round when
it divides by a negative, and returns the solution set in interval notation as
a `scalar`: `(-8,7]`, `[5/2,∞)`, `(-∞,∞)` or `∅`. An absolute value of a linear
expression is split into the two comparisons it means, so `|ax+b| <= c` is an
interval and `|ax+b| > c` is written as the union of two rays, `(-∞,1)∪(4,∞)`.
It is claimed only when the question names an inequality, writes one, and asks
for interval notation or a graph of the solution set, and it is checked against
SymPy's own reading of the same conjunction. Several displayed inequalities are
intersected only when the step asks about them together: "solve the first
inequality" answers the first alone, and naming two different ones is declined. Non-linear inequalities, several
variables, more than one absolute value, a variable outside the bars, "or", a
single-point solution, and a decimal that cannot be written exactly are declined
by name.

```json
{"route": "exact",
 "answer": {"display": "y^(23/20)", "entry": "y^(23/20)", "parts": [],
            "entry_mode": "auto"},
 "provenance": {"source": "Facet Exact", "method": "SymPy exact symbolic",
                "router": "solved", "router_detail": "",
                "runtime": "SymPy 1.14.0", "model": null, "device": null,
                "requested_backend": null, "actual_backend": null,
                "elapsed_ms": 1.4, "fallback": false, "metrics": {},
                "evidence": {"source": "facet exact solver", "model_calls": 0}}}
```

A declined question comes back with `"route": "reasoning"`, a `router_detail`
naming the gap, and the ordinary model provenance — `source` reads
`Facet Reasoning · GPU`. Every field is present on both routes, and null on the
one it does not apply to: an exact solve names no model and no processor,
because none took part.

| `problem` field | Required | Meaning |
| --------------- | -------- | ------- |
| `instruction`   | yes      | The question in words, at most 4000 characters. |
| `expressions`   | a `value` about written mathematics; `parabola_plan` | 1 to 8 exact expressions, at most 2000 characters each. |
| `result_kind`   | no       | `value`, `parabola_plan`, or `quadratic_regression`. Default `value`. |
| `answer_parts`  | `value` only | 1 to 5 separate values the answer takes. Default 1. |
| `answer_table`  | `value` only | A question that is a grid: `columns`, and `rows` of `{"value"}` or `{"blank": n}` cells. |
| `answer_representation` | `value` only | The form every separate answer must take: `{"kind": "signed-integer", "max_length": n}`. |
| `graph`         | `parabola_plan` | Normalised geometry: `family`, `orientation`, `bounds`, `snap`, `controls`. |
| `points`        | `quadratic_regression`; a `value` about data or one identified Cartesian point | 3 to 32 exact coordinates for regression; 1 exact `{"x", "y"}` coordinate for a labeled-point value. |
| `label`         | no       | The question's own label, at most 200 characters. |

A `value` question is about `expressions` or about `points` -- mathematics
somebody wrote down, or measurements nobody wrote a function for. Both at once
is two questions and neither is none, so exactly one is required.

Each kind takes its own fields and no others. Geometry on a regression, or
points on a parabola, is a question about something else and is refused rather
than ignored. There is deliberately no way to describe *where* a question came
from: a consumer that owns a browser cannot hand over a document, an element, a
picture, or an action even by accident.

### A question that is a table

A completion question states a relation and a grid: some cells carry values,
the rest are blank, and the blanks are the answer. That is not a question about
an expression with some prose around it -- the grid *is* the question, and it
crosses as one. Each blank is numbered as the answer's parts are numbered, so
part N and the Nth blank are the same cell on both sides.

```bash
echo '{"facet_protocol_version": 2, "operation": "solve_math",
       "request_id": "demo-4",
       "problem": {"instruction": "Complete the table of values below for the given equation.",
                   "expressions": ["x=y^2"], "answer_parts": 5,
                   "answer_table": {"columns": ["x", "y"],
                     "rows": [[{"value": "0"}, {"blank": 1}],
                              [{"blank": 2}, {"value": "2\\sqrt{2}"}],
                              [{"value": "64"}, {"blank": 3}],
                              [{"value": "25"}, {"blank": 4}],
                              [{"blank": 5}, {"value": "-\\sqrt{3}"}]]},
                   "answer_representation": {"kind": "signed-integer",
                                             "max_length": 4}}}' | facet-remote
```

`src/facet_runtime/exact/table.py` binds each row's stated cells into the
relation, solves for the one unknown left, and proves every candidate by
substituting it back. A row with no blank is checked too: it states the
relation a second time, and a grid that contradicts its own relation has been
misread and is declined rather than half-answered.

Where a row admits more than one exact solution -- `x = 64` is satisfied by 8
and by -8 -- choosing is unavoidable, and the rule is stated once and applied
everywhere: **the smallest in magnitude, and the non-negative one where two
share a magnitude.** The choice is a tie-break and never a claim that the other
root is wrong; the verifier below accepts either.

`answer_representation` filters the solutions and is never satisfied by
adjusting one. A row whose only exact solutions cannot be written the way the
question requires is declined, because rounding an exact answer to fit an
answer box is how a wrong answer gets typed in confidently.

```json
{"route": "exact",
 "answer": {"kind": "value", "display": "0, 8, 8, 5, 3", "entry": "",
            "parts": ["0", "8", "8", "5", "3"], "entry_mode": "math"},
 "provenance": {"source": "Facet Exact",
                "method": "SymPy exact table completion",
                "router": "solved", "model": null, "actual_backend": null,
                "evidence": {"source": "facet exact solver", "model_calls": 0,
                             "computation": {"relation": "x - y**2 = 0",
                                             "row 3": "x=64 -> y in {8, -8} -> 8",
                                             "...": "..."}}}}
```

#### Checking an answer that was reasoned

The reasoning route is stochastic, and the *shape* of its reply is not evidence
about the mathematics in it: five parts arriving is not five parts being right,
and a live run returned five that were neither. When a question carries a grid,
every reasoned answer to it is put back into the grid before it becomes an
answer -- each part into its own row, beside that row's stated values, and the
relation has to hold exactly. A part that fails is refused as
`unusable_result`, and the model is not asked again: retrying until a reply
passes would make a verifier into a filter on repeated guessing.

A grid whose relation cannot be read at all is a different claim from an answer
being wrong, and is not treated as one. Nothing is checked, the reasoned answer
stands on its own terms, and `provenance.evidence.answer_table` says which of
the three happened: `verified`, `not checkable: <reason>`, or `not-applicable`.

### A question about data

Some questions carry no expression at all: the function exists only as the fit
to the measurements. Those are still values, and the deterministic stage still
owns them.

```bash
echo '{"facet_protocol_version": 2, "operation": "solve_math",
       "request_id": "demo-3",
       "problem": {"instruction": "Treating revenue as a function of the number of photos sold, if she uses quadratic regression to fit a curve to the data, what number of photos sold and what price per photo will maximize her revenue?",
                   "points": [{"x": "4", "y": "224"}, {"x": "5", "y": "260"},
                              {"x": "12", "y": "288"}],
                   "answer_parts": 2}}' | facet-remote
```

`src/facet_runtime/exact/regression.py` fits the least-squares quadratic over
exact rationals -- three points give the interpolating parabola, more give the
true least-squares fit, and neither is ever a decimal approximation -- then
reads it where the instruction says to. A second answer is produced only when
the question asks for a rate *per the same quantity the curve is a function
of*; "per event" when the curve is a function of photos is a different
question, and is declined rather than guessed at.

```json
{"route": "exact",
 "answer": {"kind": "value", "display": "9, 36", "entry": "",
            "parts": ["9", "36"], "entry_mode": "math"},
 "provenance": {"source": "Facet Exact",
                "method": "SymPy exact least-squares regression",
                "router": "solved", "model": null, "actual_backend": null,
                "elapsed_ms": 9.5, "...": "...",
                "evidence": {"source": "facet exact solver", "model_calls": 0,
                             "computation": {"fit": "y = -4x^2 + 72x",
                                             "coefficients": "-4,72,0",
                                             "direction": "maximum",
                                             "optimum_input": "9",
                                             "optimum_value": "324",
                                             "rate_unit": "photo",
                                             "rate": "324/9 = 36"}}}}
```

The working comes back as evidence, beside the proof that no model ran. It is
there so a consumer can redo the whole computation from the same points and
compare, rather than take the answer on trust -- which is what the Hawkes
consumer does before either number reaches an answer box.

A curve that turns the wrong way is declined, not reported: a parabola opening
upwards has no maximum, and its vertex is the answer to the opposite question.

### Plans

Two families ask for geometry rather than a value: a vertical parabola somebody
will draw, and the coefficients of a quadratic regression over points somebody
measured. Neither has a deterministic route -- fitting a curve to draw is a
different request from reading one that has been fitted -- so both go straight
to their specialist in
`src/facet_runtime/graph.py`, which owns the prompt and the reply parsing.

```json
{"route": "reasoning",
 "answer": {"kind": "parabola_plan",
            "plan": {"kind": "parabola", "orientation": "vertical",
                     "opening": "up", "vertex": {"x": "3", "y": "-1"},
                     "points": [{"x": "4", "y": "0"}, {"x": "2", "y": "0"}]}},
 "provenance": {"source": "Facet Parabola Plan · GPU", "method": "gpt-oss:20b",
                "router": "not-run",
                "router_detail": "a graph plan has no deterministic route",
                "runtime": "Ollama 0.33.2", "actual_backend": "gpu", "...": "..."}}
```

A plan carries no `display`, `entry` or `parts`. It is a *proposal*: Facet
parses it strictly -- exact schema, no extra or duplicate keys, exact integer or
rational coordinates and never a decimal approximation -- and that is a check on
the model rather than a warrant. Facet does not prove the geometry, because the
authority that matters is whoever owns the surface the plan will be drawn on.
Two independent readings of an untrusted reply is the point, and the consumer's
is the one that decides.

`router: "not-run"` says the deterministic stage was never asked, which is a
different claim from having tried and declined. `source` names the specialist
that ran, because which one answered is not something a reader can infer from a
model name.

An answer stays structured. `entry` carries the single value a one-value
question takes and `parts` the separate values when it takes more than one;
exactly one of them is populated. `entry_mode` says how literally to take a
value — `verbatim`, `math`, or `auto` — because writing a value into whatever
input a consumer owns is the consumer's business and not Facet's.

`accelerator_required` is a statement about where a *model* runs. An exact
solve engages no backend at all, so it satisfies the constraint by never
needing one and reports `actual_backend` as null rather than claiming a
processor it did not use. Nothing is substituted quietly: `route` names what
happened on every result.

A reply that carries no final answer, or the wrong number of separate answers,
is refused with `unusable_result` rather than returned. An answer of the wrong
shape is worse than no answer, because a consumer would put it somewhere.

### Deploying a change

`facet-remote` on the host is a `uv tool` install, **not** the working tree.
Nothing changed here -- a solver, a prompt, a model assignment, the protocol --
reaches a consumer until it is reinstalled:

```bash
uv tool install --force --reinstall .
```

A helper left at an older protocol version refuses every request from the newer
client with `unsupported_version` rather than answering part of it, which is the
right failure but is easy to mistake for a transport problem. A helper left at
an older *solver* fails less loudly: it simply answers the way it used to.

Check what is actually installed with one request:

```bash
echo '{"facet_protocol_version": 2, "operation": "solve_math",
       "request_id": "probe-1",
       "problem": {"instruction": "Simplify.", "expressions": ["(x+1)*(x-1)"],
                   "answer_parts": 1}}' | facet-remote
```

The consumer pins this repository by commit. `facet-hawkes` imports
`facet_runtime` as a path dependency at `../facet-runtime` and records the
required commit in `facet-hawkes/deploy/facet-runtime.pin`; its
`tests/test_runtime_pin.py` fails when that sibling checkout is behind. Raise
the pin there in the same change that starts depending on something new here,
and push this repository before that pin is published.

## Model assignment

Every model Facet runs is declared once in `src/facet_runtime/models.py`, with
the runtime that serves it, its footprint, and the reason it belongs on that
device. `facet models` prints that table. Each entry can be replaced for an
experiment through its environment variable — `FACET_GPU_TEXT_MODEL` and
friends — which changes the model but never the device.

| Backend | Role   | Runtime    | Model         | Why |
| ------- | ------ | ---------- | ------------- | --- |
| CPU     | text   | Ollama     | `qwen3.5:2b`  | The installed 2.3B Q8_0 artifact occupies 2.55 GiB. CPU prefill degrades with model size far faster than decode does, so this is the smallest genuinely capable measured worker: 288 prefill and 29 decode tokens per second, against 118 and 16 for a 4B. |
| GPU     | text   | Ollama     | `gpt-oss:20b` | A mixture-of-experts model reads only its active experts per token, so this is both the largest and the fastest thing the 14.8 GiB aperture can hold: 504 prefill and 21.2 decode, against 319 and 14.4 for a dense 9B. Fully device-resident at a 16k context. Reasons unconditionally, so it is assigned low reasoning effort and a 2048-token budget. |
| NPU     | text   | FastFlowLM | `gpt-oss:20b` | The measured preferred NPU text worker: 18.7 decode against 9.3 for a dense 9B, and within 12% of the 890M on the identical model. One hard reasoning request returned an empty completion/Facet error; that failure has since been reproduced and measured on the GPU path with the same model. |
| GPU     | vision | Ollama     | `qwen3.5:9b`  | Both halves of the image pair run the same model and size class, so a difference between passes means a device difference. At 9B both devices recover a heading and the exact glyphs that 4B dropped. |
| NPU     | vision | FastFlowLM | `qwen3.5:9b`  | Matches the GPU half exactly. The two passes run one after the other, so only one vision model is resident at a time. |

`gpt-oss:20b` has no vision, which is why the image pipeline keeps `qwen3.5:9b`.

It reasons unconditionally on both runtimes and counts those tokens against
`max_output_tokens`, ahead of the answer. That is no longer a suspicion: a
quadratic regression that this repository documents answering correctly was
reproduced returning nothing, with `done_reason: length`, 1024 of 1024 tokens
spent and 4942 characters of internal reasoning behind them. Ollama ignores a
request not to reason from this model, so the assignment states an effort
instead — the same question finishes in 889 tokens including its answer at low
effort — and the budget is 2048 so a harder one has somewhere to go. Every run
now reports `stop_reason` and `output_token_limit` alongside its token counts,
and an empty completion says whether the budget ran out or the model returned
nothing, because those need opposite fixes.

FastFlowLM exposes no reasoning-effort control, so the NPU path has the budget
and that report and nothing else. Set `FACET_NPU_TEXT_MODEL=qwen3.5:9b` for an
NPU text model that is 7.7 GiB instead of 14 GiB, loads in about half the time,
and does not reason unconditionally, at roughly half the decode rate.

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

## Checking the prompts

`facet bench` measures throughput on prompts of its own, so it cannot tell you
whether Facet's *own* prompts still answer. `facet prompts` does exactly that.
Each case is a real problem answered by the real `solve_math`: the same
routing, the same prompt constructors, the same strict parsers, the same
adapters. Nothing in it writes a prompt, so a rendered prompt is byte for byte
what production would send.

```bash
facet prompts                          # render every prompt, no model, no network
facet prompts --case regression        # one of them, for reading or diffing
facet prompts --live                   # answer them all on a real model
facet prompts --live --backend gpu --show-prompt
```

Five cases cover every route: `exact`, which must reach no model at all;
`reasoning`, a question the deterministic solvers decline; `prefixed_multi`, a
question that names the variable it solves for *and* takes two answers, which
is the pairing the value prompt has to state without contradicting itself; and
`parabola` and `regression`, the two graph specialists. A case pins an expected
answer only where the answer is determined — the regression's three points lie
exactly on one parabola, and a quadratic has the roots it has — and reports
what it got where it is not. `--live` leaves a non-zero status if any case stops answering, so it
can be a gate rather than only a report. It never repairs a reply and never
relaxes a parser: a case that fails is reported failing, with the runtime's own
reason.

Alongside it, `tests/fixtures/prompts/` holds a golden file per prompt shape —
the exact text, byte for byte, rendered by the production constructors. Every
substring assertion in the suite passes no matter how the rest of a sentence is
written, which is how nine consumer-specific lines survived in the prompts as
long as they did; a golden turns any rewording into a diff a reviewer reads.
When a change to the wording is intended:

```bash
FACET_UPDATE_GOLDEN=1 uv run --frozen pytest -q tests/test_prompt_snapshots.py
git diff tests/fixtures/prompts/   # read this, then commit it
```

Twelve files cover all eight shapes the value prompt takes — its `Question:`
heading, its `x =` prefix line, one expression and several, both closing
contracts, the largest `answer_parts`, a named variable together with several
answers, and one case with every optional piece present at once — and four
plan prompts. The plan prompts get two apiece because their wording does not
branch but the payload they carry does: alongside each live case there is a
grid whose bounds and snap are not all halves and whole tens, and a point set
carrying the exact rationals the protocol accepts but no live case sends. Both
are held to be requests that could really arrive, by putting them back through
`parse_problem`.

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

Facet owns solver routing, exact deterministic mathematics, the reasoning
route, the two graph specialists, model assignment, and the proof that a run
happened where it says it did. It does not own, and will not accept, anything
about where a question came from: a document, an element, a picture, an action,
a device name, a model name, or a runtime. A consumer states a *need*; Facet
chooses.

Device routing today is the fixed GPU, NPU, CPU preference described above
rather than a workload router. When a real device router arrives it takes over
`_backend_for` in `remote.py` and the wire contract does not move, because a
consumer already asks for a constraint rather than a device. Tools, memory and
agent behaviour remain out of scope.
