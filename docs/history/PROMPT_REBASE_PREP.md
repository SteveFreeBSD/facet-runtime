# Prompt rebaseline: evidence pack

> **Historical.** This was written before the work, and §1-§7 describe the state
> as it was then: the old prompt wording, the old model assignment, and a
> failure whose cause was still unknown. It is kept as the record of what was
> known beforehand, and as the measurement behind the reasoning-effort and
> output-budget settings that shipped. §8 records what the failure turned out to
> be.
>
> Nothing here is current instruction. Since it was written, §5.1 has also been
> closed -- `tests/fixtures/prompts/` now holds a byte-exact golden file per
> prompt shape, so a rewording is a diff a reviewer reads rather than a change
> every substring assertion passes. The *Still open* note below predates that.
> For the current state read [the README](../../README.md); the head shas, test
> counts and command sheet in §7 and §9 belong to September 2026 and are no
> longer the numbers to expect.

Preparation only. Nothing here changes production behaviour, and no active
prompt was edited to write it. Heads at the time of writing:

- facet-runtime `b385fdd` — Route the two graph families to their own specialists
- facet-hawkes `2c763c1` — Ask Facet for a graph plan instead of writing it a prompt

Prompt ownership has finished moving. `grep -rn generate_text` over
`facet-hawkes/src/ethnos` matches only `facet_client.py`, and
`tests/test_facet_graph_routing.py::test_no_hawkes_path_writes_a_model_prompt_any_more`
holds that line. Every model-facing prompt now lives in this repository.

## 1. Prompt inventory

Three constructors, all reached from `solve_math`.

### 1a. Value / reasoning — `src/facet_runtime/solve.py:205` `reasoning_prompt`

Serves `result_kind: "value"`, `route: "reasoning"` — reached only after
`solve_exact` declines. Assembled from four optional pieces: a `Question:`
heading when a label was sent, the instruction, the expressions, a `prefix`
line when `ANSWER_PREFIX` matches *solve for x*, and one of two closing
contracts depending on `answer_parts`.

Single-value form (the common one):

```text
Solve this Hawkes precalculus question.
Instruction: Find the domain of the following function.
Expression(s):
- \frac{x+1}{x^2-9}
Your entire response must be one line beginning with the exact words FINAL ANSWER: followed by only what belongs in the Hawkes answer box. Do not repeat the input expression or output an equals sign. Never output angle brackets or a trailing period. Do not explain.
```

Multi-value form, with a label and a `solve for r` prefix:

```text
Solve this Hawkes precalculus question.
Question: Question 4 of 12
Instruction: Solve the following formula for the indicated variable. Solve for r.
Expression(s):
- C=2*pi*r
The page already prints `r =` beside the answer, so give only the value that follows it.
This question takes 2 separate answers.
Reply with exactly 3 labelled lines and nothing else, and keep every label exactly as written here:
FINAL ANSWER: all answers as the page would display them
PART 1: answer number 1 by itself
PART 2: answer number 2 by itself
Every line must begin with its own label, including each PART line. A PART line holds only what belongs in that one answer box: no label repeated inside it, no variable name, no equals sign, no "or", no explanation.
```

**Parser:** `extract_final_math` (`src/facet_runtime/exact/answer.py:32`) takes
the last `FINAL ANSWER:` line, or the innermost `\boxed{}`; then
`labelled_parts` (`solve.py:262`) requires exactly `answer_parts` `PART n:`
lines, numbered from one, in order, none empty. A miss on either raises
`SolveRefused("unusable_result", …)`.

**Result schema:** `{"kind": "value", "display", "entry", "parts", "entry_mode"}`.

**Tests:** `tests/test_solve_math.py::test_the_reasoning_prompt_carries_the_question_and_no_page`,
`::test_separate_answers_stay_separate_values`,
`::test_a_reply_of_the_wrong_shape_fails_closed` (4 cases),
`::test_a_reasoning_reply_with_no_final_answer_fails_closed`;
facet-hawkes `tests/test_facet_hawkes_bridge.py::test_the_reasoning_prompt_is_facets_to_write`,
`::test_the_fixed_prompt_does_not_teach_placeholder_tags`,
`::test_the_question_label_reaches_facet_when_the_page_supplied_one`,
`::test_the_page_prefix_is_stated_so_facet_does_not_repeat_it`,
`::test_a_paired_answer_shape_reaches_facet_as_a_two_part_contract`,
`tests/test_facet_solver_routing.py::test_the_reasoning_prompt_never_offers_an_action`.

### 1b. Parabola plan — `src/facet_runtime/graph.py:157` `parabola_prompt`

Serves `result_kind: "parabola_plan"`, `route: "reasoning"`, `router: "not-run"`.

```text
Produce a graph plan for the exact function. Return ONLY one JSON object, no markdown, prose, code, or extra keys. Coordinates must be exact integer or rational STRINGS (for example "-3/2"). Required schema: {"kind":"parabola","orientation":"vertical","opening":"up or down","vertex":{"x":"rational","y":"rational"},"points":[{"x":"rational","y":"rational"},{"x":"rational","y":"rational"}]}. Derive the vertex, opening and two symmetric defining points from the function. First point must be right of vertex, second left. Prefer one unit horizontal offset if it fits the bounds and snap grid. You have no browser actions.
{"instruction": "Graph the parabola.", "exact_expression": ["f(x)=(x-3)^2-1"], "graph_answer": {"family": "parabola", "orientation": "vertical", "bounds": [-10.0, 10.0, -10.0, 10.0], "snap": [0.5, 0.5], "controls": "vertex-and-symmetric-points"}}
```

**Parser:** `parse_parabola_plan` (`graph.py:222`) — `strict_json` (duplicate
keys refused), then exactly `{kind, orientation, opening, vertex, points}`,
`kind == "parabola"`, `orientation == "vertical"`, `opening ∈ {up, down}`,
exactly two points, and every coordinate matching
`RATIONAL` (`graph.py:38`, `-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?`) — a decimal is refused.

**Result schema:** `{"kind": "parabola_plan", "plan": {…}}` and nothing else.

**Tests:** `tests/test_graph_specialists.py::test_a_parabola_question_reaches_the_parabola_specialist`,
`::test_a_reply_that_is_not_a_parabola_plan_fails_closed` (12 cases),
`::test_a_plan_survives_the_wire_exactly`, `::test_the_parsers_accept_what_they_are_meant_to`,
`::test_a_repeated_key_is_refused_rather_than_resolved`;
facet-hawkes `tests/test_hawkes_graph.py::test_graph_invokes_facet_and_reports_real_provenance`,
`tests/test_facet_graph_routing.py` (whole module, 37 tests).

### 1c. Quadratic regression — `src/facet_runtime/graph.py:182` `regression_prompt`

Serves `result_kind: "quadratic_regression"`, `route: "reasoning"`,
`router: "not-run"`. **This is the one that currently fails live.**

```text
Find the quadratic least-squares regression y=a*x^2+b*x+c for these exact SVG point coordinates. Return ONLY JSON with exactly this schema: {"kind":"quadratic-regression","coefficients":["a","b","c"]}. Coefficients must be exact integer or rational strings, NOT decimal approximations. Ethnos will independently verify and round them for display. No prose, markdown, browser commands, or extra keys.
{"instruction": "Use quadratic regression. Round to three decimal places.", "points": [{"x": "-5", "y": "5"}, {"x": "-2", "y": "-4"}, {"x": "-1", "y": "5"}]}
```

**Parser:** `parse_regression_plan` (`graph.py:253`) — `strict_json`, then
exactly `{kind, coefficients}`, `kind == "quadratic-regression"`, exactly three
coefficients, each matching `RATIONAL`.

**Result schema:** `{"kind": "quadratic_regression", "plan": {…}}`.

**Tests:** `tests/test_graph_specialists.py::test_a_regression_question_reaches_the_regression_specialist`,
`::test_a_reply_that_is_not_a_regression_fails_closed` (6 cases),
`::test_an_exact_rational_coefficient_is_still_a_coefficient`;
facet-hawkes `tests/test_facet_graph_routing.py::test_a_regression_request_goes_through_solve_math`,
`::test_exact_regression_coefficients_survive_serialisation_exactly`,
`::test_a_schema_valid_regression_that_is_wrong_is_refused_here`,
`tests/test_hawkes_graph.py::test_regression_coefficients_are_checked_exactly`.

## 2. Consumer-specific wording

**Model-facing — reaches a model on every call.** Nine lines, all inside
returned prompt strings:

| Location | Text | Trigger |
| --- | --- | --- |
| `solve.py:253` | `Solve this Hawkes precalculus question.` | every value prompt |
| `solve.py:248` | `…only what belongs in the Hawkes answer box.` | single-value only |
| `solve.py:224` | ``The page already prints `r =` beside the answer…`` | when *solve for x* matches |
| `solve.py:235` | `FINAL ANSWER: all answers as the page would display them` | multi-value only |
| `solve.py:241` | `…what belongs in that one answer box:` | multi-value only |
| `graph.py:171` | `You have no browser actions.` | every parabola prompt |
| `graph.py:186` | `…for these exact SVG point coordinates.` | every regression prompt |
| `graph.py:189` | `Ethnos will independently verify and round them for display.` | every regression prompt |
| `graph.py:190` | `No prose, markdown, browser commands, or extra keys.` | every regression prompt |

**Not model-facing — docstrings and comments only, harmless.**
`remote.py:3` ("Ethnos today, others later"); `graph.py:17,56`;
`solve.py:70,84-86,102,115,200,214-215,221,269,297`;
`exact/router.py:15,197`; `exact/symbolic.py:4,300,351,688,716,746,789,962,1315,1361,1431`
and `702,1339,1341,1360` (editor notes). `exact/symbolic.py` carries the bulk of
it: the solver's rules were derived from observed Hawkes marking behaviour and
the comments say so, which is provenance for the mathematics rather than
anything a model reads.

**No occurrence in any prompt of:** a DOM id, a selector, a screenshot, a URL,
a tab, a frame, or an actuation verb. That part of the boundary is already
clean and is gated (§5).

## 3. Regression failure evidence

The case: points `(-5,5)`, `(-2,-4)`, `(-1,5)`; expected
`{"kind":"quadratic-regression","coefficients":["3","18","20"]}`, displayed as
`3x² + 18x + 20`. All three points lie exactly on that parabola.

**Recorded in the repositories:**

- `facet-hawkes/docs/HAWKES_GRAPH.md:95-118` — this exact case, with the point
  set, the required reply, the exact-fit proof (`Xᵀ(Xc−y)=0`, rows `[x²,x,1]`,
  rank three) and the three-decimal rounding rule. **It records the case
  succeeding live**: "The direct Facet request took 37410.096 ms on the same
  model/runtime/GPU/device above", then a full add-on solve at 06:14:17 UTC
  (39306 ms) and insertion at 06:14:29 UTC (10019 ms).
- `facet-hawkes/tests/test_hawkes_graph.py:252-272` and
  `tests/test_facet_graph_routing.py:51-52` — the same points and coefficients
  pinned as fixtures.
- `src/facet_runtime/adapters/ollama.py:188` — where the failure surfaces:
  `raise FacetRuntimeError("Ollama returned no response text")`.
- `src/facet_runtime/models.py:117-129` and `README.md:229,234-236` — one prior
  empty completion, **on the NPU / FastFlowLM path, not this one**, cause not
  captured, with the unconditional-reasoning output-budget risk named as a
  suspicion rather than a cause.

**Not recorded in either repository.** Neither repo tracks any `.log` or
`.jsonl`; `git ls-files` finds none. Nothing in-repo documents an empty
completion on the **GPU / Ollama** path.

**Session-observed, 2026-09-05, not repo-recorded — treat as a lead, re-measure
before relying on it.** Against the deployed `facet-remote` on casbox, this
case returned `execution_failed: Ollama returned no response text` three
consecutive times, ~54 s wall each. The **byte-identical prompt sent through
the old `generate_text` path failed the same way**, which is why the ownership
move is not the cause. The parabola prompt on the same model/runtime in the
same session succeeded in 31688.511 ms (`gpt-oss:20b`, Radeon 890M, 300 prompt
tokens, 555 generated, 20.9 decode tps, `device_resident_fraction: 1.0`).

So the honest shape of the problem: **a documented live success has become a
reproducible empty completion, with no in-repo measurement of the failure.**
Reproducing it is step one (§7), and capturing it somewhere durable is worth
doing before changing any wording.

## 4. Benchmark map

| Path | Cases | Invocation | Success criterion |
| --- | --- | --- | --- |
| `facet-hawkes/benchmarks/precalculus_model_benchmark.json` | 18 multiple-choice items, `schema_version: precalculus-model-benchmark-v1`; categories: functions 4, complex-numbers 4, algebra 2, exponential-logarithmic 2, trigonometry 2, polynomials 2, analytic-geometry 1, sequences 1 | `ethnos precalc-bench --model …` (`src/ethnos/cli/commands/bench.py:123`) | Ollama structured output validated against `PrecalculusModelAnswer` (`work`, then `selected_option`, then `final_answer`); scores `accuracy = correct / total`, counting an invalid parse as incorrect (`src/ethnos/precalc_benchmark.py:95-145`) |
| `facet-hawkes/benchmarks/hawkes_lesson_coverage.json` | 37 cases, `schema_version: hawkes-coverage-v1` | `ethnos hawkes-coverage [--strict]` (`bench.py:62`) | Which questions the **exact** solver answers; currently `36/36 answered exactly, 1 correctly declined (0 wrong, 0 unrecognized)`. Pinned by `tests/test_hawkes_coverage.py` — no wrong answer allowed, ≥20 sweep results |
| `facet-runtime/src/facet_runtime/benchmark.py` | 2 fixed cases (short prompt; ~2.3k-token prefill) | `facet bench [--backend …] [--repeat N]` | Decode/prefill throughput per backend with device proof. **Throughput only — no correctness, and it does not use any solve prompt** |
| `facet-hawkes/benchmarks/ethics_*`, `history_*` | quiz corpora | `ethnos bench` | Unrelated to precalculus or graphs |

**The gap this table exposes:** `precalc-bench` builds **its own** prompt,
`build_precalculus_prompt` (`precalc_benchmark.py:54`), and calls Ollama
directly. It never touches `reasoning_prompt`, `solve_math`, or Facet. It is a
model benchmark, **not** a rebaseline harness for the prompts in §1, and it
cannot measure a wording change to any of them. No benchmark or fixture
anywhere exercises `regression_prompt` against a live model.

## 5. Prompt test coverage gaps

Mechanical, listed so they are not rediscovered. Not fixed in this pass.

1. **No prompt is snapshot-tested.** Every assertion is a substring
   (`"FINAL ANSWER:" in prompt`) or an equality against the constructor's own
   output — `tests/test_facet_hawkes_bridge.py:69` `prompt_for()` calls
   `reasoning_prompt` and compares the result to `reasoning_prompt`, so it
   compares a prompt to itself. **Any wording change passes every current
   test.** The byte-identity of the two graph prompts against their pre-move
   originals was verified once, by hand, against `git show HEAD:…` — that check
   exists nowhere in the suite.
2. **Consumer-specific wording is not prohibited anywhere.** Three forbidden
   lists apply to prompt text, and none contains *Hawkes*, *Ethnos*, *SVG*,
   *page*, or *browser*:
   `test_solve_math.py:138` → `field, editor, selector, click, tab, submit, #`;
   `test_facet_hawkes_bridge.py:451` → `field, editor, input, box id, DOM, radio`
   (after stripping `answer box`);
   `test_facet_solver_routing.py:449-450` → word-boundary `submit, check, next,
   skip, click, press, button`.
   All three are satisfied by the current text *including* its nine
   consumer-specific lines.
3. **`regression_prompt` has no live fixture.** Its only exercise is a
   substring check and a fake adapter. There is no recorded model reply for it
   in either repo, so there is nothing to diff a new wording against.
4. **No command compares prompt variants.** Rendering a prompt requires an
   ad-hoc `python -c` (§7). There is no `facet prompts`, no golden file, and no
   way to diff two candidate wordings without writing one.
5. **The value prompt's conditional branches are unrendered anywhere.** The
   label heading, the `prefix` line and the multi-part contract are each built
   by a separate branch; only §1a of this document shows them assembled.
6. **`answer_parts > 1` and the graph specialists never co-occur**, which is
   correct — `_checked_kind` refuses `answer_parts` on a plan — but it means
   the multi-part contract wording is only ever exercised on the value route.

## 6. Candidate wording — quadratic regression

Three candidates, none applied and none recommended without re-baselining
against a reproduced failure. All preserve the parser exactly as written: the
`quadratic-regression` literal, exactly three coefficients, exact integer or
rational strings, no decimals, ONLY JSON, no extra keys. `parse_regression_plan`
needs no change for any of them. The trailing JSON payload line is unchanged in
all three.

**A — minimal scrub.** Three deletions, no new claims; the smallest change that
removes the consumer-specific wording.

```text
Find the quadratic least-squares regression y=a*x^2+b*x+c for these exact point coordinates. Return ONLY JSON with exactly this schema: {"kind":"quadratic-regression","coefficients":["a","b","c"]}. Coefficients must be exact integer or rational strings, NOT decimal approximations. No prose, markdown, or extra keys.
```

**B — scrub plus an explicit method cue.** Names the normal equations, which is
what the consumer checks anyway, and states the exactness requirement with an
example.

```text
Fit the quadratic least-squares regression y=a*x^2+b*x+c to these exact point coordinates by solving the normal equations exactly. Return ONLY JSON with exactly this schema: {"kind":"quadratic-regression","coefficients":["a","b","c"]}. Each coefficient is an exact integer or rational string such as "3" or "-9/20", never a decimal. No prose, markdown, or extra keys.
```

**C — shortest.** Fewest tokens to reason about, on the theory that an empty
completion may be an output-budget or over-long-reasoning symptom. Worth trying
first *only* if the failure is reproduced and looks budget-shaped.

```text
Return the exact quadratic least-squares fit y=a*x^2+b*x+c for these points as ONLY this JSON: {"kind":"quadratic-regression","coefficients":["a","b","c"]}. Each coefficient is an exact integer or rational string (for example "3" or "-9/20"), never a decimal. No prose, markdown, or extra keys.
```

Not proposed, deliberately: swapping the model, relaxing the rational-only
rule, accepting decimals, tolerating prose around the JSON, or letting the
consumer's proof stand in for a strict parse.

## 7. Command sheet

Fish-safe: no heredocs, no `VAR=value cmd` prefixes, no `$(…)`.

**Tests — facet-runtime**

```fish
cd /home/steve/apps/facet-runtime
uv run --frozen pytest -q
uv run --frozen pytest -q tests/test_solve_math.py tests/test_graph_specialists.py
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

Expected at `b385fdd`: 192 passed, lint and format clean.

**Tests — facet-hawkes**

```fish
cd /home/steve/apps/facet-hawkes
uv run --frozen --extra dev pytest -q
uv run --frozen --extra dev pytest -q tests/test_facet_graph_routing.py tests/test_facet_solver_routing.py tests/test_facet_hawkes_bridge.py tests/test_hawkes_graph.py
uv run --frozen --extra dev ruff check .
uv run --frozen --extra dev ruff format --check .
uv run --frozen --extra dev vulture src tests --min-confidence 80
uv run --frozen --extra dev python scripts/build_extension.py --check
```

Expected at `2c763c1`: 1049 passed, all gates clean, 31 packaged files validated.

**Benchmarks**

```fish
cd /home/steve/apps/facet-hawkes
uv run --frozen --extra dev ethnos hawkes-coverage
uv run --frozen --extra dev ethnos precalc-bench --model gpt-oss:20b --output /tmp/precalc.json
cd /home/steve/apps/facet-runtime
uv run --frozen facet bench --repeat 2
```

`hawkes-coverage` is offline and fast. `precalc-bench` needs Ollama and does
**not** exercise any prompt in §1. `facet bench` measures throughput only.

**Render a prompt without running a model**

```fish
cd /home/steve/apps/facet-runtime
uv run --frozen python -c "
from facet_runtime.graph import Point, regression_prompt
print(regression_prompt('Use quadratic regression. Round to three decimal places.', tuple(Point(x, y) for x, y in [('-5','5'), ('-2','-4'), ('-1','5')])))
"
```

Swap in `parabola_prompt` or `solve.reasoning_prompt` the same way; §1 shows
the argument shapes.

**Live regression case — the existing supported path**

`facet-remote` reads one JSON request on stdin (README, *Remote protocol*).
No new tooling is needed:

```fish
echo '{"facet_protocol_version": 2, "operation": "solve_math", "request_id": "rebase-reg", "problem": {"result_kind": "quadratic_regression", "instruction": "Use quadratic regression. Round to three decimal places.", "points": [{"x":"-5","y":"5"},{"x":"-2","y":"-4"},{"x":"-1","y":"5"}]}, "constraints": {"accelerator_required": false, "allow_fallback": false}}' | facet-remote
```

Takes roughly a minute and exits 1 on the observed failure. The parabola
comparison case, which succeeded in the same session:

```fish
echo '{"facet_protocol_version": 2, "operation": "solve_math", "request_id": "rebase-par", "problem": {"result_kind": "parabola_plan", "instruction": "Graph the parabola.", "expressions": ["f(x)=(x-3)^2-1"], "graph": {"family":"parabola","orientation":"vertical","bounds":[-10.0,10.0,-10.0,10.0],"snap":[0.5,0.5],"controls":"vertex-and-symmetric-points"}}, "constraints": {"accelerator_required": false, "allow_fallback": false}}' | facet-remote
```

`facet-remote` is a `uv tool` install, not the working tree. After changing any
prompt, redeploy before invoking it live:

```fish
cd /home/steve/apps/facet-runtime
uv tool install --force --reinstall .
```

## 8. Outcome (2026-09-05)

### Root cause

Reproduced on the first attempt, at the raw Ollama layer, with the exact
production payload. `gpt-oss:20b` **reasons unconditionally and ignores
`think: false`**, and Ollama counts those reasoning tokens against
`num_predict` *ahead of* the answer. The regression case reached the cap:

```text
done_reason: 'length'   eval_count: 1024 (= num_predict)
response:    ''          thinking: 4942 characters
```

So `response` was empty and §3's `FacetRuntimeError("Ollama returned no
response text")` fired. Nothing was wrong with the transport, the parser, the
installed tool, or the ownership move; the installed tree was byte-identical to
the checkout apart from `.pyc` files, which is why the old `generate_text` path
failed identically.

What pushed *this* question over the cap and not the parabola one is visible in
the recorded reasoning: the prompt contradicted itself. It required exact
rational coefficients while carrying the caller's instruction to *round to three
decimal places*, and the model spent hundreds of tokens on the conflict --
"That is contradictory. We need to decide which instruction to follow. ... We
need to ask for clarification?" -- before running out.

Two independent contributors, both measured, both fixed:

| Prompt | `think` | `num_predict` | `eval_count` | `done_reason` | Result |
| --- | --- | --- | --- | --- | --- |
| as shipped | `false` | 1024 | 1024 | **length** | **empty — the failure** |
| as shipped | `low` | 1024 | 889 | stop | `["3","18","20"]` |
| generalised | `low` | 1024 | 836 | stop | `["3","18","20"]` |
| generalised | `false` | 4096 | 758 | stop | `["3","18","20"]` |

### What changed

- **`Say why a completion came back empty`** — `reasoning_effort` joins the
  model assignment, because a budget and an effort only mean anything together
  and a refusal the model discards is not a control. gpt-oss takes `low` and
  2048 tokens. Every run now reports `stop_reason` and `output_token_limit`
  beside its counts, and an empty completion says whether the budget ran out or
  the model returned nothing — those need opposite fixes and were one sentence
  before. The NPU assignment's note that its own empty completion's cause "was
  not captured" is no longer true and now records the measurement.
- **`Answer Facet's own prompts against a real model`** — `facet prompts`,
  closing §4 and §5.4. Four cases answered by the real `solve_math`, so the
  routing, constructors, parsers and adapters are production's. `--live` exits
  non-zero when a prompt stops answering.
- **`Ask in no consumer's vocabulary but Facet's own`** — all nine lines in §2
  generalised, and the rounding contradiction settled in the prompt rather than
  left to the model. §5.2 is closed: `CONSUMER_VOCABULARY` and `leaks()` live in
  `prompts.py`, both repositories gate against that one list, and §5.5's three
  unrendered branches are gated too.
- facet-hawkes **`Let Facet word its own prompts`** — the one assertion that
  pinned Facet's wording, and the two vocabulary gates, now defer to Facet's
  list.

Candidate **A** of §6 was the closest to what shipped, plus the clause that
settles the rounding. **C** was not needed: the budget was the cliff, not the
prompt length.

### Live results

Reinstalled with `uv tool install --force --reinstall .` first; installed tree
verified identical to the checkout.

| Case | Route | Tokens | Stop | Result | Wall |
| --- | --- | --- | --- | --- | --- |
| regression | reasoning · GPU | 836 / 2048 | stop | `["3","18","20"]` | 44.7 s |
| parabola | reasoning · GPU | 151 / 2048 | stop | vertex `(3,-1)`, opens up | 11.8 s |
| reasoning | reasoning · GPU | 62 / 2048 | stop | `(-∞,-3)∪(-3,3)∪(3,∞)` | 7.3 s |
| exact | exact | — | — | `18i`, `model_calls: 0` | 0.05 s |

All four through `facet prompts --live --backend gpu` and, for the first two and
the last, through `facet-remote` on the exact requests in §7 — the regression
one being the command that had returned `execution_failed` three times running.
`generate_text` was checked too. The exact case satisfies
`accelerator_required: true` with `actual_backend: null`, unchanged.

The diagnosis was confirmed live as well, by squeezing the budget to 32 tokens:

```text
Ollama stopped gpt-oss:20b on its output cap after 32 of 32 tokens, all of it
internal reasoning (67 characters of it), so it never wrote an answer; raise
max_output_tokens or lower reasoning_effort for this assignment
```

### Still open

§5.1 (no prompt is snapshot-tested) and §5.3 (no recorded model reply for the
regression) are narrowed rather than closed: `facet prompts` renders each
prompt for diffing and `--live` checks the reply, but no golden file pins the
bytes. §5.6 stands as correct by design.

## 9. Sanity check

Both working trees clean, both heads unchanged from the top of this document.
Confirm before starting:

```fish
git -C /home/steve/apps/facet-runtime log --oneline -1
git -C /home/steve/apps/facet-runtime status --short
git -C /home/steve/apps/facet-hawkes log --oneline -1
git -C /home/steve/apps/facet-hawkes status --short
```

This document is the only intended change in this repository, and nothing in
facet-hawkes was modified to produce it.
