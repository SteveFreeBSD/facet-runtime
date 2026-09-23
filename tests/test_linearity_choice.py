"""Exact linearity classification is based on the simplified relation."""

import pytest

from facet_runtime.solve import MathProblem, SolveRefused, solve_math

ASK = "Determine if the following equation is linear."
CHOICES = ("Linear", "Not Linear")


def solve(equation: str, choices: tuple[str, ...] = CHOICES) -> tuple[dict, int]:
    calls = 0

    def reason(_prompt: str):
        nonlocal calls
        calls += 1
        raise AssertionError("an exact linearity classification reached a model")

    result = solve_math(
        MathProblem(
            instruction=ASK,
            expressions=(equation,),
            answer_choices=choices,
        ),
        reason=reason,
    )
    return result, calls


def test_live_nonlinear_looking_terms_cancel_to_a_linear_relation() -> None:
    result, calls = solve("7x(y + 5) = 11 - 7y(3 - x)")

    assert result["route"] == "exact"
    assert result["answer"]["display"] == "Linear"
    assert result["answer"]["entry"] == "Linear"
    assert result["answer"]["entry_mode"] == "verbatim"
    assert result["answer"]["form"] == "choice"
    assert result["provenance"]["source"] == "Facet Exact"
    assert result["provenance"]["method"] == ("SymPy exact linearity classification")
    assert result["provenance"]["evidence"]["model_calls"] == 0
    assert result["provenance"]["evidence"]["computation"] == {
        "simplified_relation": "35*x + 21*y - 11",
        "total_degree": "1",
    }
    assert calls == 0


@pytest.mark.parametrize(
    "equation",
    [
        "7x(y + 5) = 10 - 8y(3 - x)",
        "7x(y + 5) = 11 - 6y(3 - x)",
        "x^2 + y = 3",
        "x*y = 8",
    ],
)
def test_non_cancelling_nonlinear_controls_are_not_linear(equation: str) -> None:
    result, calls = solve(equation)
    assert result["answer"]["display"] == "Not Linear"
    assert result["provenance"]["evidence"]["computation"]["total_degree"] == "2"
    assert calls == 0


@pytest.mark.parametrize(
    ("equation", "answer"),
    [
        ("xy + 2x = xy + 3y + 1", "Linear"),
        ("x^2 + y = x^2 + 4", "Linear"),
        ("3(x + 2) = 2y - x", "Linear"),
        ("x^3 - x^3 + x*y = 4", "Not Linear"),
    ],
)
def test_classification_follows_the_reduced_polynomial(
    equation: str, answer: str
) -> None:
    result, calls = solve(equation)
    assert result["answer"]["display"] == answer
    assert calls == 0


@pytest.mark.parametrize(
    ("equation", "reason"),
    [
        ("x + y = x + y", "identity"),
        ("x + y = x + y + 1", "contradiction"),
        ("(x^2 - 1)/(x - 1) = y", "variable denominator"),
        ("x + y + z = 1", "one or two variables"),
    ],
)
def test_unsupported_relations_refuse_instead_of_guessing(
    equation: str, reason: str
) -> None:
    with pytest.raises(SolveRefused, match=reason):
        solve(equation)


@pytest.mark.parametrize(
    "choices",
    [("Yes", "No"), ("Linear",), ("Linear", "Not Linear", "Sometimes")],
)
def test_unknown_choice_vocabulary_refuses(choices: tuple[str, ...]) -> None:
    with pytest.raises(SolveRefused, match="published choices"):
        solve("x + y = 4", choices)


def test_page_wording_is_preserved_exactly() -> None:
    result, calls = solve("x*y = 4", ("linear", "Non-Linear"))
    assert result["answer"]["display"] == "Non-Linear"
    assert calls == 0


def test_live_standard_form_request_is_one_semantic_conditional_answer() -> None:
    result, calls = solve_math_problem(
        "(-2 + y)^2 - y^2 = -9x + 4",
        "Determine if the following equation is linear. If the equation is linear, "
        "convert it to standard form.",
        CHOICES,
    )

    answer = result["answer"]
    assert result["route"] == "exact"
    assert answer["form"] == "conditional-choice"
    assert answer["display"] == "Linear; 9x - 4y = 0"
    assert answer["choice"] == "Linear"
    assert answer["entry"] == "9x-4y=0"
    assert answer["relation"] == {"subject": "9x-4y", "value": "0"}
    assert result["provenance"]["evidence"]["model_calls"] == 0
    assert calls == 0


@pytest.mark.parametrize(
    ("equation", "entry"),
    [
        ("2*x+4*y=6", "x+2y=3"),
        ("-x+2*y=5", "x-2y=-5"),
        ("x/2-y/3=7/6", "3x-2y=7"),
    ],
)
def test_standard_form_is_primitive_integer_and_sign_normalized(
    equation: str, entry: str
) -> None:
    result, calls = solve_math_problem(
        equation,
        "Determine if the equation is linear. If linear, write it in standard form.",
        CHOICES,
    )

    assert result["answer"]["entry"] == entry
    assert result["answer"]["choice"] == "Linear"
    assert calls == 0


def test_nonlinear_standard_form_request_stays_a_plain_classification() -> None:
    result, calls = solve_math_problem(
        "x*y=4",
        "Determine if the equation is linear. If linear, write it in standard form.",
        CHOICES,
    )

    assert result["answer"]["form"] == "choice"
    assert result["answer"]["display"] == "Not Linear"
    assert result["answer"]["relation"] is None
    assert calls == 0


def solve_math_problem(
    equation: str, instruction: str, choices: tuple[str, ...]
) -> tuple[dict, int]:
    calls = 0

    def reason(_prompt: str):
        nonlocal calls
        calls += 1
        raise AssertionError("an exact linearity classification reached a model")

    result = solve_math(
        MathProblem(
            instruction=instruction,
            expressions=(equation,),
            answer_choices=choices,
        ),
        reason=reason,
    )
    return result, calls
