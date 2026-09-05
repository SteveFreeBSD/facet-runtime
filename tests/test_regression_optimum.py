"""A question about data, answered exactly and with no model at all.

The other exact solvers are given a function somebody wrote down. This one is
given three measurements and an instruction saying to fit a curve to them, so
the function it answers about does not exist until the fit is computed. That is
the whole capability: the least-squares quadratic over exact rationals, its
turning point, and the rate there -- all of it deterministic, checkable, and
free.

The live question these were written against is lesson 3.3 question 3, where
the same data appears twice on the page: in a table and as three plotted
points. Its answer is nine photos at thirty-six dollars each.
"""

from __future__ import annotations

import pytest
import sympy

from facet_runtime.exact import NotThisQuestion, fit_quadratic, rate_unit
from facet_runtime.exact.regression import regression_optimum
from facet_runtime.solve import MathProblem, SolveRefused, parse_problem, solve_math

#: The live instruction, verbatim.
REVENUE = (
    "Treating revenue as a function of the number of photos sold, a graph of the "
    "three data points is also shown. If she uses quadratic regression to fit a "
    "curve to the data, what number of photos sold and what price per photo will "
    "maximize her revenue?"
)

#: Photos sold against revenue raised, from the same three rows.
REVENUE_POINTS = [("4", "224"), ("5", "260"), ("12", "288")]

REVENUE_PAYLOAD = [
    {"x": "4", "y": "224"},
    {"x": "5", "y": "260"},
    {"x": "12", "y": "288"},
]


def refuse(prompt: str):
    raise AssertionError(f"a model was asked to answer this: {prompt[:80]}")


# The computation ------------------------------------------------------------


def test_three_points_give_the_interpolating_quadratic_exactly():
    """Three points and three coefficients: least squares is interpolation."""
    assert fit_quadratic(REVENUE_POINTS) == (-4, 72, 0)


def test_more_points_than_coefficients_give_the_least_squares_fit():
    """A fourth point off the curve moves the fit, and it stays exact."""
    a, b, c = fit_quadratic([*REVENUE_POINTS, ("9", "300")])

    assert all(value.is_Rational for value in (a, b, c))
    design = sympy.Matrix(
        [
            [sympy.Rational(x) ** 2, sympy.Rational(x), 1]
            for x, _ in [*REVENUE_POINTS, ("9", "300")]
        ]
    )
    values = sympy.Matrix(
        [sympy.Rational(y) for _, y in [*REVENUE_POINTS, ("9", "300")]]
    )
    residual = design * sympy.Matrix([a, b, c]) - values
    # The defining property of a least-squares fit: the residual is orthogonal
    # to every column of the design matrix, exactly and not approximately.
    assert design.T * residual == sympy.zeros(3, 1)


def test_the_live_question_is_nine_photos_at_thirty_six_dollars():
    values, working = regression_optimum(REVENUE, REVENUE_POINTS, 2)

    assert values == ["9", "36"]
    assert working["fit"] == "y = -4x^2 + 72x"
    assert working["optimum_input"] == "9"
    assert working["optimum_value"] == "324"
    assert working["rate"] == "324/9 = 36"


def test_one_answer_is_the_turning_point_alone():
    values, working = regression_optimum(REVENUE, REVENUE_POINTS, 1)

    assert values == ["9"]
    assert "rate" not in working


def test_a_fractional_optimum_stays_a_fraction():
    """Nothing is rounded on the way out, because nothing needs to be."""
    points = [("0", "0"), ("1", "4"), ("2", "6")]
    values, working = regression_optimum(
        "Treating total as a function of the number of items, use quadratic "
        "regression to find the number of items that will maximize the total.",
        points,
        1,
    )

    assert values == ["5/2"]
    assert working["coefficients"] == "-1,5,0"
    assert working["optimum_value"] == "25/4"


# What it refuses ------------------------------------------------------------


def test_a_curve_that_opens_upwards_has_no_maximum():
    """Fail closed rather than report a minimum as if it were a maximum."""
    with pytest.raises(NotThisQuestion, match="no maximum"):
        regression_optimum(
            REVENUE.replace("photos sold", "items sold"),
            [("1", "1"), ("2", "4"), ("3", "9")],
            1,
        )


def test_an_instruction_naming_neither_end_is_declined():
    with pytest.raises(NotThisQuestion, match="maximum or a minimum"):
        regression_optimum(
            "Use quadratic regression to fit a curve to the data.",
            REVENUE_POINTS,
            1,
        )


def test_a_linear_regression_is_a_different_computation():
    with pytest.raises(NotThisQuestion, match="quadratic regression"):
        regression_optimum(
            "Use linear regression to find the value that will maximize revenue.",
            REVENUE_POINTS,
            1,
        )


def test_points_on_one_line_do_not_determine_a_quadratic():
    with pytest.raises(NotThisQuestion, match="do not determine one quadratic"):
        fit_quadratic([("1", "1"), ("1", "2"), ("1", "3")])


def test_a_rate_per_something_the_curve_is_not_a_function_of_is_refused():
    """ "Per event" is not "per photo", and the difference is the answer.

    Revenue divided by photos is a price per photo. Revenue divided by photos
    is *not* a price per event, and producing one would be a number with no
    derivation, typed into a real answer box.
    """
    asked = REVENUE.replace("what price per photo", "what price per event")

    assert rate_unit(asked) is None
    with pytest.raises(NotThisQuestion, match="not a rate per the quantity"):
        regression_optimum(asked, REVENUE_POINTS, 2)


def test_a_rate_is_only_read_when_the_question_says_what_it_is_a_function_of():
    asked = REVENUE.replace("Treating revenue as a function of the ", "Given the ")

    assert rate_unit(asked) is None


def test_more_than_two_answers_are_not_this_question():
    with pytest.raises(NotThisQuestion, match="one or two values"):
        regression_optimum(REVENUE, REVENUE_POINTS, 3)


# Through the protocol -------------------------------------------------------


def test_the_routed_solve_is_exact_and_asks_no_model():
    problem = parse_problem(
        {"instruction": REVENUE, "answer_parts": 2, "points": REVENUE_PAYLOAD}
    )

    result = solve_math(problem, reason=refuse)

    assert result["route"] == "exact"
    assert result["answer"]["parts"] == ["9", "36"]
    # Two answers, so there is no single string to type: `entry` stays empty
    # and the parts are the answer.
    assert result["answer"]["entry"] == ""
    assert result["answer"]["entry_mode"] == "math"


def test_the_provenance_carries_the_working_and_no_processor():
    problem = parse_problem(
        {"instruction": REVENUE, "answer_parts": 2, "points": REVENUE_PAYLOAD}
    )

    provenance = solve_math(problem, reason=refuse)["provenance"]

    assert provenance["source"] == "Facet Exact"
    assert provenance["method"] == "SymPy exact least-squares regression"
    assert provenance["router"] == "solved"
    assert provenance["model"] is None
    assert provenance["actual_backend"] is None
    assert provenance["evidence"]["model_calls"] == 0
    # The working, so a consumer can redo it rather than trust it.
    assert provenance["evidence"]["computation"] == {
        "fit": "y = -4x^2 + 72x",
        "coefficients": "-4,72,0",
        "direction": "maximum",
        "optimum_input": "9",
        "optimum_value": "324",
        "rate_unit": "photo",
        "rate": "324/9 = 36",
    }


def test_a_question_about_points_that_no_solver_owns_still_reaches_reasoning():
    """A decline over points routes like any other decline, and states its data.

    The prompt has to carry the points: an instruction about "the data" with no
    data in it is a question nobody could answer.
    """
    asked = []

    def reason(prompt):
        asked.append(prompt)
        raise SolveRefused("unusable_result", "stopped here on purpose")

    problem = MathProblem(
        instruction="Use exponential regression to model the data.",
        points=parse_problem({"instruction": "x", "points": REVENUE_PAYLOAD}).points,
    )
    with pytest.raises(SolveRefused):
        solve_math(problem, reason=reason)

    assert "- (4, 224)" in asked[0]
    assert "- (12, 288)" in asked[0]


# The protocol boundary ------------------------------------------------------


def test_a_value_question_is_about_expressions_or_about_points():
    with pytest.raises(SolveRefused, match="not both and not neither"):
        parse_problem(
            {"instruction": REVENUE, "expressions": ["x"], "points": REVENUE_PAYLOAD}
        )


def test_a_value_question_about_neither_is_refused():
    with pytest.raises(SolveRefused, match="not both and not neither"):
        parse_problem({"instruction": REVENUE})


def test_points_are_still_checked_for_being_exact():
    with pytest.raises(SolveRefused, match="exact integer or rational"):
        parse_problem(
            {
                "instruction": REVENUE,
                "points": [{"x": "4", "y": "22.4"}, *REVENUE_PAYLOAD[1:]],
            }
        )


def test_a_page_still_cannot_be_described_to_facet():
    """The new field is data, not a document. Nothing else got in with it."""
    with pytest.raises(SolveRefused, match="takes no"):
        parse_problem(
            {
                "instruction": REVENUE,
                "points": REVENUE_PAYLOAD,
                "table": [["Price per Photo", "Number of Photos Sold"]],
            }
        )
