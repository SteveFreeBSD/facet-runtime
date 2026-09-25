"""Bounded integer line plotting is exact, not literal-pair transcription."""

from fractions import Fraction

import pytest

from facet_runtime.exact.intercepts import affine_coefficients
from facet_runtime.solve import SolveRefused, parse_problem, solve_math

INSTRUCTION = (
    "Graph the line by plotting any two ordered pairs with integer coordinates "
    "that satisfy the equation."
)


def solve(equation, bounds=(-7, 8, -5, 6), snap=(1, 1), **changes):
    payload = {
        "result_kind": "point_plot_plan",
        "instruction": INSTRUCTION,
        "expressions": [equation],
        "graph": {
            "family": "line",
            "orientation": "cartesian",
            "controls": "two-points",
            "bounds": list(bounds),
            "snap": list(snap),
        },
    }
    payload.update(changes)

    def no_model(_):
        raise AssertionError("integer line plotting must never use a model")

    return solve_math(parse_problem(payload), reason=no_model)


@pytest.mark.parametrize(
    "equation,bounds,snap",
    [
        ("y=-x", (-7, 8, -5, 6), (1, 1)),
        ("2*x+3*y=7", (-7, 8, -5, 6), (1, 1)),
        ("y=2*x/3-5/3", (-7, 8, -5, 6), (1, 1)),
        ("x/2+y/3=1", (-7, 8, -5, 6), (1, 1)),
        ("y=-3", (-7, 8, -5, 6), (1, 1)),
        ("x=-4", (-7, 8, -5, 6), (1, 1)),
        ("y=x", (20, 25, 21, 24), (1, 1)),
        ("y=-x", (-25, -20, 21, 24), (1, 1)),
        ("y=x", (-4.5, 4.5, -3.5, 3.5), (1.5, 0.5)),
        ("y=2*x", (-1000000, 1000000, -1000000, 1000000), (2, 3)),
    ],
)
def test_exact_distinct_integer_solutions(equation, bounds, snap):
    result = solve(equation, bounds, snap)
    assert result["route"] == "exact"
    assert result["provenance"]["evidence"]["model_calls"] == 0
    assert result["provenance"]["method"] == "SymPy exact integer line points"
    plan = result["answer"]["plan"]
    assert plan["kind"] == "points"
    points = [(Fraction(p["x"]), Fraction(p["y"])) for p in plan["points"]]
    assert len(points) == len(set(points)) == 2
    a, b, c = map(Fraction, affine_coefficients([equation]))
    for x, y in points:
        assert x.denominator == y.denominator == 1
        assert bounds[0] <= x <= bounds[1] and bounds[2] <= y <= bounds[3]
        assert a * x + b * y + c == 0
        assert (x / Fraction(str(snap[0]))).denominator == 1
        assert (y / Fraction(str(snap[1]))).denominator == 1


def test_prefers_small_points():
    assert solve("y=2*x")["answer"]["plan"]["points"] == [
        {"x": "0", "y": "0"},
        {"x": "1", "y": "2"},
    ]
    assert solve("y=10-x", bounds=(-20, 20, -20, 20))["answer"]["plan"]["points"] == [
        {"x": "5", "y": "5"},
        {"x": "4", "y": "6"},
    ]


def test_diophantine_search_matches_exhaustive_small_grids():
    for a, b in ((2, 3), (-3, 2), (0, 2), (3, 0), (-2, -4)):
        for c in range(-5, 6):
            equation = f"{a}*x+({b})*y={c}"
            expected = {
                (x, y) for x in range(-3, 5) for y in range(-2, 6) if a * x + b * y == c
            }
            if len(expected) < 2:
                with pytest.raises(SolveRefused):
                    solve(equation, bounds=(-3, 4, -2, 5))
            else:
                result = solve(equation, bounds=(-3, 4, -2, 5))
                points = {
                    (int(p["x"]), int(p["y"]))
                    for p in result["answer"]["plan"]["points"]
                }
                assert len(points) == 2 and points <= expected


@pytest.mark.parametrize(
    "equation,bounds,snap",
    [
        ("2*x+2*y=1", (-7, 8, -5, 6), (1, 1)),
        ("x=20", (-7, 8, -5, 6), (1, 1)),
        ("y=x", (0, 0.9, 0, 0.9), (1, 1)),
        ("y=1", (-7, 8, -5, 6), (1, 2)),
        ("y=x^2", (-7, 8, -5, 6), (1, 1)),
        ("x=x", (-7, 8, -5, 6), (1, 1)),
    ],
)
def test_no_safe_pair_fails_without_model(equation, bounds, snap):
    with pytest.raises(SolveRefused):
        solve(equation, bounds, snap)


def test_missing_authority_fails_without_model():
    with pytest.raises(SolveRefused):
        solve_math(
            parse_problem(
                {
                    "result_kind": "point_plot_plan",
                    "instruction": INSTRUCTION,
                    "expressions": ["y=x"],
                }
            ),
            reason=lambda _: pytest.fail("model called"),
        )


@pytest.mark.parametrize(
    "adjective", ["integer", "integer value", "integer-valued", "integer valued"]
)
def test_integer_wording_variants(adjective):
    result = solve("y=3*x+1", instruction=INSTRUCTION.replace("integer", adjective))
    assert result["route"] == "exact"
    assert result["provenance"]["evidence"]["model_calls"] == 0


def test_conflicting_stated_points_are_not_derived():
    with pytest.raises(SolveRefused):
        solve("y=x", instruction=INSTRUCTION + " Use (0,0) and (1,2).")
