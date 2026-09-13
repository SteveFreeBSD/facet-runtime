"""What the exact evaluator keeps from how an expression is written.

Three defects from the 12 September 2026 audit, all in the step that turns a
question's text into SymPy before any solver sees it:

- F02: SymPy cancels a common factor while dividing, so `(x-2)/(x-2) = x-1`
  lost the only sign that it is undefined at 2, and `x = 2` was published.
- F09: Python makes a decimal literal a binary float before it is read, so
  `0.10000000000000001 - 0.1` was simplified to exactly 0.
- F14: the exponent limit bounds degree, not width, and a 167-byte request
  `(a+b+c+d+e+f+g+h)^64` asked for 1,329,890,705 terms and exhausted memory.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from facet_runtime.exact.inequality import solve_linear_inequality
from facet_runtime.exact.polynomial import expand_first_polynomial_expression
from facet_runtime.exact.router import solve_exact

# --- F02: a restriction is read from the expression as written -----------------


@pytest.mark.parametrize(
    ("equation", "display"),
    [
        pytest.param("(x-2)/(x-2)=x-1", "No Solution", id="the-audit-reproduction"),
        pytest.param("(x-3)/(x-3)=x-2", "No Solution", id="cancelled-to-a-line"),
        pytest.param(r"\frac{x^{2}-4}{x-2}=0", "x = -2", id="cancelled-by-together"),
        pytest.param(r"\frac{x}{x-2}+1=\frac{2}{x-2}", "No Solution", id="cleared"),
        pytest.param("x/2+1=3", "One Solution (x = 4)", id="a-constant-divisor"),
    ],
)
def test_a_root_where_the_equation_is_undefined_is_never_published(equation, display):
    solution, decline = solve_exact("Solve the equation.", [equation])

    assert solution is not None, decline
    assert solution.display == display


def test_an_absolute_value_equation_holds_its_roots_to_a_cancelled_divisor():
    """`|x-1| + 1 = 3` has roots -1 and 3, and the question is undefined at 3."""
    solution, decline = solve_exact("Solve the equation.", ["|x-1|+(x-3)/(x-3)=3"])

    assert solution is not None, decline
    assert "3" not in solution.entry and all("3" not in p for p in solution.parts)


def test_an_inequality_that_divides_by_its_variable_is_declined():
    """`1 + x < 5` is `(-∞,4)`, and the question as written leaves out 2."""
    solved, decline = solve_linear_inequality(
        "Solve the inequality. Express your answer in interval notation.",
        ["(x-2)/(x-2)+x<5"],
    )

    assert solved is None
    assert "divides by its variable" in decline


# --- F09: a decimal is the digits the question wrote ---------------------------


def test_a_long_decimal_keeps_every_digit_it_was_written_with():
    solution, decline = solve_exact("Simplify.", ["0.10000000000000001-0.1"])

    assert solution is not None, decline
    assert solution.display == r"\frac{1}{100000000000000000}"


def test_the_polynomial_route_reads_the_same_digits():
    expansion = expand_first_polynomial_expression(
        "Find the product.\n(0.10000000000000001x+1)(x+1)"
    )

    assert expansion is not None
    assert "10000000000000001" in expansion.expanded


# --- F14: a short expression may not ask for an enormous one -------------------

#: Run in a child held to 256 MiB and two CPU seconds, so that a regression
#: fails this test instead of taking the machine with it.
BOUNDED = """
import json, resource, sys
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_CPU, (2, 3))
from facet_runtime.exact.router import solve_exact
from facet_runtime.exact.polynomial import expand_first_polynomial_expression
instruction, expression = json.loads(sys.argv[1])
solution, decline = solve_exact(instruction, [expression])
expansion = expand_first_polynomial_expression(instruction + "\\n" + expression)
print(json.dumps([solution is None, decline, expansion is None]))
"""


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param("(a+b+c+d+e+f+g+h)^64", id="the-audit-reproduction"),
        pytest.param(
            "(a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v+w+x+y+z)^12",
            id="wide-inside-the-polynomial-exponent-limit",
        ),
        pytest.param("((a+b)^64)^64", id="deep-rather-than-wide"),
        pytest.param("(a+b+c+d)^16*(e+f+g+h)^16", id="a-product-of-powers"),
    ],
)
def test_an_expansion_past_the_budget_is_declined_inside_hard_limits(expression):
    finished = subprocess.run(
        [sys.executable, "-c", BOUNDED, json.dumps(["Expand.", expression])],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert finished.returncode == 0, finished.stderr[-2000:]
    declined, _, not_expanded = json.loads(finished.stdout)
    assert declined and not_expanded


@pytest.mark.parametrize(
    ("instruction", "expression", "display"),
    [
        ("Expand.", "(x+1)^5", "x^5 + 5x^4 + 10x^3 + 10x^2 + 5x + 1"),
        ("Simplify.", "(x^8)^32", "x^256"),
    ],
)
def test_coursework_stays_well_inside_the_budget(instruction, expression, display):
    solution, decline = solve_exact(instruction, [expression])

    assert solution is not None, decline
    assert solution.display == display
