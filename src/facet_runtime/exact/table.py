"""Completing a table of values exactly, and checking one that was reasoned.

A completion question states a relation and a grid: some cells carry values the
question gives, the rest are blank, and the blanks are the answer. That is
enough to compute, one row at a time, with no model involved -- bind the row's
known cells into the relation, solve for the one unknown left, and prove each
candidate by substituting it back.

Two things are answered here, and they are the same computation read twice.
`complete_table` produces the answer. `verify_completion` takes an answer that
came from somewhere else -- the reasoning route, which is stochastic -- and
proves it against the same grid. A five-part reply is not an answer merely
because it has five parts, and this is what makes the difference checkable
rather than plausible.

Nothing here rounds. A value that cannot be written the way the question
requires is not adjusted until it can be; the row is refused, and so is the
table, because half a table is a different question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import sympy

from facet_runtime.exact.symbolic import _safe_sympy_expression

#: A column heading this module will treat as a variable. A table of values
#: heads its columns with the variable itself -- `x`, `y`, `f(x)` is not one --
#: and a heading that is a phrase belongs to a different kind of table
#: entirely, the one whose numbers the question *states* rather than asks for.
VARIABLE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,2}")

#: The one representation this module can enforce, published by Hawkes as a
#: character class and normalised long before it reaches Facet.
SIGNED_INTEGER = "signed-integer"

MAX_COLUMNS = 8
MAX_ROWS = 32
MAX_CELL_CHARS = 200

#: The name this route answers to, for a reader looking at where a value came
#: from. It is the same solver; what distinguishes it is what it was given.
TABLE_METHOD = "SymPy exact table completion"


class TableRefused(ValueError):
    """This table, or this answer to it, was not settled exactly."""


class TableUnverifiable(TableRefused):
    """The grid itself could not be read as mathematics.

    A different claim from "this answer is wrong", and the two must not be
    confused. A part that fails substitution is an answer nobody may type; a
    grid whose relation cannot be read is a question this module has nothing to
    say about, and rejecting a reasoned answer to it would be refusing on the
    strength of a check that never ran.
    """


@dataclass(frozen=True, slots=True)
class Cell:
    """One cell: a value the question states, or the Nth blank in it."""

    blank: int | None = None
    value: str = ""


@dataclass(frozen=True, slots=True)
class AnswerTable:
    """A grid of stated values and numbered blanks, with named columns."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Cell, ...], ...]

    @property
    def blanks(self) -> int:
        return sum(1 for row in self.rows for cell in row if cell.blank is not None)

    def as_json(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "rows": [
                [
                    {"blank": cell.blank} if cell.blank is not None
                    else {"value": cell.value}
                    for cell in row
                ]
                for row in self.rows
            ],
        }


@dataclass(frozen=True, slots=True)
class Representation:
    """What every answer to this question must look like when it is written."""

    kind: str
    max_length: int


@dataclass(frozen=True, slots=True)
class TableCompletion:
    """The blanks, filled, in the order the question numbers them."""

    parts: tuple[str, ...]
    evidence: dict[str, str] = field(default_factory=dict)


# --- reading the request ----------------------------------------------------


def parse_answer_table(payload: Any) -> AnswerTable:
    """Read the grid, or refuse it. Nothing here is repaired or guessed."""
    if not isinstance(payload, dict) or set(payload) != {"columns", "rows"}:
        raise TableRefused("an answer table is columns and rows")
    columns = payload["columns"]
    if (
        not isinstance(columns, list)
        or not 2 <= len(columns) <= MAX_COLUMNS
        or not all(isinstance(name, str) and name.strip() for name in columns)
    ):
        raise TableRefused(f"columns must be 2 to {MAX_COLUMNS} names")
    names = tuple(name.strip() for name in columns)
    if len(set(names)) != len(names):
        raise TableRefused("two columns share a name")
    rows = payload["rows"]
    if not isinstance(rows, list) or not 2 <= len(rows) <= MAX_ROWS:
        raise TableRefused(f"rows must be 2 to {MAX_ROWS}")
    read: list[tuple[Cell, ...]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(names):
            raise TableRefused("every row has one cell per column")
        read.append(tuple(_parse_cell(cell) for cell in row))
    table = AnswerTable(columns=names, rows=tuple(read))
    numbered = [cell.blank for row in table.rows for cell in row if cell.blank]
    if not numbered:
        raise TableRefused("a table to complete has a blank in it")
    # Numbered from one, in order. A gap is a part with no cell to belong to.
    if numbered != list(range(1, len(numbered) + 1)):
        raise TableRefused("blanks must be numbered from one, in order")
    return table


def _parse_cell(payload: Any) -> Cell:
    if not isinstance(payload, dict) or len(payload) != 1:
        raise TableRefused("a cell is one blank or one value")
    if "blank" in payload:
        position = payload["blank"]
        if not isinstance(position, int) or isinstance(position, bool) or position < 1:
            raise TableRefused("a blank is numbered from one")
        return Cell(blank=position)
    value = payload.get("value")
    if not isinstance(value, str) or not value.strip():
        raise TableRefused("a stated cell carries a non-empty value")
    if len(value) > MAX_CELL_CHARS:
        raise TableRefused("a stated cell exceeds the size limit")
    return Cell(value=value.strip())


def parse_representation(payload: Any) -> Representation:
    """Read the form every answer must take, or refuse it."""
    if not isinstance(payload, dict) or set(payload) != {"kind", "max_length"}:
        raise TableRefused("a representation is a kind and a max_length")
    kind, limit = payload["kind"], payload["max_length"]
    if kind != SIGNED_INTEGER:
        raise TableRefused(f"{kind} is not a representation Facet enforces")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 40:
        raise TableRefused("max_length must be 1 to 40")
    return Representation(kind=kind, max_length=limit)


# --- the mathematics --------------------------------------------------------


def _symbols(table: AnswerTable) -> dict[str, sympy.Symbol]:
    """One symbol per column, or a refusal naming what the heading was not.

    Declared real, which is how `_safe_sympy_expression` declares the symbols
    it reads out of the relation. A symbol carrying different assumptions is a
    different symbol to SymPy: an unrestricted `x` does not substitute into a
    relation written in terms of a real one, and the row then looks
    unconstrained when it is nothing of the kind.
    """
    for name in table.columns:
        if not VARIABLE_NAME.fullmatch(name):
            raise TableUnverifiable("a column heading is not a variable name")
    return {name: sympy.Symbol(name, real=True) for name in table.columns}


def _relation(expressions: tuple[str, ...] | list[str], allowed: set[str]):
    """The one equation the table is a table of, as `left - right`.

    Exactly one expression, exactly one equals sign, and every symbol in it a
    column of the table. Anything else is a question about something other than
    this grid, and answering it from these cells would be inventing a relation.
    """
    written = [item for item in expressions if item and item.strip()]
    if len(written) != 1:
        raise TableUnverifiable("a table is completed from one stated relation")
    text = written[0]
    sides = text.split("=")
    if len(sides) != 2 or not sides[0].strip() or not sides[1].strip():
        raise TableUnverifiable("the relation is not one equation")
    try:
        left = _safe_sympy_expression(sides[0])
        right = _safe_sympy_expression(sides[1])
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as error:
        raise TableUnverifiable("the relation could not be read exactly") from error
    residual = left - right
    unknown = {symbol.name for symbol in residual.free_symbols} - allowed
    if unknown:
        raise TableUnverifiable("the relation names something the table does not")
    if not residual.free_symbols:
        raise TableUnverifiable("the relation has no variable to solve for")
    return residual


def _value(cell: Cell, where: str):
    try:
        parsed = _safe_sympy_expression(cell.value)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as error:
        raise TableUnverifiable(f"{where} could not be read exactly") from error
    if not parsed.is_number or parsed.is_real is not True:
        raise TableUnverifiable(f"{where} is not an exact real number")
    return parsed


def _fits(value, representation: Representation | None) -> bool:
    """Whether this exact value can be written the way the question requires.

    A value that cannot is not adjusted until it can be. Rounding an exact
    answer to satisfy an answer box is how a wrong answer gets typed in
    confidently, and refusing is the only other honest option.
    """
    if representation is None:
        return True
    if representation.kind != SIGNED_INTEGER:
        return False
    if value.is_Integer is not True:
        return False
    return len(str(value)) <= representation.max_length


def _order(value) -> tuple:
    """The canonical order among solutions a row admits.

    Smallest in magnitude first, and the non-negative one where two share a
    magnitude -- so a row satisfied by both 8 and -8 is answered 8, and one
    satisfied by 2 and -3 is answered 2. The numbers are only ever *ordered*
    numerically; the value chosen stays exactly what SymPy computed. A final
    key on the canonical form makes the order total, so two solutions that
    agree to thirty digits still sort the same way on every run.
    """
    try:
        magnitude = float(sympy.Abs(value).evalf(30))
        signed = float(value.evalf(30))
    except (TypeError, ValueError):
        magnitude, signed = 0.0, 0.0
    return (magnitude, -signed, sympy.srepr(value))


def _solutions(residual, bindings: dict, unknown: sympy.Symbol, where: str) -> list:
    """Every exact real solution of the row, proved by substitution."""
    bound = residual.subs(bindings)
    if unknown not in bound.free_symbols:
        raise TableUnverifiable(f"{where} does not constrain its blank")
    found = sympy.solveset(bound, unknown, sympy.S.Reals)
    if found is sympy.S.EmptySet:
        raise TableRefused(f"{where} has no exact solution")
    if not isinstance(found, sympy.FiniteSet):
        # An interval, an image set or a condition set is not a list of answers
        # anyone could type. Declining is the only reading that is not a guess.
        raise TableRefused(f"{where} does not have finitely many solutions")
    proved = []
    for candidate in found:
        if not candidate.is_number or candidate.is_real is not True:
            continue
        # The proof, and the whole reason this is not merely `solve`: the value
        # goes back into the relation the question stated and the remainder has
        # to be exactly zero.
        remainder = sympy.simplify(bound.subs({unknown: candidate}))
        if remainder == 0:
            proved.append(sympy.nsimplify(candidate))
    if not proved:
        raise TableRefused(f"{where} has no exact solution")
    return sorted(proved, key=_order)


def _rows(table: AnswerTable, symbols: dict[str, sympy.Symbol], residual):
    """Each row as (its blank's position, its blank's symbol, its bindings).

    A row with no blank is not skipped. It states the relation twice over --
    once as the relation and once as a pair of values that must satisfy it --
    and a table whose stated rows do not agree with its own relation has been
    misread. Better to find that here than to answer four rows of it.
    """
    prepared = []
    for index, row in enumerate(table.rows, start=1):
        where = f"row {index}"
        blanks = [(column, cell) for column, cell in zip(table.columns, row)
                  if cell.blank is not None]
        bindings = {
            symbols[column]: _value(cell, f"{where}, column {column}")
            for column, cell in zip(table.columns, row)
            if cell.blank is None
        }
        if len(blanks) > 1:
            raise TableUnverifiable(f"{where} has more than one blank")
        if not blanks:
            if sympy.simplify(residual.subs(bindings)) != 0:
                raise TableUnverifiable(f"{where} contradicts the stated relation")
            continue
        (column, cell), = blanks
        prepared.append((cell.blank, symbols[column], bindings, where))
    return prepared


def complete_table(
    expressions: tuple[str, ...] | list[str],
    table: AnswerTable,
    *,
    answer_parts: int,
    representation: Representation | None = None,
) -> TableCompletion:
    """Fill every blank exactly, or refuse the table and say what stopped it."""
    if answer_parts != table.blanks:
        raise TableRefused("the table's blanks and the answer's parts disagree")
    symbols = _symbols(table)
    residual = _relation(expressions, set(symbols))
    evidence = {"relation": f"{sympy.sstr(residual)} = 0"}
    chosen: dict[int, str] = {}
    for position, unknown, bindings, where in _rows(table, symbols, residual):
        proved = _solutions(residual, bindings, unknown, where)
        allowed = [value for value in proved if _fits(value, representation)]
        if not allowed:
            # Every solution is exact and none can be written the way the
            # question requires. Rounding one is not available here.
            raise TableRefused(
                f"{where} has no solution the answer may be written as"
            )
        pick = allowed[0]
        chosen[position] = sympy.sstr(pick)
        stated = ", ".join(
            f"{symbol}={sympy.sstr(value)}" for symbol, value in bindings.items()
        )
        evidence[where] = (
            f"{stated} -> {unknown} in "
            f"{{{', '.join(sympy.sstr(value) for value in proved)}}} -> {chosen[position]}"
        )
    parts = tuple(chosen[position] for position in sorted(chosen))
    # The answer this route produces goes through the same proof an answer from
    # anywhere else does. One rule, applied twice, so the two can never drift.
    verify_completion(parts, expressions, table, representation=representation)
    return TableCompletion(parts=parts, evidence=evidence)


def verify_completion(
    parts: tuple[str, ...] | list[str],
    expressions: tuple[str, ...] | list[str],
    table: AnswerTable,
    *,
    representation: Representation | None = None,
) -> None:
    """Prove an answer against the grid it claims to complete, or refuse it.

    This is what a reasoning result has to survive. Each part is read as exact
    mathematics, put back into its own row beside that row's stated values, and
    the relation has to hold exactly. A part that is merely the right shape,
    or right for a different row, fails here.
    """
    if len(parts) != table.blanks:
        raise TableRefused("the answer has a part for every blank, and no more")
    symbols = _symbols(table)
    residual = _relation(expressions, set(symbols))
    for position, unknown, bindings, where in _rows(table, symbols, residual):
        written = parts[position - 1]
        try:
            value = _safe_sympy_expression(written)
        except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as error:
            raise TableRefused(f"part {position} is not exact mathematics") from error
        if not value.is_number or value.is_real is not True:
            raise TableRefused(f"part {position} is not an exact real number")
        if not _fits(value, representation):
            raise TableRefused(
                f"part {position} is not written the way this question requires"
            )
        if unknown not in residual.subs(bindings).free_symbols:
            raise TableUnverifiable(f"{where} does not constrain its blank")
        remainder = sympy.simplify(residual.subs(bindings).subs({unknown: value}))
        if remainder != 0:
            raise TableRefused(f"part {position} does not satisfy {where}")
