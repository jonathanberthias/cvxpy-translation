from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from functools import partial
from functools import wraps
from typing import Callable
from typing import Generator
from typing import Literal

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from cvxpy_translation.gurobi.translation import CVXPY_VERSION
from cvxpy_translation.gurobi.translation import GUROBI_MAJOR


@dataclass(frozen=True)
class CaseContext:
    solver: str


CONTEXTS = (CaseContext(solver=cp.GUROBI), CaseContext(solver=cp.SCIP))


@dataclass(frozen=True)
class ProblemTestCase:
    problem: cp.Problem
    group: str
    context: CaseContext
    invalid_reason: str | None = None
    skip_reason: str | None = None


_PROBLEM_GENERATOR_NAMES: list[str] = []


def group_cases(
    group: str, *, invalid_reason: str | None = None
) -> Callable[
    [Callable[[], Generator[cp.Problem]]], Callable[[], Generator[ProblemTestCase]]
]:
    def dec(
        iter_fn: Callable[[], Generator[cp.Problem]],
    ) -> Callable[[], Generator[ProblemTestCase]]:
        @wraps(iter_fn)
        def inner() -> Generator[ProblemTestCase]:
            for context in CONTEXTS:
                reset_id_counter()
                for problem in iter_fn():
                    yield ProblemTestCase(
                        problem=problem,
                        group=group,
                        context=context,
                        invalid_reason=invalid_reason,
                    )

        # Register the generator function to be used later
        _PROBLEM_GENERATOR_NAMES.append(iter_fn.__name__)

        return inner

    return dec


def skipif(
    condition: Callable[[ProblemTestCase], bool], reason: str
) -> Callable[
    [Callable[[], Generator[ProblemTestCase]]], Callable[[], Generator[ProblemTestCase]]
]:
    """Mark a group of test cases to skip if the condition is True."""

    def dec(
        iter_fn: Callable[[], Generator[ProblemTestCase]],
    ) -> Callable[[], Generator[ProblemTestCase]]:
        @wraps(iter_fn)
        def inner() -> Generator[ProblemTestCase]:
            for case in iter_fn():
                if condition(case):
                    yield replace(case, skip_reason=reason)
                else:
                    yield case

        return inner

    return dec


def all_problems() -> Generator[ProblemTestCase]:
    for problem_gen_name in _PROBLEM_GENERATOR_NAMES:
        # Make sure order of groups does not matter
        reset_id_counter()
        problem_gen = globals()[problem_gen_name]
        yield from problem_gen()


def all_valid_problems() -> Generator[ProblemTestCase]:
    yield from (case for case in all_problems() if not case.invalid_reason)


def all_invalid_problems() -> Generator[ProblemTestCase]:
    yield from (case for case in all_problems() if case.invalid_reason)


@group_cases("simple")
def simple_expressions() -> Generator[cp.Problem]:
    x = cp.Variable(name="x", nonneg=True)
    y = cp.Variable(name="y", nonneg=True)

    yield cp.Problem(cp.Minimize(x))
    yield cp.Problem(cp.Minimize(x + 1))
    yield cp.Problem(cp.Minimize(x + x))
    yield cp.Problem(cp.Minimize(x + y))

    yield cp.Problem(cp.Minimize(x - 1))
    yield cp.Problem(cp.Minimize(x - y), [y <= 1])

    yield cp.Problem(cp.Minimize(2 * x))
    yield cp.Problem(cp.Minimize(2 * x + 1))
    yield cp.Problem(cp.Minimize(2 * x + y))

    yield cp.Problem(cp.Minimize(-x), [x <= 1])
    yield cp.Problem(cp.Minimize(-x + 1), [x <= 1])
    yield cp.Problem(cp.Minimize(1 - x), [x <= 1])

    yield cp.Problem(cp.Minimize(x / 2))
    yield cp.Problem(cp.Minimize(x / 2 + 1))

    yield cp.Problem(cp.Minimize(x**2))
    yield cp.Problem(cp.Minimize((x - 1) ** 2))
    yield cp.Problem(cp.Minimize(x**2 + y**2))


@group_cases("scalar_linear")
def scalar_linear_constraints() -> Generator[cp.Problem]:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    yield cp.Problem(cp.Minimize(x), [x >= 1])
    yield cp.Problem(cp.Maximize(x), [x <= 1])
    yield cp.Problem(cp.Minimize(x), [x == 1])

    yield cp.Problem(cp.Minimize(x), [x >= 1, x <= 2])
    yield cp.Problem(cp.Minimize(x), [x >= 1, x <= 1])
    yield cp.Problem(cp.Minimize(x), [x >= 0, x <= 2, x == 1])

    yield cp.Problem(cp.Minimize(x), [x >= 1, y >= 1])
    yield cp.Problem(cp.Minimize(x), [x == 1, y == 1])

    yield cp.Problem(cp.Minimize(x), [x + y >= 1, y == 0])
    yield cp.Problem(cp.Maximize(x), [x + y <= 1, y == 0])
    yield cp.Problem(cp.Minimize(x), [x + y == 1, y == 0])

    yield cp.Problem(cp.Minimize(x), [2 * x >= 1])
    yield cp.Problem(cp.Minimize(x), [2 * x + y >= 1, y == 0])


@skipif(lambda case: case.context.solver == cp.SCIP, "TODO")
@group_cases("matrix")
def matrix_constraints() -> Generator[cp.Problem]:
    x = cp.Variable(2, name="x")
    y = cp.Variable(2, name="y")
    A = np.arange(4).reshape((2, 2))
    S = sp.csr_matrix(A)

    yield cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1, x <= 2])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [x == 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [x == 1, y == 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [x + y >= 1, y == 0])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [x + y + 1 >= 0, y == 0])

    yield cp.Problem(cp.Minimize(cp.sum(x)), [A @ x == 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [A @ x + y == 1, y == 0])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [A @ x + y + 1 == 0, y == 0])

    yield cp.Problem(cp.Minimize(cp.sum(x)), [S @ x == 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [S @ x + y == 1, y == 0])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [S @ x + y + 1 == 0, y == 0])


@group_cases("quadratic")
def quadratic_expressions() -> Generator[cp.Problem]:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    yield cp.Problem(cp.Minimize(x**2))
    yield cp.Problem(cp.Minimize(x**2 + 1))
    yield cp.Problem(cp.Minimize(x**2 + x))
    yield cp.Problem(cp.Minimize(x**2 + x**2))
    yield cp.Problem(cp.Minimize(x**2 + y**2))
    yield cp.Problem(cp.Minimize(2 * x**2))
    yield cp.Problem(cp.Minimize((2 * x) ** 2))
    yield cp.Problem(cp.Minimize((2 * x) ** 2 + 1))
    yield cp.Problem(cp.Minimize((2 * x) ** 2 + x**2))
    yield cp.Problem(cp.Minimize((2 * x) ** 2 + y**2))
    yield cp.Problem(cp.Minimize((x + y) ** 2))
    yield cp.Problem(cp.Minimize((x - y) ** 2))
    yield cp.Problem(cp.Minimize((x - y) ** 2 + x + y), [y == 0])


@skipif(lambda case: case.context.solver == cp.SCIP, "TODO")
@group_cases("matrix_quadratic")
def matrix_quadratic_expressions() -> Generator[cp.Problem]:
    x = cp.Variable(2, name="x")
    A = 2 * np.eye(2)
    S = 2 * sp.eye(2)

    yield cp.Problem(cp.Minimize(cp.sum_squares(x)))
    yield cp.Problem(cp.Minimize(cp.sum_squares(x - 1)))
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.sum_squares(x) <= 1])
    yield cp.Problem(cp.Minimize(cp.sum_squares(x)), [cp.sum_squares(x) <= 1])
    yield cp.Problem(cp.Minimize(cp.sum_squares(A @ x)))
    yield cp.Problem(cp.Minimize(cp.sum_squares(S @ x)))


@group_cases("quad_form")
def quad_form() -> Generator[cp.Problem]:
    x = cp.Variable((1,), name="x")
    A = np.array([[1]])
    yield cp.Problem(cp.Minimize(cp.quad_form(x, A)))

    x = cp.Variable(2, name="x")
    A_ = np.arange(4).reshape((2, 2))
    A = A_.T @ A_
    yield cp.Problem(cp.Minimize(cp.quad_form(x, A)))

    x = np.arange(1, 3)
    A = cp.Variable((2, 2), name="A", nonneg=True)
    yield cp.Problem(cp.Minimize(cp.quad_form(x, A)))


@skipif(
    lambda case: case.context.solver == cp.GUROBI and GUROBI_MAJOR < 11,
    "requires Gurobi 11+",
)
@group_cases("quad_form_stack")
def quad_form_stack() -> Generator[cp.Problem]:
    x = cp.Variable(2, name="x")
    y = np.array([1, 2])
    yield cp.Problem(cp.Minimize(cp.quad_form(cp.hstack([x, y]), np.eye(4))))


@skipif(
    lambda case: case.context.solver == cp.SCIP, "SCIP does not support PSD variables"
)
@skipif(
    lambda case: case.context.solver == cp.GUROBI,
    "Gurobi does not support PSD variables",
)
@group_cases("quad_form_psd")
def quad_form_psd() -> Generator[cp.Problem]:
    x = np.arange(1, 3)
    A = cp.Variable((2, 2), name="A", PSD=True)
    yield cp.Problem(cp.Minimize(cp.quad_form(x, A)))


@skipif(lambda case: case.context.solver == cp.SCIP, "TODO")
@group_cases("genexpr_abs")
def genexpr_abs() -> Generator[cp.Problem]:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    yield cp.Problem(cp.Minimize(cp.abs(x)))
    yield cp.Problem(cp.Minimize(cp.abs(x) + 1))
    yield cp.Problem(cp.Minimize(cp.abs(x) + cp.abs(y)))
    yield cp.Problem(cp.Minimize(cp.abs(x + y)))

    reset_id_counter()
    x = cp.Variable(1, name="X")
    y = cp.Variable(name="Y")

    yield cp.Problem(cp.Minimize(cp.abs(x)))
    yield cp.Problem(cp.Minimize(cp.abs(x) + 1))
    yield cp.Problem(cp.Minimize(cp.abs(x) + cp.abs(y)))
    yield cp.Problem(cp.Minimize(cp.abs(x + y)))

    reset_id_counter()
    x = cp.Variable(2, name="X", nonneg=True)
    y = cp.Variable(2, name="Y", nonneg=True)
    A = np.array([1, -2])

    yield cp.Problem(cp.Minimize(cp.sum(cp.abs(x))))
    yield cp.Problem(cp.Minimize(cp.sum(cp.abs(x + y))))
    yield cp.Problem(cp.Minimize(cp.sum(cp.abs(x) + 1)))
    yield cp.Problem(cp.Minimize(cp.sum(cp.abs(x) + A)))
    yield cp.Problem(cp.Minimize(cp.sum(cp.abs(x) + cp.abs(y))))
    yield cp.Problem(cp.Minimize(cp.sum(cp.abs(x) + cp.abs(A))))

    reset_id_counter()
    x = cp.Variable((2, 2), name="X", nonneg=True)
    y = cp.Variable((2, 2), name="Y", nonneg=True)
    A = np.array([[1, -2], [3, 4]])

    yield cp.Problem(cp.Minimize(cp.sum(cp.abs(x))))
    yield cp.Problem(cp.Minimize(cp.sum(cp.abs(x + y))))
    yield cp.Problem(cp.Minimize(cp.sum(cp.abs(x + 1))))
    yield cp.Problem(cp.Minimize(cp.sum(cp.abs(x) + A)))
    yield cp.Problem(cp.Minimize(cp.sum(cp.abs(x) + cp.abs(y))))
    yield cp.Problem(cp.Minimize(cp.sum(cp.abs(x) + cp.abs(A))))


@skipif(lambda case: case.context.solver == cp.SCIP, "TODO")
@group_cases("genexpr_min_max")
def genexpr_min_max() -> Generator[cp.Problem]:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    yield cp.Problem(cp.Maximize(cp.min(x)), [x <= 1])
    yield cp.Problem(cp.Maximize(cp.min(x) + 1), [x <= 1])
    yield cp.Problem(cp.Maximize(cp.min(x) + cp.min(y)), [x <= 1, y <= 1])
    yield cp.Problem(cp.Maximize(cp.min(x + y)), [x <= 1, y <= 1])

    reset_id_counter()

    yield cp.Problem(cp.Minimize(cp.max(x)), [x >= 1])
    yield cp.Problem(cp.Minimize(cp.max(x) + 1), [x >= 1])
    yield cp.Problem(cp.Minimize(cp.max(x) + cp.max(y)), [x >= 1, y >= 1])
    yield cp.Problem(cp.Minimize(cp.max(x + y)), [x >= 1, y >= 1])

    reset_id_counter()
    x = cp.Variable(1, name="x")
    y = cp.Variable(name="y")

    yield cp.Problem(cp.Maximize(cp.min(x)), [x <= 1])
    yield cp.Problem(cp.Maximize(cp.min(x) + 1), [x <= 1])
    yield cp.Problem(cp.Maximize(cp.min(x) + cp.min(y)), [x <= 1, y <= 1])
    yield cp.Problem(cp.Maximize(cp.min(x + y)), [x <= 1, y <= 1])

    reset_id_counter()

    yield cp.Problem(cp.Minimize(cp.max(x)), [x >= 1])
    yield cp.Problem(cp.Minimize(cp.max(x) + 1), [x >= 1])
    yield cp.Problem(cp.Minimize(cp.max(x) + cp.max(y)), [x >= 1, y >= 1])
    yield cp.Problem(cp.Minimize(cp.max(x + y)), [x >= 1, y >= 1])

    reset_id_counter()
    x = cp.Variable(2, name="X")
    y = cp.Variable(2, name="Y")
    A = np.array([1, -2])

    yield cp.Problem(cp.Maximize(cp.min(x)), [x <= 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.min(x) >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.min(x) + 1 >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.min(x) + cp.min(y) >= 1, y == 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.min(x + y) >= 1, y == 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.min(x + A) >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.min(x) + cp.min(A) >= 1])

    reset_id_counter()

    yield cp.Problem(cp.Minimize(cp.max(x)), [x >= 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.max(x) <= 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.max(x) + 1 <= 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.max(x) + cp.max(y) <= 1, y == 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.max(x + y) <= 1, y == 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.max(x + A) <= 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.max(x) + cp.max(A) <= 1])

    reset_id_counter()
    x = cp.Variable((2, 2), name="X")
    y = cp.Variable((2, 2), name="Y")
    A = np.array([[1, -2], [3, 4]])

    yield cp.Problem(cp.Maximize(cp.min(x)), [x <= 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.min(x) >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.min(x) + 1 >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.min(x) + cp.min(y) >= 1, y == 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.min(x + y) >= 1, y == 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.min(x + A) >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.min(x) + cp.min(A) >= 1])

    reset_id_counter()

    yield cp.Problem(cp.Minimize(cp.max(x)), [x >= 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.max(x) <= 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.max(x) + 1 <= 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.max(x) + cp.max(y) <= 1, y == 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.max(x + y) <= 1, y == 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.max(x + A) <= 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.max(x) + cp.max(A) <= 1])


@skipif(lambda case: case.context.solver == cp.SCIP, "TODO")
@group_cases("genexpr_minimum_maximum")
def genexpr_minimum_maximum() -> Generator[cp.Problem]:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    yield cp.Problem(cp.Minimize(x), [cp.minimum(x, 2) >= 1])
    yield cp.Problem(cp.Minimize(x + y), [cp.minimum(x, y) >= 1])
    yield cp.Problem(cp.Minimize(x + y), [cp.minimum(x, y, 3) >= 1])
    yield cp.Problem(cp.Minimize(x), [cp.minimum(1, 2) <= 1, x >= 0])
    yield cp.Problem(cp.Minimize(x), [cp.minimum(x + 1, 2) >= 1])

    reset_id_counter()

    yield cp.Problem(cp.Maximize(x), [cp.maximum(x, 1) <= 2])
    yield cp.Problem(cp.Maximize(x + y), [cp.maximum(x, y) <= 1])
    yield cp.Problem(cp.Maximize(x + y), [cp.maximum(x, y, 1) <= 3])
    yield cp.Problem(cp.Maximize(x), [cp.maximum(1, 2) <= 2, x <= 0])
    yield cp.Problem(cp.Maximize(x), [cp.maximum(x + 1, 1) <= 2])

    reset_id_counter()
    x = cp.Variable(1, name="x")
    y = cp.Variable(name="y")

    yield cp.Problem(cp.Minimize(x), [cp.minimum(x, 2) >= 1])
    yield cp.Problem(cp.Minimize(x + y), [cp.minimum(x, y) >= 1])
    yield cp.Problem(cp.Minimize(x + y), [cp.minimum(x, y, 3) >= 1])
    yield cp.Problem(cp.Minimize(x), [cp.minimum(1, 2) <= 1, x >= 0])

    reset_id_counter()

    yield cp.Problem(cp.Maximize(x), [cp.maximum(x, 1) <= 2])
    yield cp.Problem(cp.Maximize(x + y), [cp.maximum(x, y) <= 1])
    yield cp.Problem(cp.Maximize(x + y), [cp.maximum(x, y, 1) <= 3])
    yield cp.Problem(cp.Maximize(x), [cp.maximum(1, 2) <= 2, x <= 0])

    reset_id_counter()
    x = cp.Variable(2, name="X")
    y = cp.Variable(2, name="Y")
    z = cp.Variable(name="z")
    A = np.array([2, -1])

    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.minimum(x, A) >= -1])
    yield cp.Problem(cp.Minimize(cp.sum(x + y)), [cp.minimum(x, y) >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(x + y)), [cp.minimum(x, y) >= A])
    yield cp.Problem(cp.Minimize(cp.sum(x + y)), [cp.minimum(x, y, A) >= -1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.minimum(x, A) >= -y, y == 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.minimum(x + A, A) >= -1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.minimum(x, z, 1) >= 0, z == 1])

    reset_id_counter()

    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.maximum(x, A) <= 2])
    yield cp.Problem(cp.Maximize(cp.sum(x + y)), [cp.maximum(x, y) <= 1])
    yield cp.Problem(cp.Maximize(cp.sum(x + y)), [cp.maximum(x, y) <= A])
    yield cp.Problem(cp.Maximize(cp.sum(x + y)), [cp.maximum(x, y, A) <= 2])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.maximum(x, A) <= y, y == 2])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.maximum(x + A, A) <= 2])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.maximum(x, z, 1) <= 2, z == 1])

    reset_id_counter()
    x = cp.Variable((2, 2), name="X")
    y = cp.Variable((2, 2), name="Y")
    A = np.array([[1, -2], [3, 4]])
    B = np.array([[2, -1], [1, 3]])

    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.minimum(x, A) >= -2])
    yield cp.Problem(cp.Minimize(cp.sum(x + y)), [cp.minimum(x, y) >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(x + y)), [cp.minimum(x, y) >= A])
    yield cp.Problem(cp.Minimize(cp.sum(x + y)), [cp.minimum(x, y, A) >= -2])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.minimum(x, A) >= -y, y == 2])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.minimum(A, B) >= -2, x == 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.minimum(x + A, A) >= -2])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.minimum(x, z, 1) >= 0, z == 1])

    reset_id_counter()

    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.maximum(x, A) <= 4])
    yield cp.Problem(cp.Maximize(cp.sum(x + y)), [cp.maximum(x, y) <= 1])
    yield cp.Problem(cp.Maximize(cp.sum(x + y)), [cp.maximum(x, y) <= A])
    yield cp.Problem(cp.Maximize(cp.sum(x + y)), [cp.maximum(x, y, A) <= 4])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.maximum(x, A) <= y, y == 4])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.maximum(A, B) <= 4, x == 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.maximum(x + A, A) <= 4])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.maximum(x, z, 1) <= 2, z == 1])


def _genexpr_norm_problems(
    norm: Callable[[cp.Expression | float | np.ndarray], cp.Expression],
) -> Generator[cp.Problem]:
    x = cp.Variable(name="x")
    yield cp.Problem(cp.Minimize(norm(x)))
    yield cp.Problem(cp.Minimize(norm(x - 1)))
    yield cp.Problem(cp.Minimize(norm(x) + norm(-1)))
    yield cp.Problem(cp.Maximize(x), [norm(x) <= 1])

    reset_id_counter()
    x = cp.Variable(1, name="x")
    yield cp.Problem(cp.Minimize(norm(x)))
    yield cp.Problem(cp.Minimize(norm(x - 1)))
    yield cp.Problem(cp.Minimize(norm(x) + norm(-1)))
    yield cp.Problem(cp.Maximize(x), [norm(x) <= 1])

    reset_id_counter()
    x = cp.Variable(2, name="x")
    # Slightly off from [1, -1] to avoid symmetric solutions
    A = np.array([1.00001, -0.99999])
    yield cp.Problem(cp.Minimize(norm(x)))
    yield cp.Problem(cp.Minimize(norm(x - A)))
    yield cp.Problem(cp.Minimize(norm(x) + norm(A)))
    yield cp.Problem(cp.Maximize(cp.sum(cp.multiply(x, A))), [norm(x) <= np.sqrt(2)])

    reset_id_counter()
    x = cp.Variable((2, 2), name="X")
    yield cp.Problem(cp.Minimize(norm(x)))
    A = np.array([[2, 1], [-2, -4]])  # 2-norm is exactly 5
    yield cp.Problem(cp.Minimize(norm(x - A)))
    yield cp.Problem(cp.Minimize(norm(x) + norm(A)))
    # lower values in the upper bound lead to tiny differences in the solution
    # TODO: investigate which of cvxpy or this library is the most accurate
    yield cp.Problem(cp.Maximize(cp.sum(cp.multiply(x, A))), [norm(x) <= 6])


@skipif(lambda case: case.context.solver == cp.SCIP, "TODO")
@group_cases("genexpr_norm1")
def genexpr_norm1() -> Generator[cp.Problem]:
    yield from _genexpr_norm_problems(cp.norm1)


@skipif(lambda case: case.context.solver == cp.SCIP, "TODO")
@group_cases("genexpr_norm2")
def genexpr_norm2() -> Generator[cp.Problem]:
    # we use pnorm(p=2) as the norm2 function will automatically
    # use matrix norms but Gurobi only handles vector norms
    # pnorm will only create vector norms
    yield from _genexpr_norm_problems(partial(cp.pnorm, p=2))


@skipif(lambda case: case.context.solver == cp.SCIP, "TODO")
@group_cases("genexpr_norminf")
def genexpr_norminf() -> Generator[cp.Problem]:
    yield from _genexpr_norm_problems(cp.norm_inf)


@skipif(lambda case: case.context.solver == cp.SCIP, "TODO")
@group_cases("indexing")
def indexing() -> Generator[cp.Problem]:
    x = cp.Variable(2, name="x", nonneg=True)
    m = cp.Variable((2, 2), name="m", nonneg=True)
    y = x + np.array([1, 2])

    idx = 0
    yield cp.Problem(cp.Minimize(x[idx]))
    yield cp.Problem(cp.Minimize(y[idx]))
    yield cp.Problem(cp.Minimize(m[idx, idx]))

    yield cp.Problem(cp.Minimize(cp.sum(m[idx])))
    yield cp.Problem(cp.Minimize(cp.sum(m[idx, :])))

    reset_id_counter()
    idx = np.array([0])
    yield cp.Problem(cp.Minimize(cp.sum(x[idx])))
    yield cp.Problem(cp.Minimize(cp.sum(y[idx])))
    yield cp.Problem(cp.Minimize(cp.sum(m[idx])))
    yield cp.Problem(cp.Minimize(cp.sum(m[idx, :])))

    reset_id_counter()
    idx = np.array([True, False])
    yield cp.Problem(cp.Minimize(cp.sum(x[idx])))
    yield cp.Problem(cp.Minimize(cp.sum(y[idx])))
    yield cp.Problem(cp.Minimize(cp.sum(m[idx])))
    yield cp.Problem(cp.Minimize(cp.sum(m[idx, :])))

    yield cp.Problem(cp.Minimize(cp.sum(x[:])))
    yield cp.Problem(cp.Minimize(cp.sum(y[:])))
    yield cp.Problem(cp.Minimize(cp.sum(m[:, :])))


@group_cases("sum_scalar")
def sum_scalar() -> Generator[cp.Problem]:
    x = cp.Variable(name="x", nonneg=True)
    yield cp.Problem(cp.Minimize(cp.sum(x)))
    yield cp.Problem(cp.Minimize(cp.sum(x + 1)))

    x = cp.Variable(1, name="x", nonneg=True)
    yield cp.Problem(cp.Minimize(cp.sum(x)))
    yield cp.Problem(cp.Minimize(cp.sum(x + 1)))

    x = cp.Variable((1, 1), name="x", nonneg=True)
    yield cp.Problem(cp.Minimize(cp.sum(x)))
    yield cp.Problem(cp.Minimize(cp.sum(x + 1)))


@group_cases("sum_axis")
def sum_axis() -> Generator[cp.Problem]:
    x = cp.Variable((2, 2), name="x", nonneg=True)
    obj = cp.sum(cp.multiply(x, np.arange(1, 5).reshape((2, 2), order="F")))

    yield cp.Problem(cp.Minimize(obj), [cp.sum(x, axis=None) >= 1])
    yield cp.Problem(cp.Minimize(obj), [cp.sum(x, axis=0) >= 1])
    yield cp.Problem(cp.Minimize(obj), [cp.sum(x, axis=1) >= 1])
    if CVXPY_VERSION >= (1, 6, 6):
        # axis=-1 support added in 1.6.0, but it's broken until 1.6.6
        yield cp.Problem(cp.Minimize(obj), [cp.sum(x, axis=-1) >= 1])


@group_cases("reshape")
def reshape() -> Generator[cp.Problem]:
    x = cp.Variable(name="x")
    a = x + 1
    yield cp.Problem(cp.Maximize(x), [cp.reshape(x, (), order="F") <= 1])
    yield cp.Problem(cp.Maximize(x), [cp.reshape(x, 1, order="F") <= np.ones(1)])
    if CVXPY_VERSION >= (1, 4, 0):
        # -1 support added in https://github.com/cvxpy/cvxpy/pull/2061
        yield cp.Problem(cp.Maximize(x), [cp.reshape(x, -1, order="F") <= np.ones(1)])
    yield cp.Problem(cp.Maximize(x), [cp.reshape(a, (), order="F") <= 1])
    yield cp.Problem(cp.Maximize(x), [cp.reshape(a, (1,), order="F") <= np.ones(1)])
    if CVXPY_VERSION >= (1, 4, 0):
        yield cp.Problem(cp.Maximize(x), [cp.reshape(a, -1, order="F") <= np.ones(1)])

    yield cp.Problem(cp.Maximize(x), [cp.vec(x) <= np.ones(1)])

    reset_id_counter()
    x = cp.Variable(1, name="x")
    a = x + 1
    yield cp.Problem(cp.Maximize(x), [cp.reshape(x, (), order="F") <= 1])
    yield cp.Problem(cp.Maximize(x), [cp.reshape(a, (), order="F") <= 1])
    yield cp.Problem(cp.Maximize(x), [cp.vec(x) <= np.ones(1)])

    reset_id_counter()
    x = cp.Variable(2, name="x")
    a = x + np.array([1, 1])
    yield cp.Problem(
        cp.Maximize(cp.sum(x)), [cp.reshape(x, (2,), order="F") <= np.ones(2)]
    )
    if CVXPY_VERSION >= (1, 4, 0):
        yield cp.Problem(
            cp.Maximize(cp.sum(x)), [cp.reshape(x, -1, order="F") <= np.ones(2)]
        )
    yield cp.Problem(
        cp.Maximize(cp.sum(x)), [cp.reshape(x, (2, 1), order="F") <= np.ones((2, 1))]
    )
    yield cp.Problem(
        cp.Maximize(cp.sum(x)), [cp.reshape(x, (1, 2), order="F") <= np.ones((1, 2))]
    )
    yield cp.Problem(
        cp.Maximize(cp.sum(x)), [cp.reshape(a, (2,), order="F") <= np.ones(2)]
    )
    if CVXPY_VERSION >= (1, 4, 0):
        yield cp.Problem(
            cp.Maximize(cp.sum(x)), [cp.reshape(a, (-1,), order="F") <= np.ones(2)]
        )
    yield cp.Problem(
        cp.Maximize(cp.sum(x)), [cp.reshape(a, (2, 1), order="F") <= np.ones((2, 1))]
    )
    yield cp.Problem(
        cp.Maximize(cp.sum(x)), [cp.reshape(a, (1, 2), order="F") <= np.ones((1, 2))]
    )
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.vec(x) <= np.arange(2)])

    reset_id_counter()
    x = cp.Variable(4, name="x")
    c = np.ones(4)
    a = x + c
    yield cp.Problem(
        cp.Maximize(cp.sum(x)), [cp.reshape(x, (4,), order="F") <= np.ones(4)]
    )
    yield cp.Problem(
        cp.Maximize(cp.sum(x)), [cp.reshape(x, (4, 1), order="F") <= np.ones((4, 1))]
    )
    yield cp.Problem(
        cp.Maximize(cp.sum(x)), [cp.reshape(x, (2, 2), order="F") <= np.ones((2, 2))]
    )
    yield cp.Problem(
        cp.Maximize(cp.sum(x)), [cp.reshape(x, (1, 4), order="F") <= np.ones((1, 4))]
    )
    yield cp.Problem(
        cp.Maximize(cp.sum(x)), [cp.reshape(a, (4,), order="F") <= np.ones(4)]
    )
    yield cp.Problem(
        cp.Maximize(cp.sum(x)), [cp.reshape(a, (4, 1), order="F") <= np.ones((4, 1))]
    )
    yield cp.Problem(
        cp.Maximize(cp.sum(x)), [cp.reshape(a, (2, 2), order="F") <= np.ones((2, 2))]
    )
    yield cp.Problem(
        cp.Maximize(cp.sum(x)), [cp.reshape(a, (1, 4), order="F") <= np.ones((1, 4))]
    )
    yield cp.Problem(
        cp.Maximize(cp.sum(x)),
        [cp.reshape(x, (2, 2), order="F") <= cp.reshape(c, (2, 2), order="F")],
    )
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.vec(x) <= np.arange(4)])


def _stack(stack_name: Literal["vstack", "hstack"]) -> Generator[cp.Problem]:
    stack = getattr(cp, stack_name)

    if stack_name == "hstack":
        # vstack does not support scalar variables
        x = cp.Variable(name="x")
        y = cp.Variable(name="y")
        yield cp.Problem(cp.Minimize(cp.sum(stack([x]))), [x >= 1])
        yield cp.Problem(cp.Minimize(cp.sum(stack([x, y, 1]))), [x >= 1, y >= 1])
        yield cp.Problem(cp.Minimize(cp.sum(stack([2 * x]))), [x >= 1])
        yield cp.Problem(
            cp.Minimize(cp.sum(stack([2 * x, 3 * y, 1]))), [x >= 1, y >= 1]
        )

    reset_id_counter()
    x = cp.Variable(1, name="x")
    y = cp.Variable(1, name="y")
    yield cp.Problem(cp.Minimize(cp.sum(stack([x]))), [x >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(stack([x, y, 1]))), [x >= 1, y >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(stack([2 * x]))), [x >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(stack([2 * x, 3 * y, 1]))), [x >= 1, y >= 1])

    reset_id_counter()
    x = cp.Variable(2, name="x")
    yield cp.Problem(cp.Minimize(cp.sum(stack([x]))), [x >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(stack([2 * x]))), [x >= 1])
    if stack_name == "hstack":
        yield cp.Problem(cp.Minimize(cp.sum(stack([x, y, 1]))), [x >= 1, y >= 1])
        yield cp.Problem(
            cp.Minimize(cp.sum(stack([2 * x, 3 * y, 1]))), [x >= 1, y >= 1]
        )

    reset_id_counter()
    x = cp.Variable((2, 2), name="x")
    y = (
        cp.Variable((1, 2), name="y")
        if stack_name == "vstack"
        else cp.Variable((2, 1), name="y")
    )
    A = np.arange(4).reshape((2, 2))
    yield cp.Problem(cp.Minimize(cp.sum(stack([x]))), [x >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(stack([x, y, A]))), [x >= 1, y >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(stack([2 * x]))), [x >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(stack([2 * x, 3 * y, A]))), [x >= 1, y >= 1])


@skipif(lambda case: case.context.solver == cp.SCIP, "TODO")
@skipif(
    lambda case: case.context.solver == cp.GUROBI and GUROBI_MAJOR < 11,
    reason="requires Gurobi 11+",
)
@group_cases("vstack")
def vstack() -> Generator[cp.Problem]:
    yield from _stack("vstack")


@skipif(lambda case: case.context.solver == cp.SCIP, "TODO")
@skipif(
    lambda case: case.context.solver == cp.GUROBI and GUROBI_MAJOR < 11,
    reason="requires Gurobi 11+",
)
@group_cases("hstack")
def hstack() -> Generator[cp.Problem]:
    yield from _stack("hstack")


@skipif(lambda case: case.context.solver == cp.SCIP, "TODO")
@skipif(
    lambda case: case.context.solver == cp.GUROBI and GUROBI_MAJOR < 12,
    reason="requires Gurobi 12+",
)
@group_cases("nonlinear_exp")
def nonlinear_exp() -> Generator[cp.Problem]:
    x = cp.Variable(name="x")
    yield cp.Problem(cp.Minimize(cp.exp(x)), [x >= 1])
    yield cp.Problem(cp.Minimize(cp.exp(x + 1)), [x >= 1])
    yield cp.Problem(cp.Maximize(x), [cp.exp(x) <= 1])

    reset_id_counter()
    x = cp.Variable(1, name="x")
    yield cp.Problem(cp.Minimize(cp.exp(x)), [x >= 1])
    yield cp.Problem(cp.Minimize(cp.exp(x + 1)), [x >= 1])
    yield cp.Problem(cp.Maximize(x), [cp.exp(x) <= 1])

    reset_id_counter()
    x = cp.Variable(2, name="x")
    t = np.array([1, 2])
    yield cp.Problem(cp.Minimize(cp.sum(cp.exp(x))), [x >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(cp.exp(x + t))), [x >= 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.exp(x) <= 1])


@skipif(lambda case: case.context.solver == cp.SCIP, "TODO")
@skipif(
    lambda case: case.context.solver == cp.GUROBI and GUROBI_MAJOR < 12,
    reason="requires Gurobi 12+",
)
@group_cases("nonlinear_log")
def nonlinear_log() -> Generator[cp.Problem]:
    x = cp.Variable(name="x")
    yield cp.Problem(cp.Maximize(cp.log(x)), [x <= 2])
    yield cp.Problem(cp.Maximize(cp.log1p(x)), [x <= 2])
    yield cp.Problem(cp.Minimize(x), [cp.log(x) >= -1])

    reset_id_counter()
    x = cp.Variable(1, name="x")
    yield cp.Problem(cp.Maximize(cp.log(x)), [x <= 0.5])
    yield cp.Problem(cp.Maximize(cp.log1p(x + 1)), [x <= -0.5])
    yield cp.Problem(cp.Minimize(x), [cp.log(x) >= 1])

    reset_id_counter()
    x = cp.Variable(2, name="x")
    yield cp.Problem(cp.Maximize(cp.sum(cp.log(x))), [x <= 1])
    yield cp.Problem(cp.Maximize(cp.sum(cp.log1p(x))), [x <= -0.5])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.log(x) >= 1])


@group_cases("attributes")
def attributes() -> Generator[cp.Problem]:
    x = cp.Variable(nonpos=True, name="x")
    yield cp.Problem(cp.Maximize(x))

    x = cp.Variable(neg=True, name="x")
    yield cp.Problem(cp.Maximize(x))

    n = cp.Variable(name="n", integer=True)
    yield cp.Problem(cp.Minimize(n), [n >= 1])

    b = cp.Variable(name="b", boolean=True)
    yield cp.Problem(cp.Minimize(b))

    yield cp.Problem(cp.Maximize(x + n + b), [n <= 1])


@skipif(lambda case: case.context.solver == cp.SCIP, "TODO")
@group_cases("invalid", invalid_reason="unsupported expressions")
def invalid_expressions() -> Generator[cp.Problem]:
    x = cp.Variable(name="x")
    v = cp.Variable(2, name="v")
    yield cp.Problem(cp.Minimize(x**3))
    yield cp.Problem(cp.Minimize(x**4))
    yield cp.Problem(cp.Maximize(cp.sqrt(x)))
    yield cp.Problem(cp.Minimize(cp.norm(v, 4)))
    yield cp.Problem(cp.Minimize(cp.norm(v, 0.5)))


@skipif(lambda case: case.context.solver == cp.SCIP, "TODO")
@skipif(
    lambda case: case.context.solver == cp.GUROBI and GUROBI_MAJOR >= 11,
    "works in Gurobi 11+",
)
@group_cases("invalid_stack", invalid_reason="unsupported stack expressions")
def invalid_stack_expressions() -> Generator[cp.Problem]:
    x = cp.Variable(name="x")
    yield cp.Problem(cp.Minimize(cp.sum(cp.hstack([x, 1]))))
    yield cp.Problem(cp.Minimize(cp.sum(cp.vstack([x, 1]))))


def reset_id_counter() -> None:
    """Reset the counter used to assign constraint and variable ids."""
    from cvxpy.lin_ops.lin_utils import ID_COUNTER  # noqa: PLC0415

    ID_COUNTER.count = 1


if __name__ == "__main__":
    list(all_problems())
