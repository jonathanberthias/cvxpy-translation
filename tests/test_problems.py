from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from functools import wraps
from typing import Callable
from typing import Generator
from typing import Iterator
from typing import Literal

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from cvxpy_gurobi.translation import CVXPY_VERSION
from cvxpy_gurobi.translation import GUROBIPY_VERSION


@dataclass(frozen=True)
class ProblemTestCase:
    problem: cp.Problem
    group: str
    invalid_reason: str | None = None


def group_cases(
    group: str, *, invalid_reason: str | None = None
) -> Callable[
    [Callable[[], Iterator[cp.Problem]]], Callable[[], Iterator[ProblemTestCase]]
]:
    def dec(
        iter_fn: Callable[[], Iterator[cp.Problem]],
    ) -> Callable[[], Iterator[ProblemTestCase]]:
        @wraps(iter_fn)
        def inner() -> Iterator[ProblemTestCase]:
            for problem in iter_fn():
                yield ProblemTestCase(problem, group, invalid_reason=invalid_reason)

        return inner

    return dec


def all_problems() -> Generator[ProblemTestCase, None, None]:
    problem_gen: Callable[[], Generator[ProblemTestCase, None, None]]
    for problem_gen in (
        simple_expressions,
        scalar_linear_constraints,
        quadratic_expressions,
        matrix_constraints,
        matrix_quadratic_expressions,
        quad_form,
        genexpr_abs,
        genexpr_min_max,
        genexpr_minimum_maximum,
        genexpr_norm1,
        genexpr_norm2,
        genexpr_norminf,
        indexing,
        sum_scalar,
        sum_axis,
        reshape,
        hstack if GUROBIPY_VERSION >= (11,) else lambda: iter(()),
        vstack if GUROBIPY_VERSION >= (11,) else lambda: iter(()),
        attributes,
        invalid_expressions,
    ):
        # Make sure order of groups does not matter
        reset_id_counter()
        yield from problem_gen()


def all_valid_problems() -> Iterator[ProblemTestCase]:
    yield from (case for case in all_problems() if not case.invalid_reason)


@group_cases("simple")
def simple_expressions() -> Iterator[cp.Problem]:
    x = cp.Variable(name="x", nonneg=True)
    y = cp.Variable(name="y", nonneg=True)

    yield cp.Problem(cp.Minimize(x))
    yield cp.Problem(cp.Minimize(x + 1))
    yield cp.Problem(cp.Minimize(x + x))
    yield cp.Problem(cp.Minimize(x + y))

    yield cp.Problem(cp.Minimize(x - x))
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
def scalar_linear_constraints() -> Iterator[cp.Problem]:
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


@group_cases("matrix")
def matrix_constraints() -> Iterator[cp.Problem]:
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
def quadratic_expressions() -> Iterator[cp.Problem]:
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


@group_cases("matrix_quadratic")
def matrix_quadratic_expressions() -> Iterator[cp.Problem]:
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
def quad_form() -> Iterator[cp.Problem]:
    x = cp.Variable((1,), name="x")
    A = np.array([[1]])
    yield cp.Problem(cp.Minimize(cp.quad_form(x, A)))

    x = cp.Variable(2, name="x")
    y = np.array([1, 2])
    _A = np.arange(4).reshape((2, 2))
    A = _A.T @ _A
    yield cp.Problem(cp.Minimize(cp.quad_form(x, A)))
    if GUROBIPY_VERSION >= (11,):
        yield cp.Problem(cp.Minimize(cp.quad_form(cp.hstack([x, y]), np.eye(4))))

    x = np.arange(1, 3)
    # Can't constrain to PSD=True because Gurobi does not solve PSD problems
    A = cp.Variable((2, 2), name="A", nonneg=True)
    yield cp.Problem(cp.Minimize(cp.quad_form(x, A)))


@group_cases("genexpr_abs")
def genexpr_abs() -> Iterator[cp.Problem]:
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


@group_cases("genexpr_min_max")
def genexpr_min_max() -> Iterator[cp.Problem]:
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


@group_cases("genexpr_minimum_maximum")
def genexpr_minimum_maximum() -> Iterator[cp.Problem]:
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
) -> Iterator[cp.Problem]:
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


@group_cases("genexpr_norm1")
def genexpr_norm1() -> Iterator[cp.Problem]:
    yield from _genexpr_norm_problems(cp.norm1)


@group_cases("genexpr_norm2")
def genexpr_norm2() -> Iterator[cp.Problem]:
    # we use pnorm(p=2) as the norm2 function will automatically
    # use matrix norms but Gurobi only handles vector norms
    # pnorm will only create vector norms
    yield from _genexpr_norm_problems(partial(cp.pnorm, p=2))


@group_cases("genexpr_norminf")
def genexpr_norminf() -> Iterator[cp.Problem]:
    yield from _genexpr_norm_problems(cp.norm_inf)


@group_cases("indexing")
def indexing() -> Iterator[cp.Problem]:
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
def sum_scalar() -> Iterator[cp.Problem]:
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
def sum_axis() -> Iterator[cp.Problem]:
    x = cp.Variable((2, 2), name="x", nonneg=True)

    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.sum(x, axis=None) >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.sum(x, axis=0) >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.sum(x, axis=1) >= 1])


@group_cases("reshape")
def reshape() -> Iterator[cp.Problem]:
    x = cp.Variable(name="x")
    a = x + 1
    yield cp.Problem(cp.Maximize(x), [cp.reshape(x, ()) <= 1])
    yield cp.Problem(cp.Maximize(x), [cp.reshape(x, 1) <= np.ones(1)])
    if CVXPY_VERSION >= (1, 4, 0):
        # -1 support added in https://github.com/cvxpy/cvxpy/pull/2061
        yield cp.Problem(cp.Maximize(x), [cp.reshape(x, -1) <= np.ones(1)])
    yield cp.Problem(cp.Maximize(x), [cp.reshape(a, ()) <= 1])
    yield cp.Problem(cp.Maximize(x), [cp.reshape(a, (1,)) <= np.ones(1)])
    if CVXPY_VERSION >= (1, 4, 0):
        yield cp.Problem(cp.Maximize(x), [cp.reshape(a, -1) <= np.ones(1)])

    yield cp.Problem(cp.Maximize(x), [cp.vec(x) <= np.ones(1)])

    reset_id_counter()
    x = cp.Variable(1, name="x")
    a = x + 1
    yield cp.Problem(cp.Maximize(x), [cp.reshape(x, ()) <= 1])
    yield cp.Problem(cp.Maximize(x), [cp.reshape(a, ()) <= 1])
    yield cp.Problem(cp.Maximize(x), [cp.vec(x) <= np.ones(1)])

    reset_id_counter()
    x = cp.Variable(2, name="x")
    a = x + np.array([1, 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.reshape(x, (2,)) <= np.ones(2)])
    if CVXPY_VERSION >= (1, 4, 0):
        yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.reshape(x, -1) <= np.ones(2)])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.reshape(x, (2, 1)) <= np.ones((2, 1))])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.reshape(x, (1, 2)) <= np.ones((1, 2))])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.reshape(a, (2,)) <= np.ones(2)])
    if CVXPY_VERSION >= (1, 4, 0):
        yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.reshape(a, (-1,)) <= np.ones(2)])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.reshape(a, (2, 1)) <= np.ones((2, 1))])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.reshape(a, (1, 2)) <= np.ones((1, 2))])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.vec(x) <= np.arange(2)])

    reset_id_counter()
    x = cp.Variable(4, name="x")
    c = np.ones(4)
    a = x + c
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.reshape(x, (4,)) <= np.ones(4)])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.reshape(x, (4, 1)) <= np.ones((4, 1))])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.reshape(x, (2, 2)) <= np.ones((2, 2))])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.reshape(x, (1, 4)) <= np.ones((1, 4))])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.reshape(a, (4,)) <= np.ones(4)])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.reshape(a, (4, 1)) <= np.ones((4, 1))])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.reshape(a, (2, 2)) <= np.ones((2, 2))])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.reshape(a, (1, 4)) <= np.ones((1, 4))])
    yield cp.Problem(
        cp.Maximize(cp.sum(x)), [cp.reshape(x, (2, 2)) <= cp.reshape(c, (2, 2))]
    )
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.vec(x) <= np.arange(4)])


def _stack(stack_name: Literal["vstack", "hstack"]) -> Iterator[cp.Problem]:
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


@group_cases("vstack")
def vstack() -> Iterator[cp.Problem]:
    yield from _stack("vstack")


@group_cases("hstack")
def hstack() -> Iterator[cp.Problem]:
    yield from _stack("hstack")


@group_cases("attributes")
def attributes() -> Iterator[cp.Problem]:
    x = cp.Variable(nonpos=True, name="x")
    yield cp.Problem(cp.Maximize(x))

    x = cp.Variable(neg=True, name="x")
    yield cp.Problem(cp.Maximize(x))

    n = cp.Variable(name="n", integer=True)
    yield cp.Problem(cp.Minimize(n), [n >= 1])

    b = cp.Variable(name="b", boolean=True)
    yield cp.Problem(cp.Minimize(b))

    yield cp.Problem(cp.Maximize(x + n + b), [n <= 1])


@group_cases("invalid", invalid_reason="unsupported expressions")
def invalid_expressions() -> Iterator[cp.Problem]:
    x = cp.Variable(name="x")
    v = cp.Variable(2, name="v")
    yield cp.Problem(cp.Minimize(x**3))
    yield cp.Problem(cp.Minimize(x**4))
    yield cp.Problem(cp.Maximize(cp.sqrt(x)))
    yield cp.Problem(cp.Minimize(cp.norm(v, 4)))
    yield cp.Problem(cp.Minimize(cp.norm(v, 0.5)))
    if GUROBIPY_VERSION < (11,):
        yield cp.Problem(cp.Minimize(cp.sum(cp.hstack([x, 1]))))
        yield cp.Problem(cp.Minimize(cp.sum(cp.vstack([x, 1]))))


def reset_id_counter() -> None:
    """Reset the counter used to assign constraint and variable ids."""
    from cvxpy.lin_ops.lin_utils import ID_COUNTER

    ID_COUNTER.count = 1


if __name__ == "__main__":
    list(all_problems())
