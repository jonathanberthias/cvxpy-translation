from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from functools import wraps
from typing import Callable
from typing import Iterator

import cvxpy as cp
import numpy as np
import scipy.sparse as sp


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


def all_problems() -> Iterator[ProblemTestCase]:
    for problem_gen in (
        simple_expressions,
        scalar_linear_constraints,
        quadratic_expressions,
        matrix_constraints,
        matrix_quadratic_expressions,
        genexpr_abs,
        genexpr_min_max,
        genexpr_minimum_maximum,
        genexpr_norm1,
        genexpr_norm2,
        genexpr_norminf,
        indexing,
        sum_axis,
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
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    yield cp.Problem(cp.Minimize(x))
    yield cp.Problem(cp.Minimize(x + 1))
    yield cp.Problem(cp.Minimize(x + x))
    yield cp.Problem(cp.Minimize(x + y))

    yield cp.Problem(cp.Minimize(x - x))
    yield cp.Problem(cp.Minimize(x - 1))
    yield cp.Problem(cp.Minimize(x - y))

    yield cp.Problem(cp.Minimize(2 * x))
    yield cp.Problem(cp.Minimize(2 * x + 1))
    yield cp.Problem(cp.Minimize(2 * x + y))

    yield cp.Problem(cp.Minimize(-x))
    yield cp.Problem(cp.Minimize(-x + 1))
    yield cp.Problem(cp.Minimize(1 - x))

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

    yield cp.Problem(cp.Minimize(x), [x + y >= 1])
    yield cp.Problem(cp.Minimize(x), [x + y <= 1])
    yield cp.Problem(cp.Minimize(x), [x + y == 1])

    yield cp.Problem(cp.Minimize(x), [2 * x >= 1])
    yield cp.Problem(cp.Minimize(x), [2 * x + y >= 1])


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
    yield cp.Problem(cp.Minimize(cp.sum(x)), [x + y >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [x + y + 1 >= 0])

    yield cp.Problem(cp.Minimize(cp.sum(x)), [A @ x == 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [A @ x + y == 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [A @ x + y + 1 == 0])

    yield cp.Problem(cp.Minimize(cp.sum(x)), [S @ x == 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [S @ x + y == 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [S @ x + y + 1 == 0])


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
    yield cp.Problem(cp.Minimize((x - y) ** 2 + x + y))


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


@group_cases("genexpr_abs")
def genexpr_abs() -> Iterator[cp.Problem]:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    yield cp.Problem(cp.Minimize(cp.abs(x)))
    yield cp.Problem(cp.Minimize(cp.abs(x) + 1))
    yield cp.Problem(cp.Minimize(cp.abs(x) + cp.abs(y)))
    yield cp.Problem(cp.Minimize(cp.abs(x + y)))

    x = cp.Variable(1, name="X")
    y = cp.Variable(name="Y")

    yield cp.Problem(cp.Minimize(cp.abs(x)))
    yield cp.Problem(cp.Minimize(cp.abs(x) + 1))
    yield cp.Problem(cp.Minimize(cp.abs(x) + cp.abs(y)))
    yield cp.Problem(cp.Minimize(cp.abs(x + y)))

    x = cp.Variable(2, name="X", nonneg=True)
    y = cp.Variable(2, name="Y", nonneg=True)
    A = np.array([1, -2])

    yield cp.Problem(cp.Minimize(cp.sum(cp.abs(x))))
    yield cp.Problem(cp.Minimize(cp.sum(cp.abs(x + y))))
    yield cp.Problem(cp.Minimize(cp.sum(cp.abs(x) + 1)))
    yield cp.Problem(cp.Minimize(cp.sum(cp.abs(x) + A)))
    yield cp.Problem(cp.Minimize(cp.sum(cp.abs(x) + cp.abs(y))))
    yield cp.Problem(cp.Minimize(cp.sum(cp.abs(x) + cp.abs(A))))

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

    yield cp.Problem(cp.Minimize(cp.max(x)), [x >= 1])
    yield cp.Problem(cp.Minimize(cp.max(x) + 1), [x >= 1])
    yield cp.Problem(cp.Minimize(cp.max(x) + cp.max(y)), [x >= 1, y >= 1])
    yield cp.Problem(cp.Minimize(cp.max(x + y)), [x >= 1, y >= 1])

    x = cp.Variable(1, name="x")
    y = cp.Variable(name="y")

    yield cp.Problem(cp.Maximize(cp.min(x)), [x <= 1])
    yield cp.Problem(cp.Maximize(cp.min(x) + 1), [x <= 1])
    yield cp.Problem(cp.Maximize(cp.min(x) + cp.min(y)), [x <= 1, y <= 1])
    yield cp.Problem(cp.Maximize(cp.min(x + y)), [x <= 1, y <= 1])

    yield cp.Problem(cp.Minimize(cp.max(x)), [x >= 1])
    yield cp.Problem(cp.Minimize(cp.max(x) + 1), [x >= 1])
    yield cp.Problem(cp.Minimize(cp.max(x) + cp.max(y)), [x >= 1, y >= 1])
    yield cp.Problem(cp.Minimize(cp.max(x + y)), [x >= 1, y >= 1])

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

    yield cp.Problem(cp.Minimize(cp.max(x)), [x >= 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.max(x) <= 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.max(x) + 1 <= 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.max(x) + cp.max(y) <= 1, y == 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.max(x + y) <= 1, y == 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.max(x + A) <= 1])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.max(x) + cp.max(A) <= 1])

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

    yield cp.Problem(cp.Maximize(x), [cp.maximum(x, 1) <= 2])
    yield cp.Problem(cp.Maximize(x + y), [cp.maximum(x, y) <= 1])
    yield cp.Problem(cp.Maximize(x + y), [cp.maximum(x, y, 1) <= 3])
    yield cp.Problem(cp.Maximize(x), [cp.maximum(1, 2) <= 2, x <= 0])

    x = cp.Variable(1, name="x")
    y = cp.Variable(name="y")

    yield cp.Problem(cp.Minimize(x), [cp.minimum(x, 2) >= 1])
    yield cp.Problem(cp.Minimize(x + y), [cp.minimum(x, y) >= 1])
    yield cp.Problem(cp.Minimize(x + y), [cp.minimum(x, y, 3) >= 1])
    yield cp.Problem(cp.Minimize(x), [cp.minimum(1, 2) <= 1, x >= 0])

    yield cp.Problem(cp.Maximize(x), [cp.maximum(x, 1) <= 2])
    yield cp.Problem(cp.Maximize(x + y), [cp.maximum(x, y) <= 1])
    yield cp.Problem(cp.Maximize(x + y), [cp.maximum(x, y, 1) <= 3])
    yield cp.Problem(cp.Maximize(x), [cp.maximum(1, 2) <= 2, x <= 0])

    x = cp.Variable(2, name="X")
    y = cp.Variable(2, name="Y")
    A = np.array([2, -1])

    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.minimum(x, A) >= -1])
    yield cp.Problem(cp.Minimize(cp.sum(x + y)), [cp.minimum(x, y) >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(x + y)), [cp.minimum(x, y) >= A])
    yield cp.Problem(cp.Minimize(cp.sum(x + y)), [cp.minimum(x, y, A) >= -1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.minimum(x, A) >= -y, y == 1])

    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.maximum(x, A) <= 2])
    yield cp.Problem(cp.Maximize(cp.sum(x + y)), [cp.maximum(x, y) <= 1])
    yield cp.Problem(cp.Maximize(cp.sum(x + y)), [cp.maximum(x, y) <= A])
    yield cp.Problem(cp.Maximize(cp.sum(x + y)), [cp.maximum(x, y, A) <= 2])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.maximum(x, A) <= y, y == 2])

    x = cp.Variable((2, 2), name="X")
    y = cp.Variable((2, 2), name="Y")
    A = np.array([[1, -2], [3, 4]])
    B = np.array([[2, -1], [1, 3]])

    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.minimum(x, A) >= -1])
    yield cp.Problem(cp.Minimize(cp.sum(x + y)), [cp.minimum(x, y) >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(x + y)), [cp.minimum(x, y) >= A])
    yield cp.Problem(cp.Minimize(cp.sum(x + y)), [cp.minimum(x, y, A) >= -1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.minimum(x, A) >= -y, y == 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.minimum(A, B) >= -2, x == 1])

    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.maximum(x, A) <= 2])
    yield cp.Problem(cp.Maximize(cp.sum(x + y)), [cp.maximum(x, y) <= 1])
    yield cp.Problem(cp.Maximize(cp.sum(x + y)), [cp.maximum(x, y) <= A])
    yield cp.Problem(cp.Maximize(cp.sum(x + y)), [cp.maximum(x, y, A) <= 2])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.maximum(x, A) <= y, y == 2])
    yield cp.Problem(cp.Maximize(cp.sum(x)), [cp.maximum(A, B) <= 4, x == 1])


def _genexpr_norm_problems(
    norm: Callable[[cp.Expression | float | np.ndarray], cp.Expression],
) -> Iterator[cp.Problem]:
    x = cp.Variable(name="x")
    yield cp.Problem(cp.Minimize(norm(x)))
    yield cp.Problem(cp.Minimize(norm(x - 1)))
    yield cp.Problem(cp.Minimize(norm(x) + norm(-1)))
    yield cp.Problem(cp.Maximize(x), [norm(x) <= 1])

    x = cp.Variable(1, name="x")
    yield cp.Problem(cp.Minimize(norm(x)))
    yield cp.Problem(cp.Minimize(norm(x - 1)))
    yield cp.Problem(cp.Minimize(norm(x) + norm(-1)))
    yield cp.Problem(cp.Maximize(x), [norm(x) <= 1])

    x = cp.Variable(2, name="x")
    A = np.array([1, -1])
    yield cp.Problem(cp.Minimize(norm(x)))
    yield cp.Problem(cp.Minimize(norm(x - A)))
    yield cp.Problem(cp.Minimize(norm(x) + norm(A)))
    yield cp.Problem(cp.Maximize(cp.sum(cp.multiply(x, A))), [norm(x) <= np.sqrt(2)])

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

    idx = np.array([0])
    yield cp.Problem(cp.Minimize(cp.sum(x[idx])))
    yield cp.Problem(cp.Minimize(cp.sum(y[idx])))
    yield cp.Problem(cp.Minimize(cp.sum(m[idx])))
    yield cp.Problem(cp.Minimize(cp.sum(m[idx, :])))

    idx = np.array([True, False])
    yield cp.Problem(cp.Minimize(cp.sum(x[idx])))
    yield cp.Problem(cp.Minimize(cp.sum(y[idx])))
    yield cp.Problem(cp.Minimize(cp.sum(m[idx])))
    yield cp.Problem(cp.Minimize(cp.sum(m[idx, :])))

    yield cp.Problem(cp.Minimize(cp.sum(x[:])))
    yield cp.Problem(cp.Minimize(cp.sum(y[:])))
    yield cp.Problem(cp.Minimize(cp.sum(m[:, :])))


@group_cases("sum_axis")
def sum_axis() -> Iterator[cp.Problem]:
    x = cp.Variable((2, 2), name="x", nonneg=True)

    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.sum(x, axis=None) >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.sum(x, axis=0) >= 1])
    yield cp.Problem(cp.Minimize(cp.sum(x)), [cp.sum(x, axis=1) >= 1])


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

    n = cp.Variable(name="n", integer=True)
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


def reset_id_counter() -> None:
    """Reset the counter used to assign constraint ids."""
    from cvxpy.lin_ops.lin_utils import ID_COUNTER

    ID_COUNTER.count = 1
