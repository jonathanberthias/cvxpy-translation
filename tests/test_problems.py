from __future__ import annotations

from dataclasses import dataclass
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
        indexing,
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
    yield cp.Problem(cp.Minimize(x**3))
    yield cp.Problem(cp.Minimize(x**4))
    # TODO: maybe using setPWLObj?
    yield cp.Problem(cp.Minimize(cp.abs(x)))
    yield cp.Problem(cp.Maximize(cp.sqrt(x)))


def reset_id_counter() -> None:
    """Reset the counter used to assign constraint ids."""
    from cvxpy.lin_ops.lin_utils import ID_COUNTER

    ID_COUNTER.count = 1
