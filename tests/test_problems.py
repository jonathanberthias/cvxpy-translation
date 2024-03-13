from itertools import chain
from typing import Iterator

import cvxpy as cp
import numpy as np
import scipy.sparse as sp


def all_valid_problems() -> Iterator[cp.Problem]:
    yield from chain(
        simple_expressions(),
        scalar_linear_constraints(),
        quadratic_expressions(),
        matrix_constraints(),
        matrix_quadratic_expressions(),
        attributes(),
    )


def all_problems() -> Iterator[cp.Problem]:
    yield from all_valid_problems()
    yield from invalid_expressions()


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


def invalid_expressions() -> Iterator[cp.Problem]:
    x = cp.Variable(name="x")
    yield cp.Problem(cp.Minimize(x**3))
    yield cp.Problem(cp.Minimize(x**4))
    # TODO: maybe using setPWLObj?
    yield cp.Problem(cp.Minimize(cp.abs(x)))
    yield cp.Problem(cp.Maximize(cp.sqrt(x)))
