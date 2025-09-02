import cvxpy as cp
import numpy as np
import pytest

from cvxpy_translation import backfill_problem
from cvxpy_translation import build_model


def test_empty_presolved_model_scip():
    """If presolve removes the constraints, it is impossible to get their dual value."""
    x = cp.Variable()
    problem = cp.Problem(cp.Minimize(x), [x >= 0])
    model = build_model(problem, cp.SCIP)
    model.optimize()
    assert model.getNConss(transformed=True) == 0
    backfill_problem(problem, model)


def test_empty_presolved_model_grb():
    """If presolve removes the constraints, it should still be possible to get their dual value."""
    x = cp.Variable()
    problem = cp.Problem(cp.Minimize(x), [x >= 0])
    model = build_model(problem, cp.GUROBI)
    model.optimize()
    backfill_problem(problem, model)


@pytest.mark.parametrize(
    ("param", "value", "status"),
    [
        ("limits/time", 0, "timelimit"),
        ("limits/nodes", 0, "nodelimit"),
        ("limits/totalnodes", 0, "totalnodelimit"),
        ("limits/stallnodes", 0, "stallnodelimit"),
        ("limits/memory", 0, "memlimit"),
        ("limits/solutions", 0, "sollimit"),
        ("limits/bestsol", 0, "bestsollimit"),
    ],
)
def test_user_limit_scip(param: str, value: int, status: str):
    """If the user limit is reached, the problem should still be backfilled."""
    x = cp.Variable(50, boolean=True)
    # some random non-trivial problem
    problem = cp.Problem(
        cp.Minimize(cp.sum(x**2 + cp.log1p(x))), [x @ np.arange(x.size) == 13]
    )
    model = build_model(problem, cp.SCIP)
    model.setParam(param, value)
    model.optimize()
    assert model.getStatus() == status
    backfill_problem(problem, model)
    assert problem.status == cp.USER_LIMIT
