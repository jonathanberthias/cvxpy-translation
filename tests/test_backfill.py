import cvxpy as cp

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
