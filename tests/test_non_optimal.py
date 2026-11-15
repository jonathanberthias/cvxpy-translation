import cvxpy as cp
import gurobipy as gp
from cvxpy import settings as s

import cvxpy_translation.gurobi


def test_backfill_unbounded() -> None:
    problem = cp.Problem(cp.Maximize(cp.Variable()))
    cvxpy_translation.gurobi.solve(problem)
    assert problem.solver_stats is not None
    model = problem.solver_stats.extra_stats
    assert problem.status == s.UNBOUNDED
    assert isinstance(model, gp.Model)
    assert model.Status == gp.GRB.Status.UNBOUNDED


def test_backfill_infeasible() -> None:
    x = cp.Variable(nonneg=True)
    problem = cp.Problem(cp.Maximize(x), [x <= -1])
    cvxpy_translation.gurobi.solve(problem, DualReductions=0)
    assert problem.solver_stats is not None
    model = problem.solver_stats.extra_stats
    assert problem.status == s.INFEASIBLE
    assert isinstance(model, gp.Model)
    assert model.Status == gp.GRB.Status.INFEASIBLE
