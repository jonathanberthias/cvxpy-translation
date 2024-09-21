import cvxpy as cp
import gurobipy as gp
from cvxpy import settings as s

import cvxpy_gurobi


def test_backfill_unbounded() -> None:
    problem = cp.Problem(cp.Maximize(cp.Variable()))
    cvxpy_gurobi.solve(problem)
    model = problem.solver_stats.extra_stats
    assert model.Status == gp.GRB.Status.UNBOUNDED
    assert problem.status == s.UNBOUNDED


def test_backfill_infeasible() -> None:
    x = cp.Variable(nonneg=True)
    problem = cp.Problem(cp.Maximize(x), [x <= -1])
    cvxpy_gurobi.solve(problem, **{gp.GRB.Param.DualReductions: 0})
    model = problem.solver_stats.extra_stats
    assert model.Status == gp.GRB.Status.INFEASIBLE
    assert problem.status == s.INFEASIBLE
