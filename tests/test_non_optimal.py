import cvxpy as cp
import gurobipy as gp
from cvxpy import settings as s

import cvxpy_gurobi
from cvxpy_gurobi.translation import CVXPY_VERSION


def test_backfill_unbounded() -> None:
    problem = cp.Problem(cp.Maximize(cp.Variable()))
    cvxpy_gurobi.solve(problem)
    model = problem.solver_stats.extra_stats
    assert problem.status == s.UNBOUNDED
    if CVXPY_VERSION >= (1, 2, 0):
        assert model.Status == gp.GRB.Status.UNBOUNDED


def test_backfill_infeasible() -> None:
    x = cp.Variable(nonneg=True)
    problem = cp.Problem(cp.Maximize(x), [x <= -1])
    cvxpy_gurobi.solve(problem, **{gp.GRB.Param.DualReductions: 0})
    model = problem.solver_stats.extra_stats
    assert problem.status == s.INFEASIBLE
    if CVXPY_VERSION >= (1, 2, 0):
        assert model.Status == gp.GRB.Status.INFEASIBLE
