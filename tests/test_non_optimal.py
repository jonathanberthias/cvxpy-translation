import math

import cvxpy as cp
import gurobipy as gp
import pytest
from cvxpy import settings as s

import cvxpy_translation.gurobi
from cvxpy_translation import solve


@pytest.mark.parametrize(
    ("solver", "limit"), [(cp.GUROBI, "TimeLimit"), (cp.SCIP, "limits/time")]
)
def test_solve_time_limit_without_incumbent(solver: str, limit: str) -> None:
    x = cp.Variable()
    problem = cp.Problem(cp.Minimize(x), [x >= 1])

    with pytest.raises(
        cp.SolverError, match=rf"(?i){solver} returned no objective value.*user_limit"
    ):
        solve(problem, solver, **{limit: 0})


def test_solve_infeasible_or_unbounded() -> None:
    x = cp.Variable()
    y = cp.Variable()
    problem = cp.Problem(cp.Maximize(x), [y >= 1, y <= 0])

    with pytest.raises(
        cp.SolverError,
        match=r"Gurobi returned no objective value.*infeasible_or_unbounded",
    ):
        cvxpy_translation.gurobi.solve(problem)

    assert problem.status == s.INFEASIBLE_OR_UNBOUNDED
    assert problem.value is None


def test_backfill_unbounded() -> None:
    problem = cp.Problem(cp.Maximize(cp.Variable()))
    value = cvxpy_translation.gurobi.solve(problem)
    assert math.isinf(value)
    assert problem.solver_stats is not None
    model = problem.solver_stats.extra_stats
    assert problem.status == s.UNBOUNDED
    assert isinstance(model, gp.Model)
    assert model.Status == gp.GRB.Status.UNBOUNDED


def test_backfill_infeasible() -> None:
    x = cp.Variable(nonneg=True)
    problem = cp.Problem(cp.Maximize(x), [x <= -1])
    value = cvxpy_translation.gurobi.solve(problem, DualReductions=0)
    assert math.isinf(value)
    assert problem.solver_stats is not None
    model = problem.solver_stats.extra_stats
    assert problem.status == s.INFEASIBLE
    assert isinstance(model, gp.Model)
    assert model.Status == gp.GRB.Status.INFEASIBLE
