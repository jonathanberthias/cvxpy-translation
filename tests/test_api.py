from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING
from typing import Callable

import cvxpy as cp
import gurobipy as gp
import pytest

import cvxpy_gurobi
from cvxpy_gurobi import CVXPY_VERSION

if TYPE_CHECKING:
    from typing import TypeAlias

    from cvxpy_gurobi import ParamDict


@pytest.fixture
def problem() -> cp.Problem:
    x = cp.Variable(name="x", pos=True)
    return cp.Problem(cp.Minimize(x), [x * x >= 1])


@pytest.fixture(params=[False, True])
def dual(request: pytest.FixtureRequest) -> bool:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def params(dual: bool) -> ParamDict:
    if dual:
        return {gp.GRB.Param.QCPDual: 1}
    return {}


Validator: TypeAlias = Callable[[cp.Problem], None]


def validate(problem: cp.Problem, *, dual: bool) -> None:
    assert problem.value == 1.0
    x = next(v for v in problem.variables() if v.name() == "x")
    assert x.value == 1.0
    assert problem.status == cp.OPTIMAL
    assert problem.solver_stats is not None
    assert problem.solver_stats.solve_time is not None
    assert problem.solver_stats.solver_name == cvxpy_gurobi.GUROBI_TRANSLATION
    assert isinstance(problem.solver_stats.extra_stats, gp.Model)
    if CVXPY_VERSION >= (1, 4):  # didn't exist before
        assert problem.compilation_time is not None
    dual_value = problem.constraints[0].dual_value
    if dual:
        assert dual_value is not None
    else:
        assert dual_value is None


@pytest.fixture(name="validate")
def _validate(dual: bool) -> Validator:
    return partial(validate, dual=dual)


def test_registered_solver(
    problem: cp.Problem, validate: Validator, params: ParamDict
) -> None:
    cvxpy_gurobi.register_solver()
    problem.solve(method=cvxpy_gurobi.GUROBI_TRANSLATION, **params)
    validate(problem)


def test_registered_solver_with_env(
    problem: cp.Problem, validate: Validator, params: ParamDict
) -> None:
    env = gp.Env(params=params)
    cvxpy_gurobi.register_solver()
    problem.solve(method=cvxpy_gurobi.GUROBI_TRANSLATION, env=env)
    validate(problem)


def test_direct_solve(
    problem: cp.Problem, validate: Validator, params: ParamDict
) -> None:
    cvxpy_gurobi.solve(problem, **params)
    validate(problem)


def test_direct_solve_with_env(
    problem: cp.Problem, validate: Validator, params: ParamDict
) -> None:
    env = gp.Env(params=params)
    cvxpy_gurobi.solve(problem, env=env)
    validate(problem)


def test_manual(problem: cp.Problem, validate: Validator, params: ParamDict) -> None:
    model = cvxpy_gurobi.build_model(problem, params=params)
    model.optimize()
    cvxpy_gurobi.backfill_problem(problem, model, compilation_time=1.0, solve_time=1.0)
    validate(problem)


def test_manual_with_env(
    problem: cp.Problem, validate: Validator, params: ParamDict
) -> None:
    env = gp.Env(params=params)
    model = cvxpy_gurobi.build_model(problem, env=env)
    model.optimize()
    cvxpy_gurobi.backfill_problem(problem, model, compilation_time=1.0, solve_time=1.0)
    validate(problem)


def test_readme_example():
    problem = cp.Problem(cp.Maximize(cp.Variable(name="x", nonpos=True)))
    cvxpy_gurobi.solve(problem)
    assert problem.value == 0
