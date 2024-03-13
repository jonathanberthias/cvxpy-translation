from __future__ import annotations

from typing import Callable
from typing import TypeAlias

import cvxpy as cp
import gurobipy as gp
import pytest

import cvxpy_gurobi
from cvxpy_gurobi import ParamDict


@pytest.fixture
def problem() -> cp.Problem:
    x = cp.Variable(name="x", pos=True)
    return cp.Problem(cp.Minimize(x), [x * x >= 1])


@pytest.fixture(params=[False, True])
def dual(request: pytest.FixtureRequest) -> bool:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def params(dual: bool) -> ParamDict | None:
    if dual:
        return {gp.GRB.Param.QCPDual: 1}
    return None


Validator: TypeAlias = Callable[[cp.Problem], None]


@pytest.fixture
def validate(dual: bool) -> Validator:
    def validator(problem: cp.Problem) -> None:
        assert problem.value == 1.0
        assert problem.var_dict["x"].value == 1.0
        assert problem.status == cp.OPTIMAL
        dual_value = problem.constraints[0].dual_value
        if dual:
            assert dual_value is not None
        else:
            assert dual_value is None

    return validator


def test_registered_solver(
    problem: cp.Problem, validate: Validator, params: ParamDict | None
) -> None:
    cvxpy_gurobi.register_solver(params=params)
    problem.solve(method=cvxpy_gurobi.NATIVE_GUROBI)
    validate(problem)


def test_direct_solve(
    problem: cp.Problem, validate: Validator, params: ParamDict | None
) -> None:
    cvxpy_gurobi.solve(problem, params=params)
    validate(problem)


def test_manual(
    problem: cp.Problem, validate: Validator, params: ParamDict | None
) -> None:
    model = cvxpy_gurobi.build_model(problem, params=params)
    model.optimize()
    cvxpy_gurobi.backfill_problem(problem, model)
    validate(problem)


def test_manual_with_env(
    problem: cp.Problem, validate: Validator, params: ParamDict | None
) -> None:
    env = gp.Env()
    model = cvxpy_gurobi.build_model(problem, env=env, params=params)
    model.optimize()
    cvxpy_gurobi.backfill_problem(problem, model)
    validate(problem)


def test_granular(
    problem: cp.Problem, validate: Validator, params: ParamDict | None
) -> None:
    model = gp.Model()
    var_map = cvxpy_gurobi.map_variables(problem, model)
    cvxpy_gurobi.fill_model(problem, model, var_map)
    if params:
        cvxpy_gurobi.set_params(model, params=params)
    model.optimize()
    cvxpy_gurobi.backfill_problem(problem, model)
    validate(problem)
