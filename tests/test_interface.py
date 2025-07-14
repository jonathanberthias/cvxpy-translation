from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING
from typing import Callable

import cvxpy as cp
import gurobipy as gp
import pyscipopt as scip
import pytest

import cvxpy_translation.gurobi
import cvxpy_translation.scip
from cvxpy_translation import CVXPY_VERSION

if TYPE_CHECKING:
    from typing_extensions import Generator
    from typing_extensions import TypeAlias

    from cvxpy_translation.interface import ParamDict


@pytest.fixture
def problem() -> cp.Problem:
    x = cp.Variable(name="x", pos=True)
    return cp.Problem(cp.Minimize(x), [x * x >= 1])


@pytest.fixture(params=[cp.GUROBI, cp.SCIP])
def solver(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=[False, True])
def dual(request: pytest.FixtureRequest) -> bool:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def params(dual: bool, solver: str) -> ParamDict:
    if dual and solver == cp.GUROBI:
        return {gp.GRB.Param.QCPDual: 1}
    return {}


Validator: TypeAlias = Callable[[cp.Problem], None]


def validate(problem: cp.Problem, solver: str, *, dual: bool) -> None:
    assert problem.value == pytest.approx(1.0)
    x = next(v for v in problem.variables() if v.name() == "x")
    assert x.value == pytest.approx(1.0)
    assert problem.status == cp.OPTIMAL
    assert problem.solver_stats is not None
    assert problem.solver_stats.solve_time is not None
    name = (
        cvxpy_translation.gurobi.GUROBI_TRANSLATION
        if solver == cp.GUROBI
        else cvxpy_translation.scip.SCIP_TRANSLATION
    )
    assert problem.solver_stats.solver_name == name
    assert isinstance(
        problem.solver_stats.extra_stats,
        gp.Model if solver == cp.GUROBI else scip.Model,
    )
    if CVXPY_VERSION >= (1, 4):  # didn't exist before
        assert problem.compilation_time is not None
    dual_value = problem.constraints[0].dual_value
    if dual and solver == cp.GUROBI:
        assert dual_value is not None
    else:
        # SCIP does not report dual values for quadratic constraints
        assert dual_value is None


@pytest.fixture(name="validate")
def _validate(dual: bool, solver: str) -> Validator:
    return partial(validate, dual=dual, solver=solver)


@contextmanager
def register_solver(solver: str) -> Generator[str]:
    method = cvxpy_translation.register_translation_solver(solver)
    try:
        yield method
    finally:
        del cp.Problem.REGISTERED_SOLVE_METHODS[method]


def test_registered_solver(
    problem: cp.Problem, solver: str, validate: Validator, params: ParamDict
) -> None:
    with register_solver(solver) as method:
        problem.solve(method=method, **params)
    validate(problem)


def test_registered_solver_with_env(
    problem: cp.Problem, solver: str, validate: Validator, params: ParamDict
) -> None:
    if solver != cp.GUROBI:
        pytest.skip("Only GUROBI supports env parameter")
    env = gp.Env(params=params)
    with register_solver(solver) as method:
        problem.solve(method=method, env=env)
    validate(problem)


def test_direct_solve(
    problem: cp.Problem, solver: str, validate: Validator, params: ParamDict
) -> None:
    cvxpy_translation.solve(problem, solver, **params)
    validate(problem)


def test_direct_solve_with_env(
    problem: cp.Problem, solver: str, validate: Validator, params: ParamDict
) -> None:
    if solver != cp.GUROBI:
        pytest.skip("Only GUROBI supports env parameter")
    env = gp.Env(params=params)
    cvxpy_translation.gurobi.solve(problem, env=env)
    validate(problem)


def test_manual(
    problem: cp.Problem, solver: str, validate: Validator, params: ParamDict
) -> None:
    model = cvxpy_translation.build_model(problem, solver, params=params)
    model.optimize()
    cvxpy_translation.backfill_problem(
        problem, model, compilation_time=1.0, solve_time=1.0
    )
    validate(problem)


def test_manual_with_env(
    problem: cp.Problem, solver: str, validate: Validator, params: ParamDict
) -> None:
    if solver != cp.GUROBI:
        pytest.skip("Only GUROBI supports env parameter")
    env = gp.Env(params=params)
    model = cvxpy_translation.gurobi.build_model(problem, env=env)
    model.optimize()
    cvxpy_translation.gurobi.backfill_problem(
        problem, model, compilation_time=1.0, solve_time=1.0
    )
    validate(problem)


def test_readme_example(solver: str) -> None:
    problem = cp.Problem(cp.Maximize(cp.Variable(name="x", nonpos=True)))
    cvxpy_translation.solve(problem, solver)
    assert problem.value == 0
