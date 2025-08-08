from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import cvxpy as cp

try:
    import gurobipy as gp

    gurobi_available = True
except ImportError:
    gurobi_available = False

try:
    import pyscipopt as scip

    scip_available = True
except ImportError:
    scip_available = False


if TYPE_CHECKING:
    from typing_extensions import TypeAlias


Param: TypeAlias = Any
ParamDict: TypeAlias = dict[str, Param]


def solve(problem: cp.Problem, solver: str, **params: Param) -> float:
    """Solve a CVXPY problem by translating it into a solver's model.

    This function can be used to solve CVXPY problems without registering the custom solver:
        cvxpy_translation.solve(problem, solver=cp.GUROBI)
    """
    if solver == cp.GUROBI:
        from cvxpy_translation.gurobi import solve as solve_gurobi

        return solve_gurobi(problem, **params)
    if solver == cp.SCIP:
        from cvxpy_translation.scip import solve as solve_scip

        return solve_scip(problem, **params)
    msg = f"Unsupported solver: {solver}, supported solvers are: {cp.GUROBI}, {cp.SCIP}"
    raise NotImplementedError(msg)


def register_translation_solver(solver: str) -> str:
    """Register a solver with CVXPY."""
    if solver == cp.GUROBI:
        from cvxpy_translation.gurobi import register_solver as register_gurobi

        return register_gurobi()
    if solver == cp.SCIP:
        from cvxpy_translation.scip import register_solver as register_scip

        return register_scip()
    msg = f"Unsupported solver: {solver}, supported solvers are: {cp.GUROBI}, {cp.SCIP}"
    raise NotImplementedError(msg)


def build_model(
    problem: cp.Problem, solver: str, *, params: ParamDict | None = None
) -> gp.Model | scip.Model:
    """Convert a CVXPY problem to a native solver model."""
    if solver == cp.GUROBI:
        from cvxpy_translation.gurobi import build_model as build_gurobi_model

        return build_gurobi_model(problem, params=params)
    if solver == cp.SCIP:
        from cvxpy_translation.scip import build_model as build_scip_model

        return build_scip_model(problem, params=params)
    msg = f"Unsupported solver: {solver}, supported solvers are: {cp.GUROBI}, {cp.SCIP}"
    raise NotImplementedError(msg)


def backfill_problem(
    problem: cp.Problem,
    model: gp.Model | scip.Model,
    compilation_time: float | None = None,
    solve_time: float | None = None,
) -> None:
    """Update the CVXPY problem with the solution from the native model."""
    if gurobi_available and isinstance(model, gp.Model):
        from cvxpy_translation.gurobi import backfill_problem as backfill_gurobi_problem

        backfill_gurobi_problem(problem, model, compilation_time, solve_time)
    elif scip_available and isinstance(model, scip.Model):
        from cvxpy_translation.scip import backfill_problem as backfill_scip_problem

        backfill_scip_problem(problem, model, compilation_time, solve_time)
    else:
        msg = f"Unsupported model type: {type(model)}, expected gp.Model or scip.Model"
        raise NotImplementedError(msg)
