from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING
from typing import Union

import cvxpy as cp
import cvxpy.settings as cp_settings
import gurobipy as gp
import numpy as np
import numpy.typing as npt
from cvxpy.constraints.nonpos import Inequality
from cvxpy.constraints.zero import Equality
from cvxpy.problems.problem import SolverStats
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solution import failure_solution
from cvxpy.reductions.solvers.conic_solvers import gurobi_conif
from cvxpy.settings import SOLUTION_PRESENT

from cvxpy_translation import CVXPY_VERSION
from cvxpy_translation.gurobi.translation import Translater
from cvxpy_translation.utils import Timer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cvxpy.constraints.constraint import Constraint
    from typing_extensions import TypeAlias

AnyVar: TypeAlias = Union[gp.Var, gp.MVar]
Param: TypeAlias = Union[str, float]
ParamDict: TypeAlias = dict[str, Param]

# Default name for the solver when registering it with CVXPY.
GUROBI_TRANSLATION: str = "GUROBI_TRANSLATION"


def solve(problem: cp.Problem, *, env: gp.Env | None = None, **params: Param) -> float:
    """Solve a CVXPY problem using Gurobi.

    This function can be used to solve CVXPY problems without registering the solver:
        cvxpy_translation.gurobi.solve(problem)
    """
    with Timer() as compilation:
        model = build_model(problem, params=params, env=env)

    with Timer() as solve:
        model.optimize()

    backfill_problem(
        problem, model, compilation_time=compilation.time, solve_time=solve.time
    )

    return float(problem.value)  # pyright: ignore[reportArgumentType]


def register_solver(name: str = GUROBI_TRANSLATION) -> str:
    """Register the solver under the given name, defaults to `GUROBI_TRANSLATION`.

    Once this function has been called, the solver can be used as follows:
        problem.solve(method=GUROBI_TRANSLATION)
    """
    cp.Problem.register_solve(name, solve)
    return name


def build_model(
    problem: cp.Problem, *, env: gp.Env | None = None, params: ParamDict | None = None
) -> gp.Model:
    """Convert a CVXPY problem to a Gurobi model."""
    model = gp.Model(env=env)
    fill_model(problem, model)
    if params:
        set_params(model, params)
    return model


def fill_model(problem: cp.Problem, model: gp.Model) -> None:
    """Add the objective and constraints from a CVXPY problem to a Gurobi model.

    Args:
        problem: The CVXPY problem to convert.
        model: The Gurobi model to which constraints and objectives are added.

    """
    Translater(model).visit(problem)


def set_params(model: gp.Model, params: ParamDict) -> None:
    for key, param in params.items():
        model.setParam(key, param)


def backfill_problem(
    problem: cp.Problem,
    model: gp.Model,
    compilation_time: float | None = None,
    solve_time: float | None = None,
) -> None:
    """Update the CVXPY problem with the solution from the Gurobi model."""
    solution = extract_solution_from_model(model, problem)
    problem.unpack(solution)

    if CVXPY_VERSION >= (1, 4):
        # class construction changed in https://github.com/cvxpy/cvxpy/pull/2141
        solver_stats = SolverStats.from_dict(solution.attr, GUROBI_TRANSLATION)
    else:
        solver_stats = SolverStats(solution.attr, GUROBI_TRANSLATION)  # type: ignore[arg-type]
    problem._solver_stats = solver_stats  # noqa: SLF001

    if solve_time is not None:
        problem._solve_time = solve_time  # noqa: SLF001
    if CVXPY_VERSION >= (1, 4) and compilation_time is not None:
        # added in https://github.com/cvxpy/cvxpy/pull/2046
        problem._compilation_time = compilation_time  # noqa: SLF001


def extract_solution_from_model(model: gp.Model, problem: cp.Problem) -> Solution:
    attr = {
        cp_settings.EXTRA_STATS: model,
        cp_settings.SOLVE_TIME: model.Runtime,
        cp_settings.NUM_ITERS: model.IterCount,
    }
    status = gurobi_conif.GUROBI.STATUS_MAP[model.Status]
    if status not in SOLUTION_PRESENT:
        if CVXPY_VERSION >= (1, 2, 0):
            # attr was added in https://github.com/cvxpy/cvxpy/pull/1270
            return failure_solution(status, attr)
        return failure_solution(status)

    primal_vars = {}
    dual_vars = {}
    for var in problem.variables():
        primal_vars[var.id] = extract_variable_value(model, var.name(), var.shape)
    # Duals are only available for convex continuous problems
    # https://www.gurobi.com/documentation/current/refman/pi.html
    if not model.IsMIP:
        for constr in problem.constraints:
            dual = get_constraint_dual(model, constr)
            if dual is None:
                continue
            if isinstance(problem.objective, cp.Minimize) and (
                isinstance(constr, Equality)
                or (isinstance(constr, Inequality) and constr.args[1].is_constant())
            ):
                dual *= -1
            dual_vars[constr.constr_id] = dual
    return Solution(
        status=status,
        opt_val=model.ObjVal,
        primal_vars=primal_vars,
        dual_vars=dual_vars,
        attr=attr,
    )


def extract_variable_value(
    model: gp.Model, var_name: str, shape: tuple[int, ...]
) -> npt.NDArray[np.float64]:
    if shape == ():
        v = model.getVarByName(var_name)
        assert v is not None
        return np.array(v.X)

    value = np.zeros(shape)
    for idx, subvar_name in _matrix_to_gurobi_names(var_name, shape):
        subvar = model.getVarByName(subvar_name)
        assert subvar is not None, subvar_name
        value[idx] = subvar.X
    return value


def get_constraint_dual(  # noqa: PLR0911
    model: gp.Model, constraint: Constraint
) -> npt.NDArray[np.float64] | None:
    constraint_name = str(constraint.constr_id)
    shape = constraint.shape
    # quadratic constraints don't have duals computed by default
    # https://www.gurobi.com/documentation/current/refman/qcpi.html
    has_qcp_duals = model.params.QCPDual

    if shape == ():
        constr = get_constraint_by_name(model, constraint_name)
        # CVXPY returns an array of shape (1,) for quadratic constraints
        # and a scalar for linear constraints -__-
        if isinstance(constr, gp.Constr):
            return np.array(constr.Pi)
        assert isinstance(constr, gp.QConstr)
        if has_qcp_duals:
            return np.array([constr.QCPi])
        return None

    if CVXPY_VERSION < (1, 4) and shape == (1, 1) and _contains_quad_form(constraint):
        # In older versions of CVXPY, the shape of a scalar quad form is (1, 1)
        _, constr_name = next(_matrix_to_gurobi_names(constraint_name, ()))
        with suppress(LookupError):
            constr = get_constraint_by_name(model, constr_name)
            if isinstance(constr, gp.Constr):
                return np.array(constr.Pi)
            assert isinstance(constr, gp.QConstr)
            if has_qcp_duals:
                return np.array([constr.QCPi])
            return None

    dual = np.zeros(shape)
    for idx, subconstr_name in _matrix_to_gurobi_names(constraint_name, shape):
        subconstr = get_constraint_by_name(model, subconstr_name)
        if isinstance(subconstr, gp.QConstr):
            if not has_qcp_duals:
                # no need to check the other subconstraints, they should all be the same
                return None
            dual[idx] = subconstr.QCPi
        else:
            dual[idx] = subconstr.Pi
    return dual


def _contains_quad_form(constraint: Constraint) -> bool:
    return any(cp.QuadForm in arg.atoms() for arg in constraint.args)


def get_constraint_by_name(model: gp.Model, name: str) -> gp.Constr | gp.QConstr:
    try:
        constr = model.getConstrByName(name)
    except gp.GurobiError:
        # quadratic constraints are not returned by getConstrByName
        for q_constr in model.getQConstrs():
            if q_constr.QCName == name:
                return q_constr
    else:
        if constr is not None:
            return constr
    msg = f"Constraint {name} not found."  # pragma: no cover
    raise LookupError(msg)  # pragma: no cover


def _matrix_to_gurobi_names(
    base_name: str, shape: tuple[int, ...]
) -> Iterator[tuple[tuple[int, ...], str]]:
    if not shape:
        yield (), base_name
        return
    for idx in np.ndindex(shape):
        formatted_idx = ",".join(str(i) for i in idx)
        yield idx, f"{base_name}[{formatted_idx}]"
