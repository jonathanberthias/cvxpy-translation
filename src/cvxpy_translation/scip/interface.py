from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import Dict
from typing import Iterator
from typing import Union

import cvxpy as cp
import cvxpy.settings as cp_settings
import numpy as np
import numpy.typing as npt
import pyscipopt as scip
from cvxpy.constraints.nonpos import Inequality
from cvxpy.constraints.zero import Equality
from cvxpy.problems.problem import SolverStats
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solution import failure_solution
from cvxpy.reductions.solvers.conic_solvers import scip_conif
from cvxpy.settings import SOLUTION_PRESENT

from cvxpy_translation.scip.translation import CVXPY_VERSION
from cvxpy_translation.scip.translation import Translater

if TYPE_CHECKING:
    from typing_extensions import Self
    from typing_extensions import TypeAlias

AnyVar: TypeAlias = Union[scip.Variable, scip.MatrixVariable]
Param: TypeAlias = Union[str, float]
ParamDict: TypeAlias = Dict[str, Param]

# Default name for the solver when registering it with CVXPY.
SCIP_TRANSLATION: str = "SCIP_TRANSLATION"


class _Timer:
    time: float

    def __enter__(self) -> Self:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        end = time.perf_counter()
        self.time = end - self._start


def solve(problem: cp.Problem, **params: Param) -> float:
    """Solve a CVXPY problem using SCIP.

    This function can be used to solve CVXPY problems without registering the solver:
        cvxpy_translation.scip.solve(problem)
    """
    with _Timer() as compilation:
        model = build_model(problem, params=params)

    with _Timer() as solve:
        model.optimize()

    backfill_problem(
        problem, model, compilation_time=compilation.time, solve_time=solve.time
    )

    return float(problem.value)  # pyright: ignore[reportArgumentType]


def register_solver(name: str = SCIP_TRANSLATION) -> None:
    """Register the solver under the given name, defaults to `SCIP_TRANSLATION`.

    Once this function has been called, the solver can be used as follows:
        problem.solve(method=SCIP_TRANSLATION)
    """
    cp.Problem.register_solve(name, solve)


def build_model(problem: cp.Problem, *, params: ParamDict | None = None) -> scip.Model:
    """Convert a CVXPY problem to a SCIP model."""
    model = scip.Model()
    fill_model(problem, model)
    if params:
        set_params(model, params)
    return model


def fill_model(problem: cp.Problem, model: scip.Model) -> None:
    """Add the objective and constraints from a CVXPY problem to a SCIP model.

    Args:
        problem: The CVXPY problem to convert.
        model: The SCIP model to which constraints and objectives are added.

    """
    Translater(model).visit(problem)


def set_params(model: scip.Model, params: ParamDict) -> None:
    for key, param in params.items():
        model.setParam(key, param)


def backfill_problem(
    problem: cp.Problem,
    model: scip.Model,
    compilation_time: float | None = None,
    solve_time: float | None = None,
) -> None:
    """Update the CVXPY problem with the solution from the SCIP model."""
    solution = extract_solution_from_model(model, problem)
    problem.unpack(solution)

    if CVXPY_VERSION >= (1, 4):
        # class construction changed in https://github.com/cvxpy/cvxpy/pull/2141
        solver_stats = SolverStats.from_dict(solution.attr, SCIP_TRANSLATION)
    else:
        solver_stats = SolverStats(solution.attr, SCIP_TRANSLATION)  # type: ignore[arg-type]
    problem._solver_stats = solver_stats  # noqa: SLF001

    if solve_time is not None:
        problem._solve_time = solve_time  # noqa: SLF001
    if CVXPY_VERSION >= (1, 4) and compilation_time is not None:
        # added in https://github.com/cvxpy/cvxpy/pull/2046
        problem._compilation_time = compilation_time  # noqa: SLF001


def extract_solution_from_model(model: scip.Model, problem: cp.Problem) -> Solution:
    attr = {
        cp_settings.EXTRA_STATS: model,
        cp_settings.SOLVE_TIME: model.getSolvingTime(),
        cp_settings.NUM_ITERS: model.lpiGetIterations(),
    }
    status = scip_conif.STATUS_MAP[model.getStatus()]
    if status not in SOLUTION_PRESENT:
        if CVXPY_VERSION >= (1, 2, 0):
            # attr was added in https://github.com/cvxpy/cvxpy/pull/1270
            return failure_solution(status, attr)
        return failure_solution(status)

    primal_vars = {}
    dual_vars = {}
    for var in problem.variables():
        primal_vars[var.id] = extract_variable_value(model, var.name(), var.shape)
    for constr in problem.constraints:
        dual = get_constraint_dual(model, str(constr.constr_id), constr.shape)
        if dual is None:
            continue
        if isinstance(constr, Equality) or (
            isinstance(constr, Inequality) and constr.args[1].is_constant()
        ):
            dual *= -1
        dual_vars[constr.constr_id] = dual
    return Solution(
        status=status,
        opt_val=model.getObjVal(),
        primal_vars=primal_vars,
        dual_vars=dual_vars,
        attr=attr,
    )


def extract_variable_value(
    model: scip.Model, var_name: str, shape: tuple[int, ...]
) -> npt.NDArray[np.float64]:
    var_dict = model.getVarDict()
    if shape == ():
        v = var_dict[var_name]
        return np.array(v)

    value = np.zeros(shape)
    for idx, subvar_name in _matrix_to_scip_names(var_name, shape):
        value[idx] = var_dict[subvar_name]
    return value


def get_constraint_dual(
    model: scip.Model, constraint_name: str, shape: tuple[int, ...]
) -> npt.NDArray[np.float64] | None:
    if shape == ():
        constr = get_constraint_by_name(model, constraint_name)
        # CVXPY returns an array of shape (1,) for quadratic constraints
        # and a scalar for linear constraints -__-
        try:
            return np.array(model.getDualsolLinear(constr))
        except Warning:
            return None

    return None


def get_constraint_by_name(model: scip.Model, name: str) -> scip.Constraint:
    constrs = model.getConss(transformed=False)
    for constr in constrs:
        if constr.name == name:
            return constr
    msg = f"Constraint {name} not found."
    raise LookupError(msg)


def _matrix_to_scip_names(
    base_name: str, shape: tuple[int, ...]
) -> Iterator[tuple[tuple[int, ...], str]]:
    for idx in np.ndindex(shape):
        formatted_idx = "_".join(str(i) for i in idx)
        yield idx, f"{base_name}_{formatted_idx}"
