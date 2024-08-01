from __future__ import annotations

import operator
import time
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Iterator
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

from cvxpy_gurobi._version import __version__
from cvxpy_gurobi._version import __version_tuple__

if TYPE_CHECKING:
    from typing import TypeAlias

    try:
        from typing import Self
    except ImportError:
        try:
            # Use Self from typing_extensions if available
            # but we don't want to add it as a dependency
            from typing_extensions import Self
        except ImportError:
            Self: TypeAlias = Any

    from cvxpy.atoms.affine.add_expr import AddExpression
    from cvxpy.atoms.affine.binary_operators import DivExpression
    from cvxpy.atoms.affine.binary_operators import MulExpression
    from cvxpy.atoms.affine.binary_operators import multiply
    from cvxpy.atoms.affine.index import index
    from cvxpy.atoms.affine.index import special_index
    from cvxpy.atoms.affine.promote import Promote
    from cvxpy.atoms.affine.sum import Sum
    from cvxpy.atoms.affine.unary_operators import NegExpression
    from cvxpy.atoms.elementwise.power import power
    from cvxpy.atoms.quad_over_lin import quad_over_lin
    from cvxpy.constraints.constraint import Constraint


__all__ = (
    "__version__",
    "__version_tuple__",
    "backfill_problem",
    "build_model",
    "InvalidPowerError",
    "GUROBI_TRANSLATION",
    "register_solver",
    "solve",
    "UnsupportedConstraintError",
    "UnsupportedError",
    "UnsupportedExpressionError",
)

CVXPY_VERSION = tuple(map(int, cp.__version__.split(".")))

AnyVar: TypeAlias = Union[gp.Var, gp.MVar]
Param: TypeAlias = Union[str, float]
ParamDict: TypeAlias = Dict[str, Param]

# Default name for the solver when registering it with CVXPY.
GUROBI_TRANSLATION: str = "GUROBI_TRANSLATION"


class UnsupportedError(ValueError):
    msg_template = "Unsupported CVXPY node: {node}"

    def __init__(self, node: cp.Expression | cp.Constraint) -> None:
        super().__init__(self.msg_template.format(node=node, klass=type(node)))
        self.node = node


class UnsupportedConstraintError(UnsupportedError):
    msg_template = "Unsupported CVXPY constraint: {node}"


class UnsupportedExpressionError(UnsupportedError):
    msg_template = "Unsupported CVXPY expression: {node} ({klass})"


class InvalidPowerError(UnsupportedExpressionError):
    msg_template = "Unsupported power: {node}, only quadratic expressions are supported"


class _Timer:
    time: float

    def __enter__(self) -> Self:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        end = time.perf_counter()
        self.time = end - self._start


def solve(problem: cp.Problem, *, env: gp.Env | None = None, **params: Param) -> float:
    """Solve a CVXPY problem using Gurobi.

    This function can be used to solve CVXPY problems without registering the solver:
        cvxpy_gurobi.solve(problem)
    """
    with _Timer() as compilation:
        model = build_model(problem, params=params, env=env)

    with _Timer() as solve:
        model.optimize()

    backfill_problem(
        problem, model, compilation_time=compilation.time, solve_time=solve.time
    )

    return problem.value


def register_solver(name: str = GUROBI_TRANSLATION) -> None:
    """Register the solver under the given name, defaults to `NATIVE_GUROBI`.

    Once this function has been called, the solver can be used as follows:
        problem.solve(method=NATIVE_GUROBI)
    """
    cp.Problem.register_solve(name, solve)


def build_model(
    problem: cp.Problem, *, env: gp.Env | None = None, params: ParamDict | None = None
) -> gp.Model:
    """Convert a CVXPY problem to a Gurobi model."""
    model = gp.Model(env=env)
    variables = map_variables(problem, model)
    fill_model(problem, model, variables)
    if params:
        set_params(model, params)
    return model


def fill_model(
    problem: cp.Problem, model: gp.Model, variable_map: dict[str, AnyVar]
) -> None:
    """Add the objective and constraints from a CVXPY problem to a Gurobi model.

    Args:
        problem: The CVXPY problem to convert.
        model: The Gurobi model to which constraints and objectives are added.
        variable_map: A mapping from CVXPY variable names to Gurobi variables.
    """
    ObjectiveBuilder(model, variable_map).visit(problem.objective)
    ConstraintsBuilder(model, variable_map).visit_constraints(problem.constraints)
    model.update()


def set_params(model: gp.Model, params: ParamDict) -> None:
    for key, param in params.items():
        model.setParam(key, param)


def map_variables(problem: cp.Problem, model: gp.Model) -> dict[str, AnyVar]:
    return {var.name(): to_gurobi_var(var, model) for var in problem.variables()}


def to_gurobi_var(var: cp.Variable, model: gp.Model) -> AnyVar:
    lb = -gp.GRB.INFINITY
    ub = gp.GRB.INFINITY
    if var.is_nonneg():
        lb = 0
    if var.is_nonpos():
        ub = 0

    vtype = gp.GRB.CONTINUOUS
    if var.attributes["integer"]:
        vtype = gp.GRB.INTEGER
    if var.attributes["boolean"]:
        vtype = gp.GRB.BINARY

    if var.shape == ():
        return model.addVar(name=var.name(), lb=lb, ub=ub, vtype=vtype)
    return model.addMVar(var.shape, name=var.name(), lb=lb, ub=ub, vtype=vtype)


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
        solver_stats = SolverStats(solution.attr, GUROBI_TRANSLATION)
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
        return failure_solution(status, attr)

    primal_vars = {}
    dual_vars = {}
    for var in problem.variables():
        primal_vars[var.id] = extract_variable_value(model, var.name(), var.shape)
    # Duals are only available for convex continuous problems
    # https://www.gurobi.com/documentation/current/refman/pi.html
    if not model.IsMIP:
        for constr in problem.constraints:
            dual = get_constraint_dual(model, str(constr.constr_id), constr.shape)
            if dual is None:
                continue
            if isinstance(problem.objective, cp.Minimize) and (
                isinstance(constr, Equality)
                or isinstance(constr, Inequality)
                and not constr.expr.is_concave()
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


def get_constraint_dual(
    model: gp.Model, constraint_name: str, shape: tuple[int, ...]
) -> npt.NDArray[np.float64] | None:
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


def get_constraint_by_name(model: gp.Model, name: str) -> gp.Constr | gp.QConstr:
    try:
        constr = model.getConstrByName(name)
    except gp.GurobiError:
        # quadratic constraints are not returned by getConstrByName
        for q_constr in model.getQConstrs():
            if q_constr.QCName == name:
                return q_constr
        raise
    else:
        assert constr is not None
        return constr


def _matrix_to_gurobi_names(
    base_name: str, shape: tuple[int, ...]
) -> Iterator[tuple[tuple[int, ...], str]]:
    for idx in np.ndindex(shape):
        formatted_idx = ",".join(str(i) for i in idx)
        yield idx, f"{base_name}[{formatted_idx}]"


class ObjectiveBuilder:
    def __init__(self, model: gp.Model, variables: dict[str, AnyVar]):
        self.m = model
        self.vars = variables
        self.translater = ExpressionTranslater(variables)

    def visit(self, objective: cp.Objective) -> None:
        sense = (
            gp.GRB.MINIMIZE if isinstance(objective, cp.Minimize) else gp.GRB.MAXIMIZE
        )
        self.m.setObjective(self.translater.visit(objective.expr), sense=sense)


class ConstraintsBuilder:
    def __init__(self, model: gp.Model, variables: dict[str, AnyVar]):
        self.m = model
        self.vars = variables
        self.translater = ExpressionTranslater(variables)

    def translate(self, node: cp.Expression) -> Any:
        return self.translater.visit(node)

    def visit_constraints(self, constraints: list[Constraint]) -> None:
        for constraint in constraints:
            self.m.addConstr(
                self.visit_constraint(constraint), name=str(constraint.constr_id)
            )

    def visit_constraint(self, constraint: Constraint) -> Any:
        visitor = getattr(self, f"visit_{type(constraint).__name__}", None)
        if visitor is None:  # pragma: no cover
            raise UnsupportedConstraintError(constraint)
        return visitor(constraint)

    def visit_Equality(self, constraint: Equality) -> Any:
        left, right = constraint.args
        left = self.translate(left)
        right = self.translate(right)
        return left == right

    def _should_reverse_inequality(self, lower: object, upper: object) -> bool:
        # gurobipy objects don't have base classes and don't define __module__
        # This is very hacky but seems to work
        upper_from_gurobi = "'gurobipy." in str(type(upper))
        # having lower as an array raises an error when using the <= operator:
        # gurobipy.GurobiError:
        #     Constraint has no bool value (are you trying "lb <= expr <= ub"?)
        # so we need to reverse the inequality
        return upper_from_gurobi and isinstance(lower, np.ndarray)

    def visit_Inequality(self, ineq: Inequality) -> Any:
        lower, upper = ineq.args
        lower = self.translate(lower)
        upper = self.translate(upper)
        return (
            upper >= lower
            if self._should_reverse_inequality(lower, upper)
            else lower <= upper
        )


class ExpressionTranslater:
    def __init__(self, variables: dict[str, AnyVar]):
        self.vars = variables

    def visit(self, node: cp.Expression) -> Any:
        visitor = getattr(self, f"visit_{type(node).__name__}", None)
        if visitor is None:
            raise UnsupportedExpressionError(node)
        return visitor(node)

    def visit_AddExpression(self, node: AddExpression) -> Any:
        args = list(map(self.visit, node.args))
        return reduce(operator.add, args)

    def visit_Constant(self, const: cp.Constant) -> npt.NDArray[np.float64]:
        # TODO: can be a sparse array - fix annotation?
        return const.value

    def visit_DivExpression(self, node: DivExpression) -> Any:
        return self.visit(node.args[0]) / self.visit(node.args[1])

    def visit_index(self, node: index) -> Any:
        return self.visit(node.args[0])[node.key]

    def visit_MulExpression(self, node: MulExpression) -> Any:
        x, y = node.args
        x = self.visit(x)
        y = self.visit(y)
        return x @ y

    def visit_multiply(self, mul: multiply) -> Any:
        return self.visit(mul.args[0]) * self.visit(mul.args[1])

    def visit_NegExpression(self, node: NegExpression) -> Any:
        return -self.visit(node.args[0])

    def visit_power(self, node: power) -> Any:
        power = self.visit(node.p)
        if power != 2:
            raise InvalidPowerError(node.p)
        arg = self.visit(node.args[0])
        return arg**power

    def visit_Promote(self, node: Promote) -> Any:
        # FIXME: should we do something here?
        return self.visit(node.args[0])

    def visit_quad_over_lin(self, node: quad_over_lin) -> Any:
        x, y = node.args
        x = self.visit(x)
        squares = (((x[i]).item()) ** 2 for i in np.ndindex(x.shape))
        quad = gp.quicksum(squares)
        lin = self.visit(y)
        return quad / lin

    def visit_special_index(self, node: special_index) -> Any:
        return self.visit(node.args[0])[node.key]

    def visit_Sum(self, node: Sum) -> Any:
        return self.visit(node.args[0]).sum()

    def visit_Variable(self, var: cp.Variable) -> AnyVar:
        return self.vars[var.name()]
