from __future__ import annotations

import operator
import time
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterator

import cvxpy as cp
import cvxpy.settings as cp_settings
import gurobipy as gp
import numpy as np
import numpy.typing as npt
from cvxpy.constraints.nonpos import Inequality
from cvxpy.constraints.zero import Equality
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solution import failure_solution
from cvxpy.reductions.solvers.conic_solvers import gurobi_conif
from cvxpy.settings import SOLUTION_PRESENT

if TYPE_CHECKING:
    from typing import TypeAlias

    from cvxpy.atoms.affine.add_expr import AddExpression
    from cvxpy.atoms.affine.binary_operators import DivExpression
    from cvxpy.atoms.affine.binary_operators import MulExpression
    from cvxpy.atoms.affine.binary_operators import multiply
    from cvxpy.atoms.affine.promote import Promote
    from cvxpy.atoms.affine.sum import Sum
    from cvxpy.atoms.affine.unary_operators import NegExpression
    from cvxpy.atoms.elementwise.abs import abs as atoms_abs
    from cvxpy.atoms.elementwise.power import power
    from cvxpy.atoms.quad_over_lin import quad_over_lin
    from cvxpy.constraints.constraint import Constraint


__all__ = (
    "backfill_problem",
    "build_model",
    "fill_model",
    "InvalidPowerError",
    "make_solver",
    "map_variables",
    "NATIVE_GUROBI",
    "register_solver",
    "set_params",
    "solve",
    "translate",
    "UnsupportedConstraintError",
    "UnsupportedError",
    "UnsupportedExpressionError",
)


AnyVar: TypeAlias = gp.Var | gp.MVar
ParamDict: TypeAlias = dict[str, str | float]

# Default name for the solver when registering it with CVXPY.
# "Native" refers to the fact that the entire problem is solved using Gurobi,
# as opposed to the default behavior where CVXPY reformulates the problem
# before sending it to Gurobi.
NATIVE_GUROBI: str = "NATIVE_GUROBI"


class UnsupportedError(ValueError):
    msg_template = "Unsupported CVXPY node: {}"

    def __init__(self, node: cp.Expression | cp.Constraint) -> None:
        super().__init__(self.msg_template.format(node))
        self.node = node


class UnsupportedExpressionError(ValueError):
    msg_template = "Unsupported CVXPY expression: {}"


class InvalidPowerError(UnsupportedExpressionError):
    msg_template = "Unsupported power: {}, only quadratic expressions are supported"


class UnsupportedConstraintError(UnsupportedError):
    msg_template = "Unsupported CVXPY constraint: {}"


def solve(problem: cp.Problem, params: ParamDict | None = None) -> float:
    """Solves a CVXPY problem using Gurobi.

    This function can be used to solve CVXPY problems without registering the solver:
        cvxpy_gurobi.solve(problem)
    """
    start_setup = time.process_time()
    model = build_model(problem, params=params)
    setup_time = time.process_time() - start_setup
    model.optimize()
    backfill_problem(problem, model, setup_time=setup_time)
    return problem.value


def register_solver(params: ParamDict | None = None) -> None:
    """Registers the solver under the `NATIVE_GUROBI` name.

    Once this function has been called, the solver can be used as follows:
        problem.solve(solver=NATIVE_GUROBI)

    Args:
        params: A dictionary of Gurobi parameters to set on the model.
    """
    cp.Problem.register_solve(NATIVE_GUROBI, make_solver(params))


def make_solver(params: ParamDict | None = None) -> Callable[[cp.Problem], float]:
    """Returns a function that solves a CVXPY problem using Gurobi."""

    def solver(problem: cp.Problem) -> float:
        return solve(problem, params)

    return solver


def build_model(
    problem: cp.Problem, params: ParamDict | None = None, env: gp.Env | None = None
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
    problem: cp.Problem, model: gp.Model, setup_time: float | None = None
) -> None:
    """Update the CVXPY problem with the solution from the Gurobi model."""
    solution = extract_solution_from_model(model, problem, setup_time=setup_time)
    problem.unpack(solution)


def extract_solution_from_model(
    model: gp.Model, problem: cp.Problem, setup_time: float | None = None
) -> Solution:
    attr = {
        cp_settings.EXTRA_STATS: model,
        cp_settings.SOLVE_TIME: model.Runtime,
        cp_settings.NUM_ITERS: model.IterCount,
    }
    if setup_time is not None:
        attr[cp_settings.SETUP_TIME] = setup_time

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
        assert subvar is not None
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
        formatted_idx = ", ".join(str(i) for i in idx)
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

    def visit_abs(self, node: atoms_abs) -> gp.GenExprAbs:
        return gp.abs_(self.visit(node.args[0]))

    def visit_AddExpression(self, node: AddExpression) -> Any:
        args = list(map(self.visit, node.args))
        return reduce(operator.add, args)

    def visit_Constant(self, const: cp.Constant) -> npt.NDArray[np.float64]:
        # TODO: can be a sparse array - fix annotation?
        return const.value

    def visit_DivExpression(self, node: DivExpression) -> Any:
        return self.visit(node.args[0]) / self.visit(node.args[1])

    def visit_MulExpression(self, node: MulExpression) -> Any:
        x, y = node.args
        x = self.visit(x)
        y = self.visit(y)
        return x @ y

    def visit_multiply(self, mul: multiply) -> Any:
        return self.visit(mul.args[0]) * self.visit(mul.args[1])

    def visit_NegExpression(self, node: NegExpression) -> Any:
        return -self.visit(node.args[0])

    def visit_one_minus_pos(self, node: cp.one_minus_pos) -> Any:
        return 1 - self.visit(node.args[0])

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

    def visit_Sum(self, node: Sum) -> Any:
        return self.visit(node.args[0]).sum()

    def visit_Variable(self, var: cp.Variable) -> AnyVar:
        return self.vars[var.name()]


def translate(expr: cp.Expression, variables: dict[str, AnyVar]) -> Any:
    """Translate a CVXPY expression to a Gurobi expression."""
    return ExpressionTranslater(variables).visit(expr)
