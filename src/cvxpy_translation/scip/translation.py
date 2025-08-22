from __future__ import annotations

import operator
from functools import reduce
from math import prod
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import Union
from typing import overload

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import pyscipopt as scip
import scipy.sparse as sp
from cvxpy.constraints.constraint import Constraint
from pyscipopt.recipes.nonlinear import set_nonlinear_objective

from cvxpy_translation import CVXPY_VERSION
from cvxpy_translation.exceptions import InvalidParameterError
from cvxpy_translation.exceptions import UnsupportedAttributesError
from cvxpy_translation.exceptions import UnsupportedConstraintError
from cvxpy_translation.exceptions import UnsupportedError
from cvxpy_translation.exceptions import UnsupportedExpressionError
from cvxpy_translation.exceptions import UnsupportedPartialAttributesError

if TYPE_CHECKING:
    from cvxpy.atoms.affine.add_expr import AddExpression
    from cvxpy.atoms.affine.binary_operators import DivExpression
    from cvxpy.atoms.affine.binary_operators import MulExpression
    from cvxpy.atoms.affine.binary_operators import multiply
    from cvxpy.atoms.affine.hstack import Hstack
    from cvxpy.atoms.affine.index import index
    from cvxpy.atoms.affine.index import special_index
    from cvxpy.atoms.affine.promote import Promote
    from cvxpy.atoms.affine.sum import Sum
    from cvxpy.atoms.affine.unary_operators import NegExpression
    from cvxpy.atoms.affine.vstack import Vstack
    from cvxpy.atoms.elementwise.power import power
    from cvxpy.atoms.quad_over_lin import quad_over_lin
    from cvxpy.constraints.nonpos import Inequality
    from cvxpy.constraints.zero import Equality
    from cvxpy.utilities.canonical import Canonical
    from typing_extensions import TypeAlias


AnyVar: TypeAlias = Union[scip.Variable, scip.MatrixVariable]
Param: TypeAlias = Union[str, float]
ParamDict: TypeAlias = dict[str, Param]


class InvalidPowerError(UnsupportedExpressionError):
    msg_template = "Unsupported power: {node}, only quadratic expressions are supported"


class InvalidNormError(UnsupportedExpressionError):
    msg_template = (
        "Unsupported norm: {node}, only 1-norm, 2-norm and inf-norm are supported"
    )


class ComplexExpressionError(UnsupportedExpressionError):
    msg_template = "Complex expressions are not supported: {node}"


def _shape(expr: Any) -> tuple[int, ...]:
    return getattr(expr, "shape", ())


def _is_scalar_shape(shape: tuple[int, ...]) -> bool:
    return prod(shape) == 1


def _is_scalar(expr: Any) -> bool:
    return _is_scalar_shape(_shape(expr))


HANDLED_ATTRIBUTES = {"integer", "boolean", "nonneg", "nonpos", "neg", "pos", "bounds"}
NO_PARTIAL_ATTRIBUTES = {"integer", "boolean"}


def translate_variable(var: cp.Variable, model: scip.Model) -> AnyVar:
    attributes = var.attributes
    set_attributes = {k for k, v in attributes.items() if v}
    unhandled = set_attributes - HANDLED_ATTRIBUTES
    if unhandled:
        raise UnsupportedAttributesError(leaf=var, attributes=unhandled)

    for partial_attr in set_attributes & NO_PARTIAL_ATTRIBUTES:
        idx = getattr(var, f"{partial_attr}_idx")
        if var.ndim > 0 and var[idx].size != var.size:
            raise UnsupportedPartialAttributesError(leaf=var, attribute=partial_attr)

    # Bounds added in https://github.com/cvxpy/cvxpy/pull/2234
    if CVXPY_VERSION >= (1, 5, 0) and var.bounds is not None:
        lb, ub = var.bounds
    else:
        lb, ub = None, None
        if var.is_nonneg():
            lb = 0
        if var.is_nonpos():
            ub = 0

    vtype = "CONTINUOUS"
    if attributes["integer"]:
        vtype = "INTEGER"
    if attributes["boolean"]:
        vtype = "BINARY"

    return add_variable(model, var.shape, lb=lb, ub=ub, vtype=vtype, name=var.name())


@overload
def add_variable(
    model: scip.Model,
    shape: tuple[()],
    *,
    name: str,
    vtype: str = "CONTINUOUS",
    lb: float | None = None,
    ub: float | None = None,
) -> scip.Variable: ...
@overload
def add_variable(
    model: scip.Model,
    shape: tuple[int, ...],
    *,
    name: str,
    vtype: str = "CONTINUOUS",
    lb: float | None = None,
    ub: float | None = None,
) -> scip.MatrixVariable: ...
def add_variable(
    model: scip.Model,
    shape: tuple[int, ...],
    *,
    name: str,
    vtype: str = "CONTINUOUS",
    lb: float | None = None,
    ub: float | None = None,
) -> scip.Variable | scip.MatrixVariable:
    if shape == ():
        return model.addVar(name=name, lb=lb, ub=ub, vtype=vtype)
    return model.addMatrixVar(shape, name=name, lb=lb, ub=ub, vtype=vtype)


def add_constraint(
    cons: scip.scip.ExprCons | scip.scip.MatrixExprCons, model: scip.Model, *, name: str
) -> None:
    if isinstance(cons, scip.scip.ExprCons):
        model.addCons(cons, name=name)
    elif isinstance(cons, scip.scip.MatrixExprCons):
        model.addMatrixCons(cons, name=name)
    else:  # pragma: no cover
        msg = f"Unexpected constraint type: {type(cons)}"
        raise TypeError(msg)


def _should_reverse_inequality(lower: object, upper: object) -> bool:
    """Check whether lower <= upper is safe.

    When writing an inequality constraint lower <= upper,
    we get a type error if lower is an array and upper is a scip object:
        TypeError: Can't evaluate constraints as booleans.

        If you want to add a ranged constraint of the form
            lhs <= expression <= rhs
        you have to use parenthesis to break the Python syntax for chained comparisons:
            lhs <= (expression <= rhs)

    In that case, we should write upper >= lower instead.
    """
    # scip objects don't define __module__
    # This is very hacky but seems to work
    upper_from_scip = "'pyscipopt." in str(type(upper))
    return upper_from_scip and isinstance(lower, np.ndarray)


class Translater:
    def __init__(self, model: scip.Model) -> None:
        self.model = model
        self.vars: dict[int, AnyVar] = {}
        self._aux_id = 0

    def visit(self, node: Canonical) -> Any:
        visitor = getattr(self, f"visit_{type(node).__name__}", None)
        if visitor is not None:
            return visitor(node)

        if isinstance(node, Constraint):
            raise UnsupportedConstraintError(node)
        if isinstance(node, cp.Expression):
            raise UnsupportedExpressionError(node)
        raise UnsupportedError(node)

    def translate_into_scalar(self, node: cp.Expression) -> Any:
        expr = self.visit(node)
        shape = _shape(expr)
        if shape == () and not isinstance(expr, np.ndarray):
            return expr
        assert _is_scalar_shape(shape), f"Expected scalar, got shape {shape}"
        # expr can be many things: an ndarray, MVar, MLinExpr, etc.
        # but let's assume it always has an `item` method
        return expr.item()

    def make_auxilliary_variable_for(
        self,
        expr: Any,
        atom_name: str,
        *,
        desired_shape: tuple[int, ...],
        vtype: str = "CONTINUOUS",
        lb: float | None = None,
        ub: float | None = None,
    ) -> scip.MatrixVariable:
        """Add a variable constrained to the value of the given SCIP expression."""
        self._aux_id += 1
        var = add_variable(
            self.model,
            shape=desired_shape,
            name=f"{atom_name}_{self._aux_id}",
            vtype=vtype,
            lb=lb,
            ub=ub,
        )
        assert isinstance(var, scip.MatrixVariable)
        self.model.addMatrixCons(var == expr)
        return var

    def visit_abs(self, node: cp.abs) -> Any:
        (arg,) = node.args
        return np.abs(self.visit(arg))

    def visit_AddExpression(self, node: AddExpression) -> Any:
        args = list(map(self.visit, node.args))
        return reduce(operator.add, args)

    def visit_conj(self, node: cp.conj) -> Any:
        (arg,) = node.args
        if arg.is_complex():
            raise ComplexExpressionError(node)
        return self.visit(arg)

    def visit_Constant(self, const: cp.Constant) -> Any:
        val = const.value
        if sp.issparse(val):
            return val.toarray()
        return val

    def visit_Parameter(self, param: cp.Parameter) -> Any:
        value = param.value

        if value is None:
            raise InvalidParameterError(param)

        return value

    def visit_DivExpression(self, node: DivExpression) -> Any:
        return self.visit(node.args[0]) / self.visit(node.args[1])

    def visit_Equality(self, constraint: Equality) -> Any:
        left, right = constraint.args
        left = self.visit(left)
        right = self.visit(right)
        return left == right

    def visit_exp(self, node: cp.exp) -> AnyVar:
        (arg,) = node.args
        return scip.exp(self.visit(arg))

    def visit_Hstack(self, node: Hstack) -> Any:
        args = node.args
        exprs = [self.visit(arg) for arg in args]
        return np.hstack(exprs)

    def visit_index(self, node: index) -> Any:
        return self.visit(node.args[0])[node.key]

    def visit_Inequality(self, ineq: Inequality) -> Any:
        lo, up = ineq.args
        lower = self.visit(lo)
        upper = self.visit(up)
        return (
            upper >= lower
            if _should_reverse_inequality(lower, upper)
            else lower <= upper
        )

    def visit_log(self, node: cp.log) -> AnyVar:
        (arg,) = node.args
        return scip.log(self.visit(arg))

    def visit_log1p(self, node: cp.log1p) -> AnyVar:
        (arg,) = node.args
        return scip.log(self.visit(arg) + 1)

    def visit_max(self, node: cp.max) -> Any:
        (arg,) = node.args
        expr = self.visit(arg)
        if isinstance(arg, cp.Constant):
            return np.max(expr)
        if _is_scalar(arg):
            # max of a scalar is itself
            return expr
        self._aux_id += 1
        bound_var = add_variable(self.model, shape=(), name=f"max_{self._aux_id}")
        self.model.addMatrixCons(bound_var >= expr, name=f"max_{self._aux_id}")
        return bound_var

    def visit_min(self, node: cp.min) -> Any:
        (arg,) = node.args
        expr = self.visit(arg)
        if isinstance(arg, cp.Constant):
            return np.min(expr)
        if _is_scalar(arg):
            # min of a scalar is itself
            return expr
        self._aux_id += 1
        bound_var = add_variable(self.model, shape=(), name=f"min_{self._aux_id}")
        self.model.addMatrixCons(bound_var <= expr, name=f"min_{self._aux_id}")
        return bound_var

    def visit_maximum(self, node: cp.maximum) -> Any:
        args = node.args
        exprs = [self.visit(arg) for arg in args]
        self._aux_id += 1
        z = add_variable(self.model, shape=node.shape, name=f"maximum_{self._aux_id}")
        for i, expr in enumerate(exprs):
            cons = z >= expr
            add_constraint(cons, self.model, name=f"maximum_{self._aux_id}_{i}")
        return z

    def visit_minimum(self, node: cp.minimum) -> Any:
        args = node.args
        exprs = [self.visit(arg) for arg in args]
        self._aux_id += 1
        z = add_variable(self.model, shape=node.shape, name=f"minimum_{self._aux_id}")
        for i, expr in enumerate(exprs):
            cons = z <= expr
            add_constraint(cons, self.model, name=f"minimum_{self._aux_id}_{i}")
        return z

    def _visit_objective(
        self,
        objective: cp.Minimize | cp.Maximize,
        sense: Literal["minimize", "maximize"],
    ) -> None:
        """Visit an objective and set it in the model."""
        obj = self.translate_into_scalar(objective.expr)
        if obj.degree() > 1:
            set_nonlinear_objective(self.model, obj, sense=sense)
        else:
            self.model.setObjective(obj, sense=sense)

    def visit_Maximize(self, objective: cp.Maximize) -> None:
        self._visit_objective(objective, sense="maximize")

    def visit_Minimize(self, objective: cp.Minimize) -> None:
        self._visit_objective(objective, sense="minimize")

    def visit_MulExpression(self, node: MulExpression) -> Any:
        x, y = node.args
        x = self.visit(x)
        y = self.visit(y)
        return x @ y

    def visit_multiply(self, mul: multiply) -> Any:
        return self.visit(mul.args[0]) * self.visit(mul.args[1])

    def visit_NegExpression(self, node: NegExpression) -> Any:
        return -self.visit(node.args[0])

    def visit_norm1(self, node: cp.norm1) -> Any:
        (arg,) = node.args
        expr = self.visit(arg)
        if isinstance(expr, scip.Expr):
            return abs(expr)
        return np.abs(expr).sum()

    def visit_Pnorm(self, node: cp.Pnorm) -> Any:
        (arg,) = node.args
        expr = self.visit(arg)
        p = node.original_p
        if p != 2:
            raise InvalidNormError(node)
        if isinstance(expr, scip.Expr):
            return (expr**p) ** (1 / p)
        return (expr**p).sum() ** (1 / p)

    def visit_norm_inf(self, node: cp.norm_inf) -> Any:
        (arg,) = node.args
        # inf norm is max(abs(arg))
        expr = self.visit(arg)
        if isinstance(expr, scip.Expr):
            return abs(expr)
        if isinstance(arg, cp.Constant):
            return np.max(np.abs(expr))
        self._aux_id += 1
        z = add_variable(self.model, shape=(), name=f"norminf_{self._aux_id}", lb=0)
        add_constraint(z >= abs(expr), self.model, name=f"norminf_{self._aux_id}")
        return z

    def visit_power(self, node: power) -> Any:
        power = self.visit(node.p)
        if power != 2:
            raise InvalidPowerError(node.p)
        arg = self.visit(node.args[0])
        return arg**power

    def visit_Problem(self, problem: cp.Problem) -> None:
        self.visit(problem.objective)
        for constraint in problem.constraints:
            cons = self.visit(constraint)
            if isinstance(cons, scip.scip.ExprCons):
                self.model.addCons(cons, name=str(constraint.constr_id))
            elif isinstance(cons, scip.scip.MatrixExprCons):
                self.model.addMatrixCons(cons, name=str(constraint.constr_id))
            else:  # pragma: no cover
                msg = f"Unexpected constraint type: {type(cons)}"
                raise TypeError(msg)

    def visit_Promote(self, node: Promote) -> Any:
        # FIXME: should we do something here?
        return self.visit(node.args[0])

    def visit_QuadForm(self, node: cp.QuadForm) -> scip.Expr:
        vec, psd_mat = node.args
        vec_expr = self.visit(vec)
        psd_mat_expr = self.visit(psd_mat)
        quad = vec_expr @ psd_mat_expr @ vec_expr.T
        if isinstance(quad, scip.Expr):
            return quad
        # The result is a scalar wrapped in a MatrixExpr
        return quad.item()

    def visit_quad_over_lin(self, node: quad_over_lin) -> Any:
        x, y = node.args
        x = self.visit(x)
        quad = scip.quicksum(x**2)
        lin = self.visit(y)
        return quad / lin

    def visit_reshape(
        self, node: cp.reshape
    ) -> scip.MatrixVariable | npt.NDArray[np.float64]:
        """Reshape a variable or expression.

        Only MVars have a reshape method, so anything else will be proxied by an MVar.
        In all cases, the resulting MVar's shape should be exactly the target shape,
        no dimension squeezing, scalar inference should happen.
        """
        (x,) = node.args
        target_shape = node.shape
        expr = self.visit(x)
        if isinstance(expr, (int, float)):
            return np.reshape(expr, target_shape)
        if isinstance(expr, scip.Expr):
            if target_shape == ():
                return expr
            expr = np.array([expr]).view(scip.MatrixExpr)
        elif target_shape == ():
            assert isinstance(expr, scip.MatrixExpr)
            assert _is_scalar(expr)
            return expr.item()
        return expr.reshape(target_shape)

    def visit_special_index(self, node: special_index) -> Any:
        return self.visit(node.args[0])[node.key]

    def visit_Sum(self, node: Sum) -> Any:
        expr = self.visit(node.args[0])
        if _is_scalar(expr):
            return expr
        # axis is broken in PyScipOpt 5.5.0, so we handle it manually
        if node.axis is None:
            return expr.sum()
        # TODO: use numpy.lib.array_utils.normalize_axis_index when we drop support for NumPy < 2.0
        axis = node.axis + expr.ndim if node.axis < 0 else node.axis
        return np.apply_along_axis(scip.quicksum, axis, expr).view(scip.MatrixExpr)

    def visit_Variable(self, var: cp.Variable) -> AnyVar:
        if var.id not in self.vars:
            self.vars[var.id] = translate_variable(var, self.model)
        return self.vars[var.id]

    def visit_Vstack(self, node: Vstack) -> Any:
        args = node.args
        exprs = [self.visit(arg) for arg in args]
        return np.vstack(exprs)
