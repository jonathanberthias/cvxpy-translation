from __future__ import annotations

import operator
from functools import reduce
from math import prod
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Union
from typing import overload

import cvxpy as cp
import gurobipy as gp
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from cvxpy.constraints.constraint import Constraint

from cvxpy_translation import CVXPY_VERSION
from cvxpy_translation.exceptions import InvalidParameterError
from cvxpy_translation.exceptions import UnsupportedAttributesError
from cvxpy_translation.exceptions import UnsupportedConstraintError
from cvxpy_translation.exceptions import UnsupportedError
from cvxpy_translation.exceptions import UnsupportedExpressionError
from cvxpy_translation.exceptions import UnsupportedPartialAttributesError

if TYPE_CHECKING:
    from collections.abc import Iterator

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


GUROBIPY_VERSION = gp.gurobi.version()
GUROBI_MAJOR = GUROBIPY_VERSION[0]

AnyVar: TypeAlias = Union[gp.Var, gp.MVar]
Param: TypeAlias = Union[str, float]
ParamDict: TypeAlias = dict[str, Param]


class InvalidPowerError(UnsupportedExpressionError):
    msg_template = "Unsupported power: {node}, only quadratic expressions are supported"


class InvalidNormError(UnsupportedExpressionError):
    msg_template = (
        "Unsupported norm: {node}, only 1-norm, 2-norm and inf-norm are supported"
    )


class InvalidNonlinearAtomError(UnsupportedExpressionError):
    msg_template = (
        "Unsupported nonlinear atom: {node}, upgrade your version of gurobipy"
    )


class ComplexExpressionError(UnsupportedExpressionError):
    msg_template = "Complex expressions are not supported: {node}"


def _shape(expr: Any) -> tuple[int, ...]:
    return getattr(expr, "shape", ())


def _is_scalar_shape(shape: tuple[int, ...]) -> bool:
    return prod(shape) == 1


def _is_scalar(expr: Any) -> bool:
    return _is_scalar_shape(_shape(expr))


def _squeeze_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(d for d in shape if d != 1)


def iterzip_subexpressions(
    *exprs: Any, shape: tuple[int, ...]
) -> Iterator[tuple[Any, ...]]:
    for idx in np.ndindex(shape):
        idx_exprs = []
        for expr in exprs:
            if _shape(expr) == ():
                idx_exprs.append(expr)
            elif _is_scalar(expr):
                item = expr[(0,) * len(idx)]
                idx_exprs.append(item)
            else:
                idx_exprs.append(expr[idx])
        yield tuple(idx_exprs)


def iter_subexpressions(expr: Any, shape: tuple[int, ...]) -> Iterator[Any]:
    for exprs in iterzip_subexpressions(expr, shape=shape):
        yield exprs[0]


def to_subexpressions_array(expr: Any, shape: tuple[int, ...]) -> npt.NDArray:
    return np.fromiter(
        iter_subexpressions(expr, shape=shape), dtype=np.object_
    ).reshape(shape)


def to_zipped_subexpressions_array(
    *exprs: Any, shape: tuple[int, ...]
) -> npt.NDArray[np.object_]:
    return np.fromiter(
        iterzip_subexpressions(*exprs, shape=shape), dtype=np.object_
    ).reshape(shape)


def promote_array_to_gurobi_matrixapi(array: npt.NDArray[np.object_]) -> Any:
    """Promote an array of Gurobi objects to the equivalent Gurobi matrixapi object."""
    kind = type(array.flat[0])
    if issubclass(kind, gp.Var):
        return gp.MVar.fromlist(array)
    # TODO: support other types
    msg = f"Cannot promote array of {kind}"
    raise NotImplementedError(msg)  # pragma: no cover


HANDLED_ATTRIBUTES = {"integer", "boolean", "nonneg", "nonpos", "neg", "pos", "bounds"}
NO_PARTIAL_ATTRIBUTES = {"integer", "boolean"}


def translate_variable(var: cp.Variable, model: gp.Model) -> AnyVar:
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
        lb = -gp.GRB.INFINITY
        ub = gp.GRB.INFINITY
        if var.is_nonneg():
            lb = 0
        if var.is_nonpos():
            ub = 0

    vtype = gp.GRB.CONTINUOUS
    if attributes["integer"]:
        vtype = gp.GRB.INTEGER
    if attributes["boolean"]:
        vtype = gp.GRB.BINARY

    return add_variable(model, var.shape, lb=lb, ub=ub, vtype=vtype, name=var.name())


@overload
def add_variable(
    model: gp.Model, shape: tuple[()], name: str, vtype: str, lb: float, ub: float
) -> gp.Var: ...
@overload
def add_variable(
    model: gp.Model, shape: tuple[int, ...], name: str, vtype: str, lb: float, ub: float
) -> AnyVar: ...
def add_variable(
    model: gp.Model,
    shape: tuple[int, ...],
    name: str,
    vtype: str = gp.GRB.CONTINUOUS,
    lb: float = -gp.GRB.INFINITY,
    ub: float = gp.GRB.INFINITY,
) -> AnyVar:
    if shape == ():
        return model.addVar(name=name, lb=lb, ub=ub, vtype=vtype)
    return model.addMVar(shape, name=name, lb=lb, ub=ub, vtype=vtype)


def _should_reverse_inequality(lower: object, upper: object) -> bool:
    """Check whether lower <= upper is safe.

    When writing an inequality constraint lower <= upper,
    we get an error if lower is an array and upper is a gurobipy object:
        gurobipy.GurobiError:
            Constraint has no bool value (are you trying "lb <= expr <= ub"?)

    In that case, we should write upper >= lower instead.
    """
    # gurobipy objects don't have base classes and don't define __module__
    # This is very hacky but seems to work
    upper_from_gurobi = "'gurobipy." in str(type(upper))
    return upper_from_gurobi and isinstance(lower, np.ndarray)


class Translater:
    def __init__(self, model: gp.Model) -> None:
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
        if shape == ():
            return expr
        assert _is_scalar_shape(shape), f"Expected scalar, got shape {shape}"
        # expr can be many things: an ndarray, MVar, MLinExpr, etc.
        # but let's assume it always has an `item` method
        return expr.item()

    def translate_into_variable(
        self,
        node: cp.Expression,
        *,
        scalar: bool = False,
        vtype: str = gp.GRB.CONTINUOUS,
        lb: float = -gp.GRB.INFINITY,
        ub: float = gp.GRB.INFINITY,
    ) -> AnyVar | npt.NDArray[np.float64] | float:
        """Translate a CVXPY expression, and return a gurobipy variable constrained to its value.

        This is useful for gurobipy functions that only handle variables as their arguments.
        If translating the expression results in a variable, it is returned directly.
        Constants are also returned directly.
        If scalar is True, the result is guaranteed to be a scalar, otherwise its shape will be
        the shape of whatever gets generated while translating the node.
        """
        expr = self.visit(node)
        if isinstance(expr, gp.Var):
            return expr
        if isinstance(expr, (gp.MVar, np.ndarray)):
            if scalar:
                # Extract the underlying variable - will raise an error if the shape is not scalar
                return expr.item()  # type: ignore[return-value]
            return expr
        return self.make_auxilliary_variable_for(
            expr, node.__class__.__name__, vtype=vtype, lb=lb, ub=ub
        )

    def make_auxilliary_variable_for(
        self,
        expr: Any,
        atom_name: str,
        *,
        desired_shape: tuple[int, ...] | None = None,
        vtype: str = gp.GRB.CONTINUOUS,
        lb: float = -gp.GRB.INFINITY,
        ub: float = gp.GRB.INFINITY,
    ) -> AnyVar:
        """Add a variable constrained to the value of the given gurobipy expression."""
        desired_shape = (
            _squeeze_shape(_shape(expr)) if desired_shape is None else desired_shape
        )
        self._aux_id += 1
        var = add_variable(
            self.model,
            shape=desired_shape,
            name=f"{atom_name}_{self._aux_id}",
            vtype=vtype,
            lb=lb,
            ub=ub,
        )
        self.model.addConstr(var == expr)
        return var

    def apply_and_visit_elementwise(
        self, fn: Callable[[cp.Expression], Any], expr: cp.Expression
    ) -> Any:
        """Apply fn to each element of `expr` and return the array of results."""

        def visit(x: cp.Expression) -> Any:
            return self.visit(fn(x))

        vectorized_visitor = np.vectorize(visit, otypes=[np.object_])
        subarray = to_subexpressions_array(expr, shape=expr.shape)
        translated = vectorized_visitor(subarray)
        return promote_array_to_gurobi_matrixapi(translated)

    def star_apply_and_visit_elementwise(
        self, fn: Callable[[Any], Any], *exprs: cp.Expression
    ) -> Any:
        """Apply fn across all given expressions and return the array of results.

        The difference with `apply_and_visit_elementwise` is that fn is expected
        to take multiple scalar arguments.
        """

        def visit(args: tuple[cp.Expression, ...]) -> Any:
            return self.visit(fn(*args))

        vectorized_visitor = np.vectorize(visit, otypes=[np.object_])
        subarray = to_zipped_subexpressions_array(*exprs, shape=exprs[0].shape)
        translated = vectorized_visitor(subarray)
        return promote_array_to_gurobi_matrixapi(translated)

    def visit_abs(self, node: cp.abs) -> Any:
        (arg,) = node.args
        if isinstance(arg, cp.Constant):
            return np.abs(arg.value)
        if node.shape == ():
            var = self.translate_into_variable(arg, scalar=True)
            assert isinstance(var, gp.Var)
            return self.make_auxilliary_variable_for(gp.abs_(var), "abs", lb=0)
        return self.apply_and_visit_elementwise(cp.abs, arg)

    def visit_AddExpression(self, node: AddExpression) -> Any:
        args = list(map(self.visit, node.args))
        return reduce(operator.add, args)

    def visit_conj(self, node: cp.conj) -> Any:
        (arg,) = node.args
        if arg.is_complex():
            raise ComplexExpressionError(node)
        return self.visit(arg)

    def visit_Constant(self, const: cp.Constant) -> Any:
        return const.value

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
        if GUROBI_MAJOR < 12:
            raise InvalidNonlinearAtomError(node)
        from gurobipy import nlfunc

        (arg,) = node.args
        expr = self.visit(arg)
        return self.make_auxilliary_variable_for(
            nlfunc.exp(expr), "exp", desired_shape=_shape(expr)
        )

    def _stack(self, node: Hstack | Vstack, gp_fn: Callable) -> Any:
        args = node.args
        exprs = [self.visit(arg) for arg in args]
        return gp_fn(exprs)

    def visit_Hstack(self, node: Hstack) -> Any:
        if GUROBIPY_VERSION < (11,):
            raise UnsupportedExpressionError(node)
        return self._stack(node, gp.hstack)

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
        if GUROBI_MAJOR < 12:
            raise InvalidNonlinearAtomError(node)
        from gurobipy import nlfunc

        (arg,) = node.args
        expr = self.visit(arg)
        return self.make_auxilliary_variable_for(
            nlfunc.log(expr), "log", desired_shape=_shape(expr)
        )

    def visit_log1p(self, node: cp.log1p) -> AnyVar:
        if GUROBI_MAJOR < 12:
            raise InvalidNonlinearAtomError(node)
        from gurobipy import nlfunc

        (arg,) = node.args
        expr = self.visit(arg)
        return self.make_auxilliary_variable_for(
            nlfunc.log(expr + 1), "log1p", desired_shape=_shape(expr)
        )

    def _min_max(
        self,
        node: cp.min | cp.max,
        gp_fn: Callable[[list[gp.Var]], Any],
        np_fn: Callable[[Any], float],
        name: str,
    ) -> Any:
        (arg,) = node.args
        if isinstance(arg, cp.Constant):
            return np_fn(arg.value)
        if _is_scalar_shape(arg.shape):
            # min/max of a scalar is itself
            return self.visit(arg)

        var = self.translate_into_variable(arg)
        assert isinstance(var, gp.MVar)  # other cases were handled above
        return self.make_auxilliary_variable_for(gp_fn(var.reshape(-1).tolist()), name)

    def visit_max(self, node: cp.max) -> Any:
        return self._min_max(node, gp_fn=gp.max_, np_fn=np.max, name="max")

    def visit_min(self, node: cp.min) -> Any:
        return self._min_max(node, gp_fn=gp.min_, np_fn=np.min, name="min")

    def _minimum_maximum(
        self, node: cp.minimum | cp.maximum, gp_fn: Callable[[Any], Any], name: str
    ) -> Any:
        args = node.args

        if _is_scalar_shape(node.shape):
            varargs = [self.translate_into_variable(arg, scalar=True) for arg in args]
            return self.make_auxilliary_variable_for(gp_fn(varargs), name)

        return self.star_apply_and_visit_elementwise(type(node), *args)  # pyright: ignore[reportArgumentType]

    def visit_maximum(self, node: cp.maximum) -> Any:
        return self._minimum_maximum(node, gp_fn=gp.max_, name="maximum")

    def visit_minimum(self, node: cp.minimum) -> Any:
        return self._minimum_maximum(node, gp_fn=gp.min_, name="minimum")

    def visit_Maximize(self, objective: cp.Maximize) -> None:
        obj = self.translate_into_scalar(objective.expr)
        self.model.setObjective(obj, sense=gp.GRB.MAXIMIZE)

    def visit_Minimize(self, objective: cp.Minimize) -> None:
        obj = self.translate_into_scalar(objective.expr)
        self.model.setObjective(obj, sense=gp.GRB.MINIMIZE)

    def visit_MulExpression(self, node: MulExpression) -> Any:
        x, y = node.args
        x = self.visit(x)
        y = self.visit(y)
        return x @ y

    def visit_multiply(self, mul: multiply) -> Any:
        return self.visit(mul.args[0]) * self.visit(mul.args[1])

    def visit_NegExpression(self, node: NegExpression) -> Any:
        return -self.visit(node.args[0])

    def _handle_norm(
        self, node: cp.norm1 | cp.Pnorm | cp.norm_inf, p: float, name: str
    ) -> Any:
        (x,) = node.args
        if isinstance(x, cp.Constant):
            return np.linalg.norm(x.value.ravel(), p)
        arg = self.translate_into_variable(x)
        assert isinstance(arg, (gp.Var, gp.MVar))
        varargs = [arg] if isinstance(arg, gp.Var) else arg.reshape(-1).tolist()
        norm = gp.norm(varargs, p)
        return self.make_auxilliary_variable_for(norm, name, lb=0)

    def visit_norm1(self, node: cp.norm1) -> Any:
        return self._handle_norm(node, 1, "norm1")

    def visit_Pnorm(self, node: cp.Pnorm) -> Any:
        if node.p != 2:
            raise InvalidNormError(node)
        return self._handle_norm(node, 2, "norm2")

    def visit_norm_inf(self, node: cp.norm_inf) -> Any:
        return self._handle_norm(node, np.inf, "norminf")

    def visit_power(self, node: power) -> Any:
        power = self.visit(node.p)
        if power != 2:
            raise InvalidPowerError(node.p)
        arg = self.visit(node.args[0])
        return arg**power

    def visit_Problem(self, problem: cp.Problem) -> None:
        self.visit(problem.objective)
        for constraint in problem.constraints:
            self.model.addConstr(self.visit(constraint), name=str(constraint.constr_id))
        self.model.update()

    def visit_Promote(self, node: Promote) -> Any:
        # FIXME: should we do something here?
        return self.visit(node.args[0])

    def visit_QuadForm(self, node: cp.QuadForm) -> Any:
        vec, psd_mat = node.args
        vec = self.visit(vec)
        psd_mat = self.visit(psd_mat)
        return vec @ psd_mat @ vec

    def visit_quad_over_lin(self, node: quad_over_lin) -> Any:
        x, y = node.args
        x = self.visit(x)
        squares = (((x[i]).item()) ** 2 for i in np.ndindex(x.shape))
        quad = gp.quicksum(squares)
        lin = self.visit(y)
        return quad / lin

    def visit_reshape(self, node: cp.reshape) -> gp.MVar | npt.NDArray[np.float64]:
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
        if isinstance(expr, gp.Var):
            expr = gp.MVar.fromvar(expr)
        elif isinstance(expr, np.ndarray) or sp.issparse(expr):
            return expr.reshape(target_shape)
        elif not isinstance(expr, gp.MVar):
            expr_shape = _shape(expr)
            # Force creation of an MVar even if the shape is scalar
            if expr_shape == ():
                expr_shape = (1,)
            expr = self.make_auxilliary_variable_for(
                expr, "reshape", desired_shape=expr_shape
            )
            assert isinstance(expr, gp.MVar)
        return expr.reshape(target_shape)

    def visit_special_index(self, node: special_index) -> Any:
        return self.visit(node.args[0])[node.key]

    def visit_Sum(self, node: Sum) -> Any:
        expr = self.visit(node.args[0])
        if _is_scalar(expr):
            return expr
        return expr.sum(axis=node.axis)

    def visit_Variable(self, var: cp.Variable) -> AnyVar:
        if var.id not in self.vars:
            self.vars[var.id] = translate_variable(var, self.model)
            self.model.update()
        return self.vars[var.id]

    def visit_Vstack(self, node: Vstack) -> Any:
        if GUROBIPY_VERSION < (11,):
            raise UnsupportedExpressionError(node)
        return self._stack(node, gp.vstack)
