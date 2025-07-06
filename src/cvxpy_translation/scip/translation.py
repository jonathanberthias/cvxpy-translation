from __future__ import annotations

import importlib.metadata
import operator
from functools import reduce
from math import prod
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import Literal
from typing import Union
from typing import overload

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import pyscipopt as scip
import scipy.sparse as sp
from pyscipopt.recipes.nonlinear import set_nonlinear_objective

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


try:
    CVXPY_VERSION = tuple(map(int, importlib.metadata.version("cvxpy").split(".")))
except importlib.metadata.PackageNotFoundError:
    CVXPY_VERSION = tuple(map(int, importlib.metadata.version("cvxpy-base").split(".")))

AnyVar: TypeAlias = Union[scip.Variable, scip.MatrixVariable]
Param: TypeAlias = Union[str, float]
ParamDict: TypeAlias = Dict[str, Param]


class UnsupportedError(ValueError):
    msg_template = "Unsupported CVXPY node: {node}"

    def __init__(self, node: Canonical) -> None:
        super().__init__(self.msg_template.format(node=node, klass=type(node)))
        self.node = node


class UnsupportedConstraintError(UnsupportedError):
    msg_template = "Unsupported CVXPY constraint: {node}"


class UnsupportedExpressionError(UnsupportedError):
    msg_template = "Unsupported CVXPY expression: {node} ({klass})"


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


class InvalidParameterError(UnsupportedExpressionError):
    msg_template = "Unsupported parameter: value for {node} is not set"


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


def promote_array_to_scip_matrixapi(array: npt.NDArray[np.object_]) -> Any:
    """Promote an array of Gurobi objects to the equivalent Gurobi matrixapi object."""
    kind = type(array.flat[0])
    if issubclass(kind, scip.Variable):
        return scip.MatrixVariable(array)
    # TODO: support other types
    msg = f"Cannot promote array of {kind}"
    raise NotImplementedError(msg)  # pragma: no cover


def translate_variable(var: cp.Variable, model: scip.Model) -> AnyVar:
    lb = None
    ub = None
    if var.is_nonneg():
        lb = 0
    if var.is_nonpos():
        ub = 0

    vtype = "CONTINUOUS"
    if var.attributes["integer"]:
        vtype = "INTEGER"
    if var.attributes["boolean"]:
        vtype = "BINARY"

    return add_variable(model, var.shape, lb=lb, ub=ub, vtype=vtype, name=var.name())


@overload
def add_variable(
    model: scip.Model, shape: tuple[()], name: str, vtype: str, lb: float, ub: float
) -> scip.Variable: ...
@overload
def add_variable(
    model: scip.Model,
    shape: tuple[int, ...],
    name: str,
    vtype: str,
    lb: float,
    ub: float,
) -> AnyVar: ...
def add_variable(
    model: scip.Model,
    shape: tuple[int, ...],
    name: str,
    vtype: str = "CONTINUOUS",
    lb: float | None = None,
    ub: float | None = None,
) -> AnyVar:
    if shape == ():
        return model.addVar(name=name, lb=lb, ub=ub, vtype=vtype)
    return model.addMatrixVar(shape, name=name, lb=lb, ub=ub, vtype=vtype)


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

        if isinstance(node, cp.Constraint):
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

    def translate_into_variable(
        self,
        node: cp.Expression,
        *,
        scalar: bool = False,
        vtype: str = "CONTINUOUS",
        lb: float | None = None,
        ub: float | None = None,
    ) -> AnyVar | npt.NDArray[np.float64] | float:
        """Translate a CVXPY expression, and return a gurobipy variable constrained to its value.

        This is useful for gurobipy functions that only handle variables as their arguments.
        If translating the expression results in a variable, it is returned directly.
        Constants are also returned directly.
        If scalar is True, the result is guaranteed to be a scalar, otherwise its shape will be
        the shape of whatever gets generated while translating the node.
        """
        expr = self.visit(node)
        if isinstance(expr, scip.Variable):
            return expr
        if isinstance(expr, (scip.MatrixVariable, np.ndarray)):
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
        vtype: str = "CONTINUOUS",
        lb: float | None = None,
        ub: float | None = None,
    ) -> AnyVar:
        """Add a variable constrained to the value of the given SCIP expression."""
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
        if isinstance(var, scip.Variable):
            self.model.addCons(var == expr)
        else:
            assert isinstance(var, scip.MatrixVariable)
            self.model.addMatrixCons(var == expr)
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
        return promote_array_to_scip_matrixapi(translated)

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
        return promote_array_to_scip_matrixapi(translated)

    def visit_abs(self, node: cp.abs) -> Any:
        (arg,) = node.args
        if isinstance(arg, cp.Constant):
            return np.abs(arg.value)
        if node.shape == ():
            var = self.translate_into_variable(arg, scalar=True)
            assert isinstance(var, scip.Var)
            return self.make_auxilliary_variable_for(scip.abs_(var), "abs", lb=0)
        return self.apply_and_visit_elementwise(cp.abs, arg)

    def visit_AddExpression(self, node: AddExpression) -> Any:
        args = list(map(self.visit, node.args))
        return reduce(operator.add, args)

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
        expr = self.visit(arg)
        return self.make_auxilliary_variable_for(
            scip.exp(expr), "exp", desired_shape=_shape(expr)
        )

    def _stack(
        self,
        node: Hstack | Vstack,
        axis: Literal[0, 1],
        name: Literal["hstack", "vstack"],
    ) -> Any:
        self._aux_id += 1
        axis = min(axis, len(node.shape) - 1)
        var = add_variable(self.model, shape=node.shape, name=f"{name}_{self._aux_id}")
        start_idx = 0
        for arg in node.args:
            expr = self.visit(arg)
            arg_shape = arg.shape
            arg_shape += (1,) * (len(node.shape) - len(arg_shape))
            indices = np.arange(start_idx, start_idx + arg.shape[axis])
            self.model.addMatrixCons(var.take(indices, axis=axis) == expr)
            start_idx += arg.shape[axis]
        return var

    def visit_Hstack(self, node: Hstack) -> Any:
        return self._stack(node, axis=1, name="hstack")

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
        expr = self.visit(arg)
        return self.make_auxilliary_variable_for(
            scip.log(expr), "log", desired_shape=_shape(expr)
        )

    def visit_log1p(self, node: cp.log1p) -> AnyVar:
        (arg,) = node.args
        expr = self.visit(arg)
        return self.make_auxilliary_variable_for(
            scip.log(expr + 1), "log1p", desired_shape=_shape(expr)
        )

    def _min_max(
        self,
        node: cp.min | cp.max,
        scip_fn: Callable[[list[scip.Var]], Any],
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
        assert isinstance(var, scip.MVar)  # other cases were handled above
        return self.make_auxilliary_variable_for(
            scip_fn(var.reshape(-1).tolist()), name
        )

    def visit_max(self, node: cp.max) -> Any:
        return self._min_max(node, scip_fn=scip.max_, np_fn=np.max, name="max")

    def visit_min(self, node: cp.min) -> Any:
        return self._min_max(node, scip_fn=scip.min_, np_fn=np.min, name="min")

    def _minimum_maximum(
        self, node: cp.minimum | cp.maximum, scip_fn: Callable[[Any], Any], name: str
    ) -> Any:
        args = node.args

        if _is_scalar_shape(node.shape):
            varargs = [self.translate_into_variable(arg, scalar=True) for arg in args]
            return self.make_auxilliary_variable_for(scip_fn(varargs), name)

        return self.star_apply_and_visit_elementwise(type(node), *args)  # pyright: ignore[reportArgumentType]

    def visit_maximum(self, node: cp.maximum) -> Any:
        return self._minimum_maximum(node, scip_fn=scip.max_, name="maximum")

    def visit_minimum(self, node: cp.minimum) -> Any:
        return self._minimum_maximum(node, scip_fn=scip.min_, name="minimum")

    def visit_Maximize(self, objective: cp.Maximize) -> None:
        obj = self.translate_into_scalar(objective.expr)
        self.model.setObjective(obj, sense="maximize")

    def visit_Minimize(self, objective: cp.Minimize) -> None:
        obj = self.translate_into_scalar(objective.expr)
        if obj.degree() > 1:
            set_nonlinear_objective(self.model, obj, sense="minimize")
        else:
            self.model.setObjective(obj, sense="minimize")

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
        assert isinstance(arg, (scip.Var, scip.MVar))
        varargs = [arg] if isinstance(arg, scip.Var) else arg.reshape(-1).tolist()
        norm = scip.norm(varargs, p)
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
            cons = self.visit(constraint)
            if isinstance(cons, scip.scip.ExprCons):
                self.model.addCons(cons, name=str(constraint.constr_id))
            elif isinstance(cons, scip.scip.MatrixExprCons):
                self.model.addMatrixCons(cons, name=str(constraint.constr_id))
            else:
                msg = f"Unexpected constraint type: {type(cons)}"
                raise TypeError(msg)

    def visit_Promote(self, node: Promote) -> Any:
        # FIXME: should we do something here?
        return self.visit(node.args[0])

    def visit_QuadForm(self, node: cp.QuadForm) -> scip.Expr:
        vec, psd_mat = node.args
        vec = self.visit(vec)
        psd_mat = self.visit(psd_mat)
        quad = vec @ psd_mat @ vec.T
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
        if isinstance(x, cp.Constant):
            try:
                return x.value.reshape(target_shape)
            except AttributeError:
                return np.reshape(x, target_shape)
        expr = self.visit(x)
        if isinstance(expr, scip.Expr):
            if target_shape == ():
                return expr
            expr = np.array([expr]).view(scip.MatrixExpr)
        elif target_shape == ():
            assert isinstance(expr, scip.MatrixExpr)
            assert _is_scalar(expr)
            return expr.item()
        elif not isinstance(expr, scip.MatrixExpr):
            expr_shape = _shape(expr)
            # Force creation of a MatrixVariable even if the shape is scalar
            if expr_shape == ():
                expr_shape = (1,)
            expr = self.make_auxilliary_variable_for(
                expr, "reshape", desired_shape=expr_shape
            )
            assert isinstance(expr, scip.MatrixVariable)
        reshaped = expr.reshape(target_shape)
        assert reshaped.shape == target_shape
        return reshaped

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
        return self._stack(node, axis=0, name="vstack")
