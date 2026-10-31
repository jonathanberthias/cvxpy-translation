from __future__ import annotations

import inspect
from typing import Any
from typing import cast

import cvxpy as cp
import gurobipy as gp
import numpy as np
import pyscipopt as scip
import pytest
import scipy.sparse as sp

import cvxpy_translation
import test_problems
from cvxpy_translation.gurobi.translation import Translater as GrbTranslater
from cvxpy_translation.scip.translation import Translater as ScipTranslater


@pytest.fixture(params=[cp.GUROBI, cp.SCIP])
def translater(request: pytest.FixtureRequest) -> Any:
    if request.param == cp.GUROBI:
        return GrbTranslater(gp.Model())
    return ScipTranslater(scip.Model())


@pytest.mark.xfail(reason="TODO: implement all atoms")
def test_no_missing_atoms(translater: Any) -> None:
    missing = {
        atom
        for atom in cp.EXP_ATOMS + cp.PSD_ATOMS + cp.SOC_ATOMS + cp.NONPOS_ATOMS
        if inspect.isclass(atom)
        and getattr(translater, f"visit_{atom.__name__}", None) is None
    }
    assert missing == set()


@pytest.fixture(
    params=test_problems.all_invalid_problems(),
    ids=lambda case: f"{case.context.solver}_{case.idx}",
)
def invalid_case(request: pytest.FixtureRequest) -> test_problems.ProblemTestCase:
    return request.param


@pytest.fixture
def invalid_case_translater(invalid_case: test_problems.ProblemTestCase) -> Any:
    return (
        GrbTranslater(gp.Model())
        if invalid_case.context.solver == cp.GUROBI
        else ScipTranslater(scip.Model())
    )


def test_failing_atoms(
    invalid_case: test_problems.ProblemTestCase, invalid_case_translater: Any
) -> None:
    if invalid_case.skip_reason:
        pytest.skip(invalid_case.skip_reason)
    with pytest.raises(cvxpy_translation.UnsupportedExpressionError) as exc:
        invalid_case_translater.visit(invalid_case.problem)
    assert type(exc.value) is not cvxpy_translation.UnsupportedExpressionError


def test_parameter(translater: Any) -> None:
    p = cp.Parameter()
    # Non-happy path raises
    with pytest.raises(cvxpy_translation.InvalidParameterError):
        translater.visit(p)
    # Happy path succeeds
    p.value = 1
    translater.visit(p)


def test_parameter_reshape(translater: Any) -> None:
    """From https://github.com/jonathanberthias/cvxpy-translation/issues/76.

    Parameter.value is not necessarily a numpy/scipy array,
    so reshaping is not always straightforward.
    """
    p = cp.Parameter()
    p.value = 1
    translater.visit(cp.reshape(p, (1,), order="C"))


@pytest.mark.parametrize(
    "sparse_format", ["bsr", "coo", "csc", "csr", "dia", "dok", "lil"]
)
@pytest.mark.parametrize("kind", ["matrix", "array"])
def test_sparse_constant(translater: Any, sparse_format: str, kind: str) -> None:
    constructor = getattr(sp, f"{sparse_format}_{kind}", None)
    if constructor is None:
        pytest.skip("Sparse arrays are unavailable in older SciPy versions")
    dense = np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 4.0]])
    const = cp.Constant(constructor(dense))

    translated = translater.visit(const)
    if isinstance(translater, ScipTranslater):
        assert isinstance(translated, np.ndarray)
    else:
        assert sp.issparse(translated)
    np.testing.assert_array_equal(sp.csr_matrix(translated).toarray(), dense)

    reshaped = translater.visit(cp.reshape(const, (3, 2), order="C"))
    assert reshaped.shape == (3, 2)
    np.testing.assert_array_equal(
        sp.csr_matrix(reshaped).toarray(), dense.reshape(3, 2)
    )


def test_gurobi_indexed_constant() -> None:
    translater = GrbTranslater(gp.Model())
    result = translater.translate_into_variable(cp.Constant([2.0, 3.0])[0])
    assert result == 2.0
    translater.model.update()
    assert translater.model.NumVars == 0


@pytest.mark.parametrize(
    ("arg", "axis", "keepdims"),
    [
        pytest.param(cp.Variable(), None, True, id="scalar"),
        pytest.param(cp.Variable(1), None, True, id="singleton-vector"),
        pytest.param(cp.Variable((1, 1)), 0, True, id="singleton-matrix"),
        pytest.param(cp.Variable((1, 1)), 0, False, id="drop-singleton-axis"),
        pytest.param(
            cp.maximum(cp.Variable((1, 1)), 0), 0, True, id="scalarized-matrix"
        ),
        pytest.param(cp.Variable((2, 3)), None, True, id="matrix-all"),
        pytest.param(cp.Variable((2, 3)), np.int64(0), True, id="numpy-integer-axis"),
        pytest.param(cp.Variable((2, 3)) + 1, 0, True, id="affine-columns"),
        pytest.param(cp.square(cp.Variable((2, 3))), 1, True, id="quadratic-rows"),
        pytest.param(cp.Constant(np.ones((2, 3))), 0, True, id="constant-columns"),
        pytest.param(
            cp.Parameter((2, 3), value=np.ones((2, 3))), 1, True, id="parameter-rows"
        ),
        pytest.param(
            cp.Variable((2, 3)),
            -1,
            True,
            id="negative-axis",
            marks=pytest.mark.skipif(
                cvxpy_translation.CVXPY_VERSION < (1, 6, 6),
                reason="CVXPY's negative-axis support is broken before 1.6.6",
            ),
        ),
    ],
)
def test_sum_shape(
    translater: Any, arg: cp.Expression, axis: int | None, keepdims: bool
) -> None:
    # CVXPY's sum wrapper also accepts iterables and infers a possible int result.
    expr = cast("cp.Expression", cp.sum(arg, axis=axis, keepdims=keepdims))
    translated = translater.visit(expr)
    assert getattr(translated, "shape", ()) == expr.shape
    for axis_to_index in range(expr.ndim):
        key = tuple(0 if i == axis_to_index else slice(None) for i in range(expr.ndim))
        indexed = expr[key]
        translated = translater.visit(indexed)
        assert getattr(translated, "shape", ()) == indexed.shape
