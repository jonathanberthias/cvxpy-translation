import inspect
from typing import Any

import cvxpy as cp
import gurobipy as gp
import pyscipopt as scip
import pytest

import cvxpy_translation
import test_problems
from cvxpy_translation.gurobi.translation import Translater as GrbTranslater
from cvxpy_translation.scip.translation import Translater as ScipTranslater


@pytest.fixture(params=[GrbTranslater(gp.Model()), ScipTranslater(scip.Model())])
def translater(request: pytest.FixtureRequest) -> Any:
    return request.param


@pytest.mark.xfail(reason="TODO: implement all atoms")
def test_no_missing_atoms(translater: Any) -> None:
    missing = {
        atom
        for atom in cp.EXP_ATOMS + cp.PSD_ATOMS + cp.SOC_ATOMS + cp.NONPOS_ATOMS
        if inspect.isclass(atom)
        and getattr(translater, f"visit_{atom.__name__}", None) is None  # type: ignore[attr-defined]
    }
    assert missing == set()


@pytest.fixture(params=test_problems.all_invalid_problems())
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
