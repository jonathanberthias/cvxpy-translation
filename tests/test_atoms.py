import cvxpy as cp
import gurobipy as gp
import pytest

import cvxpy_translation.gurobi
import test_problems
from cvxpy_translation.gurobi.translation import Translater


@pytest.mark.xfail(reason="TODO: implement all atoms")
def test_no_missing_atoms() -> None:
    missing = [
        atom
        for atom in cp.EXP_ATOMS + cp.PSD_ATOMS + cp.SOC_ATOMS + cp.NONPOS_ATOMS
        if getattr(Translater, f"visit_{atom.__name__}", None) is None  # type: ignore[attr-defined]
    ]
    assert missing == []


@pytest.mark.parametrize("case", test_problems.all_invalid_problems())
def test_failing_atoms(case: test_problems.ProblemTestCase) -> None:
    if case.skip_reason:
        pytest.skip(case.skip_reason)
    translater = Translater(gp.Model())
    with pytest.raises(cvxpy_translation.gurobi.UnsupportedExpressionError):
        translater.visit(case.problem.objective.expr)


def test_parameter() -> None:
    translater = Translater(gp.Model())
    p = cp.Parameter()
    # Non-happy path raises
    with pytest.raises(cvxpy_translation.gurobi.InvalidParameterError):
        translater.visit(p)
    # Happy path succeeds
    p.value = 1
    translater.visit(p)


def test_parameter_reshape() -> None:
    """From https://github.com/jonathanberthias/cvxpy-translation/issues/76.

    Parameter.value is not necessarily a numpy/scipy array,
    so reshaping is not always straightforward.
    """
    translater = Translater(gp.Model())
    p = cp.Parameter()
    p.value = 1
    translater.visit(cp.reshape(p, (1,), order="C"))
