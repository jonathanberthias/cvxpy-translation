import cvxpy as cp
import gurobipy as gp
import pytest

import cvxpy_gurobi
import test_problems
from cvxpy_gurobi.translation import Translater


@pytest.mark.xfail(reason="TODO: implement all atoms")
def test_no_missing_atoms() -> None:
    missing = [
        atom
        for atom in cp.EXP_ATOMS + cp.PSD_ATOMS + cp.SOC_ATOMS + cp.NONPOS_ATOMS
        if getattr(Translater, f"visit_{atom.__name__}", None) is None  # type: ignore[attr-defined]
    ]
    assert missing == []


@pytest.mark.parametrize("case", test_problems.invalid_expressions())
def test_failing_atoms(case: test_problems.ProblemTestCase) -> None:
    translater = Translater(gp.Model())
    with pytest.raises(cvxpy_gurobi.UnsupportedExpressionError):
        translater.visit(case.problem.objective.expr)


def test_parameter() -> None:
    translater = Translater(gp.Model())
    p = cp.Parameter()
    # Non-happy path raises
    with pytest.raises(cvxpy_gurobi.InvalidParameterError):
        translater.visit(p)
    # Happy path succeeds
    p.value = 1
    translater.visit(p)


def test_parameter_reshape() -> None:
    """Parameter.value is not necessarily a numpy/scipy array,
    so reshaping is not always straightforward.

    See https://github.com/jonathanberthias/cvxpy-gurobi/issues/76
    """
    translater = Translater(gp.Model())
    p = cp.Parameter()
    p.value = 1
    translater.visit(cp.reshape(p, (1,), order="C"))
