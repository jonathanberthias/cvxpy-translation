import cvxpy as cp
import pytest

import cvxpy_gurobi
import test_problems
from cvxpy_gurobi import ExpressionTranslater


@pytest.mark.xfail(reason="TODO: implement all atoms")
def test_no_missing_atoms() -> None:
    missing = [
        atom
        for atom in cp.EXP_ATOMS + cp.PSD_ATOMS + cp.SOC_ATOMS + cp.NONPOS_ATOMS
        if getattr(ExpressionTranslater, f"visit_{atom.__name__}", None) is None  # type: ignore[attr-defined]
    ]
    assert missing == []


@pytest.mark.parametrize("case", test_problems.invalid_expressions())
def test_failing_atoms(case) -> None:
    translater = ExpressionTranslater({})
    with pytest.raises(cvxpy_gurobi.UnsupportedError):
        translater.visit(case.problem.objective.expr)
