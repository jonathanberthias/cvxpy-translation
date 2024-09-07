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
