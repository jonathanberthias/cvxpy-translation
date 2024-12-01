from __future__ import annotations

import math
import warnings
from itertools import chain
from typing import TYPE_CHECKING

import cvxpy as cp
import cvxpy.settings as s
import gurobipy as gp
import pytest

import cvxpy_gurobi
from test_problems import ProblemTestCase
from test_problems import all_valid_problems

if TYPE_CHECKING:
    from pathlib import Path

    from cvxpy.reductions.solution import Solution
    from pytest_insta.fixture import SnapshotFixture


@pytest.fixture(params=all_valid_problems(), ids=lambda case: case.group)
def case(request: pytest.FixtureRequest) -> ProblemTestCase:
    return request.param


def test_lp(case: ProblemTestCase, snapshot: SnapshotFixture, tmp_path: Path) -> None:
    """Generate LP output for CVXPY and Gurobi..

    This test requires human intervention to check the differences in the
    generated snapshot files.
    """
    problem = case.problem
    cvxpy_lines = lp_from_cvxpy(problem)

    quiet_solve(problem)
    generated_model = problem.solver_stats.extra_stats
    cvxpy_gurobi_lines = lp_from_gurobi(generated_model, tmp_path)

    model = cvxpy_gurobi.build_model(problem)
    gurobi_lines = lp_from_gurobi(model, tmp_path)

    divider = ["-" * 40]
    output = "\n".join(
        chain(
            ["CVXPY"],
            cvxpy_lines,
            divider,
            ["AFTER COMPILATION"],
            cvxpy_gurobi_lines,
            divider,
            ["GUROBI"],
            gurobi_lines,
        )
    )

    if cvxpy_gurobi.CVXPY_VERSION[:2] == (1, 5):
        assert snapshot() == output
    else:
        # don't update snapshots nor delete them
        snapshot.session.strategy = "update-none"


def test_backfill(case: ProblemTestCase) -> None:
    problem = case.problem
    cvxpy_gurobi.solve(problem, **{gp.GRB.Param.QCPDual: 1})
    our_sol: Solution = problem.solution
    our_model: gp.Model = our_sol.attr[s.EXTRA_STATS]
    assert our_model.Status == gp.GRB.Status.OPTIMAL

    quiet_solve(problem)
    cp_sol: Solution = problem.solution

    assert our_sol.status == cp_sol.status
    assert our_sol.opt_val == pytest.approx(cp_sol.opt_val, abs=1e-7, rel=1e-6)
    assert set(our_sol.primal_vars) == set(cp_sol.primal_vars)
    for key in our_sol.primal_vars:
        assert our_sol.primal_vars[key] == pytest.approx(
            cp_sol.primal_vars[key], rel=2e-4
        )
    # Dual values are not available for MIPs
    # Sometimes, the Gurobi model is a MIP even though the CVXPY problem is not,
    # notably when using genexprs
    # So we only check the dual values if the model is not a MIP
    # This is one point where we cannot guarantee that our solution is the same as CVXPY's
    # if the dual values are important
    if not our_model.IsMIP:
        assert set(our_sol.dual_vars) == set(cp_sol.dual_vars)
        for key in our_sol.dual_vars:
            assert our_sol.dual_vars[key] == pytest.approx(cp_sol.dual_vars[key])
    assert set(our_sol.attr) >= set(cp_sol.attr)
    # In some cases, iteration count can be negative??
    cp_iters = max(cp_sol.attr.get(s.NUM_ITERS, math.inf), 0)
    assert our_sol.attr[s.NUM_ITERS] <= cp_iters


def lp_from_cvxpy(problem: cp.Problem) -> list[str]:
    sense, expr = str(problem.objective).split(" ", 1)
    out = [sense.capitalize(), f"  {expr}", "Subject To"]
    for constraint in problem.constraints:
        out += [f" {constraint.constr_id}: {constraint}"]
    bounds: list[str] = []
    binaries: list[str] = []
    generals: list[str] = []
    for variable in problem.variables():
        integer = variable.attributes["integer"]
        boolean = variable.attributes["boolean"]
        if variable.domain:
            bounds.extend(f" {d}" for d in variable.domain)
        elif not boolean:
            bounds.append(f" {variable} free")
        if integer:
            generals.append(f" {variable}")
        elif boolean:
            binaries.append(f" {variable}")
    out.extend(["Bounds", *bounds])
    if binaries:
        out.extend(["Binaries", *binaries])
    if generals:
        out.extend(["Generals", *generals])
    out.append("End")
    return out


def lp_from_gurobi(model: gp.Model, tmp_path: Path) -> list[str]:
    out_path = tmp_path / "gurobi.lp"
    model.write(str(out_path))
    return out_path.read_text().splitlines()[1:]


def quiet_solve(problem: cp.Problem) -> None:
    with warnings.catch_warnings():
        # Some problems are unbounded
        warnings.filterwarnings("ignore", category=UserWarning)
        problem.solve(solver=cp.GUROBI)
