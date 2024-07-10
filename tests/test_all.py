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
from test_problems import all_valid_problems

if TYPE_CHECKING:
    from pathlib import Path

    from cvxpy.reductions.solution import Solution
    from pytest_insta.fixture import SnapshotFixture


@pytest.fixture(params=all_valid_problems())
def problem(request: pytest.FixtureRequest) -> cp.Problem:
    return request.param


@pytest.fixture
def solved_problem(problem: cp.Problem) -> cp.Problem:
    problem.solve(solver=cp.GUROBI)
    return problem


@pytest.fixture
def model(problem: cp.Problem) -> gp.Model:
    return cvxpy_gurobi.build_model(problem)


@pytest.fixture
def solved_model(model: gp.Model) -> gp.Model:
    model.optimize()
    return model


def test_lp_output(
    solved_problem: cp.Problem,
    solved_model: gp.Model,
    snapshot: SnapshotFixture,
    tmp_path: Path,
) -> None:
    """Generate LP output for CVXPY and Gurobi..

    This test requires human intervention to check the differences in the
    generated snapshot files.
    """
    cvxpy_lines = lp_from_cvxpy(solved_problem)
    cvxpy_gurobi_lines = lp_from_gurobi(
        solved_problem.solver_stats.extra_stats, tmp_path
    )
    gurobi_lines = lp_from_gurobi(solved_model, tmp_path)

    divider = ["----------------------------------------"]
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

    if cp.__version__.startswith("1.5"):
        assert snapshot() == output
    else:
        # don't update snapshots nor delete them
        snapshot.session.strategy = "update-none"


def test_backfill(problem: cp.Problem) -> None:
    cvxpy_gurobi.solve(problem, {gp.GRB.Param.QCPDual: 1})
    our_sol: Solution = problem.solution

    with warnings.catch_warnings():
        # Some problems are unbounded
        warnings.filterwarnings("ignore", category=UserWarning)
        problem.solve(solver=cp.GUROBI)
    cp_sol: Solution = problem.solution

    assert our_sol.status == cp_sol.status
    assert our_sol.opt_val == pytest.approx(cp_sol.opt_val, abs=1e-7, rel=1e-6)
    assert set(our_sol.primal_vars) == set(cp_sol.primal_vars)
    for key in our_sol.primal_vars:
        assert our_sol.primal_vars[key] == pytest.approx(cp_sol.primal_vars[key])
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
