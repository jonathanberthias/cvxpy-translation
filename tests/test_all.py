from __future__ import annotations

import math
import warnings
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Final
from unittest.mock import patch

import cvxpy as cp
import cvxpy.settings as s
import gurobipy as gp
import pyscipopt as scip
import pytest
from cvxpy.reductions.solvers.conic_solvers.scip_conif import SCIP
from pytest_insta.fixture import SnapshotFixture

import cvxpy_translation.gurobi
import cvxpy_translation.scip
from cvxpy_translation import CVXPY_VERSION
from test_problems import ProblemTestCase
from test_problems import all_valid_problems

if TYPE_CHECKING:
    from collections.abc import Generator

    from cvxpy.reductions.solution import Solution
    from pytest_insta.session import SnapshotSession


PARAMS: Final = {
    cp.GUROBI: {gp.GRB.Param.QCPDual: 1},
    cp.SCIP: {
        "numerics/feastol": 1e-10,
        "numerics/dualfeastol": 1e-10,
        "limits/time": 10,
    },
}

SNAPSHOT_DIR: Final = Path(__file__).parent / "snapshots"


def _format_test_id(case: ProblemTestCase) -> str:
    """Format the test ID used by pytest for a given test case."""
    return f"{case.context.solver.lower()}_{case.group}_{case.idx:02d}"


@pytest.fixture(params=all_valid_problems(), ids=_format_test_id)
def case(request: pytest.FixtureRequest) -> ProblemTestCase:
    test_case: ProblemTestCase = request.param
    if test_case.skip_reason:
        pytest.skip(test_case.skip_reason)
    return test_case


@pytest.fixture
def snapshot(
    request: pytest.FixtureRequest, case: ProblemTestCase
) -> Generator[SnapshotFixture]:
    """Replace SnapshotFixture.from_request to inject the solver name in the path.

    Yields:
        SnapshotFixture: A fixture that can be used to create snapshots for the test.

    """
    path = SNAPSHOT_DIR / case.context.solver.lower() / "lp"
    session: SnapshotSession = request.config._no_deletion_snapshot_session  # noqa: SLF001  # pyright: ignore[reportAttributeAccessIssue]
    with SnapshotFixture(session[path], session) as fixture:
        yield fixture


def test_lp(case: ProblemTestCase, snapshot: SnapshotFixture, tmp_path: Path) -> None:
    """Generate LP output for CVXPY and the tested solver.

    This test requires human intervention to check the differences in the
    generated snapshot files.
    """
    problem = case.problem
    cvxpy_lines = lp_from_cvxpy(problem)

    try:
        generated_model = quiet_solve(
            problem, case.context.solver, params=PARAMS[case.context.solver]
        )
    except Exception as e:  # noqa: BLE001
        # The solver interfaces in cvxpy can't solve some problems
        cvxpy_interface_lines = [str(e)]
    else:
        cvxpy_interface_lines = lp_from_solver(
            generated_model, case.context.solver, tmp_path
        )

    model = cvxpy_translation.build_model(problem, solver=case.context.solver)
    model_lines = lp_from_solver(model, case.context.solver, tmp_path)

    divider = ["-" * 40]
    output = "\n".join(
        chain(
            ["CVXPY"],
            cvxpy_lines,
            divider,
            ["AFTER COMPILATION"],
            cvxpy_interface_lines,
            divider,
            [case.context.solver.upper()],
            model_lines,
        )
    )

    if CVXPY_VERSION[:2] == (1, 7):
        assert snapshot(f"{case.group}_{case.idx:02d}.txt") == output
    else:
        # don't update snapshots nor delete them
        snapshot.session.strategy = "update-none"


def test_backfill(case: ProblemTestCase) -> None:
    if case.context.solver == cp.GUROBI:
        check_backfill_gurobi(case)
    elif case.context.solver == cp.SCIP:
        check_backfill_scip(case)
    else:
        msg = f"Unsupported solver: {case.context.solver}"
        raise ValueError(msg)


def check_backfill_gurobi(case: ProblemTestCase) -> None:
    problem = case.problem
    params = PARAMS[case.context.solver]
    cvxpy_translation.gurobi.solve(problem, **params)
    our_sol: Solution = problem.solution
    our_model: gp.Model = our_sol.attr[s.EXTRA_STATS]
    assert our_model.Status == gp.GRB.Status.OPTIMAL
    assert our_sol.opt_val is not None
    assert our_sol.primal_vars

    try:
        quiet_solve(problem, case.context.solver, params=params)
    except cp.SolverError:
        pytest.skip(f"Problem can't be solved through CVXPY with {case.context.solver}")

    cp_sol: Solution = problem.solution
    if not case.invalid_reason and cp_sol.status not in s.SOLUTION_PRESENT:
        pytest.skip(f"CVXPY's solution status is unexpected: {cp_sol.status}")

    assert our_sol.status == cp_sol.status
    assert our_sol.opt_val == pytest.approx(cp_sol.opt_val, abs=1e-7, rel=1e-6)
    assert set(our_sol.primal_vars) == set(cp_sol.primal_vars)
    for key in our_sol.primal_vars:
        assert our_sol.primal_vars[key] == pytest.approx(
            cp_sol.primal_vars[key], rel=2e-4, abs=1e-6
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
            assert our_sol.dual_vars[key] == pytest.approx(
                cp_sol.dual_vars[key], abs=2e-6
            )
    assert set(our_sol.attr) >= set(cp_sol.attr)
    # In some cases, iteration count can be negative??
    cp_iters = max(cp_sol.attr.get(s.NUM_ITERS, math.inf), 0)
    assert our_sol.attr[s.NUM_ITERS] <= cp_iters


def check_backfill_scip(case: ProblemTestCase) -> None:
    problem = case.problem
    params = PARAMS[case.context.solver]
    our_model = cvxpy_translation.scip.build_model(problem)
    # Same settings as in cvxpy's SCIP interface
    our_model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
    our_model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
    our_model.disablePropagation()
    our_model.setParams(params)
    our_model.optimize()
    cvxpy_translation.scip.backfill_problem(problem, our_model)
    our_sol: Solution = problem.solution
    assert our_model.getStatus() == "optimal"
    assert our_sol.opt_val is not None
    assert our_sol.primal_vars

    try:
        quiet_solve(problem, case.context.solver, params=params)
    except Exception:  # noqa: BLE001
        pytest.skip(f"Problem can't be solved through CVXPY with {case.context.solver}")

    cp_sol: Solution = problem.solution

    assert our_sol.status == cp_sol.status
    assert our_sol.opt_val == pytest.approx(cp_sol.opt_val, abs=1e-6, rel=1e-6)
    assert set(our_sol.primal_vars) == set(cp_sol.primal_vars)
    for key in our_sol.primal_vars:
        assert our_sol.primal_vars[key] == pytest.approx(
            cp_sol.primal_vars[key], abs=1e-5, rel=1e-4
        )

    skip_duals = (
        # CVXPY's SCIP interface does not return dual values for MIPs
        case.problem.is_mixed_integer()
        # Dual values are not available for nonlinear constraints
        # The CVXPY SCIP interface does return dual values but I doubt they are correct
        or any(c.getConshdlrName() == "nonlinear" for c in our_model.getConss())
    )
    if not skip_duals:
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


def lp_from_solver(model: Any, solver: str, tmp_path: Path) -> list[str]:
    if isinstance(model, gp.Model):
        return lp_from_gurobi(model, tmp_path)
    if isinstance(model, scip.Model):
        return lp_from_scip(model, tmp_path)
    msg = f"Unsupported solver model type: {type(model)} for {solver}"
    raise ValueError(msg)


def lp_from_gurobi(model: gp.Model, tmp_path: Path) -> list[str]:
    out_path = tmp_path / "gurobi.lp"
    model.write(str(out_path))
    return out_path.read_text().splitlines()[1:]


def lp_from_scip(model: scip.Model, tmp_path: Path) -> list[str]:
    out_path = tmp_path / "scip.lp"
    model.writeProblem(str(out_path), verbose=False)
    return out_path.read_text().splitlines()[4:]


def quiet_solve(
    problem: cp.Problem, solver: str, params: dict
) -> gp.Model | scip.Model:
    with warnings.catch_warnings():
        # Some problems are unbounded
        warnings.filterwarnings("ignore", category=UserWarning)

        if solver == cp.GUROBI:
            # Gurobi's solve method returns a Model object
            problem.solve(solver=cp.GUROBI, **params)
            generated_model = problem.solution.attr[s.EXTRA_STATS]

        elif solver == cp.SCIP:
            old_opt = SCIP._solve  # noqa: SLF001
            generated_model = None

            def new_solve(
                self: SCIP, model: scip.Model, *args: Any, **kwargs: Any
            ) -> Any:
                """Capture the model generated during the solving process."""
                nonlocal generated_model
                generated_model = model
                return old_opt(self, model, *args, **kwargs)

            with patch.object(SCIP, "_solve", new=new_solve):
                problem.solve(solver=cp.SCIP, **params)

        else:
            msg = f"Unsupported solver: {solver}"
            raise ValueError(msg)

        assert generated_model is not None
        return generated_model
