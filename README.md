# CVXPY translation

This small library provides an alternative way to solve CVXPY problems by
building solver's native models. It currently supports Gurobi and SCIP.

## Usage

The library provides a solver that will translate a CVXPY `Problem` into a
`gurobipy.Model` or a `pyscipopt.Model`, and solve using the direct interface:

```python
import cvxpy as cp

problem = cp.Problem(cp.Maximize(cp.Variable(name="x", nonpos=True)))
cvxpy_translation.solve(problem, solver=cp.GUROBI)
assert problem.value == 0
```

This solver is a simple wrapper for the most common use case:

```python
from cvxpy_translation import build_model, backfill_problem

model = build_model(problem, solver=cp.SCIP)
model.optimize()
backfill_problem(problem, model)
assert model.getObjVal() == problem.value
```

The `build_model` function provided by this library translates the `Problem`
instance into an equivalent `Model`, and `backfill_problem` sets the optimal
values on the original problem.

<!-- prettier-ignore -->
> [!NOTE]
> Both functions must be used together as they rely on naming conventions to
> map variables and constraints between the problem and the model.

The output of the `build_model` function is a `Model` instance, which can be
further customized prior to solving. This approach enables you to manage how the
model will be optimized, set parameters, or use features that aren't available
through CVXPY's interface.

## Installation

```sh
pip install cvxpy-translation
```

## CVXPY has an interface to Gurobi and SCIP, why is this needed?

When using CVXPY's interface, the problems fed to the solver have been
pre-compiled by CVXPY, meaning the model is not exactly the same as the one you
have written. This is great for solvers with low-level APIs, such as SCS or
OSQP, but `gurobipy` and `pyscipopt` allow you to express your models at a
higher-level.

Providing the raw model to the solver can be a better idea in general to let the
solver use its own heuristics. The chosen algorithm can be different depending
on the way it is modelled.

In addition, CVXPY does not give access to the model before solving it. CVXPY
must therefore make some choices for you, such as setting some parameters on the
generated model. Having access to the model can help if you want to handle the
call to `.optimize()` in a non-standard way, e.g. by calling `.optimizeAsync()`
in `gurobipy` or `solveConcurrent()` in `pyscipopt`. It is also required to set
callbacks.

Another feature is the ability to use the latest features of the solvers, such
as non-linear expressions in Gurobi, which are not yet supported by the Gurobi
interface in CVXPY.

### Example with Gurobi

Consider this QP problem:

```python
import cvxpy as cp

x = cp.Variable(name="x")
problem = cp.Problem(cp.Minimize((x-1) ** 2))
```

The problem will be sent to Gurobi as (in LP format):

```
Minimize
 [ 2 C0 ^2 ] / 2
Subject To
 R0: - C0 + C1 = 1
Bounds
 C0 free
 C1 free
End
```

Using this package, it will instead send:

```
Minimize
  - 2 x + Constant + [ 2 x ^2 ] / 2
Subject To
Bounds
 x free
 Constant = 1
End
```

Note that:

- the variable's name matches the user-defined problem;
- no extra (free) variables;
- no extra constraints.

## Why not use `gurobipy` or `pyscipopt` directly?

CVXPY has 2 main features: a modelling API and interfaces to many solvers. The
modelling API has a great design, whereas `gurobipy` and `pyscipopt` feel like a
thin layer over the C API. The interfaces to other solvers can be useful to not
have to rewrite the problem when switching solvers.

# Supported versions

All supported versions of Python and CVXPY should work.

The same goes for `gurobipy`. However, due to licensing restrictions, old
versions of `gurobipy` cannot be tested in CI. If you run into a bug, please
open an issue in this repo specifying the versions used.

Only versions of `pyscipopt` after 5.5.0 are supported as that is when the
matrix API was introduced.

# Contributing

[Hatch](https://hatch.pypa.io/latest/) is used for development. It will handle
all the dependencies when testing on multiple versions.

For testing, run:

```sh
hatch run latest:tests
```

This will test the latest version of dependencies. You can also run
`hatch run oldest:tests` to test the minimum required dependency versions.

Make sure any change is tested through a snapshot test. To add a new test case,
build a simple CVXPY problem in `tests/test_problems.py` in the appropriate
category, then run:

```sh
hatch run update-snapshots
```

You can then check the output in the `tests/snapshots` folder is as expected.

To lint the code, run:

```sh
ruff check
```

To format the code, run:

```sh
ruff format
```
