# CVXPY x GUROBI

This small library provides an alternative way to solve CVXPY problems with
Gurobi.

## Usage

The library provides a solver that will translate a CVXPY `Problem` into a
`gurobipy.Model`, and optimize using Gurobi:

```python
import cvxpy as cp
import cvxpy_gurobi

problem = cp.Problem(cp.Maximize(cp.Variable(name="x", nonpos=True)))
cvxpy_gurobi.solve(problem)
assert problem.value == 0
```

The solver can also be registered with CVXPY and used as any other solver:

```python
import cvxpy as cp
from cvxpy_gurobi import GUROBI_TRANSLATION, solver

cvxpy_gurobi.register_solver()
# ^ this is the same as:
cp.Problem.register_solve_method(GUROBI_TRANSLATION, solver())

problem.solve(method=GUROBI_TRANSLATION)
```

This solver is a simple wrapper for the most common use case:

```python
from cvxpy_gurobi import build_model, backfill_problem

model = build_model(problem)
model.optimize()
backfill_problem(problem, model)
assert model.optVal == problem.value
```

The `build_model` function provided by this library translates the
`cvxpy.Problem` instance into an equivalent `gurobipy.Model`, and
`backfill_problem` sets the optimal values on the original problem.

> [!NOTE] Both functions must be used together as they rely on naming
> conventions to map variables and constraints between CVXPY and Gurobi.

The output of the `build_model` function is a standard `gurobipy.Model`
instance, which can be further customized prior to solving. This approach
enables you to manage how the model will be optimized.

## Installation

```sh
pip install cvxpy-gurobi
```

## CVXPY has an interface to Gurobi, why is this needed?

When using CVXPY's interface to Gurobi, the problems fed to Gurobi have been
pre-compiled by CVXPY, meaning the model is not exactly the same as the one you
have written. This is great for solvers with low-level APIs, such as SCS or
OSQP, but `gurobipy` allows you to express your models at a higher-level.

Providing the raw model to Gurobi is a better idea in general since the Gurobi
solver is able to compile the problem with a better accuracy. The chosen
algorithm can also be different depending on the way it is modelled, potentially
leading to better performance.

In addition, CVXPY does not give access to the model before solving it. CVXPY
must therefore make some choices for you, such as setting `QCPDual` to 1 on all
non-MIP models. Having access to the model can help if you want to handle the
call to `.optimize()` in a non-standard way, e.g. by sending it to an async
loop.

### Example

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

## Why not use `gurobipy` directly?

CVXPY has 2 main features: a modelling API and interfaces to many solvers. The
modelling API has a great design, whereas `gurobipy` feels like a thin layer
over the C API. The interfaces to other solvers can be useful to not have to
rewrite the problem when switching solvers.

# Supported versions

All supported versions of Python, CVXPY and `gurobipy` should work. However, due
to licensing restrictions, old versions of `gurobipy` cannot be tested in CI. If
you run into a bug, please open an issue in this repo specifying the versions
used.

# Contributing

It is _highly recommended_ to use [Hatch](https://hatch.pypa.io/latest/) for
development. It will handle all virtual environment management.

To lint and format the code, run:

```sh
hatch fmt
```

For testing, run:

```sh
hatch test
```

This will test the latest version of dependencies. You can also run
`hatch test --all` to test several combinations of the supported version range.

Make sure any change is tested through a snapshot test. To add a new test case,
build a simple CVXPY problem in `tests/test_problems.py` in the appropriate
category, then run:

```sh
hatch run update-snapshots
```

You can then check the output in the `tests/snapshots` folder is as expected.
