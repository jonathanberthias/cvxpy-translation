# Changelog

## Unreleased

## [1.1.1] - 2025-02-01

This small release fixes a bug with manually set parameter values and adds
testing for 3.13 now that Gurobi supports it.

### Fixed

- Reshaping a constant with a Python scalar value no longer errors due to
  missing `reshape` method
  ([#77](https://github.com/jonathanberthias/cvxpy-gurobi/pull/77)). Thanks to
  Halil Sen for reporting the bug!

### New

- Add support for Python 3.13
  ([#81](https://github.com/jonathanberthias/cvxpy-gurobi/pull/81))

## [1.1.0] - 2024-12-01

### Newly supported atoms

- `cp.quad_form` expressions are handled, both when the vector is a variable and
  when the PSD matrix is a variable
  ([#60](https://github.com/jonathanberthias/cvxpy-gurobi/pull/60)).
- `cp.Parameter`s that have a value assigned are treated like constants
  ([#67](https://github.com/jonathanberthias/cvxpy-gurobi/pull/67)). Thanks to
  Halil Sen for contributing this feature!

### Dependencies

Add support for CVXPY 1.6 and Gurobi 12.

## [1.0.0] - 2024-09-28

### Newly supported atoms

- CVXPY atoms that have an equivalent generalized expression in `gurobipy` are
  correctly translated. This is done by adding auxilliary variables constrained
  to the value of the arguments of the atom to the problem:
  - `abs` ([#27](https://github.com/jonathanberthias/cvxpy-gurobi/pull/27),
    [#30](https://github.com/jonathanberthias/cvxpy-gurobi/pull/30)),
  - `min`/`max`
    ([#31](https://github.com/jonathanberthias/cvxpy-gurobi/pull/31)),
  - `minimum`/`maximum`
    ([#34](https://github.com/jonathanberthias/cvxpy-gurobi/pull/34),
    [#45](https://github.com/jonathanberthias/cvxpy-gurobi/pull/45),
    [#51](https://github.com/jonathanberthias/cvxpy-gurobi/pull/51),
    [#58](https://github.com/jonathanberthias/cvxpy-gurobi/pull/58)),
  - `norm1`/`norm2`/`norm_inf`
    ([#35](https://github.com/jonathanberthias/cvxpy-gurobi/pull/35),
    [#36](https://github.com/jonathanberthias/cvxpy-gurobi/pull/36),
    [#37](https://github.com/jonathanberthias/cvxpy-gurobi/pull/37)).
- `reshape` atoms are handled during translation
  ([#42](https://github.com/jonathanberthias/cvxpy-gurobi/pull/42)).
- The `hstack` and `vstack` atoms are translated into their `gurobipy`
  counterparts, available from Gurobi 11
  ([#43](https://github.com/jonathanberthias/cvxpy-gurobi/pull/43),
  [#44](https://github.com/jonathanberthias/cvxpy-gurobi/pull/44)).

### Fixed

- The `axis` argument to `cp.sum` is no longer ignored
  ([#39](https://github.com/jonathanberthias/cvxpy-gurobi/pull/39)).
- If a scalar expression is given to `cp.sum`, it no longer raises an error
  ([#48](https://github.com/jonathanberthias/cvxpy-gurobi/pull/48)).
- The dual values should be more correct in cases where the sign is reversed
  between `cvxpy` and `gurobipy`
  ([#50](https://github.com/jonathanberthias/cvxpy-gurobi/pull/50)).

### Dependencies

The `numpy` and `scipy` dependencies have lower bounds, set willingly to fairly
old versions ([#56](https://github.com/jonathanberthias/cvxpy-gurobi/pull/56)).

### Testing

- The library is tested in CI against the oldest supported versions and the
  latest releases
  ([#56](https://github.com/jonathanberthias/cvxpy-gurobi/pull/56)).
- All test problems must be feasible and bounded to ensure they have a unique
  solution ([#50](https://github.com/jonathanberthias/cvxpy-gurobi/pull/50)).
- Backfilling infeasible and unbounded problems is explicitly tested
  ([#53](https://github.com/jonathanberthias/cvxpy-gurobi/pull/53)).

### Removed

The `variable_map` argument used when filling a `Model` was removed. Instead,
the variable map is handled by the `Translater` internally
([#24](https://github.com/jonathanberthias/cvxpy-gurobi/pull/24)). In the
future, there will be an official way to provide custom translations which is
not limited to variables.

## [0.1.0] - 2024-08-01

This is the first release of `cvxpy-gurobi`!

The core idea of the package is in place and the solver API is not expected to
change. However, only basic expressions and constraints are easily manageable
and many internal changes will be required to add support for expressions which
cannot be translated in a straightforward way, such as `cp.abs` that requires
`gurobipy`'s `GenExpr`.

In this release, the following elements are already covered:

- `AddExpression`
- `Constant`
- `DivExpression`
- `index` (indexing with integers)
- `MulExpression` (multiplication by a constant)
- `multiply` (element-wise multiplication)
- `NegExpression`
- `power` (only if `p` is 2)
- `Promote` (broadcasting)
- `quad_over_lin` (`sum_squares`)
- `special_index` (indexing with arrays)
- `Sum`
- `Variable` (duh)

[0.1.0]:
  https://github.com/jonathanberthias/cvxpy-gurobi/compare/7d97aaf...v0.1.0
[1.0.0]:
  https://github.com/jonathanberthias/cvxpy-gurobi/compare/v0.1.0...v1.0.0
[1.1.0]:
  https://github.com/jonathanberthias/cvxpy-gurobi/compare/v1.0.0...v1.1.0
[1.1.1]:
  https://github.com/jonathanberthias/cvxpy-gurobi/compare/v1.1.0...v1.1.1
