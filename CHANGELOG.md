# Changelog

## Unreleased

## [1.0.0] - 2024-09-28

### Newly supported atoms

- CVXPY atoms that have an equivalent generalized expression in
  `gurobipy` are correctly translated. This is done by adding auxilliary
  variables constrained to the value of the arguments of the atom to
  the problem:
  - `abs` (#27, #30),
  - `min`/`max` (#31),
  - `minimum`/`maximum` (#34, #45, #51, #58),
  - `norm1`/`norm2`/`norm_inf` (#35, #36, #37).
- `reshape` atoms are handled during translation (#42).
- The `hstack` and `vstack` atoms are translated into their
  `gurobipy` counterparts, available from Gurobi 11 (#43, #44).

### Fixed

- The `axis` argument to `cp.sum` is no longer ignored (#39).
- If a scalar expression is given to `cp.sum`, it no longer raises an error (#48).
- The dual values should be more correct in cases where the sign is reversed
  between `cvxpy` and `gurobipy` (#50).

### Dependencies

The `numpy` and `scipy` dependencies have lower bounds, set willingly
to fairly old versions (#56).

### Testing

- The library is tested in CI against the oldest supported versions and
  the latest releases (#56).
- All test problems must be feasible and bounded to ensure they have a
  unique solution (#50).
- Backfilling infeasible and unbounded problems is explicitly tested (#53).

### Removed

The `variable_map` argument used when filling a `Model` was removed.
Instead, the variable map is handled by the `Translater` internally (#24).
In the future, there will be an official way to provide custom translations
which is not limited to variables.


## [0.1.0] - 2024-08-01

This is the first release of `cvxpy-gurobi`!

The core idea of the package is in place and the solver API
is not expected to change. However, only basic expressions
and constraints are easily manageable and many internal changes
will be required to add support for expressions which cannot
be translated in a straightforward way, such as `cp.abs` that
requires `gurobipy`'s `GenExpr`.

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


[0.1.0]: https://github.com/jonathanberthias/cvxpy-gurobi/compare/7d97aaf...v0.1.0
[1.0.0]: https://github.com/jonathanberthias/cvxpy-gurobi/compare/v0.1.0...v1.0.0
