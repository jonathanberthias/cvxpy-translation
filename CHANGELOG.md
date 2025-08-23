# Changelog

## [2.0.0] - 2025-08-23

### New name!

The library has been renamed to `cvxpy-translation`! This was to reflect that we
are now able to translate problems into SCIP models in addition to Gurobi
models.

### New interface: SCIP

Problems can be translated to `pyscipopt.Model` instances. The feature set is on
par with Gurobi.

### Breaking changes

- The top-level functions `solve`, `build_model` and
  `register_translation_solver` now require a `solver` argument that must be the
  solver name from CVXPY. The previous functions for Gurobi are still available
  by importing from `cvxpy_translation.gurobi`.
- Exceptions are now split between general exceptions defined in the top-level
  `cvxpy_translation`, and solver specific exceptions available from the solver
  submodules, `cvxpy_translation.gurobi` and `cvxpy_translation.scip`.
- Unhandled attributes set on `Variable` and `Parameter` will now raise an error
  instead of being silently ignored
  ([#185](https://github.com/jonathanberthias/cvxpy-translation/pull/185),
  [#190](https://github.com/jonathanberthias/cvxpy-translation/pull/190))
- Support for EOL Python 3.8 has been dropped
  ([#179](https://github.com/jonathanberthias/cvxpy-translation/pull/179)).

### New features

- Support variable bounds
  ([#184](https://github.com/jonathanberthias/cvxpy-translation/pull/184))
- Support CVXPY 1.7
  ([#165](https://github.com/jonathanberthias/cvxpy-translation/pull/165))
- Add support for `cp.conj` which is used by `cp.quad_form` in CVXPY 1.7
  ([#165](https://github.com/jonathanberthias/cvxpy-translation/pull/165))

### Bug fixes

- Scalar quadratic forms were incorrectly handled in CVXPY versions before
  1.4.0, this now works as expected
  ([#146](https://github.com/jonathanberthias/cvxpy-translation/pull/146))
- An unnecessary variable was generated when parameters appeared in a
  `cp.reshape` expression
  ([#182](https://github.com/jonathanberthias/cvxpy-translation/pull/182))

## [1.2.0] - 2025-03-23

- Add support for `cp.exp`, `cp.log` and `cp.log1p` through the non-linear
  expressions added in Gurobi 12
  ([#86](https://github.com/jonathanberthias/cvxpy-translation/pull/86),
  [#87](https://github.com/jonathanberthias/cvxpy-translation/pull/87))

## [1.1.1] - 2025-02-01

This small release fixes a bug with manually set parameter values and adds
testing for 3.13 now that Gurobi supports it.

### Fixed

- Reshaping a constant with a Python scalar value no longer errors due to
  missing `reshape` method
  ([#77](https://github.com/jonathanberthias/cvxpy-translation/pull/77)). Thanks
  to Halil Sen for reporting the bug!

### New

- Add support for Python 3.13
  ([#81](https://github.com/jonathanberthias/cvxpy-translation/pull/81))

## [1.1.0] - 2024-12-01

### Newly supported atoms

- `cp.quad_form` expressions are handled, both when the vector is a variable and
  when the PSD matrix is a variable
  ([#60](https://github.com/jonathanberthias/cvxpy-translation/pull/60)).
- `cp.Parameter`s that have a value assigned are treated like constants
  ([#67](https://github.com/jonathanberthias/cvxpy-translation/pull/67)). Thanks
  to Halil Sen for contributing this feature!

### Dependencies

Add support for CVXPY 1.6 and Gurobi 12.

## [1.0.0] - 2024-09-28

### Newly supported atoms

- CVXPY atoms that have an equivalent generalized expression in `gurobipy` are
  correctly translated. This is done by adding auxilliary variables constrained
  to the value of the arguments of the atom to the problem:
  - `abs` ([#27](https://github.com/jonathanberthias/cvxpy-translation/pull/27),
    [#30](https://github.com/jonathanberthias/cvxpy-translation/pull/30)),
  - `min`/`max`
    ([#31](https://github.com/jonathanberthias/cvxpy-translation/pull/31)),
  - `minimum`/`maximum`
    ([#34](https://github.com/jonathanberthias/cvxpy-translation/pull/34),
    [#45](https://github.com/jonathanberthias/cvxpy-translation/pull/45),
    [#51](https://github.com/jonathanberthias/cvxpy-translation/pull/51),
    [#58](https://github.com/jonathanberthias/cvxpy-translation/pull/58)),
  - `norm1`/`norm2`/`norm_inf`
    ([#35](https://github.com/jonathanberthias/cvxpy-translation/pull/35),
    [#36](https://github.com/jonathanberthias/cvxpy-translation/pull/36),
    [#37](https://github.com/jonathanberthias/cvxpy-translation/pull/37)).
- `reshape` atoms are handled during translation
  ([#42](https://github.com/jonathanberthias/cvxpy-translation/pull/42)).
- The `hstack` and `vstack` atoms are translated into their `gurobipy`
  counterparts, available from Gurobi 11
  ([#43](https://github.com/jonathanberthias/cvxpy-translation/pull/43),
  [#44](https://github.com/jonathanberthias/cvxpy-translation/pull/44)).

### Fixed

- The `axis` argument to `cp.sum` is no longer ignored
  ([#39](https://github.com/jonathanberthias/cvxpy-translation/pull/39)).
- If a scalar expression is given to `cp.sum`, it no longer raises an error
  ([#48](https://github.com/jonathanberthias/cvxpy-translation/pull/48)).
- The dual values should be more correct in cases where the sign is reversed
  between `cvxpy` and `gurobipy`
  ([#50](https://github.com/jonathanberthias/cvxpy-translation/pull/50)).

### Dependencies

The `numpy` and `scipy` dependencies have lower bounds, set willingly to fairly
old versions
([#56](https://github.com/jonathanberthias/cvxpy-translation/pull/56)).

### Testing

- The library is tested in CI against the oldest supported versions and the
  latest releases
  ([#56](https://github.com/jonathanberthias/cvxpy-translation/pull/56)).
- All test problems must be feasible and bounded to ensure they have a unique
  solution
  ([#50](https://github.com/jonathanberthias/cvxpy-translation/pull/50)).
- Backfilling infeasible and unbounded problems is explicitly tested
  ([#53](https://github.com/jonathanberthias/cvxpy-translation/pull/53)).

### Removed

The `variable_map` argument used when filling a `Model` was removed. Instead,
the variable map is handled by the `Translater` internally
([#24](https://github.com/jonathanberthias/cvxpy-translation/pull/24)). In the
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
  https://github.com/jonathanberthias/cvxpy-translation/compare/7d97aaf...v0.1.0
[1.0.0]:
  https://github.com/jonathanberthias/cvxpy-translation/compare/v0.1.0...v1.0.0
[1.1.0]:
  https://github.com/jonathanberthias/cvxpy-translation/compare/v1.0.0...v1.1.0
[1.1.1]:
  https://github.com/jonathanberthias/cvxpy-translation/compare/v1.1.0...v1.1.1
[1.2.0]:
  https://github.com/jonathanberthias/cvxpy-translation/compare/v1.1.1...v1.2.0
[2.0.0]:
  https://github.com/jonathanberthias/cvxpy-translation/compare/v1.2.0...v2.0.0
