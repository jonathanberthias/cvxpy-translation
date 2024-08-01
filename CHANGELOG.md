# [0.1.0] - 2024-08-01

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


[0.1.0]: https://github.com/jonathanberthias/cvxpy-gurobi/compare/7d97aaf...0.1.0
