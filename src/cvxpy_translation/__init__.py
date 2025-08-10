import importlib.metadata

from cvxpy_translation._version import __version__ as __version__
from cvxpy_translation._version import __version_tuple__ as __version_tuple__
from cvxpy_translation.exceptions import InvalidParameterError as InvalidParameterError
from cvxpy_translation.exceptions import (
    UnsupportedConstraintError as UnsupportedConstraintError,
)
from cvxpy_translation.exceptions import UnsupportedError as UnsupportedError
from cvxpy_translation.exceptions import (
    UnsupportedExpressionError as UnsupportedExpressionError,
)
from cvxpy_translation.interface import backfill_problem as backfill_problem
from cvxpy_translation.interface import build_model as build_model
from cvxpy_translation.interface import (
    register_translation_solver as register_translation_solver,
)
from cvxpy_translation.interface import solve as solve

try:
    CVXPY_VERSION = tuple(map(int, importlib.metadata.version("cvxpy").split(".")[:3]))
except importlib.metadata.PackageNotFoundError:
    CVXPY_VERSION = tuple(
        map(int, importlib.metadata.version("cvxpy-base").split(".")[:3])
    )
