from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from cvxpy.expressions.leaf import Leaf
    from cvxpy.utilities.canonical import Canonical


class UnsupportedError(ValueError):
    msg_template = "Unsupported CVXPY node: {node}"

    def __init__(self, node: Canonical, **kwargs: Any) -> None:
        super().__init__(
            self.msg_template.format(node=node, klass=type(node), **kwargs)
        )
        self.node = node
        self.kwargs = kwargs


class UnsupportedConstraintError(UnsupportedError):
    msg_template = "Unsupported CVXPY constraint: {node}"


class UnsupportedExpressionError(UnsupportedError):
    msg_template = "Unsupported CVXPY expression: {node} ({klass})"


class UnsupportedAttributesError(UnsupportedExpressionError):
    msg_template = "Unsupported attributes for {node}: {attributes}"

    def __init__(self, leaf: Leaf, attributes: set[str]) -> None:
        super().__init__(leaf, attributes=sorted(attributes))
        self.unhandled = attributes


class UnsupportedPartialAttributesError(UnsupportedExpressionError):
    msg_template = (
        "Unsupported partial attribute {attribute} for {node}. Split the leaf instead."
    )

    def __init__(self, leaf: Leaf, attribute: str) -> None:
        super().__init__(leaf, attribute=attribute)


class InvalidParameterError(UnsupportedExpressionError):
    msg_template = "Unsupported parameter: value for {node} is not set"
