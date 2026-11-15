from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self


class Timer:
    # Initialized by __exit__ and __enter__, respectively.
    time: float  # pyrefly: ignore[uninitialized-instance-variable]
    _start: float  # pyrefly: ignore[uninitialized-instance-variable]

    def __enter__(self) -> Self:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        end = time.perf_counter()
        self.time = end - self._start
