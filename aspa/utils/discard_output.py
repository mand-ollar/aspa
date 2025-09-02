import sys
import warnings
from copy import copy
from io import TextIOWrapper
from typing import Any, Pattern, Sequence


class DiscardOutput:
    def write(self, txt: str) -> None:
        """Pass Write"""

    def flush(self) -> None:
        """Pass Flush"""


class SuppressOutput:
    original_stdout: TextIOWrapper | Any = sys.stdout
    warning_filters: Sequence[tuple[str, Pattern[str] | None, type[Warning], Pattern[str] | None, int]] = copy(
        warnings.filters
    )

    def suppress(self) -> None:
        """Suppress stdout."""
        self.original_stdout = sys.stdout
        sys.stdout = DiscardOutput()

    def restore(self) -> None:
        """Restore stdout."""
        sys.stdout = self.original_stdout

    def suppress_warnings(self) -> None:
        warnings.filterwarnings("ignore")

    def restore_warnings(self) -> None:
        warnings.resetwarnings()
        warnings.filters = self.warning_filters
