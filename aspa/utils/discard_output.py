import sys
import warnings
from io import TextIOWrapper
from typing import Any


class DiscardOutput:
    def write(self, txt: str) -> None:
        """Pass Write"""

    def flush(self) -> None:
        """Pass Flush"""


class SuppressOutput:
    original_stdout: TextIOWrapper | Any = sys.stdout

    def suppress(self) -> None:
        """Suppress stdout."""
        self.original_stdout = sys.stdout
        sys.stdout = DiscardOutput()

        warnings.filterwarnings("ignore")

    def restore(self) -> None:
        """Restore stdout."""
        sys.stdout = self.original_stdout

        warnings.resetwarnings()
