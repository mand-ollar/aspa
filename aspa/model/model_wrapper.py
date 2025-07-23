import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


class ModelWrapper(ABC):
    """Model Wrapper for inference."""

    def __init__(
        self,
        ckpt_path: str | Path | None,
        classes: list[str],
        thresholds: dict[str, float],
        sr: int,
        target_length: int,
        model_name: str,
        version: str = "0.1.0",
        gpu_id: int | None = None,
        task: Literal["classification", "tagging"] = "tagging",
    ) -> None:
        # Device setup
        self.device: torch.device
        if torch.cuda.is_available():
            if gpu_id is None:
                gpu_id = 0
            self.device = torch.device(f"cuda:{gpu_id}")
            self._print(f"Using GPU: {gpu_id}")
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self._print("Using MPS")
            else:
                self.device = torch.device("cpu")
                self._print("Using CPU")

        self.ckpt_path: str | Path | None = ckpt_path
        self.model_name: str = model_name
        self.version: str = version
        self._print(f"Model name: {self.model_name} @ version: {self.version}")

        self.task: Literal["classification", "tagging"] = task
        self._print(f"Task set to: {self.task}")

        self.apply_configurations(classes=classes, thresholds=thresholds, sr=sr, target_length=target_length)
        self._print(
            f"\n{pd.DataFrame(data={'classes': self.classes, 'thresholds': self.thresholds.values()}).set_index('classes').T.to_markdown(tablefmt='grid', index=False)}"  # noqa: E501
        )

        self._model: nn.Module | nn.Sequential

    def apply_configurations(
        self, classes: list[str], thresholds: dict[str, float], sr: int, target_length: int
    ) -> None:
        self.classes = classes
        self.thresholds = thresholds
        self.sr = sr
        self.target_length = target_length

    @property
    def model(self) -> nn.Module | nn.Sequential:
        return self._model.to(self.device).eval()

    @model.setter
    def model(self, model: nn.Module | nn.Sequential) -> None:
        self._model = model

    @abstractmethod
    def set_model(self, ckpt_path: str | Path | None) -> Any:
        """Set the model.
        Additionally, set `self.classes`, `self.sr`, and `self.target_length`.

        Args:
            ckpt_path: Path to the checkpoint file.

        Returns:
            The model.
        """

    def _pre_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._format_tensor(x=x)
        x = x.to(self.device)

        return x

    @abstractmethod
    def forward_impl(self, x: torch.Tensor) -> torch.Tensor | Any:
        """Forward pass implementation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor | Any:
        x = self._pre_forward(x=x)

        return self.forward_impl(x=x)

    def _format_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.reshape(1, 1, -1)
        elif x.dim() == 2:
            x = x.reshape(x.size(0), 1, -1)
        assert x.dim() == 3, "Input tensor must be 3D (batch_size, channel, seq_len)"

        return x

    @torch.no_grad()
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.forward(x=x)

        return logits

    def confidences(self, x: torch.Tensor, bypass: bool = False) -> torch.Tensor:
        logits: torch.Tensor = self.logits(x)
        if bypass:
            return logits

        confidences: torch.Tensor
        if self.task == "classification":
            confidences = torch.softmax(logits, dim=-1)
        elif self.task == "tagging":
            confidences = torch.sigmoid(logits)
        else:
            raise ValueError(f"Invalid task: {self.task}")

        confidences = confidences.reshape(-1, len(self.classes)) if self.classes else confidences

        return confidences

    @staticmethod
    def _print(msg: str = "", end="\n") -> None:
        print(f"[ModelWrapper] {msg}", end=end)
        print()

    def test_logits(self) -> None:
        self._print("Testing logits with zero tensor", end="")
        print(f"Result: {self.logits(torch.zeros(1, self.target_length, dtype=torch.float32))}\n")

        self._print("Testing logits with one tensor", end="")
        print(f"Result: {self.logits(torch.ones(1, self.target_length, dtype=torch.float32))}\n")
