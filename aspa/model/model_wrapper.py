import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import torch

warnings.filterwarnings("ignore")


@runtime_checkable
class ModelProtocol(Protocol):
    def __call__(self, x: torch.Tensor) -> Any: ...
    def forward(self, x: torch.Tensor) -> Any: ...
    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> "ModelProtocol": ...
    def eval(self) -> "ModelProtocol": ...


class ModelWrapper(ABC):
    """Model Wrapper for inference."""

    def __init__(
        self,
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

        self.model_name: str = model_name
        self.version: str = version
        self._print(f"Model name: {self.model_name} @ version: {self.version}")

        self.task: Literal["classification", "tagging"] = task
        self._print(f"Task set to: {self.task}")

        # Model setup
        self._model: ModelProtocol
        self._classes: list[str] = []
        self._thresholds: dict[str, float] = {}
        self._sr: int = 0
        self._target_length: int = 0

    @property
    def classes(self) -> list[str]:
        assert self._classes is not None, "Model classes must be set before use"
        return self._classes

    @classes.setter
    def classes(self, classes: list[str]) -> None:
        self._classes = classes

    @property
    def thresholds(self) -> dict[str, float]:
        assert self._thresholds is not None, "Model thresholds must be set before use"
        return self._thresholds

    @thresholds.setter
    def thresholds(self, thresholds: dict[str, float]) -> None:
        self._thresholds = thresholds

    @property
    def sr(self) -> int:
        assert self._sr is not None, "Model sample rate must be set before use"
        return self._sr

    @sr.setter
    def sr(self, sr: int) -> None:
        self._sr = sr

    @property
    def target_length(self) -> int:
        assert self._target_length is not None, "Model target length must be set before use"
        return self._target_length

    @target_length.setter
    def target_length(self, target_length: int) -> None:
        self._target_length = target_length

    @property
    def model(self) -> Any:
        assert self._model is not None, "Model must be set before use"
        if self.classes == [] or self.thresholds == {} or self.sr == 0 or self.target_length == 0:
            raise ValueError("Model classes, thresholds, sample rate, and target length must be set before use")

        return self._model.to(self.device).eval()

    @model.setter
    def model(self, ckpt_path: str | Path | None) -> None:
        """Set the model. Put the model in self._model."""
        self._model = self.set_model(ckpt_path=ckpt_path)

        # Validate that all necessary properties are set before setting the model
        if not self.classes:
            raise ValueError("Model classes must be set before setting the model")
        if not self.thresholds:
            raise ValueError("Model thresholds must be set before setting the model")
        if self.sr == 0:
            raise ValueError("Model sample rate must be set before setting the model")
        if self.target_length == 0:
            raise ValueError("Model target length must be set before setting the model")

        self._print(f"Model set with checkpoint: {ckpt_path}")

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

    def confidences(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.logits(x)
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
