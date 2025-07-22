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
            self.device = torch.device("cpu")
            self._print("Using CPU")

        self.task: Literal["classification", "tagging"] = task
        self._print(f"Task set to: {self.task}")

        # Model setup
        self._model: ModelProtocol
        self.classes: list[str]
        self.sr: int
        self.target_length: int

    @property
    def model(self) -> Any:
        assert self._model is not None, "Model must be set before use"
        if self.classes == [] or self.sr is None or self.target_length is None:
            raise ValueError("Model classes, sample rate, and target length must be set before use")

        return self._model.to(self.device).eval()

    @model.setter
    def model(self, ckpt_path: str | Path | None) -> None:
        """Set the model. Put the model in self._model."""
        self._model = self.set_model(ckpt_path=ckpt_path)
        self._print(f"Model set with checkpoint: {ckpt_path}")

    @abstractmethod
    def set_model(self, ckpt_path: str | Path | None) -> Any: ...

    def _pre_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._format_tensor(x=x)
        x = x.to(self.device)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pre_forward(x=x)

        return self.model(x)[0]

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
        assert self.target_length is not None, "Target length must be set before testing logits"

        self._print("Testing logits with zero tensor", end="")
        print(f"Result: {self.logits(torch.zeros(1, self.target_length, dtype=torch.float32))}\n")
        self._print("Testing logits with one tensor", end="")
        print(f"Result: {self.logits(torch.ones(1, self.target_length, dtype=torch.float32))}\n")
