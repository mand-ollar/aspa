import random
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from aspa.utils.audio import format_audio


class AudioAug(ABC, nn.Module):
    def __init__(self, sr: int, p: float) -> None:
        super(AudioAug, self).__init__()

        self.sr: int = sr
        self.p: float = p

    @abstractmethod
    def process(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("process method must be implemented in subclasses")

    def _format_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return format_audio(audio=x, sr=self.sr, target_dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor"
        device: torch.device = x.device

        if random.random() <= self.p:
            x = self.process(self._format_tensor(x=x))
        x = self._format_tensor(x=x).to(device=device)

        return x
