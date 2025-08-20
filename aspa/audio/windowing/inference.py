from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import WindowingDataset


class ModelInference(ABC):
    def __init__(
        self,
        model: nn.Module,
        dataset: WindowingDataset,
        device: str | torch.device,
        batch_size: int,
        num_workers: int,
    ) -> None:
        self.model: nn.Module = model
        self.dataset: WindowingDataset = dataset
        self.device: str | torch.device = device
        self.model.to(self.device).eval()

        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def filter(self, logits: torch.Tensor) -> torch.Tensor: ...

    @torch.no_grad()
    def __call__(self) -> torch.Tensor:
        dataloader: DataLoader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        audio: torch.Tensor
        logits_list: list[torch.Tensor] = []
        for audio, _ in dataloader:
            audio = audio.to(self.device)
            logits_list.append(self.forward(x=audio))

        logits: torch.Tensor = torch.cat(logits_list, dim=0)

        return self.filter(logits=logits).cpu().detach()
