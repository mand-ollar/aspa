import warnings
from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.distributions as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from aspa.audio.sound_level import SoundLevel

from .dataset import BaseWindowingDataset


class BaseFilter(ABC):
    dataset: BaseWindowingDataset

    @abstractmethod
    def apply(self, dataset: BaseWindowingDataset) -> torch.Tensor:
        raise NotImplementedError


class ModelInferenceFilter(BaseFilter):
    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device,
        batch_size: int,
        num_workers: int,
    ) -> None:
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="multiprocessing")

        self.model: nn.Module = model
        self.device: str | torch.device = device
        self.model.to(self.device).eval()

        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def filter(self, logits: torch.Tensor) -> torch.Tensor: ...

    @torch.no_grad()
    def apply(self, dataset: BaseWindowingDataset) -> torch.Tensor:
        self.dataset = dataset

        dataloader: DataLoader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        audio: torch.Tensor
        logits_list: list[torch.Tensor] = []
        for audio, _ in tqdm(dataloader, desc="Inference", leave=False, ncols=80):
            audio = audio.to(self.device)
            logits_list.append(self.forward(x=audio))

        logits: torch.Tensor = torch.cat(logits_list, dim=0)

        return self.filter(logits=logits).cpu().detach()


class RMSFilter(BaseFilter):
    def __init__(
        self,
        orientation: Literal["le", "ge"],
        sound_level_meter: SoundLevel,
        absolute_threshold: float | None = None,
        relative_threshold: float | None = None,
    ) -> None:
        self.orientation: Literal["le", "ge"] = orientation
        self.sound_level_meter: SoundLevel = sound_level_meter

        self.mode: Literal["absolute", "relative"]
        self.threshold: float
        if absolute_threshold is not None:
            self.mode = "absolute"
            self.threshold = absolute_threshold
        elif relative_threshold is not None:
            self.mode = "relative"
            self.threshold = relative_threshold
        else:
            raise ValueError("Either absolute_threshold or relative_threshold must be provided")

    def _get_rms(self, sound_level_meter: SoundLevel) -> torch.Tensor:
        dataloader: DataLoader = DataLoader(
            dataset=self.dataset,
            batch_size=2048,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        audio: torch.Tensor
        rms_list: list[torch.Tensor] = []
        for audio, _ in tqdm(dataloader, desc="RMS", leave=False, ncols=80):
            rms_list.append(sound_level_meter(audio=audio))

        rms: torch.Tensor = torch.cat(rms_list, dim=0)

        return rms

    def apply(self, dataset: BaseWindowingDataset) -> torch.Tensor:
        self.dataset = dataset

        rms: torch.Tensor = self._get_rms(sound_level_meter=self.sound_level_meter)

        if self.mode == "absolute":
            if self.orientation == "le":
                return torch.where(rms <= self.threshold)[0]
            elif self.orientation == "ge":
                return torch.where(rms >= self.threshold)[0]
            else:
                raise ValueError(f"Invalid orientation: {self.orientation}")

        elif self.mode == "relative":
            z_score: torch.Tensor = (rms - rms.mean()) / rms.std()

            normal_dist = dist.Normal(0, 1)
            z_threshold: float = normal_dist.icdf(torch.tensor(self.threshold)).item()

            if self.orientation == "le":
                return torch.where(z_score <= z_threshold)[0]
            elif self.orientation == "ge":
                return torch.where(z_score >= z_threshold)[0]
            else:
                raise ValueError(f"Invalid orientation: {self.orientation}")

        else:
            raise ValueError(f"Invalid mode: {self.mode}")
