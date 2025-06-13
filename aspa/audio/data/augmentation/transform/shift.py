from typing import Literal

import torch
from audiomentations.augmentations.shift import Shift  # type: ignore[import-untyped]

from ..audio_augs import AudioAug


class TimeShift(AudioAug):
    def __init__(
        self,
        sr: int,
        p: float,
        min_shift: float = -0.5,
        max_shift: float = 0.5,
        shift_unit: Literal["fraction", "samples", "seconds"] = "fraction",
        rollover: bool = True,
        fade_duration: float = 0.005,
    ) -> None:
        super().__init__(sr=sr, p=p)

        self.min_shift: float = min_shift
        self.max_shift: float = max_shift

        self.shift: Shift = Shift(
            min_shift=min_shift,
            max_shift=max_shift,
            shift_unit=shift_unit,
            rollover=rollover,
            fade_duration=fade_duration,
            p=1.0,
        )

    def process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.shift(samples=x.numpy(), sample_rate=self.sr))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sr={self.sr}, p={self.p}, "
            f"min_shift={self.min_shift}, max_shift={self.max_shift}, "
            f"shift_unit='{self.shift.shift_unit}', rollover={self.shift.rollover}, "
            f"fade_duration={self.shift.fade_duration})"
        )
