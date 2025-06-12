from typing import Literal

import torch
from audiomentations.augmentations.time_mask import TimeMask as _TimeMask  # type: ignore[import-untyped]

from ..audio_augs import AudioAug


class TimeMask(AudioAug):
    def __init__(
        self,
        sr: int,
        p: float,
        min_band_part: float = 0.01,
        max_band_part: float = 0.2,
        fade_duration: float = 0.005,
        mask_location: Literal["start", "end", "random"] = "random",
    ) -> None:
        super().__init__(sr=sr, p=p)

        self.time_mask: _TimeMask = _TimeMask(
            min_band_part=min_band_part,
            max_band_part=max_band_part,
            fade_duration=fade_duration,
            mask_location=mask_location,
            p=1.0,
        )

    def process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.time_mask(samples=x.numpy(), sample_rate=self.sr))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sr={self.sr}, p={self.p}, "
            f"min_band_part={self.time_mask.min_band_part}, "
            f"max_band_part={self.time_mask.max_band_part}, "
            f"fade_duration={self.time_mask.fade_duration}, "
            f"mask_location='{self.time_mask.mask_location}')"
        )
