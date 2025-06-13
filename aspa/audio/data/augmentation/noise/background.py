from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import torch
from audiomentations.augmentations.add_background_noise import AddBackgroundNoise  # type: ignore[import-untyped]
from numpy.typing import NDArray

from ..audio_augs import AudioAug


class BackgroundNoise(AudioAug):
    def __init__(
        self,
        sr: int,
        p: float,
        sounds_path: list[Path] | list[str] | Path | str,
        min_snr_db: float = 3,
        max_snr_db: float = 30,
        noise_rms: Literal["relative", "absolute"] = "relative",
        min_absolute_rms_db: float = -45,
        max_absolute_rms_db: float = -15,
        noise_transform: Optional[
            Callable[[NDArray[np.float32], int], NDArray[np.float32]]
        ] = None,
        lru_cache_size: int = 2,
    ) -> None:
        super().__init__(sr=sr, p=p)

        self.add_background_noise: AddBackgroundNoise = AddBackgroundNoise(
            sounds_path=sounds_path,
            min_snr_db=min_snr_db,
            max_snr_db=max_snr_db,
            noise_rms=noise_rms,
            min_absolute_rms_db=min_absolute_rms_db,
            max_absolute_rms_db=max_absolute_rms_db,
            noise_transform=noise_transform,
            lru_cache_size=lru_cache_size,
            p=1.0,
        )

    def process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(
            self.add_background_noise(samples=x.numpy(), sample_rate=self.sr)
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sr={self.sr}, p={self.p}, "
            f"sounds_path={self.add_background_noise.sounds_path}, "
            f"min_snr_db={self.add_background_noise.min_snr_db}, "
            f"max_snr_db={self.add_background_noise.max_snr_db}, "
            f"noise_rms={self.add_background_noise.noise_rms}, "
            f"min_absolute_rms_db={self.add_background_noise.min_absolute_rms_db}, "
            f"max_absolute_rms_db={self.add_background_noise.max_absolute_rms_db})"
        )
