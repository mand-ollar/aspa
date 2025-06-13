import torch
from audiomentations.augmentations.polarity_inversion import (  # type: ignore[import-untyped]
    PolarityInversion as _PolarityInversion,
)

from ..audio_augs import AudioAug


class PolarityInversion(AudioAug):
    def __init__(self, sr: int, p: float) -> None:
        super().__init__(sr=sr, p=p)

        self.polarity_inversion: _PolarityInversion = _PolarityInversion(p=1.0)

    def process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(
            self.polarity_inversion(samples=x.numpy(), sample_rate=self.sr)
        )

    def __repr__(self) -> str:
        return f"PolarityInversion(sr={self.sr}, p={self.p})"
