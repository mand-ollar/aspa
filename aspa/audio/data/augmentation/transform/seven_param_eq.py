import torch
from audiomentations.augmentations.seven_band_parametric_eq import SevenBandParametricEQ  # type: ignore[import-untyped]

from ..audio_augs import AudioAug


class Equalizer(AudioAug):
    def __init__(
        self,
        sr: int,
        p: float,
        min_gain_db: float = -12,
        max_gain_db: float = 12,
    ) -> None:
        super().__init__(sr=sr, p=p)

        self.eq: SevenBandParametricEQ = SevenBandParametricEQ(min_gain_db=min_gain_db, max_gain_db=max_gain_db, p=1.0)

    def process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.eq(samples=x.numpy(), sample_rate=self.sr))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sr={self.sr}, p={self.p}, "
            f"min_gain_db={self.eq.min_gain_db}, max_gain_db={self.eq.max_gain_db})"
        )
