import torch
from audiomentations.augmentations.add_gaussian_snr import AddGaussianSNR  # type: ignore[import-untyped]

from ..audio_augs import AudioAug


class GaussianNoise(AudioAug):
    def __init__(self, sr: int, p: float, min_snr_db: float = 5, max_snr_db: float = 40) -> None:
        super().__init__(sr=sr, p=p)

        self.add_gaussian_snr: AddGaussianSNR = AddGaussianSNR(min_snr_db=min_snr_db, max_snr_db=max_snr_db, p=1.0)

    def process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.add_gaussian_snr(samples=x.numpy(), sample_rate=self.sr))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sr={self.sr}, p={self.p}, "
            f"min_snr_db={self.add_gaussian_snr.min_snr_db}, "
            f"max_snr_db={self.add_gaussian_snr.max_snr_db})"
        )
