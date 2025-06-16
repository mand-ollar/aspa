import torch
from audiomentations.augmentations.loudness_normalization import LoudnessNormalization  # type: ignore[import-untyped]

from ..audio_augs import AudioAug


class Normalize(AudioAug):
    def __init__(self, sr: int, p: float, min_lufs: float = -31, max_lufs: float = -13) -> None:
        super().__init__(sr=sr, p=p)

        self.loudness_normalization: LoudnessNormalization = LoudnessNormalization(
            min_lufs=-31,
            max_lufs=-13,
            p=1.0,
        )

    def process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.loudness_normalization(samples=x.numpy(), sample_rate=self.sr))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sr={self.sr}, p={self.p}, "
            f"min_lufs={self.loudness_normalization.min_lufs}, "
            f"max_lufs={self.loudness_normalization.max_lufs})"
        )
