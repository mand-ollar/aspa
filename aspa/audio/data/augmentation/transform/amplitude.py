import torch
from audiomentations.augmentations.gain import Gain  # type: ignore[import-untyped]

from ..audio_augs import AudioAug


class Amplitude(AudioAug):
    def __init__(self, sr: int, p: float, min_factor: float = 0.2, max_factor: float = 2.0) -> None:
        super(Amplitude, self).__init__(sr=sr, p=p)

        self.min_factor: float = min_factor
        self.max_factor: float = max_factor

        self.gain: Gain = Gain(min_gain_db=self.min_factor, max_gain_db=self.max_factor, p=1.0)

    def process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.gain(samples=x.numpy(), sample_rate=self.sr))

    def __repr__(self) -> str:
        return f"Amplitude(sr={self.sr}, p={self.p}, min_factor={self.min_factor}, max_factor={self.max_factor})"
