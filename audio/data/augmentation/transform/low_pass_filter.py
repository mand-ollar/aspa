import torch
from audiomentations.augmentations.low_pass_filter import LowPassFilter  # type: ignore[import-untyped]

from ..audio_augs import AudioAug


class LPF(AudioAug):
    def __init__(
        self,
        sr: int,
        p: float,
        min_cutoff_freq: float = 150,
        max_cutoff_freq: float = 7500,
        min_rolloff: int = 12,
        max_rolloff: int = 24,
        zero_phase: bool = False,
    ) -> None:
        super().__init__(sr=sr, p=p)

        self.lpf: LowPassFilter = LowPassFilter(
            min_cutoff_freq=min_cutoff_freq,
            max_cutoff_freq=max_cutoff_freq,
            min_rolloff=min_rolloff,
            max_rolloff=max_rolloff,
            zero_phase=zero_phase,
            p=1.0,
        )

    def process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.lpf(samples=x.numpy(), sample_rate=self.sr))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sr={self.sr}, p={self.p}, "
            f"min_cutoff_freq={self.lpf.min_cutoff_freq}, max_cutoff_freq={self.lpf.max_cutoff_freq}, "
            f"min_rolloff={self.lpf.min_rolloff}, max_rolloff={self.lpf.max_rolloff}, "
            f"zero_phase={self.lpf.zero_phase})"
        )
