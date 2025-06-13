import torch
from audiomentations.augmentations.add_color_noise import AddColorNoise  # type: ignore[import-untyped]

from ..audio_augs import AudioAug


class ColorNoise(AudioAug):
    def __init__(
        self,
        sr: int,
        p: float,
        min_snr_db: float = 5,
        max_snr_db: float = 40,
        min_f_decay: float = -6,
        max_f_decay: float = 6,
        p_apply_a_weighting: float = 0,
        n_fft: int = 128,
    ) -> None:
        super().__init__(sr=sr, p=p)

        self.add_color_noise: AddColorNoise = AddColorNoise(
            min_snr_db=min_snr_db,
            max_snr_db=max_snr_db,
            min_f_decay=min_f_decay,
            max_f_decay=max_f_decay,
            p_apply_a_weighting=p_apply_a_weighting,
            n_fft=n_fft,
            p=1.0,
        )

    def process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(
            self.add_color_noise(samples=x.numpy(), sample_rate=self.sr)
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sr={self.sr}, p={self.p}, "
            f"min_snr_db={self.add_color_noise.min_snr_db}, "
            f"max_snr_db={self.add_color_noise.max_snr_db}, "
            f"min_f_decay={self.add_color_noise.min_f_decay}, "
            f"max_f_decay={self.add_color_noise.max_f_decay}, "
            f"p_apply_a_weighting={self.add_color_noise.p_apply_a_weighting}, "
            f"n_fft={self.add_color_noise.n_fft})"
        )
