from typing import Literal

import torch
from audiomentations.augmentations.pitch_shift import PitchShift as _PitchShift  # type: ignore[import-untyped]

from ..audio_augs import AudioAug


class PitchShift(AudioAug):
    def __init__(
        self,
        sr: int,
        p: float,
        min_semitones: float = -4.0,
        max_semitones: float = 4.0,
        method: Literal[
            "librosa_phase_vocoder", "signalsmith_stretch"
        ] = "signalsmith_stretch",
    ) -> None:
        super().__init__(sr=sr, p=p)

        self.min_semitones: float = min_semitones
        self.max_semitones: float = max_semitones

        self.pitch_shift: _PitchShift = _PitchShift(
            min_semitones=min_semitones,
            max_semitones=max_semitones,
            method=method,
            p=1.0,
        )

    def process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(
            self.pitch_shift(samples=x.numpy(), sample_rate=self.sr)
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sr={self.sr}, p={self.p}, "
            f"min_semitones={self.min_semitones}, max_semitones={self.max_semitones}, "
            f"method='{self.pitch_shift.method}')"
        )
