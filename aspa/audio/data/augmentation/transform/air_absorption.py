import torch
from audiomentations.augmentations.air_absorption import AirAbsorption as _AirAbsorption  # type: ignore[import-untyped]

from ..audio_augs import AudioAug


class AirAbsorption(AudioAug):
    def __init__(
        self,
        sr: int,
        p: float,
        min_temperature: float = 10,
        max_temperature: float = 20,
        min_humidity: float = 30,
        max_humidity: float = 90,
        min_distance: float = 10,
        max_distance: float = 100,
    ) -> None:
        super().__init__(sr=sr, p=p)

        self.air_absorption: _AirAbsorption = _AirAbsorption(
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            min_humidity=min_humidity,
            max_humidity=max_humidity,
            min_distance=min_distance,
            max_distance=max_distance,
            p=1.0,
        )

    def process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.air_absorption(samples=x.numpy(), sample_rate=self.sr))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sr={self.sr}, p={self.p}, "
            f"min_temperature={self.air_absorption.min_temperature}, "
            f"max_temperature={self.air_absorption.max_temperature}, "
            f"min_humidity={self.air_absorption.min_humidity}, "
            f"max_humidity={self.air_absorption.max_humidity}, "
            f"min_distance={self.air_absorption.min_distance}, "
            f"max_distance={self.air_absorption.max_distance})"
        )
