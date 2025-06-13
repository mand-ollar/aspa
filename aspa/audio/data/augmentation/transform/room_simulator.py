from typing import Literal

import torch
from audiomentations.augmentations.room_simulator import RoomSimulator as _RoomSimulator  # type: ignore[import-untyped]

from ..audio_augs import AudioAug


class RoomSimulator(AudioAug):
    def __init_(
        self,
        sr: int,
        p: float,
        min_size: float = 2.0,
        max_size: float = 6.0,
        min_absorption: float = 0.05,
        max_absorption: float = 0.4,
        min_rt60: float = 0.15,
        max_rt60: float = 0.8,
        min_source: float = 0.1,
        max_source: float = 4.0,
        min_mic_distance: float = 0.15,
        max_mic_distance: float = 0.35,
        calculation_mode: Literal["rt60", "absorption"] = "absorption",
        leave_length_unchanged: bool = True,
    ) -> None:
        super().__init__(sr=sr, p=p)

        self.room_simulator: _RoomSimulator = _RoomSimulator(
            min_size_x=min_size,
            max_size_x=max_size,
            min_size_y=min_size,
            max_size_y=max_size,
            min_size_z=min_size,
            max_size_z=max_size,
            min_absorption_value=min_absorption,
            max_absorption_value=max_absorption,
            min_target_rt60=min_rt60,
            max_target_rt60=max_rt60,
            min_source_x=min_source,
            max_source_x=max_source,
            min_source_y=min_source,
            max_source_y=max_source,
            min_source_z=min_source,
            max_source_z=max_source,
            min_mic_distance=min_mic_distance,
            max_mic_distance=max_mic_distance,
            calculation_mode=calculation_mode,
            leave_length_unchanged=leave_length_unchanged,
        )

    def process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(
            self.room_simulator(samples=x.numpy(), sample_rate=self.sr)
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sr={self.sr}, p={self.p}, "
            f"min_size={self.room_simulator.min_size_x}, "
            f"max_size={self.room_simulator.max_size_x}, "
            f"min_absorption={self.room_simulator.min_absorption_value}, "
            f"max_absorption={self.room_simulator.max_absorption_value}, "
            f"min_rt60={self.room_simulator.min_target_rt60}, "
            f"max_rt60={self.room_simulator.max_target_rt60}, "
            f"min_source={self.room_simulator.min_source_x}, "
            f"max_source={self.room_simulator.max_source_x}, "
            f"min_mic_distance={self.room_simulator.min_mic_distance}, "
            f"max_mic_distance={self.room_simulator.max_mic_distance}, "
            f"calculation_mode={self.room_simulator.calculation_mode}, "
            f"leave_length_unchanged={self.room_simulator.leave_length_unchanged})"
        )
