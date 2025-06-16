import torch

from ..audio_augs import AudioAug
from ..helper import EffectChain


class Reverb(AudioAug):
    def __init__(
        self,
        sr: int = 16000,
        p: float = 0.5,
        reverberance_min: int = 0,
        reverberance_max: int = 100,
        dumping_factor_min: int = 0,
        dumping_factor_max: int = 100,
        room_size_min: int = 0,
        room_size_max: int = 100,
    ) -> None:
        super().__init__(sr=sr, p=p)

        self.reverberance_min = reverberance_min
        self.reverberance_max = reverberance_max
        self.dumping_factor_min = dumping_factor_min
        self.dumping_factor_max = dumping_factor_max
        self.room_size_min = room_size_min
        self.room_size_max = room_size_max

        self.src_info: dict[str, int | float] = {"rate": sr}
        self.target_info: dict[str, int | float] = {"channels": 1, "rate": sr}

    def process(self, x: torch.Tensor) -> torch.Tensor:
        reverberance: float = torch.randint(self.reverberance_min, self.reverberance_max, size=(1,)).item()
        dumping_factor: float = torch.randint(self.dumping_factor_min, self.dumping_factor_max, size=(1,)).item()
        room_size: float = torch.randint(self.room_size_min, self.room_size_max, size=(1,)).item()

        num_channels = x.shape[0]
        effect_chain: EffectChain = (
            EffectChain()
            .reverb(reverberance, dumping_factor, room_size)  # type: ignore
            .channels(num_channels)
        )

        x = x.to(torch.float32)
        x = effect_chain.apply(x, src_info=self.src_info, target_info=self.target_info)
        x = x.unsqueeze(dim=1)

        return x
