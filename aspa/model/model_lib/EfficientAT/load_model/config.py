from dataclasses import dataclass, field

from aspa.utils.base_dataclass import BaseDataClass


@dataclass
class EfficientDyATParams(BaseDataClass):
    pretrained_model: list[str] = field(default_factory=lambda: ["dymn20_as"])
    n_mels: int = 128
    target_sr: int = 48000
    data_length: int = 24000
    win_length: int = 384
    hop_length: int = 96
    n_fft: int = 768
    fmin: int = 100
    fmax: int | None = None
    freqm: int = 3
    timem: int = 1
    fmin_aug_range: int = 1
    fmax_aug_range: int = 100
    num_time_masks: int = 2
    num_freq_masks: int = 2
    input_normalization: bool = False
    classes: list[str] | None = None
