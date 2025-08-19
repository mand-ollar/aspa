import wave
from collections import OrderedDict
from pathlib import Path

import librosa
import torch


class OneItemCache(OrderedDict):
    def __setitem__(self, key: Path, value: torch.Tensor) -> None:
        if len(self) >= 1 and key not in self:
            self.popitem(last=False)
        super().__setitem__(key, value)


def get_duration_sec(filepath: str | Path) -> float:
    """Get the duration time of the given audio file in seconds."""

    filepath = Path(filepath)

    if filepath.suffix.lower() == ".wav":
        with wave.open(f=str(filepath), mode="rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate
    else:
        return librosa.get_duration(path=str(filepath))
