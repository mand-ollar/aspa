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


def pad_audio(audio: torch.Tensor, target_length: int, dim: int) -> torch.Tensor:
    audio_dim: int = audio.dim()

    if audio.size(dim) == target_length:
        return audio

    elif audio.size(dim) < target_length:
        pad_length: list[int] = [0 for _ in range(audio_dim * 2)]
        pad_length[2 * dim + 1] = target_length - audio.size(dim)
        return torch.nn.functional.pad(input=audio, pad=pad_length, mode="constant", value=0)
    else:
        raise ValueError(f"Audio length {audio.size(dim)} is greater than the target length {target_length}.")
