from pathlib import Path

import soundfile as sf  # type: ignore
import torch
from torch.utils.data import Dataset

from utils.format_audio import format_audio


class LabelSlaveDataset(Dataset):
    def __init__(
        self,
        wav_path: str | Path | None,
        audio: torch.Tensor | None = None,
        audio_sr: int | None = None,
        sr: int = 48000,
        window_sec: float = 1.0,
        hop_sec: float = 0.5,
    ) -> None:
        assert wav_path is not None or audio is not None, "Either wav_path or audio must be provided."
        assert wav_path is None or audio is None, "Only one of wav_path or audio can be provided."

        self.audio: torch.Tensor
        if wav_path is not None:
            _audio, _sr = sf.read(file=wav_path, dtype="float32")
            self.audio = torch.from_numpy(_audio).float()
            audio_sr = _sr
        elif audio is not None:
            self.audio = audio.float()
            if audio_sr is None:
                audio_sr = sr
        assert isinstance(audio_sr, int)

        self.audio = format_audio(audio=self.audio, sr=audio_sr, new_sr=sr, target_dim=1)

        # Windowing
        self.window_size: int = int(sr * window_sec)
        self.hop_size: int = int(sr * hop_sec)
        self.num_windows: int = (len(self.audio) - self.window_size) // self.hop_size + 1

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a tuple of (audio, tensor[window_st, window_en])."""
        st_pnt: int = idx * self.hop_size
        en_pnt: int = st_pnt + self.window_size

        return self.audio[st_pnt:en_pnt].reshape(1, -1), torch.tensor([st_pnt, en_pnt], dtype=torch.int64)
