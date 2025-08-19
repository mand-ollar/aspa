from pathlib import Path

import torch
import torchaudio  # type: ignore
from torch.utils.data import Dataset

from aspa.utils.audio.format_audio import format_audio

from .config import WindowingConfig
from .types import WindowingResult
from .utils import OneItemCache


class WindowingDataset(Dataset):
    def __init__(
        self,
        config: WindowingConfig,
        windows_dict: dict[Path, dict[int, WindowingResult]],
        classes: list[str] | None = None,
    ) -> None:
        self.config: WindowingConfig = config
        self.windows_dict: dict[Path, dict[int, WindowingResult]] = windows_dict
        self.classes: list[str] = classes or config.classes

        assert all(class_name in self.classes for class_name in self.config.classes), (
            "classes in the windowing result must be included in the classes argument."
        )

        self.labels_list: list[torch.Tensor] = []
        self.windowing_results: list[tuple[Path, WindowingResult]] = []
        for audio_path, windows in self.windows_dict.items():
            for window in windows.values():
                self.windowing_results.append((audio_path, window))

        self.cache_audio: dict[Path, torch.Tensor] = OneItemCache()

    def __len__(self) -> int:
        return len(self.windowing_results)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        audio_path, windowing_result = self.windowing_results[idx]

        audio: torch.Tensor
        if audio_path in self.cache_audio:
            audio = self.cache_audio[audio_path]
        else:
            audio, sr = torchaudio.load(uri=audio_path)
            audio = format_audio(audio=audio, sr=sr, new_sr=self.config.target_sr, target_dim=2)
            self.cache_audio[audio_path] = audio

        windowed_audio: torch.Tensor = audio[:, windowing_result.window_st : windowing_result.window_en]
        windowed_label: torch.Tensor = torch.zeros(len(self.classes), dtype=torch.float32)
        for label_name in windowing_result.iv_name:
            if label_name is not None:
                windowed_label[self.classes.index(label_name)] = 1.0

        return windowed_audio, windowed_label
