import warnings
from pathlib import Path

import torch
import torchaudio  # type: ignore
from torch.utils.data import Dataset
from tqdm import tqdm

from aspa.utils.audio.format_audio import format_audio

from .config import WindowingConfig
from .types import WindowingResult
from .utils import OneItemCache, pad_audio

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


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

        self.labels: list[torch.Tensor] = []
        self.windowing_results: list[tuple[Path, WindowingResult]] = []
        for audio_path, windows in tqdm(self.windows_dict.items(), desc="Windowing dataset", leave=False, ncols=80):
            for window in windows.values():
                windowed_label: torch.Tensor = torch.zeros(len(self.classes), dtype=torch.float32)
                for label_name in window.iv_name:
                    if label_name is not None and label_name in self.classes:
                        windowed_label[self.classes.index(label_name)] = 1.0

                self.windowing_results.append((audio_path, window))
                self.labels.append(windowed_label)

        self.cache_audio: dict[Path, torch.Tensor] = OneItemCache()

    def __len__(self) -> int:
        return len(self.windowing_results)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        audio_path, windowing_result = self.windowing_results[idx]

        audio: torch.Tensor
        if audio_path in self.cache_audio:
            audio = self.cache_audio[audio_path]
        else:
            audio, sr = torchaudio.load(uri=str(audio_path))
            audio = format_audio(audio=audio, sr=sr, new_sr=self.config.target_sr, target_dim=2)
            self.cache_audio[audio_path] = audio

        windowed_audio: torch.Tensor = audio[:, windowing_result.window_st : windowing_result.window_en]
        windowed_audio = pad_audio(audio=windowed_audio, target_length=self.config.window_size, dim=1)
        windowed_label: torch.Tensor = self.labels[idx]

        return windowed_audio, windowed_label


class IndexedWindowingDataset:
    def __init__(self, dataset: "WindowingDataset | IndexedWindowingDataset", indices: list[int]) -> None:
        self.dataset: WindowingDataset | IndexedWindowingDataset = dataset
        self.indices: list[int] = indices

        self.classes: list[str] = dataset.classes
        self.labels: list[torch.Tensor] = [dataset.labels[i] for i in indices]
        self.windowing_results: list[tuple[Path, WindowingResult]] = [dataset.windowing_results[i] for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[self.indices[idx]]
