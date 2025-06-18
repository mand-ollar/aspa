from pathlib import Path

import soundfile as sf  # type: ignore
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchaudio.functional import resample  # type: ignore
from tqdm import tqdm

from .config import WindowingConfig
from .types import WindowingResult


class WindowingDataset(Dataset):
    def __init__(self, config: WindowingConfig, windows_dict: dict[Path, dict[int, WindowingResult]]) -> None:
        self.config: WindowingConfig = config

        self.audios_list: list[torch.Tensor] = []
        self.labels_list: list[torch.Tensor] = []
        self.windowing_results: list[WindowingResult] = []

        for audio_path, windows in windows_dict.items():
            audio_np, sr = sf.read(file=str(audio_path))
            audio: torch.Tensor = torch.from_numpy(audio_np).float()
            if sr != config.target_sr:
                audio = resample(waveform=audio, orig_freq=sr, new_freq=config.target_sr)
            audio = audio.squeeze()
            assert audio.dim() == 1, "Audio must be mono."

            for window in tqdm(windows.values(), desc="Creating window dataset", leave=False, ncols=80):
                if (config.others is not None and set(window.iv_name) == set([config.others])) or (
                    config.others is None and [element for element in window.iv_name if element is not None] == []
                ):
                    if config.include_others == "lb":
                        if window.label_name == []:
                            continue
                    elif config.include_others == "ulb":
                        if window.label_name != []:
                            continue
                    elif config.include_others == "all":
                        pass
                    elif config.include_others == "none":
                        continue
                    else:
                        raise NotImplementedError(f"Unknown include_others: {config.include_others}")

                windowed_audio: torch.Tensor = audio[window.window_st : window.window_en]
                windowed_audio = self._pad_or_truncate(audio=windowed_audio)
                self.audios_list.append(windowed_audio)

                label: torch.Tensor = torch.zeros(len(config.classes), dtype=torch.float32)
                if window.iv_name != []:
                    for i, iv_name in enumerate(window.iv_name):
                        if iv_name == config.others:
                            continue
                        assert iv_name is not None, "iv_name must not be None."
                        label[config.classes.index(iv_name)] = 1.0

                    if label.sum() == 0 and config.others is not None:
                        label[config.classes.index(config.others)] = 1.0
                else:
                    if config.others is not None and config.others in config.classes:
                        label[config.classes.index(config.others)] = 1.0

                self.labels_list.append(label)

                self.windowing_results.append(window)

        self.audios: torch.Tensor = torch.stack(self.audios_list, dim=0)
        self.labels: torch.Tensor = torch.stack(self.labels_list, dim=0)

    def __len__(self) -> int:
        return len(self.audios)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.audios[idx].unsqueeze(dim=0), self.labels[idx]

    def _pad_or_truncate(self, audio: torch.Tensor) -> torch.Tensor:
        window_size: int = int(self.config.window_size * self.config.target_sr)

        if audio.shape[-1] < window_size:
            audio = F.pad(
                input=audio,
                pad=(0, window_size - audio.shape[-1]),
            )
        elif audio.shape[-1] > window_size:
            audio = audio[:window_size]

        return audio
