from typing import Literal

import torch


class SoundLevel:
    reference_level: float = 20e-6

    def __init__(
        self,
        mode: Literal["RMS", "A-weighted"] = "RMS",
        sr: int = 16000,
        kernel_size: int = 1024,
        avg_mode: Literal["median", "mean", "max", "none"] = "median",
        scale: Literal["dB_SPL", "raw"] = "dB_SPL",
    ) -> None:
        """Sound Level Calculator.
        Two different Sound Level calculation modes: RMS and A-weighted.

        Args:
            - mode: Sound Level calculation mode. Default: "RMS".
            - sr: Sampling rate. Default: 16000.
            - kernel_size: Window size for Sound Level calculation. Default: 1024.
              (stride_size = kernel_size // 2)

        Call:
            - Set the arguments when initializing the class, and call the class instance with audio tensor.

        Output:
            - Sound Level in dB.
        """

        self.mode: Literal["RMS", "A-weighted"] = mode
        self.sr: int = sr
        self.kernel_size: int = kernel_size
        self.stride: int = int(kernel_size // 2)
        self.avg_mode: Literal["median", "mean", "max", "none"] = avg_mode
        self.scale: Literal["dB_SPL", "raw"] = scale

    def _format_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Format audio tensor to [batch, time]"""

        if audio.dim() == 1:
            audio = audio.reshape(1, -1)
        elif audio.dim() == 3:
            if audio.size(1) == 1:
                audio = audio.squeeze(dim=1)
            else:
                audio = audio.mean(dim=1)

        assert audio.dim() == 2, "Audio must have size [batch, time]."

        return audio

    def _n_windows(self, audio: torch.Tensor) -> int:
        """Calculate the number of windows for the given audio tensor, and kernel size."""
        return (audio.size(-1) - self.kernel_size) // self.stride + 1

    def _a_weight(self, min_db=-80.0) -> torch.Tensor:
        """Returns A-weighting filter."""
        freq: torch.Tensor = torch.linspace(0, self.sr // 2, self.kernel_size // 2 + 1, dtype=torch.float32)
        freq_sq = torch.pow(freq, 2)
        freq_sq[0] = 1.0

        weight: torch.Tensor = 2.0 + 20.0 * (
            2 * torch.log10(torch.tensor(12194.0))
            + 2 * torch.log10(freq_sq)
            - torch.log10(freq_sq + 12194.0**2)
            - torch.log10(freq_sq + 20.6**2)
            - 0.5 * torch.log10(freq_sq + 107.7**2)
            - 0.5 * torch.log10(freq_sq + 737.9**2)
        )
        weight = torch.maximum(weight, torch.tensor(min_db, dtype=weight.dtype))

        return weight

    def _rms(self, audio: torch.Tensor) -> torch.Tensor:
        """Calculate RMS[dB]."""
        audio = self._format_audio(audio=audio)
        squared_audio: torch.Tensor = torch.pow(audio, 2)

        squared_windows: torch.Tensor = torch.stack(
            [
                squared_audio[:, i * self.stride : i * self.stride + self.kernel_size]
                for i in range(self._n_windows(audio))
            ],
            dim=1,
        )
        mean_squares: torch.Tensor = squared_windows.mean(dim=-1)
        root_mean_squares: torch.Tensor = torch.sqrt(mean_squares)

        rms_db: torch.Tensor = self._get_spl_dB(spls=root_mean_squares)

        return rms_db

    def _a_weighted_spl(self, audio: torch.Tensor) -> torch.Tensor:
        """Calculate A-weighted SPL[dB]."""
        audio = self._format_audio(audio=audio)

        hann_window: torch.Tensor = torch.hann_window(
            self.kernel_size, periodic=True, device=audio.device, dtype=audio.dtype
        )
        squared_spectrograms: torch.Tensor = torch.stack(
            [
                torch.abs(torch.fft.rfft(hann_window * audio[:, i * self.stride : i * self.stride + self.kernel_size]))
                ** 2
                for i in range(self._n_windows(audio))
            ],
            dim=1,
        )
        a_weighted_spectrograms: torch.Tensor = squared_spectrograms * torch.pow(10, self._a_weight() / 10)
        a_weighted_spls: torch.Tensor = torch.sum(a_weighted_spectrograms, dim=-1)

        spl_db: torch.Tensor = self._get_spl_dB(spls=a_weighted_spls)

        return spl_db

    def _get_spl_dB(self, spls: torch.Tensor) -> torch.Tensor:
        if self.scale == "dB_SPL":
            spls = 20 * torch.log10(spls / self.reference_level)
        elif self.scale == "raw":
            pass
        else:
            raise ValueError(f"Invalid scale: {self.scale}")

        if self.avg_mode == "median":
            return spls.median(dim=1).values
        elif self.avg_mode == "mean":
            return spls.mean(dim=1)
        elif self.avg_mode == "max":
            return spls.max(dim=1).values
        elif self.avg_mode == "none":
            return spls
        else:
            raise ValueError(f"Invalid avg_mode: {self.avg_mode}")

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if self.mode == "RMS":
            return self._rms(audio=audio)
        elif self.mode == "A-weighted":
            return self._a_weighted_spl(audio=audio)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
