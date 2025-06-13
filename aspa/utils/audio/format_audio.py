import numpy as np
import torch
import torchaudio  # type: ignore


def _format(audio: torch.Tensor, sr: int, new_sr: int | None = None, target_dim: int | None = None) -> torch.Tensor:
    # Resample if needed
    if new_sr is not None and sr != new_sr:
        audio = torchaudio.functional.resample(waveform=audio, orig_freq=sr, new_freq=new_sr)

    # Match the dimensions
    audio_dim: int = audio.dim()
    audio_size: torch.Size = audio.size()
    if target_dim is not None and audio_dim != target_dim:
        assert audio_size[1] == 1 if audio_dim == 3 else True, "Audio must be mono (single channel)."

        audio = audio.squeeze()
        if audio.dim() == 1:
            audio = audio.reshape(1, -1)
        assert audio.dim() == 2, f"Audio must be mono. Got {audio_size} sized audio tensor."

        if target_dim == 1:
            audio = audio.squeeze(dim=0)
            assert audio.dim() == 1, f"Batch size must be 1 for target_dim=1, but got {audio_size[0]}."
        elif target_dim == 2:
            audio = audio
        elif target_dim == 3:
            audio = audio.unsqueeze(dim=1)
        else:
            raise NotImplementedError(f"Target dimension {target_dim} is not supported.")

    return audio


def format_audio(
    audio: torch.Tensor | np.ndarray, sr: int, new_sr: int | None = None, target_dim: int | None = None
) -> torch.Tensor:
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()
    assert isinstance(audio, torch.Tensor)

    return _format(audio=audio, sr=sr, new_sr=new_sr, target_dim=target_dim)
