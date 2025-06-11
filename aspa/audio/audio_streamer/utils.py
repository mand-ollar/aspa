from typing import Mapping

import pyaudio
from utils.select_option import select_option

from .microphones import LocalMicrophone
from .service import LocalMicrophoneService


def _load_local_microphones() -> list[LocalMicrophone]:
    local_mic_list: list[LocalMicrophone] = []
    p: pyaudio.PyAudio = pyaudio.PyAudio()

    for device_idx in range(p.get_device_count()):
        device: Mapping[str, str | int | float] = p.get_device_info_by_index(device_index=device_idx)

        max_input_channels: str | int | float | None = device.get("maxInputChannels")
        assert isinstance(max_input_channels, int)

        if max_input_channels > 0:
            mic = LocalMicrophone(**device)  # type: ignore
            local_mic_list.append(mic)

    return local_mic_list


def setup_microphone(chunk_sec: float, sr: int) -> LocalMicrophoneService:
    mic_list: list[LocalMicrophone] = _load_local_microphones()
    mic_name_list: list[str] = [mic.name for mic in mic_list]

    selected_idx: int = select_option(options=mic_name_list, description="Select microphone")

    selected_mic = mic_list[selected_idx]
    selected_channel: int
    if selected_mic.maxInputChannels > 1:
        selected_channel = select_option(
            options=[str(i) for i in range(selected_mic.maxInputChannels)],
            description=f"Select channel for {selected_mic.name}",
        )
    else:
        selected_channel = 0

    mic_service: LocalMicrophoneService = LocalMicrophoneService(
        mic=selected_mic,
        chunk_sec=chunk_sec,
        sr=sr,
        channel_index=selected_channel,
    )

    return mic_service


def _print(msg: str = "", end: str = "\n") -> None:
    print(f"[AudioStreamer] {msg}", end=end)
    print()
