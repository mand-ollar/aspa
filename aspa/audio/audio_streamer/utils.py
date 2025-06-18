from typing import Mapping

import pyaudio

from aspa.utils.printings import Colors, clear_screen
from aspa.utils.select_option import select_option

from .microphones import LocalMicrophone, RemoteMicrophone
from .services import LocalMicrophoneService, MicrophoneService, RemoteMicrophoneService


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


def setup_local_microphone(chunk_sec: float, sr: int) -> LocalMicrophoneService | None:
    mic_list: list[LocalMicrophone] = _load_local_microphones()
    mic_name_list: list[str] = [mic.name for mic in mic_list]

    if not mic_name_list:
        _print("No local microphones available.")
        return None

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


def setup_remote_microphone(
    chunk_sec: float, sr: int, available_mics: dict[str, list[RemoteMicrophone]]
) -> RemoteMicrophoneService | None:
    if not available_mics:
        _print("No remote microphones available.")
        return None

    category: str = list(available_mics.keys())[
        select_option(options=list(available_mics.keys()), description="Select microphone category")
    ]
    mic_list: list[RemoteMicrophone] = available_mics[category]
    mic: RemoteMicrophone = mic_list[
        select_option(options=[mic.name for mic in mic_list], description="Select remote microphone")
    ]

    mic_service: RemoteMicrophoneService = RemoteMicrophoneService(mic=mic, chunk_sec=chunk_sec, sr=sr)

    return mic_service


def setup_microphone(
    chunk_sec: float, n_mics: int, sr: int, available_remote_mics: dict[str, list[RemoteMicrophone]] | None = None
) -> list[MicrophoneService]:
    mic_services: list[MicrophoneService] = []

    mic_cnt: int = 0
    while mic_cnt <= n_mics:
        clear_screen()
        _print(f"Setting up {n_mics} microphones...")
        if mic_cnt > 0:
            _print("Set Microphones:")
            for i, mic_service in enumerate(mic_services):
                print(f"  - {i}: {mic_service.name} {Colors.GREEN}[Connected]{Colors.END}")
            print()
        if mic_cnt == n_mics:
            break

        mic_type_idx: int  # 0: Local, 1: Remote
        if available_remote_mics is None:
            mic_type_idx = 0
        else:
            mic_type_idx = select_option(options=["Local", "Remote"], description="Select microphone type")

        if mic_type_idx == 0:  # Local microphone
            local_mic_output: LocalMicrophoneService | None = setup_local_microphone(chunk_sec=chunk_sec, sr=sr)
            if local_mic_output is not None:
                local_mic_service: LocalMicrophoneService = local_mic_output
                _print(f"Working on {local_mic_service.mic.name} {Colors.GREEN}[Connected]{Colors.END}")
                mic_services.append(local_mic_service)
                mic_cnt += 1

        elif mic_type_idx == 1:  # Remote microphone
            assert available_remote_mics is not None, "Remote microphones are not available."
            remote_mic_output: RemoteMicrophoneService | None = setup_remote_microphone(
                chunk_sec=chunk_sec, sr=sr, available_mics=available_remote_mics
            )
            if remote_mic_output is not None:
                remote_mic_service: RemoteMicrophoneService = remote_mic_output
                _print(f"Working on {remote_mic_service.mic.name} {Colors.GREEN}[Connected]{Colors.END}")
                mic_services.append(remote_mic_service)
                mic_cnt += 1

        else:
            raise ValueError(f"Invalid microphone type index: {mic_type_idx}")

    return mic_services


def _print(msg: str = "", end: str = "\n") -> None:
    print(f"[AudioStreamer] {msg}", end=end)
    print()
