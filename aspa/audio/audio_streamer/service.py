import string
import subprocess
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import numpy as np
import pyaudio
import soundfile as sf  # type: ignore

from aspa.utils.printings import Colors

from .microphones import LocalMicrophone, RemoteMicrophone


class MicrophoneService(ABC):
    def __init__(self, mic: LocalMicrophone | RemoteMicrophone, chunk_sec: float = 0.05, sr: int = 48000) -> None:
        self.mic: LocalMicrophone | RemoteMicrophone = mic
        self.chunk_sec: float = chunk_sec
        self.sr: int = sr
        self.name: str | None = None

    @abstractmethod
    def listen(self, listen_event: threading.Event) -> Iterator[np.ndarray]:
        """Listen to the microphone and yield audio chunks."""


class LocalMicrophoneService(MicrophoneService):
    def __init__(self, mic: LocalMicrophone, chunk_sec: float = 0.05, sr: int = 48000, channel_index: int = 1) -> None:
        super().__init__(mic=mic, chunk_sec=chunk_sec, sr=sr)

        if mic.maxInputChannels < 1:
            raise ValueError("The microphone must have at least one input channel.")

        self.channel_index: int = channel_index
        self.channel_count: int = mic.maxInputChannels

        self.name = f"{Colors.YELLOW}[Local] MIC <{mic.name}> - CH <{self.channel_index}>{Colors.END}"

    def listen(self, listen_event: threading.Event) -> Iterator[np.ndarray]:
        return self._listen_local_microphone(listen_event=listen_event)

    def _listen_local_microphone(self, listen_event: threading.Event) -> Iterator[np.ndarray]:
        assert isinstance(self.mic, LocalMicrophone), "The microphone must be a LocalMicrophone instance."

        p: pyaudio.PyAudio = pyaudio.PyAudio()
        stream: pyaudio.Stream = p.open(
            format=pyaudio.paFloat32,
            channels=self.channel_count,
            rate=self.sr,
            input=True,
            frames_per_buffer=int(self.sr * self.chunk_sec),
            input_device_index=self.mic.index,
        )

        frames: int = int(self.sr * self.chunk_sec)
        while listen_event.is_set():
            raw_data: bytes = stream.read(frames, exception_on_overflow=False)
            all_channels: np.ndarray = np.frombuffer(raw_data, dtype=np.float32)
            all_channels = all_channels.reshape(frames, self.channel_count)
            selected: np.ndarray = all_channels[:, self.channel_index]
            yield selected

        stream.stop_stream()
        stream.close()
        p.terminate()


class RemoteMicrophoneService(MicrophoneService):
    def __init__(self, mic: RemoteMicrophone, chunk_sec: float = 0.05, sr: int = 48000) -> None:
        super().__init__(mic=mic, chunk_sec=chunk_sec, sr=sr)

        self.proc: subprocess.Popen | None = None
        self.name = f"{Colors.YELLOW}[Remote] MIC <{mic.name}> @{mic.ip}{Colors.END}"

    def listen(self, listen_event: threading.Event) -> Iterator[np.ndarray]:
        return self._listen_remote_microphone(listen_event=listen_event)

    def _listen_remote_microphone(self, listen_event: threading.Event) -> Iterator[np.ndarray]:
        assert isinstance(self.mic, RemoteMicrophone), "The microphone must be a RemoteMicrophone instance."

        command: list[str] = [
            "ffmpeg",
            "-loglevel",
            "fatal",
            "-rtsp_transport",
            "tcp",
            "-i",
            self.mic.rtsp_url,
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-",
        ]
        self.proc = subprocess.Popen(command, stdout=subprocess.PIPE)

        while listen_event.is_set():
            if self.proc.poll() is not None:
                self.proc = subprocess.Popen(command, stdout=subprocess.PIPE)
                continue

            assert self.proc.stdout is not None, "Process stdout is None."

            yield np.frombuffer(
                self.proc.stdout.read(np.dtype(np.float32).itemsize * int(self.chunk_sec * self.sr)), dtype=np.float32
            )


class RecordService:
    def __init__(self, filepath: Path, mic_service: MicrophoneService, record_event: threading.Event) -> None:
        self.filepath: Path = filepath
        self.mic_service: MicrophoneService = mic_service
        self.sr: int = mic_service.sr
        self.record_event: threading.Event = record_event

        self.tmp_filepath: Path
        self.record_thread: threading.Thread

    def _record(self) -> None:
        self.record_event.wait()

        assert self.filepath is not None

        with open(file=self.tmp_filepath, mode="wb") as f:
            for chunk in self.mic_service.listen(listen_event=self.record_event):
                f.write(chunk.tobytes())

                if not self.record_event.is_set():
                    self._save()
                    break

    def _save(self) -> None:
        assert self.filepath is not None

        with open(file=self.tmp_filepath, mode="rb") as f:
            audio: np.ndarray = np.frombuffer(f.read(), dtype=np.float32).copy()
            sf.write(file=self.filepath, data=audio, samplerate=self.sr)

        self.tmp_filepath.unlink()

    def __call__(self) -> None:
        tmp_filename = "".join(np.random.choice(list(string.ascii_letters + string.digits), 10))

        self.filepath = Path(self.filepath)
        self.tmp_filepath = Path(f"/tmp/{tmp_filename}.tmp")

        self.record_thread = threading.Thread(target=self._record, daemon=True)
        self.record_thread.start()
        self.record_thread.join(timeout=0)
