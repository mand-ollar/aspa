import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
from art import text2art  # type: ignore

from aspa.utils.printings import Colors, clear_lines, clear_screen

from .microphones import RemoteMicrophone
from .services import MicrophoneService, RecordService
from .utils import _print, setup_microphone


class AfterRecordingCallback(ABC):
    @staticmethod
    def _print(msg: str = "") -> None:
        print(f"[AfterRecording] {msg}\n")

    @abstractmethod
    def __call__(self, record_services: list[RecordService]) -> None:
        raise NotImplementedError(
            "AfterRecordingCallback must implement the __call__ method to handle post-recording actions."
        )


class RecordingStreamer:
    def __init__(
        self,
        recording_sec: float,
        buffering_sec: int,
        sr: int,
        n_mics: int,
        after_recording_callback: AfterRecordingCallback | None = None,
        avaiable_remote_mics: dict[str, list[RemoteMicrophone]] | None = None,
    ) -> None:
        clear_screen()

        date_time: str = datetime.now().strftime(r"%Y%m%d-%H%M%S")
        self.save_folder = Path("./data/saved_audio")
        self.recording_sec: float = recording_sec
        self.buffering_sec: int = buffering_sec
        self.sr: int = sr

        self.after_recording_callback: AfterRecordingCallback | None = after_recording_callback

        self.mic_services: list[MicrophoneService] = setup_microphone(
            n_mics=n_mics, sr=sr, available_remote_mics=avaiable_remote_mics
        )

        folder_name: str = date_time
        self.folder_name_append: str = input(
            f"Enter name for the recording folder - {Colors.BLUE}default: {folder_name}_{Colors.END}"
        )
        if self.folder_name_append:
            folder_name += f"_{self.folder_name_append}"

        self.save_folder = self.save_folder / folder_name
        self.save_folder.mkdir(parents=True, exist_ok=True)

        self.record_services: list[RecordService]
        clear_screen()

    def _print_intro(self, idx: int) -> None:
        print(text2art("Recording"))

    def _run_recording_process(self, idx: int) -> bool:
        clear_screen()

        record_event: threading.Event = threading.Event()

        self.record_services = []
        for i, mic_service in enumerate(self.mic_services):
            self.record_services.append(
                RecordService(
                    filepath=self.save_folder / f"{idx}_mic{i}.wav",
                    mic_service=mic_service,
                    record_event=record_event,
                )
            )
            self.record_services[-1]()

        self._print_intro(idx=idx)

        # Trigger
        _print(
            f"Press `{Colors.BOLD}{Colors.GREEN}Enter{Colors.END}` for recording, "
            f"enter `{Colors.BOLD}{Colors.RED}b{Colors.END}` for undo the last recording",
            end="",
        )
        flag: str = input(f"Ready for recording {idx} :)")
        print()

        if flag.lower() == "b":
            return False

        for i in range(self.buffering_sec)[::-1]:
            _print(f"Buffering... {Colors.YELLOW}{i + 1}{Colors.END}")
            clear_lines(2)
            time.sleep(1)

        record_event.set()
        _print(f"Recording {idx} started")

        print("Recorded seconds:")
        while True:
            recorded_sec_list = []
            for i, record_service in enumerate(self.record_services):
                if record_event.is_set() and record_service.tmp_filepath.exists():
                    with open(file=record_service.tmp_filepath, mode="rb") as f:
                        f.seek(0, 2)
                        recorded_sec_list.append(f"{f.tell() / self.sr / np.dtype(np.float32).itemsize:.2f}")

            print(" / ".join(recorded_sec_list), end="\r")

            if recorded_sec_list and float(recorded_sec_list[-1]) >= self.recording_sec:
                print("\n")
                break

        record_event.clear()

        _print("Waiting for the recording to finish")
        constant_cnt: int = 0
        for record_service in self.record_services:
            file_size: int = 0
            while constant_cnt < 100:
                if record_service.filepath.exists():
                    new_file_size: int = record_service.filepath.stat().st_size
                else:
                    continue

                if file_size == new_file_size and new_file_size > 0:
                    constant_cnt += 1
                file_size = new_file_size

                print(constant_cnt, end="\r")
        _print("Recording & saving process finished")

        if self.after_recording_callback is not None:
            self.after_recording_callback(record_services=self.record_services)

        input("Enter any key to continue...")
        print()

        return True

    def stream(self, n_iter: int) -> None:
        cnt: int = 0
        while cnt < n_iter:
            if self._run_recording_process(idx=cnt):
                cnt += 1
            else:
                cnt = max(0, cnt - 1)

        clear_screen()
        _print(f"Finished recording {cnt} iterations")

        _print("Saving...")
        time.sleep(3)
        _print(f"Saved audio files @{self.save_folder}!")
