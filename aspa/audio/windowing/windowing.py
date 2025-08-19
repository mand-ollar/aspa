from pathlib import Path
from types import MethodType
from typing import Optional

import numpy as np
from rich.progress import track

from .config import WindowingConfig
from .dataset import WindowingDataset
from .types import WindowingResult
from .utils import get_duration_sec


class Windowing:
    annotation_format_priority: tuple[str, ...] = ("txt", "tsv")

    def __init__(self, config: WindowingConfig) -> None:
        self.config: WindowingConfig = config

        self.oov_list: list[str] = []
        self.excluded_labels: list[str] = []

        self.audio_folders: list[Path]
        if isinstance(self.config.audio_folders, (str, Path)):
            self.audio_folders = [Path(self.config.audio_folders)]
        else:
            self.audio_folders = [Path(folder) for folder in self.config.audio_folders]

        if self.config.label_file_finder is not None:
            setattr(self, "_get_label_file_for_audio", MethodType(self.config.label_file_finder, self))
        if self.config.label_line_parser is not None:
            setattr(self, "_parse_label_line", MethodType(self.config.label_line_parser, self))
        if self.config.label_file_processor is not None:
            setattr(self, "_get_labels_for_audio", MethodType(self.config.label_file_processor, self))

    def _gather_audio_files(self) -> list[Path]:
        """Gather audio files in the folder."""

        audio_files: list[Path] = []
        for audio_folder in self.audio_folders:
            if audio_folder.is_dir():
                audio_files += list(audio_folder.rglob("*.wav"))
            elif audio_folder.is_file():
                audio_files.append(audio_folder)
            else:
                raise ValueError(f"Invalid audio folder: {audio_folder}")

        return audio_files

    def _get_label_file_for_audio(self, audio_path: Path) -> Path:
        for file_format in self.annotation_format_priority:
            file_format = file_format.replace(".", "")
            label_path = audio_path.with_suffix(f".{file_format}")
            if label_path.exists():
                return label_path

        if self.config.ignore_missing_label_files:
            (label_path := Path("~/.cache/dummy_label.txt")).touch()
            return label_path
        else:
            raise RuntimeError(
                f"No label file found for {audio_path}.",
                "Consider configuring the annotation file rules by overriding the attribute.",
            )

    def _parse_label_line(self, label_line: str) -> tuple[str, str, str]:
        st, en, label_name = label_line.split("\t")

        return st, en, label_name

    def _get_labels_for_audio(self, audio_path: Path, label_path: Optional[Path] = None) -> np.ndarray:
        if label_path is None:
            raise NotImplementedError

        is_pnt_unit: bool = True
        with open(file=label_path, mode="r") as f:
            labels: list[str] = f.read().strip().split("\n")
            try:
                st_str, en_str, label_name = self._parse_label_line(label_line=labels[0])
                if not st_str.isdigit() or not en_str.isdigit():
                    is_pnt_unit = False
            except ValueError:
                is_pnt_unit = True

            split_labels: list[list[int | str]] = []

            for label in labels:
                if label == "":
                    continue

                st_str, en_str, label_name = self._parse_label_line(label_line=label)

                st_number: float = float(st_str)
                en_number: float = float(en_str)

                assert st_number <= en_number, (
                    "Start time must be smaller than end time: \n\t"
                    f"- Line {labels.index(label) + 1} of {audio_path}, {label}.\n"
                )
                if st_number == en_number:
                    continue

                if is_pnt_unit:
                    split_labels.append([int(st_number), int(en_number), label_name])
                else:
                    split_labels.append([
                        int(st_number * self.config.target_sr),
                        int(en_number * self.config.target_sr),
                        label_name,
                    ])

        return np.array(split_labels, dtype=object)

    def _num_windows(self, audio_length: int) -> int:
        # If the audio is shorter than the window size, pad the audio to the window size.
        if audio_length < self.config.window_size:
            audio_length = self.config.window_size

        # To match the number of windows when start_offset > 0
        if self.config.start_offset > 0:
            audio_length += self.config.start_offset

        return (audio_length - self.config.window_size) // self.config.hop_size + int(not self.config.drop_last_window)

    def _windowing_for_single_audio(self, audio_path: Path, labels: np.ndarray) -> dict[int, WindowingResult] | None:
        st_int: int
        en_int: int
        label_name: str

        st_int, en_int, label_name = labels[0]

        if label_name in self.config.exclude_labels:
            return {}

        if en_int - st_int < self.config.window_size:
            found: bool = False
            for iv_label_name, similars in self.config.similar_labels.items():
                if label_name in similars:
                    found = True
                    break

            if found:
                result: WindowingResult = WindowingResult(
                    audio_path=audio_path,
                    window_st=(en_int + st_int) // 2 - self.config.window_size // 2,
                    window_en=(en_int + st_int) // 2 - self.config.window_size // 2 + self.config.window_size,
                    iv_name=[iv_label_name],
                    label_name=[label_name],
                    relative_ratio=[1.0],
                    absolute_ratio=[1.0],
                    label_id=[0],
                )

                return {0: result}

        return None

    def _others_decision(self, result: WindowingResult) -> bool:
        if self.config.include_others == "lb":
            if result.label_name:
                return True
        elif self.config.include_others == "ulb":
            if not result.label_name:
                return True
        elif self.config.include_others == "all":
            return True
        elif self.config.include_others == "none":
            return False
        else:
            raise ValueError(f"Invalid include_others value: {self.config.include_others}")

        return False

    def _windowing(self, audio_path: Path, verbose: bool = False) -> dict[int, WindowingResult]:
        audio_length: int = int(get_duration_sec(filepath=audio_path) * self.config.target_sr)
        _label_path: Path = self._get_label_file_for_audio(audio_path=audio_path)
        labels: np.ndarray = self._get_labels_for_audio(audio_path=audio_path, label_path=_label_path)
        num_windows: int = self._num_windows(audio_length=audio_length)

        # If the audio is slightly longer than the window size,
        # and the target audio is shorter than the window size,
        # place the target audio in the middle and make the window.
        # Only for target audio.
        if len(labels) == 1 and self.config.window_size < audio_length < self.config.window_size * 3:
            if (result := self._windowing_for_single_audio(audio_path=audio_path, labels=labels)) is not None:
                return result

        windowed_results: dict[int, WindowingResult] = {}
        cnt: int = 0

        for i in track(range(num_windows), description="Windowing", transient=True):
            others: bool = False

            windowing_result: WindowingResult = WindowingResult(
                audio_path=audio_path,
                window_st=self.config.start_offset + i * self.config.hop_size,
                window_en=self.config.start_offset + i * self.config.hop_size + self.config.window_size,
            )

            skip_window: bool = False

            # Get the indices of the labels that overlap with the window.
            overlapped_mask: np.ndarray = np.logical_and(
                labels[:, 0] < windowing_result.window_en,
                labels[:, 1] > windowing_result.window_st,
            )
            masked_indices: np.ndarray = np.where(overlapped_mask)[0]

            if not overlapped_mask.any():
                others = True
            else:
                # Get relative ratio and absolute ratio of the overlapped labels.
                overlapped_labels: np.ndarray = labels[overlapped_mask]
                _overlap: np.ndarray = np.minimum(overlapped_labels[:, 1], windowing_result.window_en) - np.maximum(
                    overlapped_labels[:, 0], windowing_result.window_st
                )
                relative_ratios: np.ndarray = _overlap / (windowing_result.window_en - windowing_result.window_st)
                absolute_ratios: np.ndarray = _overlap / (overlapped_labels[:, 1] - overlapped_labels[:, 0])

                # Ratio thresholding
                ratio_mask: np.ndarray = np.logical_or(
                    relative_ratios >= self.config.relative_ratio_threshold,
                    absolute_ratios >= self.config.absolute_ratio_threshold,
                )
                masked_indices = masked_indices[ratio_mask]
                relative_ratios = relative_ratios[ratio_mask]
                absolute_ratios = absolute_ratios[ratio_mask]
                overlapped_labels = overlapped_labels[ratio_mask]

                for j, relative_ratio, absolute_ratio, label_name in zip(
                    masked_indices, relative_ratios, absolute_ratios, overlapped_labels[:, 2]
                ):
                    windowing_result.label_name.append(label_name)
                    windowing_result.relative_ratio.append(relative_ratio)
                    windowing_result.absolute_ratio.append(absolute_ratio)
                    windowing_result.label_id.append(j)

                    found: bool = False
                    for iv_label_name, similars in self.config.similar_labels.items():
                        if label_name in similars:
                            # Check exclude labels
                            if label_name in self.config.exclude_labels:
                                skip_window = True
                                if label_name not in self.excluded_labels:
                                    self.excluded_labels.append(label_name)
                                break

                            windowing_result.iv_name.append(iv_label_name)
                            found = True
                            break

                    if not found:
                        # Check exclude labels
                        if label_name in self.config.exclude_labels:
                            skip_window = True
                            if label_name not in self.excluded_labels:
                                self.excluded_labels.append(label_name)
                            break

                        # Not found from the similar labels, add as others
                        windowing_result.iv_name.append(self.config.others)
                        others = True

                    if (
                        not found
                        and label_name not in self.oov_list
                        and label_name not in list(self.config.similar_labels.keys())
                    ):
                        if verbose:
                            print(f"Considering\n{label_name}\nas others.\n")
                        self.oov_list.append(label_name)

                # If exclude label is found
                if skip_window:
                    continue

            assert (
                len(windowing_result.label_name)
                == len(windowing_result.relative_ratio)
                == len(windowing_result.absolute_ratio)
                == len(windowing_result.label_id)
            ), "Window information must be added for every key."

            if others and not self._others_decision(result=windowing_result):
                continue

            windowed_results[cnt] = windowing_result
            cnt += 1

        return windowed_results

    def get_windows(self, verbose: bool = False) -> dict[Path, dict[int, WindowingResult]]:
        """
        Window the audio files and save as wav files in the folder.

        Returns:
            A dictionary of windowed results.
            The key is the audio file name and the value is a dictionary of windowed results.
            The key of the inner dictionary is the window index and the value is the windowed result.
        """

        audio_paths: list[Path] = self._gather_audio_files()
        windows_dict: dict[Path, dict[int, WindowingResult]] = {}

        for audio_path in audio_paths:
            windows_dict[audio_path] = self._windowing(audio_path=audio_path, verbose=verbose)

        print("Windowing oov list:")
        print(self.oov_list)
        print()

        print("Windowing excluded labels:")
        print(self.excluded_labels)
        print()

        return windows_dict

    def get_dataset(self, classes: list[str] | None = None) -> WindowingDataset:
        windows_dict: dict[Path, dict[int, WindowingResult]] = self.get_windows()

        return WindowingDataset(config=self.config, windows_dict=windows_dict, classes=classes)
