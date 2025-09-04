from pathlib import Path
from types import MethodType
from typing import Literal, Optional

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
from tqdm import tqdm

from .config import WindowingConfig
from .dataset import BaseWindowingDataset, IndexedWindowingDataset, WindowingDataset
from .types import WindowingResult
from .utils import get_duration_sec


class Windowing:
    annotation_format_priority: tuple[str, ...] = ("txt", "tsv")

    def __init__(self, config: WindowingConfig) -> None:
        self.config: WindowingConfig = config

        self.iv_list: set[str] = set()
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
                audio_files += list(audio_folder.rglob(f"*.{self.config.audio_format}"))
            elif audio_folder.is_file():
                audio_files.append(audio_folder)
            else:
                raise ValueError(f"Invalid audio folder: {audio_folder}")

        return audio_files

    def _get_label_file_for_audio(self, audio_path: Path) -> Path:
        label_path: Path

        for file_format in self.annotation_format_priority:
            file_format = file_format.replace(".", "")
            label_path = audio_path.with_suffix(f".{file_format}")
            if label_path.exists():
                return label_path

        if self.config.ignore_missing_label_files:
            label_path = Path("./.cache/dummy_label.txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.touch()

            return label_path
        else:
            raise RuntimeError(
                f"No label file found for {audio_path}.",
                "Consider configuring the annotation file rules by overriding the attribute.",
            )

    def _parse_label_line(self, label_line: str) -> tuple[str, str, str]:
        st, en, label_name = label_line.split("\t")

        return st, en, label_name

    def _get_labels_for_audio(self, audio_path: Path, label_path: Optional[Path] = None) -> list[tuple[int, int, str]]:
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

            split_labels: list[tuple[int, int, str]] = []

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
                    split_labels.append((int(st_number), int(en_number), label_name))
                else:
                    split_labels.append((
                        int(st_number * self.config.target_sr),
                        int(en_number * self.config.target_sr),
                        label_name,
                    ))

        return split_labels

    def _num_windows(self, audio_length: int) -> int:
        # If the audio is shorter than the window size, pad the audio to the window size.
        if audio_length < self.config.window_size:
            audio_length = self.config.window_size

        # To match the number of windows when start_offset > 0
        if self.config.start_offset > 0:
            audio_length += self.config.start_offset

        return (audio_length - self.config.window_size) // self.config.hop_size + int(not self.config.drop_last_window)

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

    def _windowing_for_single_audio(
        self, audio_path: Path, labels: list[tuple[int, int, str]], audio_length: int
    ) -> dict[int, WindowingResult] | None:
        st_int: int
        en_int: int
        label_name: str

        st_int, en_int, label_name = labels[0]

        if label_name in self.config.exclude_labels:
            return {}

        if en_int - st_int < self.config.window_size:
            found: bool = False
            for iv_label_name, similars in self.config.similar_labels.items():
                if label_name.strip() in similars:
                    found = True
                    break

            if found:
                result: WindowingResult = WindowingResult(
                    audio_path=audio_path,
                    window_st=st_int,
                    window_en=min(en_int, audio_length),
                    iv_name=[iv_label_name],
                    label_name=[label_name],
                    relative_ratio=[1.0],
                    absolute_ratio=[1.0],
                    label_id=[0],
                )

                return {0: result}

        return None

    def _windowing_for_large_labels(
        self,
        audio_path: Path,
        labels: list[tuple[int, int, str]],
        num_windows: int,
        show_progress: bool,
        verbose: bool,
    ) -> dict[int, WindowingResult]:
        cnt: int = 0
        windowed_results: dict[int, WindowingResult] = {}

        n_labels: int = len(labels)

        labels_np: np.ndarray = np.array(labels, dtype=[("start", "i4"), ("end", "i4"), ("label", "S100")])

        for i in tqdm(range(num_windows), desc="Windowing", leave=False, ncols=80, disable=not show_progress):
            others: bool = False
            skip_window: bool = False

            result: WindowingResult = WindowingResult(
                audio_path=audio_path,
                window_st=self.config.start_offset + i * self.config.hop_size,
                window_en=self.config.start_offset + i * self.config.hop_size + self.config.window_size,
            )

            if n_labels != 0:
                overlapped_mask: np.ndarray = (labels_np["start"] < result.window_en) & (
                    labels_np["end"] > result.window_st
                )

            if sum(overlapped_mask) == 0 or n_labels == 0:
                if self.config.include_others in ["all", "ulb"]:
                    windowed_results[cnt] = result
                    cnt += 1
                continue

            overlapped_labels: np.ndarray = labels_np[overlapped_mask]
            _overlaps: np.ndarray = np.minimum(overlapped_labels["end"], result.window_en) - np.maximum(
                overlapped_labels["start"], result.window_st
            )
            _window_size: int = result.window_en - result.window_st
            _label_sizes: np.ndarray = overlapped_labels["end"] - overlapped_labels["start"]
            relative_ratios: np.ndarray = _overlaps / _window_size
            absolute_ratios: np.ndarray = _overlaps / _label_sizes

            for j, relative_ratio, absolute_ratio, label_name_np in zip(
                overlapped_labels, relative_ratios, absolute_ratios, overlapped_labels["label"]
            ):
                label_name: str = label_name_np.decode("utf-8")
                if label_name in self.config.exclude_labels:
                    if label_name not in self.excluded_labels:
                        self.excluded_labels.append(label_name)
                    skip_window = True
                    break

                result.label_name.append(label_name)
                result.relative_ratio.append(relative_ratio)
                result.absolute_ratio.append(absolute_ratio)
                result.label_id.append(j)

                found: bool = False
                for iv_label_name, similars in self.config.similar_labels.items():
                    if label_name.strip() in similars:
                        if (
                            relative_ratio >= self.config.relative_ratio_threshold
                            or absolute_ratio >= self.config.absolute_ratio_threshold
                        ):
                            found = True
                            break

                if found:
                    result.iv_name.append(iv_label_name)
                else:
                    others = True
                    result.iv_name.append(self.config.others)
                    if label_name not in self.oov_list + list(self.config.similar_labels.keys()):
                        if verbose:
                            print(f"Considering\n{label_name}\nas others.\n")
                        self.oov_list.append(label_name)

            # After the label loop.
            if skip_window or (others and not self._others_decision(result=result)):
                continue

            windowed_results[cnt] = result
            cnt += 1

        return windowed_results

    def _windowing_for_small_labels(
        self,
        audio_path: Path,
        labels: list[tuple[int, int, str]],
        num_windows: int,
        show_progress: bool,
        verbose: bool,
    ) -> dict[int, WindowingResult]:
        cnt: int = 0
        windowed_results: dict[int, WindowingResult] = {}

        n_labels: int = len(labels)

        if n_labels > 0:
            st_sorted_labels: list[tuple[int, int, str]] = sorted(labels, key=lambda x: x[0])
            en_sorted_labels: list[tuple[int, int, str]] = sorted(labels, key=lambda x: x[1])
            min_st: int = st_sorted_labels[0][0]
            max_en: int = en_sorted_labels[-1][1]

            iterator_dict: dict[Literal["start", "end"], list[tuple[int, tuple[int, int, str]]]] = {
                "start": list(enumerate(st_sorted_labels)),
                "end": list(enumerate(en_sorted_labels))[::-1],
            }

        for i in tqdm(range(num_windows), desc="Windowing", leave=False, ncols=80, disable=not show_progress):
            others: bool = False
            skip_window: bool = False

            result: WindowingResult = WindowingResult(
                audio_path=audio_path,
                window_st=self.config.start_offset + i * self.config.hop_size,
                window_en=self.config.start_offset + i * self.config.hop_size + self.config.window_size,
            )

            # No labels or the window is out of the label range,
            # and if including others "all" or "ulb", add the window and continue.
            if n_labels == 0 or min_st > result.window_en or max_en < result.window_st:
                if self.config.include_others in ["all", "ulb"]:
                    windowed_results[cnt] = result
                    cnt += 1
                continue

            # Decide whether to start from the start or the end.
            iterator_mode: Literal["start", "end"]
            if abs(result.window_st - min_st) < abs(result.window_en - max_en):
                iterator_mode = "start"
            else:
                iterator_mode = "end"

            oov_list: list[str] = []
            iv_list: list[str] = []
            for j, (st_int, en_int, label_name) in iterator_dict[iterator_mode]:
                # If there's no chance to overlap with the window, break the loop.
                if iterator_mode == "start" and st_int > result.window_en:
                    break
                elif iterator_mode == "end" and en_int < result.window_st:
                    break

                if st_int < result.window_en and en_int > result.window_st:
                    # No matter how much the ratio is, if the label is in the exclude labels, skip the window.
                    if label_name in self.config.exclude_labels:
                        if label_name not in self.excluded_labels:
                            self.excluded_labels.append(label_name)
                        skip_window = True
                        break

                    overlap: float = min(en_int, result.window_en) - max(st_int, result.window_st)
                    relative_ratio: float = overlap / self.config.window_size
                    absolute_ratio: float = overlap / (en_int - st_int)

                    result.label_name.append(label_name)
                    result.relative_ratio.append(relative_ratio)
                    result.absolute_ratio.append(absolute_ratio)
                    result.label_id.append(j)

                    found: bool = False
                    for iv_label_name, similars in self.config.similar_labels.items():
                        if label_name.strip() in similars:
                            if (
                                relative_ratio >= self.config.relative_ratio_threshold
                                or absolute_ratio >= self.config.absolute_ratio_threshold
                            ):
                                found = True
                                break

                    if found:
                        result.iv_name.append(iv_label_name)
                        iv_list.append(iv_label_name)
                    else:
                        others = True
                        result.iv_name.append(self.config.others)
                        if label_name not in self.oov_list + oov_list + list(self.config.similar_labels.keys()):
                            oov_list.append(label_name)

            # After the label loop.
            if skip_window or (others and not self._others_decision(result=result)):
                continue

            self.oov_list.extend(oov_list)
            self.iv_list.update(iv_list)

            windowed_results[cnt] = result
            cnt += 1

        return windowed_results

    def _windowing(
        self,
        audio_path: Path,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> dict[int, WindowingResult]:
        _label_path: Path = self._get_label_file_for_audio(audio_path=audio_path)
        labels = self._get_labels_for_audio(audio_path=audio_path, label_path=_label_path)

        audio_length: int = int(get_duration_sec(filepath=audio_path) * self.config.target_sr)
        num_windows: int = self._num_windows(audio_length=audio_length)

        # If the audio is slightly longer than the window size,
        # and the target audio is shorter than the window size,
        # place the target audio in the middle and make the window.
        # Only for target audio.
        if len(labels) == 1 and self.config.window_size < audio_length < self.config.window_size * 2:
            if (
                result := self._windowing_for_single_audio(
                    audio_path=audio_path, labels=labels, audio_length=audio_length
                )
            ) is not None:
                return result

        if len(labels) > 1000:
            return self._windowing_for_large_labels(
                audio_path=audio_path,
                labels=labels,
                num_windows=num_windows,
                show_progress=show_progress,
                verbose=verbose,
            )
        else:
            return self._windowing_for_small_labels(
                audio_path=audio_path,
                labels=labels,
                num_windows=num_windows,
                show_progress=show_progress,
                verbose=verbose,
            )

    def get_windows(
        self, verbose: bool = False, show_progress: Literal["overall", "each"] = "overall"
    ) -> dict[Path, dict[int, WindowingResult]]:
        """
        Window the audio files and save as wav files in the folder.

        Returns:
            A dictionary of windowed results.
            The key is the audio file name and the value is a dictionary of windowed results.
            The key of the inner dictionary is the window index and the value is the windowed result.
        """

        audio_paths: list[Path] = self._gather_audio_files()
        windows_dict: dict[Path, dict[int, WindowingResult]] = {}

        for audio_path in tqdm(
            audio_paths, desc="Windowing Progress", leave=False, disable=(show_progress != "overall"), ncols=80
        ):
            windows_dict[audio_path] = self._windowing(
                audio_path=audio_path, verbose=verbose, show_progress=(show_progress == "each")
            )

        self.oov_list.sort()
        self.excluded_labels.sort()

        if self.iv_list:
            print("Windowing iv list:")
            for lb in self.iv_list:
                print(f" - {lb}")
            print()

        if self.oov_list:
            print("Windowing oov list:")
            for lb in self.oov_list:
                print(f" - {lb}")
            print()

        if self.excluded_labels:
            print("Windowing excluded labels:")
            for lb in self.excluded_labels:
                print(f" - {lb}")
            print()

        return windows_dict

    def get_dataset(
        self,
        verbose: bool = False,
        show_progress: Literal["overall", "each"] = "overall",
        classes: list[str] | None = None,
    ) -> WindowingDataset:
        windows_dict: dict[Path, dict[int, WindowingResult]] = self.get_windows(
            verbose=verbose, show_progress=show_progress
        )

        return WindowingDataset(config=self.config, windows_dict=windows_dict, classes=classes)

    def get_stratified_grouped_split_dataset(
        self,
        dataset: BaseWindowingDataset,
        n_splits: int,
        classes: list[str] | None = None,
        seed: int = 42,
    ) -> dict[str, BaseWindowingDataset]:
        x: np.ndarray = np.zeros((len(dataset),))

        group_paths: list[Path] = [result[0] for result in dataset.windowing_results]
        unique_group_paths: list[Path] = list(set(group_paths))
        groups: np.ndarray = np.array([unique_group_paths.index(group_path) for group_path in group_paths])

        labels_tensor: torch.Tensor = torch.stack(dataset.labels, dim=0)
        labels: np.ndarray = (labels_tensor.sum(dim=-1) * labels_tensor.argmax(dim=-1)).numpy()

        sgkf: StratifiedGroupKFold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        sgkf.get_n_splits(X=x, y=labels, groups=groups)

        for i, (train_idx, valid_idx) in enumerate(sgkf.split(X=x, y=labels, groups=groups)):
            if i == 0:
                best_ratio: float = len(train_idx) / len(valid_idx)
                best_train_idx: list[int] = train_idx.tolist()
                best_valid_idx: list[int] = valid_idx.tolist()

            ratio: float = len(train_idx) / len(valid_idx)
            if abs(ratio - n_splits + 1) > abs(best_ratio - n_splits + 1):
                best_ratio = ratio
                best_train_idx = train_idx.tolist()
                best_valid_idx = valid_idx.tolist()

        if best_ratio == 0:
            raise ValueError("Check the dataset. There might be a problem on the split process.")

        return {
            "train": IndexedWindowingDataset(dataset=dataset, indices=best_train_idx),
            "valid": IndexedWindowingDataset(dataset=dataset, indices=best_valid_idx),
        }
