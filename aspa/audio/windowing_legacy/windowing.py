import time
from pathlib import Path

import librosa
from tqdm import tqdm

from .config import WindowingConfig
from .dataset import WindowingDataset
from .types import WindowingResult


class Windowing:
    def __init__(self, config: WindowingConfig) -> None:
        self.config: WindowingConfig = config
        self.oov_list: list[str] = []
        self.excluded_labels: list[str] = []

        self.audio_folders: list[Path]
        if isinstance(self.config.audio_folders, (str, Path)):
            self.audio_folders = [Path(self.config.audio_folders)]
        else:
            self.audio_folders = [Path(folder) for folder in self.config.audio_folders]

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

    def _windowing(
        self,
        audio_filepath: str | Path,
        label_filepath: str | Path,
        verbose: bool = False,
    ) -> dict[int, WindowingResult]:
        """
        Window the long audio file and save as wav files in the folder.

        Arguments:
            audio_filepath: Path to the audio file.
            label_filepath: Path to the label file. In sample point unit.
        """

        st_time: float = time.time()
        audio_length: int = int(librosa.get_duration(path=audio_filepath) * self.config.target_sr)
        audio_filepath = Path(audio_filepath)

        with open(file=label_filepath, mode="r") as f:
            labels: list[str] = f.read().strip().split("\n")
            is_pnt_unit: bool = True
            try:
                st, en, _ = labels[0].split("\t")
                if not st.isdigit() or not en.isdigit():
                    is_pnt_unit = False
            except ValueError:
                is_pnt_unit = True

        split_labels: list[tuple[int, int, str]] = []
        for label in labels:
            if label == "":
                continue

            st, en, label_name = label.split("\t")
            label_name = label_name.strip()

            st_number: float = float(st)
            en_number: float = float(en)

            assert st_number <= en_number, (
                f"Start time must be smaller than end time: \n\t - Line {labels.index(label) + 1} of {audio_filepath}, {label}.\n"
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

        window_size: int = int(self.config.window_size * self.config.target_sr)
        hop_size: int = int(self.config.hop_size * self.config.target_sr)
        start_offset: int = int(self.config.start_offset * self.config.target_sr)

        # If the audio is shorter than the window size,
        # pad the audio to the window size.
        if audio_length < window_size:
            audio_length += window_size - audio_length

        # To match the number of windows when start_offset > 0
        if start_offset > 0:
            audio_length += start_offset

        num_windows: int = (audio_length - window_size) // hop_size + 1
        print(f"Time taken: {time.time() - st_time:.2f} seconds")

        result: WindowingResult
        windowed_results: dict[int, WindowingResult]
        iv_label_name: str
        st_int: int
        en_int: int
        found: bool

        # If the audio is slightly longer than the window size,
        # and the target audio is shorter than the window size,
        # place the target audio in the middle and make the window.
        # Only for target audio.
        if len(split_labels) == 1 and audio_length > window_size:
            st_int, en_int, label_name = split_labels[0]

            if label_name in self.config.exclude_labels:
                return {}

            if en_int - st_int < window_size:
                found = False
                for iv_label_name, similars in self.config.similar_labels.items():
                    if label_name in similars:
                        found = True
                        break

                if found:
                    result = WindowingResult()
                    result.audio_path = audio_filepath
                    result.window_st = (en_int + st_int) // 2 - window_size // 2
                    result.window_en = (en_int + st_int) // 2 - window_size // 2 + window_size
                    result.iv_name = [iv_label_name]
                    result.label_name = [label_name]
                    result.relative_ratio = [1.0]
                    result.absolute_ratio = [1.0]
                    result.label_id = 0

                    return {0: result}

        windowed_results = {}
        cnt: int = 0
        for i in tqdm(range(num_windows), desc="Windowing", leave=False, ncols=80):
            result = WindowingResult()
            result.audio_path = audio_filepath
            result.window_st = start_offset + i * hop_size
            result.window_en = result.window_st + window_size
            result.iv_name = []
            result.label_name = []
            result.relative_ratio = []
            result.absolute_ratio = []

            skip_window: bool = False

            if not split_labels or (
                min([split_label[0] for split_label in split_labels]) > result.window_en
                or max([split_label[1] for split_label in split_labels]) < result.window_st
            ):
                break

            for j, (st_int, en_int, label_name) in enumerate(split_labels):
                if st_int < result.window_en and en_int > result.window_st:  # Target overlapped with the window
                    overlap: float = min(en_int, result.window_en) - max(st_int, result.window_st)
                    relative_ratio: float = overlap / window_size
                    absolute_ratio: float = overlap / (en_int - st_int)

                    result.label_name.append(label_name)
                    result.relative_ratio.append(relative_ratio)
                    result.absolute_ratio.append(absolute_ratio)
                    result.label_id = j

                    found = False
                    for iv_label_name, similars in self.config.similar_labels.items():
                        if label_name in similars:
                            if (
                                relative_ratio >= self.config.relative_ratio_threshold
                                or absolute_ratio >= self.config.absolute_ratio_threshold
                            ):
                                if label_name in self.config.exclude_labels:
                                    skip_window = True
                                    if label_name not in self.excluded_labels:
                                        self.excluded_labels.append(label_name)
                                    break
                                result.iv_name.append(iv_label_name)
                                found = True
                                break
                            else:
                                skip_window = True
                                break

                    if skip_window:
                        break

                    if not found:
                        if label_name in self.config.exclude_labels:
                            skip_window = True
                            if label_name not in self.excluded_labels:
                                self.excluded_labels.append(label_name)
                            break

                        result.iv_name.append(self.config.others)

                    if (
                        not found
                        and label_name not in self.oov_list
                        and label_name not in list(self.config.similar_labels.keys())
                    ):
                        if verbose:
                            print(f"Considering\n{label_name}\nas others.\n")
                        self.oov_list.append(label_name)

            if skip_window:
                continue

            assert (
                len(result.iv_name)
                == len(result.label_name)
                == len(result.relative_ratio)
                == len(result.absolute_ratio)
            ), "Window information must be added for every key."

            windowed_results[cnt] = result
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

        audio_files: list[Path] = self._gather_audio_files()
        windows_dict: dict[Path, dict[int, WindowingResult]] = {}

        # for audio_file in tqdm(audio_files, desc="Windowing audio files", leave=False, ncols=80):
        for audio_file in audio_files:
            label_file: Path
            if audio_file.with_suffix(".tsv").exists():
                label_file = audio_file.with_suffix(".tsv")
            elif audio_file.with_suffix(".txt").exists():
                label_file = audio_file.with_suffix(".txt")
            else:
                label_file = audio_file.with_suffix(".tsv")

                if self.config.make_missing_label_files:
                    label_file.touch()
                else:
                    raise FileNotFoundError(
                        f"Label file not found: {audio_file.with_suffix('.tsv')} or {audio_file.with_suffix('.txt')}"
                    )

            windows_dict[audio_file] = self._windowing(
                audio_filepath=audio_file,
                label_filepath=label_file,
                verbose=verbose,
            )

            for _, result in windows_dict[audio_file].items():
                assert result.window_en - result.window_st == int(self.config.window_size * self.config.target_sr), (
                    f"Window size mismatch: {result.window_en - result.window_st} != {int(self.config.window_size * self.config.target_sr)}"
                )

        print("Windowing oov list:")
        print(self.oov_list)
        print()

        print("Windowing excluded labels:")
        print(self.excluded_labels)
        print()

        return windows_dict

    def get_dataset(self) -> WindowingDataset:
        """
        Get the dataset of windowed audio files and labels.

        Returns:
            WindowingDataset of windowed audio files and labels.
        """

        dataset: WindowingDataset = WindowingDataset(config=self.config, windows_dict=self.get_windows())

        return dataset
