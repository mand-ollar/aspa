from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from aspa.utils.base_dataclass import BaseDataClass


@dataclass
class WindowingConfig(BaseDataClass):
    """Windowing configurations.

    Arguments:
        audio_folder: Path to the folder that contains the audio files. It can be str or Path, or a list of str or Path.
        classes: List of class labels.
        similar_labels: Dictionary of similar labels.
            The key is the label name, and the value is a list of similar labels you want to group together.
        window_size: Window size in seconds.
        hop_size: Hop size in seconds.
        start_offset: Start offset in seconds.
        relative_ratio_threshold: Relative ratio threshold.
            If the relative ratio of the target audio is greater than this value, it will be considered as a target audio.
        absolute_ratio_threshold: Absolute ratio threshold.
            If the absolute ratio of the target audio is greater than this value, it will be considered as a target audio.
        target_sr: Target sample rate.
        include_others: Whether to include others in the dataset.
            If it is not classified as target data, it will be classified as others by the following rules:
            - "lb": Include only labeled data as out-of-vocab as others.
            - "ulb": Include only unlabeled data as others.
            - "all": Include labeled and unlabeled data as others.
            - "none": Do not include others.
        others: Name of the others label. If there is no explicit others label, set it to None.
    """

    audio_folders: str | Path | list[str] | list[Path]
    classes: list[str] = field(default_factory=list)
    similar_labels: dict[str, list[str]] = field(default_factory=dict)
    window_size: float = 1.0
    hop_size: float = 0.5
    start_offset: float = 0.0
    relative_ratio_threshold: float = 1.0
    absolute_ratio_threshold: float = 1.0
    target_sr: int = 32000
    include_others: Literal["lb", "ulb", "all", "none"] = "all"
    others: str | None = None
    make_missing_label_files: bool = False

    def __post_init__(self) -> None:
        """Post initialization of the dataclass.
        This method is called after the dataclass is initialized.

        1. If the label name is not in the similar_labels, add it to the similar_labels.
        2. If there are missing labels in the similar_labels, add the label to the key of similar_labels.
        """
        for k, v in self.similar_labels.items():
            if k not in v:
                v.append(k)

        for c in self.classes:
            if c not in self.similar_labels:
                self.similar_labels[c] = [c]

        for k, v in self.similar_labels.items():
            self.similar_labels[k] = [str(element) for element in v]
