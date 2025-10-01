import json
from pathlib import Path
from typing import Optional


class WindowingResult:
    audio_path: Path
    window_st: int
    window_en: int
    iv_name: list[str | None]
    label_name: list[str]
    relative_ratio: list[float]
    absolute_ratio: list[float]
    label_id: list[int]

    def __init__(
        self,
        audio_path: Path,
        window_st: int,
        window_en: int,
        iv_name: Optional[list[str | None]] = None,
        label_name: Optional[list[str]] = None,
        relative_ratio: Optional[list[float]] = None,
        absolute_ratio: Optional[list[float]] = None,
        label_id: Optional[list[int]] = None,
    ) -> None:
        self.audio_path = audio_path
        self.window_st = window_st
        self.window_en = window_en
        self.iv_name = iv_name if iv_name is not None else []
        self.label_name = label_name if label_name is not None else []
        self.relative_ratio = relative_ratio if relative_ratio is not None else []
        self.absolute_ratio = absolute_ratio if absolute_ratio is not None else []
        self.label_id = label_id if label_id is not None else []

    def print_result(self) -> None:
        print(f"Audio path: {self.audio_path}")
        print(f"Window start: {self.window_st}")
        print(f"Window end: {self.window_en}")
        print(f"IV name: {self.iv_name}")
        print(f"Label name: {self.label_name}")
        print(f"Relative ratio: {self.relative_ratio}")
        print(f"Absolute ratio: {self.absolute_ratio}")
        print()

    def __str__(self) -> str:
        return json.dumps({
            "audio_path": str(self.audio_path),
            "window_st": self.window_st,
            "window_en": self.window_en,
            "iv_name": self.iv_name,
            "label_name": self.label_name,
            "relative_ratio": self.relative_ratio,
            "absolute_ratio": self.absolute_ratio,
        })
