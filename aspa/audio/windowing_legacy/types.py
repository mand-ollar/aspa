from pathlib import Path


class WindowingResult:
    audio_path: Path
    window_st: int
    window_en: int
    iv_name: list[str | None]
    label_name: list[str]
    relative_ratio: list[float]
    absolute_ratio: list[float]
    label_id: int

    def print_result(self) -> None:
        print(f"Audio path: {self.audio_path}")
        print(f"Window start: {self.window_st}")
        print(f"Window end: {self.window_en}")
        print(f"IV name: {self.iv_name}")
        print(f"Label name: {self.label_name}")
        print(f"Relative ratio: {self.relative_ratio}")
        print(f"Absolute ratio: {self.absolute_ratio}")
