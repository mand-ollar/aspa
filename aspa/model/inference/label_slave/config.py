from pathlib import Path

from pydantic import BaseModel, model_validator
from typing_extensions import Self


class LabelSlaveConfig(BaseModel):
    name: str
    ckpt_path: str | Path | list[str] | list[Path]
    gpu_id: int | None = None
    window_sec: float = 2.0
    hop_sec: float = 0.5
    sr: int = 16000

    threshold: float = 0.5

    batch_size: int = 1024
    num_workers: int = 16

    @model_validator(mode="after")
    def validation(self) -> Self:
        if not isinstance(self.ckpt_path, list):
            self.ckpt_path = [Path(self.ckpt_path)]

        for ckpt_path in self.ckpt_path:
            ckpt_path = Path(ckpt_path)
            assert ckpt_path.exists(), f"File {ckpt_path} does not exist."

        return self
