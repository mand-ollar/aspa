from typing import Literal

from pydantic import BaseModel

from aspa.model.model_wrapper import ModelWrapper


class LabelSlaveConfig(BaseModel):
    name: str
    window_sec: float = 2.0
    hop_sec: float = 1.0
    sr: int = 16000

    model_wrapper: list[ModelWrapper]
    threshold: float = 0.5

    batch_size: int = 1024
    num_workers: int = 16
    task: Literal["tagging", "classification"] = "tagging"
