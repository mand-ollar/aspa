from pathlib import Path

import torch.nn as nn

from model.model_lib.EfficientAT.load_model import EfficientDyATParams, ModelOutput, load_efficientdyat
from model.model_wrapper import ModelWrapper


class EfficientIndustrial(ModelWrapper):
    def __init__(self, ckpt_path: str | Path | None, gpu_id: int | None = None) -> None:
        super().__init__(gpu_id=gpu_id)

        assert ckpt_path is not None, "Checkpoint path must be provided."
        self.model = ckpt_path

    def set_model(self, ckpt_path: str | Path | None) -> nn.Sequential:
        model_output: ModelOutput = load_efficientdyat(
            model_cfg=EfficientDyATParams(), ckpt_path=ckpt_path, verbose=False
        )
        self.classes = model_output.model_config.classes
        self.target_length = model_output.model_config.data_length
        self.sr = model_output.model_config.target_sr

        return model_output.model
