from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from aspa.model.model_lib.EfficientAT.architecture.ensemble import get_ensemble_model
from aspa.model.model_lib.EfficientAT.load_model.config import EfficientDyATParams
from aspa.model.model_lib.EfficientAT.load_model.preprocess import AugmentMelSTFT
from aspa.utils.discard_output import SuppressOutput


class ModelOutput:
    model: nn.Sequential
    model_config: EfficientDyATParams


def get_mel_DyMN(model_cfg: EfficientDyATParams) -> torch.nn.Sequential:
    mel = AugmentMelSTFT(model_cfg=model_cfg)

    if model_cfg.ensemble_model:
        assert model_cfg.classes is not None, (
            "Argument 'classes' should be provided if 'ensemble_model' is not None"
        )
        model = get_ensemble_model(
            model_cfg.ensemble_model, num_classes=len(model_cfg.classes)
        )
    else:
        raise ValueError("ensemble_model should be provided")

    return torch.nn.Sequential(mel, model)


def load_efficientdyat(
    model_cfg: EfficientDyATParams,
    ckpt_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    verbose: bool = False,
) -> ModelOutput:
    model_output: ModelOutput = ModelOutput()
    model: nn.Sequential

    if isinstance(device, str):
        device = torch.device(device)

    if not verbose:
        suppress_output: SuppressOutput = SuppressOutput()
        suppress_output.suppress()

    # Load default model
    if ckpt_path is None:
        assert model_cfg.classes is not None, (
            "Argument 'classes' should be provided if 'ckpt_path' is None"
        )
        model = get_mel_DyMN(model_cfg=model_cfg)

        if not verbose:
            suppress_output.restore()

        model_output.model = model
        model_output.model_config = model_cfg

        return model_output

    ckpt: dict = torch.load(f=ckpt_path, map_location=device, weights_only=False)

    # model configs
    ensemble: bool = True
    model_cfg_dict: dict[str, Any]
    if "hyper_parameters" in ckpt:
        model_cfg_dict = ckpt["hyper_parameters"]["hparams"]
    elif "args" in ckpt:
        model_cfg_dict = ckpt["args"]
        if not isinstance(model_cfg_dict["ensemble_model"], list):
            model_cfg_dict["ensemble_model"] = [model_cfg_dict["pretrained_model_name"]]
            print(model_cfg_dict["ensemble_model"])
        ensemble = False
    else:
        raise ValueError("No hyperparameter found in the checkpoint")

    model_cfg = EfficientDyATParams()
    model_cfg.update_from_dict(**model_cfg_dict)
    model = get_mel_DyMN(model_cfg=model_cfg)

    # load weight
    trained_model_weight: dict[str, torch.Tensor] = ckpt["state_dict"]

    # remove loss_fn.weight
    if "loss_fn.weight" in trained_model_weight:
        trained_model_weight.pop("loss_fn.weight")
    if "loss_fn.ce.weight" in trained_model_weight:
        trained_model_weight.pop("loss_fn.ce.weight")

    # remove "model." prefix
    key: str
    new_state_dict = OrderedDict()
    for key, value in trained_model_weight.items():
        new_key: str
        if key.startswith("model."):
            new_key = key[8:]  # removing "model.x."
        else:
            new_key = key
        if not ensemble:
            new_key = f"1.models.0.{new_key}"
        new_state_dict[new_key] = value

    # load model
    model.load_state_dict(new_state_dict)

    if not verbose:
        suppress_output.restore()

    model_output.model = model.to(device=device)
    model_output.model_config = model_cfg

    return model_output
