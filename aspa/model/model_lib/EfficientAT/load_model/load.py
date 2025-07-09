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

    if model_cfg.pretrained_model:
        assert model_cfg.classes is not None, "Argument 'classes' should be provided if 'pretrained_model' is not None"
        model = get_ensemble_model(model_cfg.pretrained_model, num_classes=len(model_cfg.classes))
    else:
        raise ValueError("pretrained_model should be provided")

    return torch.nn.Sequential(mel, model)


def load_efficientdyat(
    model_cfg: EfficientDyATParams | None = None,
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
        suppress_output.suppress_warnings()

    # Load default model
    if ckpt_path is None:
        assert model_cfg is not None, "Argument 'model_cfg' should be provided if 'ckpt_path' is None"
        assert model_cfg.classes is not None, "Argument 'classes' should be provided if 'ckpt_path' is None"
        model = get_mel_DyMN(model_cfg=model_cfg)

        if not verbose:
            suppress_output.restore()

        model_output.model = model
        model_output.model_config = model_cfg

        return model_output

    ckpt: dict = torch.load(f=ckpt_path, map_location=device, weights_only=False)

    # model configs
    model_cfg_dict: dict[str, Any]
    if "hyper_parameters" in ckpt:
        model_cfg_dict = ckpt["hyper_parameters"]["hparams"]
    elif "args" in ckpt:
        model_cfg_dict = ckpt["args"]
        if not isinstance(model_cfg_dict["ensemble_model"], list):
            model_cfg_dict["ensemble_model"] = [model_cfg_dict["pretrained_model_name"]]
    else:
        raise ValueError("No hyperparameter found in the checkpoint")

    if "pretrained_model" not in model_cfg_dict.keys():
        if "ensemble_model" in model_cfg_dict.keys():
            model_cfg_dict["pretrained_model"] = model_cfg_dict["ensemble_model"]
        elif "pretrained_model_name" in model_cfg_dict.keys():
            model_cfg_dict["pretrained_model"] = [model_cfg_dict["pretrained_model_name"]]
        else:
            raise ValueError("No pretrained model key found in the checkpoint hyperparameters")

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

    key: str
    new_state_dict = OrderedDict()
    common_prefix: str
    if len(trained_model_weight) > 0:
        split_keys: list[list[str]] = [k.split(".") for k in trained_model_weight.keys()]

        common_parts: list[str] = []
        for parts in zip(*split_keys):
            if all(p == parts[0] for p in parts):
                common_parts.append(parts[0])
            else:
                break
        common_prefix = ".".join(common_parts)
    else:
        common_prefix = ""

    for key, value in trained_model_weight.items():
        new_key: str = key.replace(common_prefix, "").lstrip(".")
        new_state_dict[new_key] = value

    # load model
    if isinstance(model, nn.Sequential):
        model[1].load_state_dict(new_state_dict)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}. Expected nn.Sequential or nn.Module.")

    if not verbose:
        suppress_output.restore()
        suppress_output.restore_warnings()

    model = model.to(device=device)

    if not verbose:
        suppress_output.restore()

    model_output.model = model
    model_output.model_config = model_cfg

    return model_output
