import torch
import torch.nn as nn

from .dymn.model import DyMN
from .dymn.model import get_model as get_dymn
from .helpers.utils import NAME_TO_WIDTH
from .mn.model import MN
from .mn.model import get_model as get_mobilenet


class EnsemblerModel(nn.Module):
    def __init__(self, models: list[nn.Module]):
        super(EnsemblerModel, self).__init__()
        self.models: nn.ModuleList = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        all_out: torch.Tensor | None = None
        all_feat: torch.Tensor | None = None

        out: torch.Tensor
        feat: torch.Tensor
        for m in self.models:
            out, feat = m(x)
            if all_out is None:
                all_out = out
            else:
                all_out = out + all_out
            if all_feat is None:
                all_feat = feat
            else:
                all_feat = feat + all_feat

        assert all_out is not None
        assert all_feat is not None

        all_out = all_out / len(self.models)
        all_feat = all_feat / len(self.models)

        return all_out, all_feat


def get_ensemble_model(model_names: list[str], num_classes: int) -> EnsemblerModel:
    models: list[nn.Module] = []

    model: MN | DyMN
    for model_name in model_names:
        if model_name.startswith("dymn"):
            model = get_dymn(
                width_mult=NAME_TO_WIDTH(model_name),
                pretrained_name=model_name,
                num_classes=num_classes,
            )
        else:
            model = get_mobilenet(
                width_mult=NAME_TO_WIDTH(model_name),
                pretrained_name=model_name,
                num_classes=num_classes,
            )
        models.append(model)

    return EnsemblerModel(models)
