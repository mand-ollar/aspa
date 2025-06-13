import torch
import torch.nn as nn
import torchaudio  # type: ignore

from aspa.model.model_lib.EfficientAT.load_model.config import EfficientDyATParams


class AugmentMelSTFT(nn.Module):
    def __init__(self, model_cfg: EfficientDyATParams) -> None:
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e
        # Similar config to the spectrograms used in AST: https://github.com/YuanGongND/ast

        torch.nn.Module.__init__(self)

        self.win_length = model_cfg.win_length
        self.n_mels = model_cfg.n_mels
        self.n_fft = model_cfg.n_fft
        self.sr = model_cfg.target_sr
        self.fmin = model_cfg.fmin
        if model_cfg.fmax is None or True:
            model_cfg.fmax = model_cfg.target_sr // 2 - model_cfg.fmax_aug_range // 2
            print(f"Warning: FMAX is None setting to {model_cfg.fmax}")
        self.fmax = model_cfg.fmax
        self.hopsize = model_cfg.hop_length

        self.window: torch.Tensor
        self.register_buffer(
            "window",
            torch.hann_window(model_cfg.win_length, periodic=False),
            persistent=False,
        )

        assert model_cfg.fmin_aug_range >= 1, (
            f"fmin_aug_range={model_cfg.fmin_aug_range} should be >=1; 1 means no augmentation"
        )
        assert model_cfg.fmin_aug_range >= 1, (
            f"fmax_aug_range={model_cfg.fmax_aug_range} should be >=1; 1 means no augmentation"
        )
        self.fmin_aug_range = model_cfg.fmin_aug_range
        self.fmax_aug_range = model_cfg.fmax_aug_range

        self.preemphasis_coefficient: torch.Tensor
        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-0.97, 1]]]), persistent=False)

        self.num_time_masks = model_cfg.num_time_masks
        self.num_freq_masks = model_cfg.num_freq_masks

        if model_cfg.freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(model_cfg.freqm, iid_masks=True)

        if model_cfg.timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(model_cfg.timem, iid_masks=True)

        assert model_cfg is not None, "model_cfg should not be None"
        self.model_cfg = model_cfg

    def forward(self, x: torch.Tensor):
        # input normalization
        if self.model_cfg.input_normalization:
            rms_x: torch.Tensor = torch.sqrt(torch.mean(x**2, dim=-1)).unsqueeze(dim=-1)

            x = x / (rms_x + 1e-6)

        # pre-emphasis
        x = nn.functional.conv1d(x, self.preemphasis_coefficient).squeeze(1)

        x = torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hopsize,
            win_length=self.win_length,
            center=True,
            normalized=False,
            window=self.window,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = (x**2).sum(dim=-1)  # power mag
        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()

        # don't augment eval data
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels,
            self.n_fft,
            float(self.sr),
            fmin,
            fmax,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        mel_basis = torch.as_tensor(
            torch.nn.functional.pad(input=mel_basis, pad=(0, 1), mode="constant", value=0.0),
            device=x.device,
        )

        melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 0.00001).log()

        if self.training:
            for _ in range(self.num_freq_masks):
                melspec = self.freqm(melspec)
            for _ in range(self.num_time_masks):
                melspec = self.timem(melspec)

        melspec = (melspec + 4.5) / 5.0  # fast normalization

        return melspec.unsqueeze(1)
