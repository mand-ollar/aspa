from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aspa.utils.printings import Colors

from .config import LabelSlaveConfig
from .dataset import LabelSlaveDataset
from .model import EfficientAT, ModelWrapper


class LabelSlave:
    def __init__(self, config: LabelSlaveConfig) -> None:
        self.config: LabelSlaveConfig = config

        self.dataset: LabelSlaveDataset
        self.model: list[ModelWrapper] = []

        assert isinstance(self.config.ckpt_path, list), (
            "ckpt_path must be a list of paths."
        )
        self._set_model(ckpt_paths=self.config.ckpt_path)

    @property
    def name(self) -> str:
        return self.config.name

    def _set_model(self, ckpt_paths: list[str] | list[Path]) -> None:
        self.model = []

        for ckpt_path in ckpt_paths:
            model: EfficientAT = EfficientAT(
                ckpt_path=ckpt_path, gpu_id=self.config.gpu_id
            )
            assert model.classes is not None, "Model classes are not set."
            self.classes: list[str] = model.classes

            assert model.sr is not None, "Model sample rate is not set."
            self.config.sr = model.sr
            self.model.append(model)

    def _get_model_results(
        self, x: torch.Tensor, mode: Literal["logit", "confidence"]
    ) -> torch.Tensor:
        results: list[torch.Tensor] = []

        for model in self.model:
            if mode == "logit":
                results.append(model.logits(x=x))
            elif mode == "confidence":
                results.append(model.confidences(x=x))
            else:
                raise ValueError(
                    f"Invalid mode: {mode}. Choose either 'logit' or 'confidence'."
                )

        result: torch.Tensor = torch.mean(torch.stack(results), dim=0)

        return result

    def _set_dataset(
        self,
        wav_path: str | Path | None = None,
        audio: torch.Tensor | None = None,
        audio_sr: int | None = None,
    ) -> None:
        assert wav_path is not None or audio is not None, (
            "Either wav_path or audio must be provided."
        )

        self.dataset = LabelSlaveDataset(
            wav_path=wav_path,
            audio=audio,
            audio_sr=audio_sr,
            sr=self.config.sr,
            window_sec=self.config.window_sec,
            hop_sec=self.config.hop_sec,
        )
        assert len(self.dataset) > 0, (
            "Dataset is empty. Check the wav file path and parameters."
        )

    def _create_labels(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a tuple of (confidence, tensor[window_st, window_en])."""
        dataloader: DataLoader = DataLoader(
            dataset=self.dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )
        confidence_list: list[torch.Tensor] = []
        window_list: list[torch.Tensor] = []
        for audio, window in tqdm(
            dataloader, ncols=80, desc="Creating labels", leave=False
        ):
            confidence: torch.Tensor = self._get_model_results(
                x=audio, mode="confidence"
            )
            confidence_list.append(confidence)
            window_list.append(window)

        confidences: torch.Tensor = torch.cat(confidence_list, dim=0)
        windows: torch.Tensor = torch.cat(window_list, dim=0)

        return confidences, windows

    def _get_label_names_from_confidence(self, confidence: torch.Tensor) -> list[str]:
        if confidence.ndim != 1:
            raise ValueError(f"Expected batch size 1, got {confidence.size()}.")

        predictions: torch.Tensor
        if self.model[0].task == "tagging":
            predictions = (confidence >= self.config.threshold).int()
        elif self.model[0].task == "classification":
            predictions = (confidence == confidence.max(dim=-1).values).int()
        else:
            raise ValueError(
                f"Invalid task type: {self.model[0].task}. Choose either 'tagging' or 'classification'."
            )
        labels_list: list[str] = [
            self.classes[i] for i in torch.where(predictions == 1)[0]
        ]

        return labels_list

    def _stringify_labels(self, label: tuple[torch.Tensor, torch.Tensor]) -> str:
        """Converts the labels to a string format."""
        label_str_list: list[str] = []
        confidences, windows = label
        for confidence, window in zip(confidences, windows):
            st_pnt: int = int(window[0].item())
            en_pnt: int = int(window[1].item())
            str_labels: list[str] = self._get_label_names_from_confidence(
                confidence=confidence
            )

            for str_label in str_labels:
                label_str_list.append(f"{st_pnt}\t{en_pnt}\t{str_label}")

        label_str: str = "\n".join(label_str_list)

        return label_str

    @staticmethod
    def _merge_labels(label: str) -> str:
        label_list: list[str] = label.strip().split("\n")

        class_name_dict: dict[str, list[str]] = {}
        for label in label_list:
            _, _, class_name = label.split("\t")

            if class_name not in class_name_dict:
                class_name_dict[class_name] = []
            class_name_dict[class_name].append(label)

        merged_label_list: list[str] = []

        for class_name, labels in class_name_dict.items():
            st_pnts: list[int] = []
            en_pnts: list[int] = []
            for label in labels:
                st_pnt, en_pnt, _ = label.split("\t")
                st_pnts.append(int(st_pnt))
                en_pnts.append(int(en_pnt))

            st_pnts.sort()
            en_pnts.sort()

            merged_st_pnts: list[int] = []
            merged_en_pnts: list[int] = []

            merged_st_pnts.append(st_pnts.pop(0))

            while len(st_pnts) > 0 and len(en_pnts) > 1:
                temp_st_pnt: int = st_pnts.pop(0)
                temp_en_pnt: int = en_pnts.pop(0)

                if temp_st_pnt <= temp_en_pnt:
                    continue
                else:
                    merged_st_pnts.append(temp_st_pnt)
                    merged_en_pnts.append(temp_en_pnt)

            merged_en_pnts.append(en_pnts.pop(0))

            for st_pnt_int, en_pnt_int in zip(merged_st_pnts, merged_en_pnts):
                merged_label_list.append(f"{st_pnt_int}\t{en_pnt_int}\t{class_name}")

            merged_label: str = "\n".join(merged_label_list)

        return merged_label

    @staticmethod
    def _sort_labels(label: str) -> str:
        label_list: list[str] = label.split("\n")
        label_list = sorted(label_list, key=lambda x: int(x.split("\t")[0]))

        new_label: str = "\n".join(label_list)

        return new_label

    @staticmethod
    def _as_txt(label: str, sr: int) -> str:
        new_lines: list[str] = []
        lines: list[str] = label.strip().split("\n")
        for line in lines:
            st_pnt_str, en_pnt_str, class_name = line.split("\t")
            st_sec: float = int(st_pnt_str) / sr
            en_sec: float = int(en_pnt_str) / sr
            new_lines.append(f"{st_sec:.2f}\t{en_sec:.2f}\t{class_name}")

        new_label: str = "\n".join(new_lines)

        return new_label

    @staticmethod
    def _print(msg: str = "") -> None:
        print(f"[ModelWrapper] {msg}\n")

    def work(
        self,
        wav_path: str | Path | None = None,
        audio: torch.Tensor | None = None,
        audio_sr: int | None = None,
        label_save_path: str | Path | None = None,
        as_txt: bool = False,
        print_result: bool = False,
    ) -> None:
        self._print(
            f"Slave {Colors.BOLD}{Colors.RED}{self.name}{Colors.END} is working!"
        )

        assert wav_path is not None or audio is not None, (
            "Either wav_path or audio must be provided."
        )

        self._set_dataset(wav_path=wav_path, audio=audio, audio_sr=audio_sr)
        result: str = LabelSlave._sort_labels(
            label=LabelSlave._merge_labels(
                label=self._stringify_labels(label=self._create_labels())
            )
        )

        save_path: Path | None
        if wav_path is not None:
            save_path = Path(wav_path).with_suffix(".txt" if as_txt else ".tsv")

        elif wav_path is None and label_save_path is not None:
            if Path(label_save_path).suffix == "":
                self._print(
                    f"Appending suffix {'.txt' if as_txt else '.tsv'} to {label_save_path}"
                )
            elif as_txt and Path(label_save_path).suffix != ".txt":
                self._print(
                    f"Changing suffix from {Path(label_save_path).suffix} to .txt"
                )
            elif not as_txt and Path(label_save_path).suffix != ".tsv":
                self._print(
                    f"Changing suffix from {Path(label_save_path).suffix} to .tsv"
                )
            save_path = Path(label_save_path).with_suffix(".txt" if as_txt else ".tsv")

        else:
            save_path = None

        if save_path is not None:
            if as_txt:
                with open(file=save_path, mode="w") as f:
                    result = self._as_txt(label=result, sr=self.config.sr)
                    f.write(result)
            else:
                with open(file=save_path, mode="w") as f:
                    f.write(result)
            self._print(f"Label file saved to {save_path}")
            self._print(
                "No save path provided. Labels will be printed but not be saved."
            )

        if save_path is None or print_result:
            print("Model inference result:")
            print(self._as_txt(label=result, sr=self.config.sr))
            print()
