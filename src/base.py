import wandb
import torch

from typing import Dict, Set
from pathlib import Path
from abc import abstractmethod


class BaseTrainer(object):
    def __init__(self, output_folder: str, has_wandb_writer: bool = False) -> None:
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)

        self.has_writer = has_wandb_writer
        self._wandb_defined_metrics: Set[str] = set()
        if self.has_writer:
            self._init_wandb()

    def _get_wandb_step_name(self, wandb_section: str = "trainer_state") -> str:
        return f"{wandb_section}/{self.__class__.__name__}_step"

    def _init_wandb(self):
        wandb.define_metric(self._get_wandb_step_name())
        self._wandb_defined_metrics.add(self._get_wandb_step_name())

    def log(self, log_dict: dict, step: int = None, section: str = "train"):
        if not self.has_writer:
            return

        # add section to each metric
        log_dict = {f"{section}/{k}": v for k, v in log_dict.items()}

        # define metrics against custom step name
        for k in log_dict.keys():
            if k not in self._wandb_defined_metrics:
                wandb.define_metric(k, step_metric=self._get_wandb_step_name())
                self._wandb_defined_metrics.add(k)

        # add step metrics
        if step is not None:
            log_dict.update({self._get_wandb_step_name(): step})
        else:
            log_dict.update({self._get_wandb_step_name(): self.global_step})

        wandb.log(log_dict)

    def save(self, milestone: str):
        state = self.get_state()
        torch.save(state, str(self.output_folder / f"model-{milestone}.pt"))

    def load(self, milestone: str):
        state = torch.load(
            str(self.output_folder / f"model-{milestone}.pt"), map_location=self.device
        )
        self.load_state(state)

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def set_model_state(self, train: bool = True):
        raise NotImplementedError

    @torch.inference_mode()
    @abstractmethod
    def eval(self, dataloader: torch.utils.data.DataLoader):
        raise NotImplementedError

    @torch.inference_mode()
    @abstractmethod
    def eval_during_training(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> torch.DeviceObjType:
        raise NotImplementedError

    @property
    @abstractmethod
    def global_step(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> Dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def load_state(self, state: Dict[str, object]):
        raise NotImplementedError
