import glob
import logging
import os
import torch
import torch.nn as nn
import numpy as np
import json
import time
import random

from typing import List, Literal, Tuple, Union
from dataclasses import dataclass, field
from torch.nn import MSELoss, L1Loss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import random_split, Subset
from torch.nn.functional import softplus
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from accelerate import Accelerator
from src.model import PerceptronLoss
from src.utils import (
    group_pkl_by_mol,
    mae_score,
    eval_r2_score,
    eval_mape_score,
    divisible_by,
    cycle,
    move_dict_to_device,
)
from src.base import BaseTrainer
from src.data import (
    ImageMlsesDataset,
    SparseImageMlsesDatasetConfig,
    SparseImageMlsesDataset,
    TranslationLabelTransformer,
    Image3dMlsesDataset,
    RotateImage3dMlsesDataset,
    Image3dEnergyDataset,
)


class Trainer:
    def __init__(self, args, model):
        # set parsed arguments
        self.args = args

        # init logger and tensorboard
        self._init_logger()
        self._set_writer()

        # init ingredients
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # init model
        self.model = model
        self.model = self.model.to(self.device)

        # init optimizer and learning rate scheduler
        self.optimizer = SGD(self.model.parameters(), lr=self.args.lr)

        # log status
        self.logger.info("Experiment setting:")
        for k, v in sorted(vars(self.args).items()):
            self.logger.info(f"{k}: {v}")

    def _get_lr(self, epoch_index, min_lr=1e-6) -> float:
        start_reduce_epoch = self.args.epoch // 2
        if epoch_index < start_reduce_epoch:
            return self.args.lr

        delta_lr = (self.args.lr - min_lr) / (self.args.epoch - start_reduce_epoch)
        next_lr = self.args.lr - delta_lr * (epoch_index - start_reduce_epoch)
        return next_lr

    def resume(self, resume_ckpt_path: str):
        # resume checkpoint
        self.logger.info(f"Resume model checkpoint from {resume_ckpt_path}...")
        self.model.load_state_dict(torch.load(resume_ckpt_path))

    def train_loop(self, train_dataset, eval_dataset, step_func):
        """Training loop function for model training and finetuning.

        :param train_dataset: training dataset
        :param eval_dataset: evaluation dataset
        :param step_func: a callable function doing forward and optimize step and return loss log
        """
        self.model.train()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.number_worker,
        )

        global_step = 0
        for epoch in range(0, self.args.epoch):
            # update learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self._get_lr(epoch)

            # train steps
            for step, (feat, label, mask) in enumerate(train_dataloader):
                feat: torch.Tensor = feat.to(self.device)
                label: torch.Tensor = label.to(self.device)
                mask: torch.Tensor = mask.to(self.device)

                # run step
                input_feats = {"feat": feat, "label": label, "mask": mask}
                loss_log = step_func(input_feats)

                # print loss
                if step % self.args.log_freq == 0:
                    loss_str = " ".join([f"{k}: {v:.4f}" for k, v in loss_log.items()])
                    self.logger.info(f"Epoch: {epoch} Step: {step} | Loss: {loss_str}")
                    for k, v in loss_log.items():
                        self.writer.add_scalar(f"train/{k}", v, global_step)

                    # log current learning rate
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("train/lr", current_lr, global_step)

                # increase step
                global_step += 1

            if epoch % self.args.eval_freq == 0:
                self.logger.info(f"Evaluate eval dataset at epoch {epoch}...")
                eval_output, _ = self.eval(eval_dataset)
                for k, v in eval_output.items():
                    self.logger.info(f"{k}: {v}")
                    self.writer.add_scalar(f"train/eval_{k}", v, epoch)

                torch.save(
                    self.model.state_dict(), f"{self.args.ckpt_dir}/model_{epoch}.pth"
                )

        # save the final model after training
        torch.save(self.model.state_dict(), f"{self.args.ckpt_dir}/model_final.pth")

    def train_step(self, input_feats):
        feat, label, mask = (
            input_feats["feat"],
            input_feats["label"],
            input_feats["mask"],
        )

        # clean gradient and forward
        self.optimizer.zero_grad()
        pred = self.model(feat)

        # compute weighted loss
        negative_mask = torch.logical_and(mask == 1, label < 1.5)
        positive_mask = torch.logical_and(mask == 1, torch.logical_not(negative_mask))
        regr_loss_fn = MSELoss(reduction="none")
        regr_loss_tn = regr_loss_fn(pred, label)
        weighted_regr_loss_tn = (
            regr_loss_tn * 0.5 * negative_mask + regr_loss_tn * positive_mask
        )
        regr_loss = torch.mean(torch.masked_select(weighted_regr_loss_tn, mask == 1))
        # sign_loss_fn = PerceptronLoss(self.args.sign_threshold)
        # sign_loss = sign_loss_fn(pred, label)
        sign_loss = torch.zeros_like(regr_loss)
        loss = regr_loss + sign_loss * self.args.lambda1

        # backward and update parameters
        loss.backward()
        self.optimizer.step()

        # prepare log dict
        log = {
            "loss": loss.item(),
            "regr_loss": regr_loss.item(),
            "sign_loss": sign_loss.item(),
        }
        return log

    def finetune_step(self, input_feats):
        feat, label = input_feats["feat"], input_feats["label"]

        # clean gradient and forward
        self.optimizer.zero_grad()
        pred = self.model(feat)

        # compute loss
        # regr_loss_fn = L1Loss()
        # regr_loss = regr_loss_fn(pred, label)
        sign_loss_fn = PerceptronLoss(self.args.sign_threshold)
        sign_loss = sign_loss_fn(pred, label)
        regr_loss = torch.zeros_like(sign_loss)
        loss = sign_loss

        # backward and update parameters
        loss.backward()
        self.optimizer.step()

        # prepare log dict
        log = {
            "loss": loss.item(),
            "regr_loss": regr_loss.item(),
            "sign_loss": sign_loss.item(),
        }
        return log

    def train(self, train_dataset, eval_dataset):
        self.train_loop(train_dataset, eval_dataset, self.train_step)

    def finetune(self, train_dataset, eval_dataset):
        self.train_loop(train_dataset, eval_dataset, self.finetune_step)

    @torch.inference_mode()
    def eval(self, dataset, num_workers: int = 0):
        self.model.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        y_pred_list = []
        y_true_list = []
        for feat, label, mask in tqdm(dataloader):
            feat: torch.Tensor = feat.to(self.device)
            pred = self.model(feat)

            y_pred_list.append(torch.masked_select(pred.cpu(), mask == 1))
            y_true_list.append(torch.masked_select(label, mask == 1))

        y_pred = torch.cat(y_pred_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)
        score = self.eval_on_prediction(y_pred, y_true)

        output = {}
        output["y_pred"] = y_pred
        output["y_true"] = y_true
        return score, output

    @torch.inference_mode()
    def eval_on_prediction(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        mae = mae_score(y_pred, y_true)
        r2 = eval_r2_score(y_pred, y_true)

        threshold = self.args.sign_threshold
        diff_sign_mask = torch.logical_or(
            torch.logical_and(y_true < threshold, y_pred > threshold),
            torch.logical_and(y_true > threshold, y_pred < threshold),
        )
        sign_error_num = diff_sign_mask.float().sum().item()

        score = {}
        score["absolute_mae"] = mae
        score["r2"] = r2
        score["sign_error_num"] = sign_error_num
        return score

    def _set_writer(self):
        self.logger.info("Create writer at '{}'".format(self.args.ckpt_dir))
        self.writer = SummaryWriter(self.args.ckpt_dir)

    def _init_logger(self):
        logging.basicConfig(
            filename=os.path.join(self.args.ckpt_dir, f"mlses_{self.args.mode}.log"),
            level=logging.INFO,
            datefmt="%Y/%m/%d %H:%M:%S",
            format="%(asctime)s: %(name)s [%(levelname)s] %(message)s",
        )
        formatter = logging.Formatter(
            "%(asctime)s: %(name)s [%(levelname)s] %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(stream_handler)


@dataclass
class EPBSurfaceTrainerConfig:
    # dataset path
    dataset_path: str = ""

    # patch size
    patch_size: int = 64

    # output path
    output_folder: str = "outputs"

    # dataset split proportions
    dataset_split_proportions: Tuple[float, float, float] = (0.7, 0.1, 0.2)

    # model lr
    train_lr: float = 1e-4

    # use perceptron loss
    use_perceptron_loss: bool = False

    # loss weight for sign loss
    lambda1: float = 1

    # morel adam
    adam_betas: Tuple[float, float] = (0.9, 0.99)

    # train step
    train_num_steps: int = 1000

    # model training batch size
    train_batch_size: int = 64

    # model evaluation batch size
    eval_batch_size: Union[int, None] = 64

    # eval stop at step (used for debugging)
    eval_early_stop: int = -1

    # eval and save model every
    save_and_eval_every: int = 1000

    # probe radius upperbound
    probe_radius_upperbound: float = 1.5

    # probe radius lowerbound
    probe_radius_lowerbound: float = -5

    # dataloader num workers
    num_workers: int = 0

    # device
    use_cuda: bool = True

    # sparse surface dataset
    use_sparse_surface_dataset: bool = False

    # sparse train dataset
    sparse_train_dataset_config: SparseImageMlsesDatasetConfig = field(
        default_factory=SparseImageMlsesDatasetConfig
    )

    # sparse test dataset
    sparse_test_dataset_config: SparseImageMlsesDatasetConfig = field(
        default_factory=SparseImageMlsesDatasetConfig
    )


class EPBSurfaceTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        seed: int,
        *,
        args: EPBSurfaceTrainerConfig,
        has_wandb_writer: bool = False,
    ) -> None:
        super().__init__(args.output_folder, has_wandb_writer)
        # device setting
        self.use_cuda = args.use_cuda
        self.seed = seed
        self.args = args

        self.dataloader_worker = args.num_workers
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.save_and_eval_every = args.save_and_eval_every
        self.lambda1 = args.lambda1
        self.eval_early_stop = args.eval_early_stop
        self.use_perceptron_loss = args.use_perceptron_loss
        self.use_sparse_surface_dataset = args.use_sparse_surface_dataset

        self.label_transformer = TranslationLabelTransformer(
            args.probe_radius_upperbound,
            args.probe_radius_lowerbound,
        )

        (
            self.sign_threshold,
            train_dataset,
            eval_dataset,
            test_dataset,
        ) = self.build_dataset(
            args.dataset_path,
            args.patch_size,
            args.dataset_split_proportions,
        )
        self.dl = cycle(
            DataLoader(
                train_dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=self.dataloader_worker,
            )
        )
        self.eval_dl = DataLoader(
            eval_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.dataloader_worker,
        )
        self.test_dl = DataLoader(
            test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.dataloader_worker,
        )

        # model settings
        self.model = model.to(self.device)
        self.opt = Adam(
            self.model.parameters(), lr=args.train_lr, betas=args.adam_betas
        )

        # step counter state
        self.step = 0
        self.train_num_steps = args.train_num_steps

    def build_dataset(
        self,
        dataset_path: str,
        patch_size: int,
        split_proportions: Tuple[float, float, float],
    ):
        sign_threshold = self.label_transformer.transform(torch.zeros(1)).item()

        if not self.use_sparse_surface_dataset:
            dataset = ImageMlsesDataset(
                dataset_path,
                patch_size=patch_size,
                label_transformer=self.label_transformer,
            )
            # split training, developing, and testing datasets
            lengths = [int(p * len(dataset)) for p in split_proportions]
            lengths[-1] = len(dataset) - sum(lengths[:-1])

            generator = torch.Generator().manual_seed(self.seed)
            train_dataset, eval_dataset, test_dataset = random_split(
                dataset, lengths, generator
            )
        else:
            train_dataset = SparseImageMlsesDataset(
                dataset_path,
                patch_size=patch_size,
                split="train",
                config=self.args.sparse_train_dataset_config,
                label_transformer=self.label_transformer,
            )
            eval_dataset = SparseImageMlsesDataset(
                dataset_path,
                patch_size=patch_size,
                split="eval",
                config=self.args.sparse_test_dataset_config,
                label_transformer=self.label_transformer,
            )
            test_dataset = SparseImageMlsesDataset(
                dataset_path,
                patch_size=patch_size,
                split="test",
                config=self.args.sparse_test_dataset_config,
                label_transformer=self.label_transformer,
            )

        return sign_threshold, train_dataset, eval_dataset, test_dataset

    def get_state(self):
        state = {
            "step": self.step,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
        }
        return state

    def load_state(self, state):
        self.model.load_state_dict(state["model"])
        self.step = state["step"]
        self.opt.load_state_dict(state["opt"])

    @property
    def device(self):
        return torch.device("cuda") if self.use_cuda else torch.device("cpu")

    @property
    def global_step(self) -> int:
        return self.step

    def set_model_state(self, train: bool = True):
        self.model.train(train)

    @torch.inference_mode()
    def get_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        mae = mae_score(y_pred, y_true)
        r2 = eval_r2_score(y_pred, y_true)

        threshold = self.sign_threshold
        diff_sign_mask = torch.logical_or(
            torch.logical_and(y_true < threshold, y_pred > threshold),
            torch.logical_and(y_true > threshold, y_pred < threshold),
        )
        sign_error_num = diff_sign_mask.float().sum().item()

        score = {}
        score["absolute_mae"] = mae
        score["r2"] = r2
        score["sign_error_num"] = sign_error_num
        score["sign_error_ratio"] = sign_error_num / len(y_true) * 100
        return score

    @torch.inference_mode()
    def eval_during_training(self):
        outputs = self.eval(self.eval_dl)
        self.set_model_state(True)
        return outputs

    @torch.inference_mode()
    def eval(self, dataloader: DataLoader):
        self.set_model_state(False)
        y_pred_list = []
        y_true_list = []
        preds = []
        labels = []
        masks = []

        step = 0
        for feat, label, mask in tqdm(dataloader):
            if self.eval_early_stop != -1 and step == self.eval_early_stop:
                break

            feat: torch.Tensor = feat.to(self.device)
            pred = self.model(feat)

            preds.append(pred.cpu().numpy())
            labels.append(label.cpu().numpy())
            masks.append(mask.cpu().numpy())
            y_pred_list.append(torch.masked_select(pred.cpu(), mask == 1))
            y_true_list.append(torch.masked_select(label, mask == 1))
            step += 1

        y_pred = torch.cat(y_pred_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)
        score = self.get_metrics(y_pred, y_true)

        output = {}
        output["y_pred"] = y_pred
        output["y_true"] = y_true
        output["preds"] = np.concatenate(preds, axis=0)
        output["labels"] = np.concatenate(labels, axis=0)
        output["masks"] = np.concatenate(masks, axis=0)
        return score, output

    def compute_loss(self, pred, label, mask):
        # compute weighted loss
        negative_mask = torch.logical_and(mask == 1, label < self.sign_threshold)
        positive_mask = torch.logical_and(mask == 1, torch.logical_not(negative_mask))
        regr_loss_fn = MSELoss(reduction="none")
        regr_loss_tn = regr_loss_fn(pred, label)
        weighted_regr_loss_tn = (
            regr_loss_tn * 0.5 * negative_mask + regr_loss_tn * positive_mask
        )
        regr_loss = torch.mean(torch.masked_select(weighted_regr_loss_tn, mask == 1))

        if self.use_perceptron_loss:
            sign_loss_fn = PerceptronLoss(self.sign_threshold)
            sign_loss = sign_loss_fn(pred, label)
        else:
            sign_loss = torch.zeros_like(regr_loss)
        loss = regr_loss + sign_loss * self.lambda1

        loss_dict = {
            "loss": loss.item(),
            "regr_loss": regr_loss.item(),
            "sign_loss": sign_loss.item(),
        }
        return loss, loss_dict

    def train(self):
        self.set_model_state(True)

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                feat, label, mask = next(self.dl)
                feat: torch.Tensor = feat.to(self.device)
                label: torch.Tensor = label.to(self.device)
                mask: torch.Tensor = mask.to(self.device)

                pred = self.model(feat)
                loss, loss_dict = self.compute_loss(pred, label, mask)

                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                pbar.set_description(f"loss: {loss.item():.4f}")
                self.log(loss_dict, section="train")

                if self.step != 0 and divisible_by(self.step, self.save_and_eval_every):
                    scores, _ = self.eval_during_training()
                    self.log(scores, section="eval")
                    milestone = self.step // self.save_and_eval_every
                    self.save(milestone)

                self.step += 1
                pbar.update(1)

        scores, _ = self.eval(self.eval_dl)
        self.log(scores, section="eval")
        scores, _ = self.eval(self.test_dl)
        self.log(scores, section="test")
        self.save("final")
        print("Training done!")


class GENIUSESTrainer(EPBSurfaceTrainer):
    @staticmethod
    def flatten(img: torch.Tensor) -> torch.Tensor:
        """Flatten input tensor into the channel-last format

        :param img: _description_
        :return: _description_
        """
        img = img.permute(0, 2, 3, 1)  # NCHW -> NHWC
        img = img.reshape(-1, img.shape[-1])  # N * H * W, C
        return img

    @staticmethod
    def inv_flatten(img: torch.Tensor, *shape) -> torch.Tensor:
        N, H, W = shape
        img = img.reshape((N, H, W, -1))  # NHWC
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return img

    @torch.inference_mode()
    def eval(self, dataloader: DataLoader):
        self.set_model_state(False)
        y_pred_list = []
        y_true_list = []
        preds = []
        labels = []
        masks = []

        step = 0
        for feat, label, mask in tqdm(dataloader):
            if self.eval_early_stop != -1 and step == self.eval_early_stop:
                break

            N, _, H, W = feat.shape
            feat = self.flatten(feat)
            label = self.flatten(label)
            mask = self.flatten(mask)

            feat: torch.Tensor = feat.to(self.device)
            pred = self.model(feat)

            preds.append(self.inv_flatten(pred, N, H, W).cpu().numpy())
            labels.append(self.inv_flatten(label, N, H, W).numpy())
            masks.append(self.inv_flatten(mask, N, H, W).numpy())

            y_pred_list.append(torch.masked_select(pred.cpu(), mask == 1))
            y_true_list.append(torch.masked_select(label, mask == 1))
            step += 1

        y_pred = torch.cat(y_pred_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)
        score = self.get_metrics(y_pred, y_true)

        output = {}
        output["y_pred"] = y_pred
        output["y_true"] = y_true
        output["preds"] = np.concatenate(preds, axis=0)
        output["labels"] = np.concatenate(labels, axis=0)
        output["masks"] = np.concatenate(masks, axis=0)
        return score, output

    def train(self):
        self.set_model_state(True)

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                feat, label, mask = next(self.dl)
                feat: torch.Tensor = feat.to(self.device)
                label: torch.Tensor = label.to(self.device)
                mask: torch.Tensor = mask.to(self.device)

                feat = self.flatten(feat)
                label = self.flatten(label)
                mask = self.flatten(mask)

                pred = self.model(feat)

                # compute weighted loss
                regr_loss_fn = L1Loss()

                regr_loss = regr_loss_fn(pred, label)
                sign_loss = torch.zeros_like(regr_loss)
                loss = regr_loss + sign_loss * self.lambda1

                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                pbar.set_description(f"loss: {loss.item():.4f}")
                self.log({"loss": loss.item()}, section="train")

                if self.step != 0 and divisible_by(self.step, self.save_and_eval_every):
                    scores, _ = self.eval_during_training()
                    self.log(scores, section="eval")
                    milestone = self.step // self.save_and_eval_every
                    self.save(milestone)

                self.step += 1
                pbar.update(1)

        scores, _ = self.eval(self.eval_dl)
        self.log(scores, section="eval")
        scores, _ = self.eval(self.test_dl)
        self.log(scores, section="test")
        self.save("final")
        print("Training done!")


class EPB3dSurfaceTrainer(EPBSurfaceTrainer):
    def build_dataset(
        self,
        dataset_path: str,
        patch_size: int,
        split_proportions: Tuple[float, float, float],
    ):
        sign_threshold = self.label_transformer.transform(torch.zeros(1)).item()

        if not self.use_sparse_surface_dataset:
            dataset = Image3dMlsesDataset(
                dataset_path,
                patch_size=patch_size,
                label_transformer=self.label_transformer,
            )

            # split training, developing, and testing datasets
            lengths = [int(p * len(dataset)) for p in split_proportions]
            lengths[-1] = len(dataset) - sum(lengths[:-1])

            generator = torch.Generator().manual_seed(self.seed)
            train_dataset, eval_dataset, test_dataset = random_split(
                dataset, lengths, generator
            )
        else:
            dataset = RotateImage3dMlsesDataset(
                dataset_path,
                patch_size=patch_size,
                config=self.args.sparse_train_dataset_config,
                label_transformer=self.label_transformer,
            )
            dataset_on_test = RotateImage3dMlsesDataset(
                dataset_path,
                patch_size=patch_size,
                config=self.args.sparse_test_dataset_config,
                label_transformer=self.label_transformer,
            )

            # split training, developing, and testing datasets
            lengths = [int(p * len(dataset)) for p in split_proportions]
            lengths[-1] = len(dataset) - sum(lengths[:-1])

            generator = torch.Generator().manual_seed(self.seed)
            train_dataset, _eval_dataset, _test_dataset = random_split(
                dataset, lengths, generator
            )

            eval_dataset = Subset(dataset_on_test, _eval_dataset.indices)
            test_dataset = Subset(dataset_on_test, _test_dataset.indices)
        return sign_threshold, train_dataset, eval_dataset, test_dataset

    @torch.inference_mode()
    def sample_eval(
        self,
        dataloader: DataLoader,
        sample_number: int = 35035,
        along_axis: Literal[2, 3, 4] = 2,
    ):
        """sample several image slices to eval the model

        :param dataloader: same as eval
        :param sample_number: sampled image slices, defaults to 35035, which is the number of image slices in the test dataset of conv2d
        :return: results
        """
        self.set_model_state(False)
        y_pred_list = []
        y_true_list = []
        preds = []
        labels = []
        masks = []

        step = 0
        for feat, label, mask in tqdm(dataloader):
            if self.eval_early_stop != -1 and step == self.eval_early_stop:
                break

            feat: torch.Tensor = feat.to(self.device)
            pred = self.model(feat)

            preds.append(pred.cpu().numpy())
            labels.append(label.cpu().numpy())
            masks.append(mask.cpu().numpy())

            total_slices = feat.shape[2]
            pred = pred.cpu()
            for i in range(total_slices):
                if torch.sum(mask[:, :, i]) < 1:
                    continue

                y_pred_list.append(
                    torch.masked_select(pred[:, :, i], mask[:, :, i] == 1)
                )
                y_true_list.append(
                    torch.masked_select(label[:, :, i], mask[:, :, i] == 1)
                )
            step += 1

        rng = random.Random(self.seed)
        indices = rng.sample(list(range(len(y_pred_list))), sample_number)

        y_pred = torch.cat([y_pred_list[i] for i in indices], dim=0)
        y_true = torch.cat([y_true_list[i] for i in indices], dim=0)
        score = self.get_metrics(y_pred, y_true)

        output = {}
        output["y_pred"] = y_pred
        output["y_true"] = y_true
        output["preds"] = np.concatenate(preds, axis=0)
        output["labels"] = np.concatenate(labels, axis=0)
        output["masks"] = np.concatenate(masks, axis=0)
        return score, output


class GENIUSES3dTrainer(EPB3dSurfaceTrainer):
    @staticmethod
    def flatten(img: torch.Tensor) -> torch.Tensor:
        """Flatten input tensor into the channel-last format

        :param img: _description_
        :return: _description_
        """
        img = img.permute(0, 2, 3, 4, 1)  # NCDHW -> NDHWC
        img = img.reshape(-1, img.shape[-1])  # N * D * H * W, C
        return img

    @staticmethod
    def inv_flatten(img: torch.Tensor, *shape) -> torch.Tensor:
        N, D, H, W = shape
        img = img.reshape((N, D, H, W, -1))  # NDHWC
        img = img.permute(0, 4, 1, 2, 3)  # NDHWC -> NCDHW
        return img

    @torch.inference_mode()
    def eval(self, dataloader: DataLoader):
        self.set_model_state(False)
        y_pred_list = []
        y_true_list = []
        preds = []
        labels = []
        masks = []

        step = 0
        for feat, label, mask in tqdm(dataloader):
            if self.eval_early_stop != -1 and step == self.eval_early_stop:
                break

            N, _, D, H, W = feat.shape
            feat = self.flatten(feat)
            label = self.flatten(label)
            mask = self.flatten(mask)

            feat: torch.Tensor = feat.to(self.device)
            pred = self.model(feat)

            preds.append(self.inv_flatten(pred, N, D, H, W).cpu().numpy())
            labels.append(self.inv_flatten(label, N, D, H, W).numpy())
            masks.append(self.inv_flatten(mask, N, D, H, W).numpy())

            y_pred_list.append(torch.masked_select(pred.cpu(), mask == 1))
            y_true_list.append(torch.masked_select(label, mask == 1))
            step += 1

        y_pred = torch.cat(y_pred_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)
        score = self.get_metrics(y_pred, y_true)

        output = {}
        output["y_pred"] = y_pred
        output["y_true"] = y_true
        output["preds"] = np.concatenate(preds, axis=0)
        output["labels"] = np.concatenate(labels, axis=0)
        output["masks"] = np.concatenate(masks, axis=0)
        return score, output
