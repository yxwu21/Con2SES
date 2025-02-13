import math
import torch
import torch.nn as nn
import json

from typing import List, Literal, Union, Optional, Dict
from dataclasses import dataclass, field, asdict
from torch.nn import functional as F


def squared_hinge_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
    """ref: https://www.tensorflow.org/api_docs/python/tf/keras/losses/SquaredHinge"""
    hinge_loss = torch.maximum(1 - y_true * y_pred, torch.zeros_like(y_true))
    squared_loss = torch.square(hinge_loss)
    return squared_loss


class SquaredHingeLoss(nn.Module):
    def __init__(self):
        super(SquaredHingeLoss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = squared_hinge_loss(y_pred, y_true)
        return loss.mean()


class PerceptronLoss(nn.Module):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, y_pred, y_true):
        true_sign = torch.where(y_true > self.threshold, 1.0, -1.0)
        pred_sign = y_pred - self.threshold
        sign_loss = torch.max(torch.zeros_like(pred_sign), -pred_sign * true_sign)
        return sign_loss.mean()


class MLSESModel(nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super().__init__()
        self.layer1 = nn.Linear(dim1, dim2)
        self.layer2 = nn.Linear(dim2, dim3)
        self.layer3 = nn.Linear(dim3, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return x

    def load_ckpt(self, ckpt_file):
        ckpt = json.load(open(ckpt_file))

        with torch.no_grad():
            for i in range(1, 4):
                layer: nn.Linear = getattr(self, f"layer{i}")
                layer.weight.copy_(layer.weight.new_tensor(ckpt[f"W{i}"]))
                layer.bias.copy_(layer.bias.new_tensor(ckpt[f"b{i}"]))


class RefinedMLSESModel(nn.Module):
    def __init__(self, dim1, dim2, dim3, probe_radius):
        super().__init__()
        self.layer1 = nn.Linear(dim1, dim2)
        self.layer2 = nn.Linear(dim2, dim3)
        self.layer3 = nn.Linear(dim3, 1)

        self.probe_radius = probe_radius

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x


class SimpleConvMlsesModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
        )

        self.conv_out = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1))

    def forward(self, input_x):
        x = self.convs(input_x)

        # do prediction
        out = self.conv_out(x)
        return out


@dataclass(eq=False)
class MultiScaleConvMlsesModel(nn.Module):
    kernel_sizes: List[int] = field(default_factory=lambda: [1, 3, 5], hash=False)
    model_type: Literal["light", "medium", "medium_v2", "medium_v3", "heavy"] = "light"

    def __post_init__(self) -> None:
        super().__init__()
        if len(self.kernel_sizes) != 3:
            raise Exception("kernel_sizes must be a list of 3 elements")

        if self.model_type == "light":
            self.convs_1 = nn.Sequential(
                nn.Conv2d(96, 32, kernel_size=self.kernel_sizes[0]),
                nn.ReLU(),
            )

            self.convs_3 = nn.Sequential(
                nn.Conv2d(32, 8, kernel_size=self.kernel_sizes[1], padding="same"),
                nn.ReLU(),
            )

            self.convs_5 = nn.Sequential(
                nn.Conv2d(8, 8, kernel_size=self.kernel_sizes[2], padding="same"),
                nn.ReLU(),
            )

            self.conv_out = nn.Sequential(nn.Conv2d(48, 1, kernel_size=1))
        elif self.model_type == "medium":
            self.convs_1 = nn.Sequential(
                nn.Conv2d(96, 32, kernel_size=self.kernel_sizes[0]),
                nn.ReLU(),
            )

            self.convs_3 = nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=self.kernel_sizes[1], padding="same"),
                nn.ReLU(),
            )

            self.convs_5 = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=self.kernel_sizes[2], padding="same"),
                nn.ReLU(),
            )

            self.conv_out = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1))
        elif self.model_type == "medium_v2":
            self.convs_1 = nn.Sequential(
                nn.Conv2d(96, 48, kernel_size=self.kernel_sizes[0]),
                nn.ReLU(),
            )

            self.convs_3 = nn.Sequential(
                nn.Conv2d(48, 8, kernel_size=self.kernel_sizes[1], padding="same"),
                nn.ReLU(),
            )

            self.convs_5 = nn.Sequential(
                nn.Conv2d(8, 8, kernel_size=self.kernel_sizes[2], padding="same"),
                nn.ReLU(),
            )

            self.conv_out = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1))
        elif self.model_type == "medium_v3":
            self.convs_1 = nn.Sequential(
                nn.Conv2d(96, 64, kernel_size=self.kernel_sizes[0]),
                nn.ReLU(),
                nn.Conv2d(64, 32, kernel_size=self.kernel_sizes[0]),
                nn.ReLU(),
            )

            self.convs_3 = nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=self.kernel_sizes[1], padding="same"),
                nn.ReLU(),
            )

            self.convs_5 = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=self.kernel_sizes[2], padding="same"),
                nn.ReLU(),
            )

            self.conv_out = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1))
        elif self.model_type == "heavy":
            self.convs_1 = nn.Sequential(
                nn.Conv2d(96, 64, kernel_size=self.kernel_sizes[0]),
                nn.ReLU(),
                nn.Conv2d(64, 32, kernel_size=self.kernel_sizes[0]),
                nn.ReLU(),
            )

            self.convs_3 = nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=self.kernel_sizes[1], padding="same"),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=self.kernel_sizes[1], padding="same"),
                nn.ReLU(),
            )

            self.convs_5 = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=self.kernel_sizes[2], padding="same"),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=self.kernel_sizes[2], padding="same"),
                nn.ReLU(),
            )

            self.conv_out = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1))

    def forward(self, input_x):
        x_1 = self.convs_1(input_x)
        x_3 = self.convs_3(x_1)
        x_5 = self.convs_5(x_3)

        # do prediction
        out = self.conv_out(torch.cat([x_1, x_3, x_5], dim=1))
        return out


@dataclass(eq=False)
class MultiScaleConv3dMlsesModel(nn.Module):
    kernel_sizes: List[int] = field(default_factory=lambda: [1, 3, 5], hash=False)
    model_type: Literal["light", "medium", "heavy"] = "light"

    def __post_init__(self) -> None:
        super().__init__()
        if len(self.kernel_sizes) != 3:
            raise Exception("kernel_sizes must be a list of 3 elements")

        if self.model_type == "light":
            self.convs_1 = nn.Sequential(
                nn.Conv3d(96, 32, kernel_size=self.kernel_sizes[0]),
                nn.ReLU(),
            )

            self.convs_3 = nn.Sequential(
                nn.Conv3d(32, 8, kernel_size=self.kernel_sizes[1], padding="same"),
                nn.ReLU(),
            )

            self.convs_5 = nn.Sequential(
                nn.Conv3d(8, 8, kernel_size=self.kernel_sizes[2], padding="same"),
                nn.ReLU(),
            )

            self.conv_out = nn.Sequential(nn.Conv3d(48, 1, kernel_size=1))
        elif self.model_type == "medium":
            self.convs_1 = nn.Sequential(
                nn.Conv3d(96, 32, kernel_size=self.kernel_sizes[0]),
                nn.ReLU(),
            )

            self.convs_3 = nn.Sequential(
                nn.Conv3d(32, 16, kernel_size=self.kernel_sizes[1], padding="same"),
                nn.ReLU(),
            )

            self.convs_5 = nn.Sequential(
                nn.Conv3d(16, 16, kernel_size=self.kernel_sizes[2], padding="same"),
                nn.ReLU(),
            )

            self.conv_out = nn.Sequential(nn.Conv3d(64, 1, kernel_size=1))
        elif self.model_type == "heavy":
            self.convs_1 = nn.Sequential(
                nn.Conv3d(96, 64, kernel_size=self.kernel_sizes[0]),
                nn.ReLU(),
                nn.Conv3d(64, 32, kernel_size=self.kernel_sizes[0]),
                nn.ReLU(),
            )

            self.convs_3 = nn.Sequential(
                nn.Conv3d(32, 16, kernel_size=self.kernel_sizes[1], padding="same"),
                nn.ReLU(),
                nn.Conv3d(16, 16, kernel_size=self.kernel_sizes[1], padding="same"),
                nn.ReLU(),
            )

            self.convs_5 = nn.Sequential(
                nn.Conv3d(16, 16, kernel_size=self.kernel_sizes[2], padding="same"),
                nn.ReLU(),
                nn.Conv3d(16, 16, kernel_size=self.kernel_sizes[2], padding="same"),
                nn.ReLU(),
            )

            self.conv_out = nn.Sequential(nn.Conv3d(64, 1, kernel_size=1))

    def forward(self, input_x):
        x_1 = self.convs_1(input_x)
        x_3 = self.convs_3(x_1)
        x_5 = self.convs_5(x_3)

        # do prediction
        out = self.conv_out(torch.cat([x_1, x_3, x_5], dim=1))
        return out

    def encode(self, input_x):
        x_1 = self.convs_1(input_x)
        x_3 = self.convs_3(x_1)
        x_5 = self.convs_5(x_3)
        out = self.conv_out(torch.cat([x_1, x_3, x_5], dim=1))
        return torch.cat([x_1, x_3, x_5], dim=1), out
