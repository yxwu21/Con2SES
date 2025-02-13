import torch
import numpy as np
import math
import matplotlib.pyplot as plt

from torch.nn import L1Loss
from sklearn.metrics import f1_score, r2_score, mean_absolute_percentage_error


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def divisible_by(numer, denom):
    return (numer % denom) == 0


def tanh_acc(y_pred: torch.Tensor, y_true: torch.Tensor):
    y_pred_mapped = torch.where(y_pred > 0, 1, -1)
    mask = y_pred_mapped == y_true
    acc = mask.float().mean().item()
    return acc


def tanh_f1(y_pred: torch.Tensor, y_true: torch.Tensor, positive=1):
    y_pred_mapped = torch.where(y_pred > 0, 1, -1)
    score = f1_score(y_true.tolist(), y_pred_mapped.tolist(), pos_label=positive)
    return score


@torch.no_grad()
def mae_score(y_pred: torch.Tensor, y_true: torch.Tensor):
    mae_loss = L1Loss()
    score = mae_loss(y_pred, y_true)
    return score.item()


@torch.no_grad()
def relative_mae_score(y_pred: torch.Tensor, y_true: torch.Tensor):
    mae_loss = L1Loss(reduction="none")
    score = (mae_loss(y_pred, y_true) / torch.abs(y_true)).mean()
    return score.item()


@torch.no_grad()
def eval_r2_score(y_pred: torch.Tensor, y_true: torch.Tensor):
    score = r2_score(y_true.tolist(), y_pred.tolist())
    return score


@torch.no_grad()
def eval_mape_score(y_pred: torch.Tensor, y_true: torch.Tensor):
    score = mean_absolute_percentage_error(y_true.tolist(), y_pred.tolist())
    return score


def pad_array(array, x0, y0, patch_size):
    patch_array = array[x0 : x0 + patch_size, y0 : y0 + patch_size]
    pad_x = patch_size - patch_array.shape[0]
    pad_y = patch_size - patch_array.shape[1]

    pad_x_before = math.floor(pad_x / 2)
    pad_y_before = math.floor(pad_y / 2)
    pad_width = (
        (pad_x_before, pad_x - pad_x_before),
        (pad_y_before, pad_y - pad_y_before),
        (0, 0),
    )
    pad_patch_array = np.pad(patch_array, pad_width)
    return pad_patch_array


def pad_3d_array(array, x0, y0, z0, patch_size):
    patch_array = array[
        x0 : x0 + patch_size, y0 : y0 + patch_size, z0 : z0 + patch_size
    ]
    pad_x = patch_size - patch_array.shape[0]
    pad_y = patch_size - patch_array.shape[1]
    pad_z = patch_size - patch_array.shape[2]

    pad_x_before = math.floor(pad_x / 2)
    pad_y_before = math.floor(pad_y / 2)
    pad_z_before = math.floor(pad_z / 2)
    pad_width = (
        (pad_x_before, pad_x - pad_x_before),
        (pad_y_before, pad_y - pad_y_before),
        (pad_z_before, pad_z - pad_z_before),
        (0, 0),
    )
    pad_patch_array = np.pad(patch_array, pad_width)
    return pad_patch_array


def pad_3d_tensor(tensor, x0, y0, z0, patch_size):
    patch_tensor = tensor[
        x0 : x0 + patch_size, y0 : y0 + patch_size, z0 : z0 + patch_size
    ]
    pad_x = patch_size - patch_tensor.shape[0]
    pad_y = patch_size - patch_tensor.shape[1]
    pad_z = patch_size - patch_tensor.shape[2]

    pad_x_before = math.floor(pad_x / 2)
    pad_y_before = math.floor(pad_y / 2)
    pad_z_before = math.floor(pad_z / 2)

    pad = (
        pad_z_before,
        pad_z - pad_z_before,
        pad_y_before,
        pad_y - pad_y_before,
        pad_x_before,
        pad_x - pad_x_before,
    )

    pad_patch_tensor = torch.nn.functional.pad(patch_tensor, pad)
    return pad_patch_tensor


def cycle(dl):
    while True:
        for data in dl:
            yield data


def exists(x):
    return x is not None


def visualize_level_set_image(
    preds: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    select_indices: list,
    sign_threshold: float,
    probe_radius_lowerbound: float,
    probe_radius_upperbound: float,
):
    num_sample = len(select_indices)
    fig, axes = plt.subplots(2, num_sample, figsize=(2 * num_sample, 4))

    for i, ind in enumerate(select_indices):
        preds[ind][masks[ind] == 0] = sign_threshold
        axes[0, i].imshow(
            preds[ind][0],
            vmin=probe_radius_lowerbound,
            vmax=probe_radius_upperbound,
            cmap="RdBu",
        )
        axes[0, i].set_title(f"pred {i}")
        axes[0, i].set_axis_off()

        axes[1, i].imshow(
            labels[ind][0],
            vmin=probe_radius_lowerbound,
            vmax=probe_radius_upperbound,
            cmap="RdBu",
        )
        axes[1, i].set_title(f"label {i}")
        axes[1, i].set_axis_off()
    return fig


def visualize_joint_level_set_image(
    preds1: np.ndarray,
    preds2: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    select_indices: list,
    sign_threshold: float,
    probe_radius_lowerbound: float,
    probe_radius_upperbound: float,
):
    num_sample = len(select_indices)
    fig, axes = plt.subplots(3, num_sample, figsize=(2 * num_sample, 6))

    for i, ind in enumerate(select_indices):
        preds1[ind][masks[ind] == 0] = sign_threshold
        axes[0, i].imshow(
            preds1[ind][0],
            vmin=probe_radius_lowerbound,
            vmax=probe_radius_upperbound,
            cmap="RdBu",
        )
        # axes[0, i].set_title(f"pred {i}")
        axes[0, i].set_axis_off()

        preds2[ind][masks[ind] == 0] = sign_threshold
        axes[1, i].imshow(
            preds2[ind][0],
            vmin=probe_radius_lowerbound,
            vmax=probe_radius_upperbound,
            cmap="RdBu",
        )
        # axes[1, i].set_title(f"pred {i}")
        axes[1, i].set_axis_off()

        axes[2, i].imshow(
            labels[ind][0],
            vmin=probe_radius_lowerbound,
            vmax=probe_radius_upperbound,
            cmap="RdBu",
        )
        # axes[2, i].set_title(f"label {i}")
        axes[2, i].set_axis_off()
    return fig


def round_half_up(array: np.ndarray):
    # This will shift the decimal point in your number, rounding it away from zero, and
    # then floor it.
    rounded_array = np.floor(np.where(array >= 0, array + 0.5, array - 0.5))
    return rounded_array


def initialize_grid_space(
    atom_xyz: np.ndarray, atom_rad: np.ndarray, fill_ratio: float, h: float
):
    atom_plus = atom_xyz + atom_rad[:, None]
    atom_minu = atom_xyz - atom_rad[:, None]

    max_pos = atom_plus.max(0)
    min_pos = atom_minu.min(0)

    tmp_xmymzm = round_half_up((max_pos - min_pos) * fill_ratio / h)
    xmymzm = (
        2 * round_half_up(tmp_xmymzm / 2) + 1
    )  # maximum grid number for x, y, and z
    xbybzb = (
        round_half_up((max_pos + min_pos) / 2 / h) * h
    )  # the center points after mapping in grid step
    goxgoygoz = (
        xbybzb - ((xmymzm + 1) / 2) * h
    )  # the origin point after mapping in grid
    gcrd = (
        atom_xyz - goxgoygoz[None, :]
    ) / h  # from the grid origin, find the grid index

    return xmymzm, goxgoygoz, gcrd


def group_pkl_by_mol(pkl_list: list, pattern: str = "_raw.pkl.gz"):
    mol_dict = {}
    for pkl in pkl_list:
        mol_id = pkl.split("/")[-1].replace(pattern, "")
        if mol_id not in mol_dict:
            mol_dict[mol_id] = [pkl]
        else:
            mol_dict[mol_id].append(pkl)
    return mol_dict


def move_dict_to_device(tensor_dict, device):
    """
    Recursively move all tensors in a dictionary to the specified device.

    Args:
        tensor_dict (dict): A dictionary containing tensors or other nested dictionaries/lists.
        device (torch.device): The target device (e.g., torch.device('cuda') or torch.device('cpu')).

    Returns:
        dict: A new dictionary with all tensors moved to the specified device.
    """
    if isinstance(tensor_dict, dict):
        return {
            key: move_dict_to_device(value, device)
            for key, value in tensor_dict.items()
        }
    elif isinstance(tensor_dict, list):
        return [move_dict_to_device(item, device) for item in tensor_dict]
    elif isinstance(tensor_dict, torch.Tensor):
        return tensor_dict.to(device)
    else:
        return tensor_dict  # Return non-tensor types as-is
