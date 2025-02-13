import os
import glob
import torch
import math
import numpy as np
import json
import gzip
import pickle
import torch.nn.functional as F

from scipy.sparse import csr_matrix
from typing import List, Literal, Tuple, Union
from torch.utils.data import Dataset
from tqdm import tqdm
from dataclasses import dataclass
from .utils import (
    pad_array,
    pad_3d_array,
)


class LabelTransformer:
    """
    Normalize label value to range -1 to 1
    """

    def __init__(self, probe_radius_upperbound: float, probe_radius_lowerbound: float):
        self.upperbound = probe_radius_upperbound
        self.lowerbound = probe_radius_lowerbound

    def transform(self, x: torch.Tensor):
        clamp_x = torch.clamp_(x, min=self.lowerbound, max=self.upperbound)
        trans_x = (clamp_x - self.lowerbound) / (
            self.upperbound - self.lowerbound
        ) * 2 - 1
        return trans_x

    def inv_transform(self, x: torch.Tensor):
        inv_x = (x + 1) / 2 * (self.upperbound - self.lowerbound) + self.lowerbound
        return inv_x

    @property
    def sign_threshold(self):
        return self.transform(torch.zeros(1)).item()


class ExpLabelTransformer(LabelTransformer):
    """
    Normalize label value to range -1 to 1
    """

    def __init__(
        self,
        probe_radius_upperbound: float,
        probe_radius_lowerbound: float,
        offset: float,
    ):
        self.upperbound = probe_radius_upperbound
        self.lowerbound = probe_radius_lowerbound
        self.offset = offset

        self.exp_upperbound = math.exp(probe_radius_upperbound + self.offset)
        self.exp_lowerbound = math.exp(probe_radius_lowerbound + self.offset)

    def transform(self, x: torch.Tensor):
        clamp_x = torch.clamp_(x, min=self.lowerbound, max=self.upperbound)
        trans_x = torch.exp(clamp_x + self.offset)
        return trans_x

    def inv_transform(self, x: torch.Tensor):
        clamp_x = torch.clamp(x, min=self.exp_lowerbound, max=self.exp_upperbound)
        inv_x = torch.log(clamp_x) - self.offset
        return inv_x


class BipartScaleTransformer(LabelTransformer):
    """
    Normalize label value to range -1 to 1
    """

    def __init__(self, probe_radius_upperbound: float, probe_radius_lowerbound: float):
        self.upperbound = probe_radius_upperbound
        self.lowerbound = probe_radius_lowerbound

    def transform(self, x: torch.Tensor):
        clamp_x = torch.clamp_(x, min=self.lowerbound, max=self.upperbound)
        trans_x = torch.where(
            clamp_x > 0, clamp_x / abs(self.upperbound), clamp_x / abs(self.lowerbound)
        )
        return trans_x

    def inv_transform(self, x: torch.Tensor):
        inv_x = torch.where(x > 0, x * abs(self.upperbound), x * abs(self.lowerbound))
        return inv_x


class TranslationLabelTransformer(LabelTransformer):
    """
    Normalize label value to range -1 to 1
    """

    def __init__(
        self,
        probe_radius_upperbound: float,
        probe_radius_lowerbound: float,
        do_truncate: bool = True,
    ):
        self.upperbound = probe_radius_upperbound
        self.lowerbound = probe_radius_lowerbound
        self.do_truncate = do_truncate

    def transform(self, x: torch.Tensor):
        if self.do_truncate:
            x = torch.clamp_(x, min=self.lowerbound, max=self.upperbound)
        trans_x = x - self.lowerbound
        return trans_x

    def inv_transform(self, x: torch.Tensor):
        inv_x = x + self.lowerbound
        return inv_x


class RefinedMlsesDataset(Dataset):
    """
    Load all data in dat files into the memory. Comsume too much memory.
    """

    def __init__(self, path, input_dim=200):
        self.path = path
        self.input_dim = input_dim
        self.dat_files = glob.glob(f"{path}/*/*.dat")

        print("Number of data files loading:", len(self.dat_files))

        self.labels = []
        self.features = []
        self.features_length = []
        for file in tqdm(self.dat_files):
            with open(file, "r") as f:
                for line in f:
                    row = line.split()

                    # for each sample, we have at least four elements
                    if len(row) > 3:
                        self.labels.append(float(row[0]))
                        feature = [float(i) for i in row[1:]]
                        self.features.append(feature)
                        self.features_length.append(len(feature))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        feature = self.features[index]
        feature_length = self.features_length[index]

        feature_array = np.array(feature, dtype=np.float32)
        feature_array = np.pad(
            feature_array, (0, self.input_dim - feature_length), "constant"
        )

        label_tensor = torch.LongTensor(
            [
                label,
            ]
        )
        feature_tensor = torch.from_numpy(feature_array)

        return feature_tensor, label_tensor


class RefinedMlsesMapDataset(Dataset):
    """
    Record entries offset inside the data file.
    """

    def __init__(
        self,
        dat_files,
        input_dim=200,
        label_transformer: LabelTransformer = None,
        label_only=False,
        dummy_columns=0,
    ):
        self.input_dim = input_dim
        self.dat_files = dat_files
        self.label_transformer = label_transformer
        self.label_only = label_only
        self.index_byte = 8
        self.dummy_columns = dummy_columns

        print("Number of data files loading:", len(self.dat_files))

        self.files_length = []
        for file in tqdm(self.dat_files):
            # check if offset index file has built
            index_file = self.__get_index_file(file)
            if os.path.isfile(index_file):
                with open(index_file, "rb") as f:
                    head_line = f.read(self.index_byte)
                    total_line = int.from_bytes(head_line, "big")
                    self.files_length.append(total_line)
            else:
                # build offset index file
                with open(file, "r") as f:
                    offset = 0
                    file_offset = []
                    for line in f:
                        file_offset.append(offset)
                        offset += len(line)

                total_line = len(file_offset)
                self.files_length.append(total_line)

                with open(index_file, "wb") as f:
                    f.write(total_line.to_bytes(self.index_byte, "big"))
                    for line_offset in file_offset:
                        f.write(line_offset.to_bytes(self.index_byte, "big"))

        # build offset indexing
        self.files_cumsum = np.cumsum(self.files_length)
        print("Total line:", self.files_cumsum[-1])

    def __len__(self):
        return self.files_cumsum[-1].item()

    def __get_index_file(self, file):
        return f"{file}.offset_index"

    def get_file_name_by_index(self, index):
        file_index = np.searchsorted(self.files_cumsum, index, side="right")
        file_name = self.dat_files[file_index]
        infile_offset = index - (
            0 if file_index == 0 else self.files_cumsum[file_index - 1].item()
        )

        # get index file and read offset
        index_file = self.__get_index_file(file_name)
        with open(index_file, "rb") as f:
            f.seek(self.index_byte * (infile_offset + 1))  # +1 for skip head line
            line_offset = f.read(self.index_byte)
        file_offset = int.from_bytes(line_offset, "big")
        return file_name, file_offset

    def __getitem__(self, index):
        # read line by offset
        file_name, file_offset = self.get_file_name_by_index(index)
        with open(file_name, "r") as f:
            f.seek(file_offset)
            line = f.readline()

        # process line data
        if not self.label_only:
            row = line.split()
            label = float(row[0])
            # feature_num = math.log(len(row) - 1)
            feature = [
                float(i) for i in row[self.dummy_columns + 1 :]
            ]  # here we skip dummy columns

            # TODO: here is an extra colum for some training data files. Need to be fixed in the future.
            if len(feature) % 4 != 0:
                feature = feature[1:]
            assert (
                len(feature) % 4 == 0
            ), f"Feature len ({len(feature)}) cannot divided by 4."

            # truncate feature
            feature = feature[: self.input_dim]

            feature_array = np.array(feature, dtype=np.float32, copy=False)
            feature_array = np.pad(
                feature_array, (0, self.input_dim - len(feature)), "constant"
            )

            label_tensor = torch.FloatTensor(
                [
                    label,
                ]
            )
            feature_tensor = torch.from_numpy(feature_array)
        else:
            row = line.split()
            label = float(row[0])
            label_tensor = torch.FloatTensor(
                [
                    label,
                ]
            )
            feature_num_tensor = torch.LongTensor(
                [
                    len(row) - 1,
                ]
            )

        # do label transform if needed
        if self.label_transformer is not None:
            label_tensor = self.label_transformer.transform(label_tensor)

        if self.label_only:
            return feature_num_tensor, label_tensor
        else:
            return feature_tensor, label_tensor


class Subset(Dataset):
    def __init__(self, dataset: RefinedMlsesMapDataset, indices_path, split) -> None:
        self.dataset = dataset
        self.indices_file = f"{indices_path}/{split}.indices"

        self.length = 0
        with open(self.indices_file, "rb") as f:
            head_line = f.read(self.dataset.index_byte)
            total_line = int.from_bytes(head_line, "big")
            self.length = total_line

        print(f"{split} size:", self.length)

    def __getitem__(self, idx):
        with open(self.indices_file, "rb") as f:
            f.seek(self.dataset.index_byte * (idx + 1))  # +1 for skip head line
            indice_bytes = f.read(self.dataset.index_byte)
        indice = int.from_bytes(indice_bytes, "big")
        return self.dataset[indice]

    def __len__(self):
        return self.length


class RefinedMlsesMemoryMapDataset(Dataset):
    """
    Dataset from Numpy memmap
    """

    def __init__(
        self,
        dat_file,
        input_dim,
        sample_num,
        sample_dim,
        label_transformer: LabelTransformer = None,
        label_only=False,
    ):
        self.input_dim = input_dim
        self.sample_num = sample_num
        self.sample_dim = sample_dim
        self.dat_file = dat_file
        self.label_transformer = label_transformer
        self.label_only = label_only

        self.np_map = np.memmap(
            self.dat_file, dtype=np.float32, mode="r+", shape=(sample_num, sample_dim)
        )
        print(
            "Sample Num:",
            sample_num,
            "Sample Dim:",
            sample_dim,
            "Feature Size:",
            input_dim,
        )

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        # read sample from memory map
        sample_row = self.np_map[index]
        feature_array = sample_row[: self.input_dim]
        label_array = sample_row[self.input_dim :]

        # process line data
        if not self.label_only:
            label_tensor = torch.from_numpy(label_array)
            feature_tensor = torch.from_numpy(feature_array)
        else:
            label_tensor = torch.from_numpy(label_array)
            feature_tensor = None

        # do label transform if needed
        if self.label_transformer is not None:
            label_tensor = self.label_transformer.transform(label_tensor)

        if self.label_only:
            return label_tensor
        else:
            return feature_tensor, label_tensor


class MultitaskRefinedMlsesMapDataset(RefinedMlsesMapDataset):
    def __init__(
        self,
        dat_files,
        input_dim=200,
        lowerbound=-1.0,
        label_transformer: LabelTransformer = None,
        label_only=False,
    ):
        super().__init__(dat_files, input_dim, label_transformer, label_only)
        self.input_dim = input_dim
        self.lowerbound = lowerbound

    def __getitem__(self, index):
        # read line by offset
        file_name, file_offset = self.get_file_name_by_index(index)
        with open(file_name, "r") as f:
            f.seek(file_offset)
            line = f.readline()

        # process line data
        if not self.label_only:
            row = line.split()
            label = float(row[0])
            feature = [float(i) for i in row[1:]]

            # add the length as the first feature
            feature = [len(feature)] + feature

            # truncate feature
            feature = feature[: self.input_dim]

            feature_array = np.array(feature, dtype=np.float32)
            feature_array = np.pad(
                feature_array, (0, self.input_dim - len(feature)), "constant"
            )

            reg_label_tensor = torch.FloatTensor(
                [
                    label,
                ]
            )
            cls_label_tensor = torch.LongTensor(
                [
                    1 if label < self.lowerbound else -1,
                ]
            )  # 1 for trivial samples, -1 for nontrivial samples
            feature_tensor = torch.from_numpy(feature_array)
        else:
            row = line.split()
            label = float(row[0])
            reg_label_tensor = torch.FloatTensor(
                [
                    label,
                ]
            )
            feature_tensor = None

        # do label transform if needed
        if self.label_transformer is not None:
            reg_label_tensor = self.label_transformer.transform(reg_label_tensor)

        if self.label_only:
            return reg_label_tensor
        else:
            return feature_tensor, reg_label_tensor, cls_label_tensor


class MultitaskRefinedMlsesMemoryMapDataset(RefinedMlsesMemoryMapDataset):
    def __init__(
        self,
        dat_file,
        input_dim,
        sample_num,
        sample_dim,
        lowerbound=-1.0,
        label_transformer: LabelTransformer = None,
        label_only=False,
    ):
        super().__init__(dat_file, input_dim, sample_num, sample_dim, None, label_only)
        self.input_dim = input_dim
        self.multitask_label_transformer = label_transformer
        self.sample_num = sample_num
        self.sample_dim = sample_dim
        self.lowerbound = lowerbound

    def __getitem__(self, index):
        if self.label_only:
            return super().__getitem__(index)
        else:
            feature_tensor, label_tensor = super().__getitem__(index)

            if self.multitask_label_transformer is not None:
                reg_label_tensor = self.multitask_label_transformer.transform(
                    label_tensor
                )
            else:
                reg_label_tensor = label_tensor

            cls_label_tensor = torch.LongTensor(
                [
                    1 if label_tensor.item() < self.lowerbound else -1,
                ]
            )  # 1 for trivial samples, -1 for nontrivial samples
            return feature_tensor, reg_label_tensor, cls_label_tensor


class ImageMlsesDataset(Dataset):
    def __init__(
        self,
        patches_index_file,
        patch_size: int,
        label_transformer: LabelTransformer = None,
    ) -> None:
        super().__init__()
        with open(patches_index_file, "r") as f:
            patches_index = json.load(f)

        self.patches_index = np.array(
            [[p["image_path"], p["x0"], p["y0"]] for p in patches_index]
        )
        self.patch_size = patch_size
        self.label_transformer = label_transformer

    def __len__(self):
        return len(self.patches_index)

    def __getitem__(self, index):
        patch = self.patches_index[index]
        image_npz = np.load(patch[0])
        x0 = int(patch[1])
        y0 = int(patch[2])

        pad_image = pad_array(image_npz["image"], x0, y0, self.patch_size)
        pad_label = pad_array(image_npz["label"], x0, y0, self.patch_size)
        pad_mask = pad_array(image_npz["mask"], x0, y0, self.patch_size)

        # prepare tensors and transform them from NHWC to NCHW
        image_tensor = torch.from_numpy(pad_image).permute(2, 0, 1)
        label_tensor = torch.from_numpy(pad_label).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(pad_mask).permute(2, 0, 1)

        # do label transform if needed
        if self.label_transformer is not None:
            label_tensor = self.label_transformer.transform(label_tensor)
        return image_tensor, label_tensor, mask_tensor


@dataclass
class SparseImageMlsesDatasetConfig:
    # filter patch by slice axis
    patch_slice_axis: Literal["x", "y", "z", "all"] = "z"
    random_rotate: bool = False
    random_rotate_interval: int = 10
    given_rotate_angle: Union[List[int], None] = None
    given_rotate_axis: Union[Literal["x", "y", "z"], None] = None


class SparseImageMlsesDataset(Dataset):
    def __init__(
        self,
        patches_index_file: str,
        patch_size: int,
        split: Literal["train", "eval", "test"],
        config: SparseImageMlsesDatasetConfig,
        label_transformer: LabelTransformer = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.patch_size = patch_size
        self.label_transformer = label_transformer
        self.patch_slice_axis = config.patch_slice_axis

        with open(patches_index_file, "r") as f:
            patches_index = json.load(f)
        if self.patch_slice_axis == "all":
            self.patches_index = np.array(
                [
                    [p["voxel_path"], p["uzi"], p["x0"], p["y0"], "x"]
                    for p in patches_index["x"][split]
                ]
                + [
                    [p["voxel_path"], p["uzi"], p["x0"], p["y0"], "y"]
                    for p in patches_index["y"][split]
                ]
                + [
                    [p["voxel_path"], p["uzi"], p["x0"], p["y0"], "z"]
                    for p in patches_index["z"][split]
                ]
            )

        else:
            self.patches_index = np.array(
                [
                    [p["voxel_path"], p["uzi"], p["x0"], p["y0"], self.patch_slice_axis]
                    for p in patches_index[self.patch_slice_axis][split]
                ]
            )

    def __len__(self):
        return len(self.patches_index)

    @staticmethod
    def grid2image_loc(abs_grid_feats, span_indices: Tuple[int, int]):
        abs_grid_xs = abs_grid_feats[:, span_indices[0]]
        abs_grid_ys = abs_grid_feats[:, span_indices[1]]
        w, h = np.unique(abs_grid_xs).shape[0], np.unique(abs_grid_ys).shape[0]

        # map abs index into the image
        img_grid_xs = np.zeros_like(abs_grid_xs, dtype=np.int32)
        img_grid_ys = np.zeros_like(abs_grid_ys, dtype=np.int32)
        for i, u_x in enumerate(np.unique(abs_grid_xs)):
            mask = abs_grid_xs == u_x
            img_grid_xs[mask] = i

        for i, u_y in enumerate(np.unique(abs_grid_ys), 1):
            mask = abs_grid_ys == u_y
            img_grid_ys[mask] = -i

        return w, h, img_grid_xs, img_grid_ys

    @staticmethod
    def make_image(
        abs_grid_feats,
        rel_grid_feats,
        grid_labels,
        span_indices: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate input image, level-set image, and a mask based on the absolute location of grid point and relative grid features

        :param abs_grid_feats: absolute location (x, y, z) of grid point [N, 3]
        :param rel_grid_feats: accoding relative grid features of grid point [N, feat_dim]
        :param grid_labels: level-set value at each grid point [N,]
        :return: a tuple of input image, level-set image, and a mask
        """
        w, h, img_grid_xs, img_grid_ys = SparseImageMlsesDataset.grid2image_loc(
            abs_grid_feats, span_indices
        )

        image = np.zeros((h, w, rel_grid_feats.shape[-1]), dtype=np.float32)
        label = np.zeros((h, w, 1), dtype=np.float32)
        mask = np.zeros((h, w, 1), dtype=np.float32)
        image[img_grid_ys.tolist(), img_grid_xs.tolist()] = rel_grid_feats
        label[img_grid_ys.tolist(), img_grid_xs.tolist(), 0] = grid_labels
        mask[img_grid_ys.tolist(), img_grid_xs.tolist(), 0] = 1
        return image, label, mask

    @staticmethod
    def __map_axis2int(axis: Literal["x", "y", "z"]) -> int:
        if axis == "x":
            return 0
        elif axis == "y":
            return 1
        elif axis == "z":
            return 2
        else:
            raise ValueError(f"Unknown axis {axis}")

    @staticmethod
    def __map_axis2span(axis: Literal["x", "y", "z"]) -> Tuple[int, int]:
        return tuple(
            i for i in range(3) if i != SparseImageMlsesDataset.__map_axis2int(axis)
        )

    @staticmethod
    def __rotation_matrix(axis: Literal["x", "y", "z"], theta):
        """Rotation matrix along the x-axis."""
        if axis == "x":
            return np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)],
                ],
                dtype=np.float32,
            )

        elif axis == "y":
            """Rotation matrix along the y-axis."""
            return np.array(
                [
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)],
                ],
                dtype=np.float32,
            )
        elif axis == "z":
            """Rotation matrix along the z-axis."""
            return np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unknown axis {axis}")

    @staticmethod
    def __build_input_image(
        grid_xyz: np.ndarray,
        image_npz,
        slice_index: int,
        slice_axis: Literal["x", "y", "z"] = "z",
        feat_num: int = 96,
    ):
        """create input image from sparse dataset

        :param image_npz: sparse data
        :param slice_index: slice along axis at index
        :param slice_axis: slice along axis
        :param feat_num: grid features, defaults to 96
        :return: constructed image, label, and mask
        """
        grid_label = image_npz["grid_label"]
        grid_num = grid_xyz.shape[0]

        sparse_matrix = csr_matrix(
            (
                image_npz["grid_rel_feats_vals"],
                image_npz["grid_rel_feats_cols"],
                image_npz["grid_rel_feats_rows"],
            ),
            shape=(grid_num, feat_num),
        )

        slice_axis_index = SparseImageMlsesDataset.__map_axis2int(slice_axis)
        span_indices = SparseImageMlsesDataset.__map_axis2span(slice_axis)
        z0 = np.unique(grid_xyz[:, slice_axis_index])[slice_index]
        grid_mask = grid_xyz[:, slice_axis_index] == z0

        image, label, mask = SparseImageMlsesDataset.make_image(
            grid_xyz[grid_mask],
            sparse_matrix[grid_mask].toarray(),
            grid_label[grid_mask],
            span_indices,
        )
        return image, label, mask

    def __transform_grid(
        self, grid_xyz: np.ndarray, along_axis: Literal["x", "y", "z"] = "z"
    ):
        # Randomly sample a single value
        if self.config.given_rotate_angle is not None:
            random_angle = self.config.given_rotate_angle
        elif self.config.random_rotate:
            angles = np.arange(
                0, 361, self.config.random_rotate_interval
            )  # 361 to include 360
            random_angle = np.random.choice(angles).item()
        else:
            random_angle = 0

        if random_angle != 0:
            angle = np.radians(random_angle)
            R = SparseImageMlsesDataset.__rotation_matrix(along_axis, angle)
            grid_xyz_rotated = grid_xyz @ R.T
            grid_xyz = grid_xyz_rotated
        return grid_xyz

    # def __choose_slice_index(self, image_npz, along_axis: Literal["x", "y", "z"] = "z"):
    #     grid_xyz = image_npz["grid_xyz"]
    #     slice_axis_index = SparseImageMlsesDataset.__map_axis2int(along_axis)
    #     slice_indices = np.unique(grid_xyz[:, slice_axis_index])
    #     return np.random.choice(slice_indices).item()

    # def __sample_anchor(self, image, mask):
    #     height = image.shape[0]
    #     width = image.shape[1]
    #     success = False
    #     while not success:
    #         x0 = np.random.choice(np.arange(0, height + 1, self.patch_size)).item()
    #         y0 = np.random.choice(np.arange(0, width + 1, self.patch_size)).item()
    #         if np.sum(mask[x0 : x0 + self.patch_size, y0 : y0 + self.patch_size]) > 0:
    #             success = True

    #     return x0, y0

    def __rotate_and_tensorize_image(
        self, image, label, mask, along_axis: Literal["x", "y", "z"] = "z"
    ):
        # Randomly sample a single value
        if self.config.given_rotate_angle is not None:
            random_angle = self.config.given_rotate_angle
        elif self.config.random_rotate:
            angles = np.arange(
                0, 361, self.config.random_rotate_interval
            )  # 361 to include 360
            random_angle = np.random.choice(angles).item()
        else:
            random_angle = 0

        if random_angle != 0:
            # angle_rad = np.radians(random_angle).item()
            # R = SparseImageMlsesDataset.__rotation_matrix(along_axis, angle)

            # Convert angle to radians
            angle_rad = torch.tensor(random_angle * torch.pi / 180.0)
            cos_theta = torch.cos(angle_rad)
            sin_theta = torch.sin(angle_rad)
            rotation_matrix = torch.tensor(
                [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0]],
                dtype=torch.float,
            )
            rotation_matrix = rotation_matrix.unsqueeze_(0)

            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze_(0)
            label = torch.from_numpy(label).permute(2, 0, 1).unsqueeze_(0)
            mask = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze_(0)

            # Get input dimensions
            N, C, H, W = image.shape

            # Generate the grid
            grid = F.affine_grid(rotation_matrix, size=(N, C, H, W), align_corners=True)

            # Apply the grid to the input tensor
            rotated_image = F.grid_sample(
                image, grid, mode="nearest", padding_mode="zeros", align_corners=True
            )
            rotated_label = F.grid_sample(
                label, grid, mode="nearest", padding_mode="zeros", align_corners=True
            )
            rotated_mask = F.grid_sample(
                mask, grid, mode="nearest", padding_mode="zeros", align_corners=True
            )

            image_tensor = rotated_image.squeeze_(0)
            label_tensor = rotated_label.squeeze_(0)
            mask_tensor = rotated_mask.squeeze_(0)
        else:
            # prepare tensors and transform them from NHWC to NCHW
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)
            label_tensor = torch.from_numpy(label).permute(2, 0, 1)
            mask_tensor = torch.from_numpy(mask).permute(2, 0, 1)

        return image_tensor, label_tensor, mask_tensor

    def __getitem__(self, index):
        patch = self.patches_index[index]
        patch_path = patch[0]

        # get image and label from sparse data
        image_npz = np.load(patch_path)
        uzi = int(patch[1])
        x0 = int(patch[2])
        y0 = int(patch[3])
        slice_axis = patch[4]

        image, label, mask = self.__build_input_image(
            image_npz["grid_xyz"], image_npz, slice_index=uzi, slice_axis=slice_axis
        )

        pad_image = pad_array(image, x0, y0, self.patch_size)
        pad_label = pad_array(label, x0, y0, self.patch_size)
        pad_mask = pad_array(mask, x0, y0, self.patch_size)

        # do rotation if needed
        image_tensor, label_tensor, mask_tensor = self.__rotate_and_tensorize_image(
            pad_image, pad_label, pad_mask, slice_axis
        )

        # do label transform if needed
        if self.label_transformer is not None:
            label_tensor = self.label_transformer.transform(label_tensor)
        return image_tensor, label_tensor, mask_tensor


class Image3dMlsesDataset(Dataset):
    def __init__(
        self,
        patches_index_file,
        patch_size: int,
        label_transformer: LabelTransformer = None,
    ) -> None:
        super().__init__()
        with open(patches_index_file, "r") as f:
            patches_index = json.load(f)

        self.patches_index = np.array(
            [[p["patch_file_path"], p["x0"], p["y0"], p["z0"]] for p in patches_index]
        )
        self.patch_size = patch_size
        self.label_transformer = label_transformer

    def __len__(self):
        return len(self.patches_index)

    def __getitem__(self, index):
        patch = self.patches_index[index]
        with gzip.GzipFile(patch[0], "rb") as f:
            image_npz = pickle.load(f)

        pad_image = pad_3d_array(image_npz["image"], 0, 0, 0, self.patch_size)
        pad_label = pad_3d_array(image_npz["label"], 0, 0, 0, self.patch_size)
        pad_mask = pad_3d_array(image_npz["mask"], 0, 0, 0, self.patch_size)

        # prepare tensors and transform them from DHWC to CDHW
        dim_inds = (3, 0, 1, 2)
        image_tensor = torch.from_numpy(pad_image).permute(*dim_inds)
        label_tensor = torch.from_numpy(pad_label).permute(*dim_inds)
        mask_tensor = torch.from_numpy(pad_mask).permute(*dim_inds)

        # do label transform if needed
        if self.label_transformer is not None:
            label_tensor = self.label_transformer.transform(label_tensor)
        return image_tensor, label_tensor, mask_tensor


class RotateImage3dMlsesDataset(Image3dMlsesDataset):
    def __init__(
        self,
        patches_index_file,
        patch_size: int,
        config: SparseImageMlsesDatasetConfig,
        label_transformer: LabelTransformer = None,
    ):
        super().__init__(patches_index_file, patch_size, label_transformer)
        self.config = config

    def __rotate_and_tensorize_image(self, image, label, mask):
        # Randomly sample a single value
        if self.config.given_rotate_angle is not None:
            random_angle = self.config.given_rotate_angle
            random_axis = self.config.given_rotate_axis
        elif self.config.random_rotate:
            angles = np.arange(
                0, 361, self.config.random_rotate_interval
            )  # 361 to include 360
            random_angle = np.random.choice(angles).item()
            random_axis = np.random.choice(["x", "y", "z"])

        else:
            random_angle = 0
            random_axis = "x"

        if random_angle != 0:
            # Convert angle to radians
            angle_rad = torch.tensor(random_angle * torch.pi / 180.0)

            # Define rotation matrices for each axis
            axis = random_axis
            if axis == "x":  # Rotation around the X-axis
                rotation_matrix = torch.tensor(
                    [
                        [1, 0, 0],
                        [0, torch.cos(angle_rad), -torch.sin(angle_rad)],
                        [0, torch.sin(angle_rad), torch.cos(angle_rad)],
                    ]
                )
            elif axis == "y":  # Rotation around the Y-axis
                rotation_matrix = torch.tensor(
                    [
                        [torch.cos(angle_rad), 0, torch.sin(angle_rad)],
                        [0, 1, 0],
                        [-torch.sin(angle_rad), 0, torch.cos(angle_rad)],
                    ]
                )
            elif axis == "z":  # Rotation around the Z-axis
                rotation_matrix = torch.tensor(
                    [
                        [torch.cos(angle_rad), -torch.sin(angle_rad), 0],
                        [torch.sin(angle_rad), torch.cos(angle_rad), 0],
                        [0, 0, 1],
                    ]
                )
            else:
                raise ValueError("Axis must be one of 'x', 'y', or 'z'.")

            # Add the translation column (no translation)
            translation = torch.tensor([0, 0, 0]).unsqueeze(1)  # Shape (3, 1)
            affine_matrix = torch.cat(
                [rotation_matrix, translation], dim=1
            )  # Shape (3, 4)
            affine_matrix = affine_matrix.unsqueeze(
                0
            )  # Add batch dimension -> (1, 3, 4)

            # Remove the homogeneous coordinate (for affine_grid compatibility)
            rotation_matrix = affine_matrix

            dim_inds = (3, 0, 1, 2)
            image = torch.from_numpy(image).permute(*dim_inds).unsqueeze_(0)
            label = torch.from_numpy(label).permute(*dim_inds).unsqueeze_(0)
            mask = torch.from_numpy(mask).permute(*dim_inds).unsqueeze_(0)

            # Get input dimensions
            N, C, D, H, W = image.shape

            # Generate the grid
            grid = F.affine_grid(
                rotation_matrix, size=(N, C, D, H, W), align_corners=True
            )

            # Apply the grid to the input tensor
            rotated_image = F.grid_sample(
                image, grid, mode="nearest", padding_mode="zeros", align_corners=True
            )
            rotated_label = F.grid_sample(
                label, grid, mode="nearest", padding_mode="zeros", align_corners=True
            )
            rotated_mask = F.grid_sample(
                mask, grid, mode="nearest", padding_mode="zeros", align_corners=True
            )

            image_tensor = rotated_image.squeeze_(0)
            label_tensor = rotated_label.squeeze_(0)
            mask_tensor = rotated_mask.squeeze_(0)
        else:
            # prepare tensors and transform them from DHWC to CDHW
            dim_inds = (3, 0, 1, 2)
            image_tensor = torch.from_numpy(image).permute(*dim_inds)
            label_tensor = torch.from_numpy(label).permute(*dim_inds)
            mask_tensor = torch.from_numpy(mask).permute(*dim_inds)

        return image_tensor, label_tensor, mask_tensor

    def __getitem__(self, index):
        patch = self.patches_index[index]
        with gzip.GzipFile(patch[0], "rb") as f:
            image_npz = pickle.load(f)

        pad_image = pad_3d_array(image_npz["image"], 0, 0, 0, self.patch_size)
        pad_label = pad_3d_array(image_npz["label"], 0, 0, 0, self.patch_size)
        pad_mask = pad_3d_array(image_npz["mask"], 0, 0, 0, self.patch_size)

        # do rotation if needed
        image_tensor, label_tensor, mask_tensor = self.__rotate_and_tensorize_image(
            pad_image, pad_label, pad_mask
        )

        # do label transform if needed
        if self.label_transformer is not None:
            label_tensor = self.label_transformer.transform(label_tensor)
        return image_tensor, label_tensor, mask_tensor


class Image3dEnergyDataset(Dataset):
    def __init__(
        self,
        patch_pkls: list[str],
        patch_size: int,
        epb_mean: float,
        epb_std: float,
    ) -> None:
        super().__init__()
        self.patch_pkls = patch_pkls
        self.patch_size = patch_size
        self.epb_mean = epb_mean
        self.epb_std = epb_std

    def __len__(self):
        return len(self.patch_pkls)

    def process_one_patch(self, patch_path):
        with gzip.open(patch_path, "rb") as f:
            patch = pickle.load(f)

        if "bench_full_0.35_abs" in patch_path:
            grid_space = 0.35
        elif "bench_full_0.55_abs" in patch_path:
            grid_space = 0.55
        else:
            raise ValueError("Unknown grid space")

        feat_dict = patch["feat_info"]
        pad_level_set = pad_3d_array(feat_dict["level_set"], 0, 0, 0, self.patch_size)
        pad_atom_charge = pad_3d_array(
            feat_dict["atom_charge"], 0, 0, 0, self.patch_size
        )
        pad_atom_type = pad_3d_array(feat_dict["atom_type"], 0, 0, 0, self.patch_size)
        pad_atom_mask = pad_3d_array(feat_dict["atom_mask"], 0, 0, 0, self.patch_size)
        pad_potential = pad_3d_array(
            feat_dict["atom_potential"], 0, 0, 0, self.patch_size
        )

        # prepare tensors and transform them from DHWC to CDHW
        dim_inds = (3, 0, 1, 2)
        level_set_tensor = torch.from_numpy(pad_level_set).permute(*dim_inds)
        atom_charge_tensor = torch.from_numpy(pad_atom_charge).permute(*dim_inds)
        atom_type_tensor = torch.from_numpy(pad_atom_type).permute(*dim_inds)
        atom_mask_tensor = torch.from_numpy(pad_atom_mask).permute(*dim_inds)
        potential_tensor = torch.from_numpy(pad_potential).permute(*dim_inds)
        return (
            level_set_tensor,
            atom_charge_tensor,
            atom_type_tensor,
            atom_mask_tensor,
            potential_tensor,
            grid_space,
        )

    def __getitem__(self, index):
        patch_file = self.patch_pkls[index]
        outputs = self.process_one_patch(patch_file)
        return outputs
