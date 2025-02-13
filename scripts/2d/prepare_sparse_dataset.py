import tyro
import json
import os
import glob
import math
import numpy as np
import torch

from torch.utils.data import random_split
from typing import Literal, Tuple
from tqdm import tqdm
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from zipfile import BadZipFile
from scipy.sparse import csr_matrix


def read_dat(file, max_length: int = 100, skip_column: bool = False):
    """
    Read data from given path
    """
    data = []
    vals = []
    with open(f"{file}", "r") as f:
        reader = f.readlines()
        for line in reader:
            row = line.strip()
            get_col = row.split()
            row_vals = [float(i) for i in get_col]
            vals.append(row_vals[0])

            row_feats = row_vals[1:]
            if skip_column and len(row_feats) % 4 != 0:
                row_feats = row_feats[1:]

            tmp_array = np.zeros((max_length,), dtype=np.float32)
            tmp_row_feats = row_feats[:max_length]
            tmp_array[: len(tmp_row_feats)] = tmp_row_feats
            data.append(tmp_array)

    feat_matrix = np.stack(data, axis=0)
    return np.array(vals), feat_matrix


def make_image(
    abs_grid_feats, rel_grid_feats, grid_labels
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate input image, level-set image, and a mask based on the absolute location of grid point and relative grid features

    :param abs_grid_feats: absolute location (x, y, z) of grid point [N, 3]
    :param rel_grid_feats: accoding relative grid features of grid point [N, feat_dim]
    :param grid_labels: level-set value at each grid point [N,]
    :return: a tuple of input image, level-set image, and a mask
    """
    abs_grid_xs = abs_grid_feats[:, 0]
    abs_grid_ys = abs_grid_feats[:, 1]
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

    image = np.zeros((h, w, rel_grid_feats.shape[-1]), dtype=np.float32)
    label = np.zeros((h, w, 1), dtype=np.float32)
    mask = np.zeros((h, w, 1), dtype=np.float32)
    image[img_grid_ys.tolist(), img_grid_xs.tolist()] = rel_grid_feats
    label[img_grid_ys.tolist(), img_grid_xs.tolist(), 0] = grid_labels
    mask[img_grid_ys.tolist(), img_grid_xs.tolist(), 0] = 1
    return image, label, mask


def make_image_mask(abs_grid_feats, span_indices: Tuple[int, int]) -> np.ndarray:
    """Generate input image, level-set image, and a mask based on the absolute location of grid point and relative grid features

    :param abs_grid_feats: absolute location (x, y, z) of grid point [N, 3]
    :param rel_grid_feats: accoding relative grid features of grid point [N, feat_dim]
    :param grid_labels: level-set value at each grid point [N,]
    :return: a tuple of input image, level-set image, and a mask
    """
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

    mask = np.zeros((h, w, 1), dtype=np.float32)
    mask[img_grid_ys.tolist(), img_grid_xs.tolist(), 0] = 1
    return mask


def slice_dat_to_sparse(abs_dat_feats, abs_dat_labels, rel_dat_feats):
    grid_xyz = abs_dat_feats[:, :3]
    grid_label = abs_dat_labels

    grid_feat_sparse_matrix = csr_matrix(rel_dat_feats)
    grid_rel_feats = grid_feat_sparse_matrix.data
    grid_rel_feats_rows = grid_feat_sparse_matrix.indptr
    grid_rel_feats_cols = grid_feat_sparse_matrix.indices

    sparse_data = {
        "grid_xyz": grid_xyz,
        "grid_label": grid_label,
        "grid_rel_feats_rows": grid_rel_feats_rows,
        "grid_rel_feats_cols": grid_rel_feats_cols,
        "grid_rel_feats_vals": grid_rel_feats,
    }
    return sparse_data


def slice_sparse_to_patches_index(
    image_npz_path, patch_size: int, along_axis: Literal["x", "y", "z"] = "z"
):
    image_npz = np.load(image_npz_path)
    grid_xyz = image_npz["grid_xyz"]

    slice_axis = {
        "x": 0,
        "y": 1,
        "z": 2,
    }.get(along_axis)
    span_indices = tuple([i for i in range(3) if i != slice_axis])
    abs_grid_slice_axis = grid_xyz[:, slice_axis]
    patch_lefttop_corners = []
    for uzi, u_z in enumerate(np.unique(abs_grid_slice_axis)):
        grid_mask = abs_grid_slice_axis == u_z

        abs_grid_feats = grid_xyz[grid_mask]
        mask = make_image_mask(abs_grid_feats, span_indices)
        image_height = mask.shape[0]
        image_width = mask.shape[1]

        # make extended img so that it contains integer number of patches
        npatches_vertical = math.ceil(image_height / patch_size)
        npatches_horizontal = math.ceil(image_width / patch_size)
        for i in range(npatches_vertical):
            for j in range(npatches_horizontal):
                x0 = i * patch_size
                y0 = j * patch_size

                # skip thoes empty patches
                if np.sum(mask[x0 : x0 + patch_size, y0 : y0 + patch_size]) < 1:
                    continue

                patch_lefttop_corners.append(
                    {"voxel_path": image_npz_path, "uzi": uzi, "x0": x0, "y0": y0}
                )
    return patch_lefttop_corners


def parse_dat_path(dat_path):
    abs_head, dat_name = os.path.split(dat_path)
    abs_head, case = os.path.split(abs_head)
    abs_head, category = os.path.split(abs_head)
    return category, case, dat_name


def per_sample_read_dat_then_process(
    abs_dat_path, rel_dataset, image_dataset_path, processing_log
):
    category, case, dat_name = parse_dat_path(abs_dat_path)
    mole_name = dat_name[:6]
    rel_dat_path = glob.glob(
        os.path.join(rel_dataset, category.replace("_abs", ""), case)
        + f"/{mole_name}*.dat"
    )[
        0
    ]  # remove `_abs`

    abs_dat_labels, abs_dat_feats = read_dat(abs_dat_path, max_length=3)
    _, rel_dat_feats = read_dat(rel_dat_path, max_length=96, skip_column=True)

    try:
        assert len(abs_dat_feats) == len(rel_dat_feats)
        sparsed_data = slice_dat_to_sparse(abs_dat_feats, abs_dat_labels, rel_dat_feats)
    except:
        print("Runtime Error:", file=processing_log)
        print(f"abs: {abs_dat_path}", file=processing_log)
        print(f"rel: {rel_dat_path}", file=processing_log)
        return

    saved_dir = os.path.join(image_dataset_path, category, case, mole_name)
    os.makedirs(saved_dir, exist_ok=True)
    np.savez(os.path.join(saved_dir, f"{mole_name}"), **sparsed_data)


def read_dat_then_process(
    abs_dat_paths, rel_dataset, image_dataset_path, thread_num: int = 1
):
    if os.path.exists(image_dataset_path):
        raise Exception(f"Path {image_dataset_path} already exists")

    os.makedirs(image_dataset_path)
    processing_log = open(f"{image_dataset_path}/log.txt", "w")

    print(f"Using {thread_num} threads to process data")
    if thread_num == 1:
        for abs_dat_path in tqdm(abs_dat_paths):
            per_sample_read_dat_then_process(
                abs_dat_path, rel_dataset, image_dataset_path, processing_log
            )
    elif thread_num > 1:
        processing_fn = partial(
            per_sample_read_dat_then_process,
            rel_dataset=rel_dataset,
            image_dataset_path=image_dataset_path,
            processing_log=processing_log,
        )
        # Using ThreadPoolExecutor to process items in parallel
        with ThreadPoolExecutor(max_workers=thread_num) as executor:
            # Submit all the tasks and get their futures
            futures = [
                executor.submit(processing_fn, abs_dat_path)
                for abs_dat_path in abs_dat_paths
            ]

            # Iterate over the futures as they complete (as_completed) and wrap it with tqdm for the progress bar
            for future in tqdm(as_completed(futures), total=len(abs_dat_paths)):
                future.result()


def read_sparse_then_process(image_dataset_path, image_npz_paths, patch_size):
    patches_list = {
        "x": [],
        "y": [],
        "z": [],
    }
    with tqdm(total=len(image_npz_paths)) as pbar:
        for image_npz_path in image_npz_paths:
            try:
                for axis in ["x", "y", "z"]:
                    patches = slice_sparse_to_patches_index(
                        image_npz_path, patch_size, axis
                    )
                    patches_list[axis].extend(patches)
            except (ValueError, BadZipFile):
                print(f"Error: {image_npz_path}")

            pbar.set_description(
                f"Patches num: x {len(patches_list['x'])} y {len(patches_list['y'])} z {len(patches_list['z'])}"
            )
            pbar.update(1)

    print(f"Total patches num: {len(patches_list)}")
    with open(f"{image_dataset_path}/patches_index_sz{patch_size}.json", "w") as f:
        json.dump(patches_list, f)


@dataclass
class Args:
    mode: Literal["gen_image", "gen_patch", "split"] = "gen_image"
    abs_dat_paths: str = "datasets/benchmark_data_0.9/*_abs/*/*.dat"
    rel_dataset_path: str = "datasets/benchmark_data_0.5_all"
    image_dataset_path: str = "datasets/benchmark_image_0.9_sparse"
    image_npz_paths: str = "datasets/benchmark_image_0.9_sparse/*/*/*/*.npz"
    patch_json_path: str = "datasets/benchmark_image_0.9_sparse/patches_index_sz64.json"
    patch_size: int = 64
    thread_num: int = 1


if __name__ == "__main__":
    args = tyro.parse(Args)

    if args.mode == "gen_image":
        abs_dat_paths = glob.glob(args.abs_dat_paths)
        rel_dataset_path = args.rel_dataset_path
        image_dataset_path = args.image_dataset_path

        read_dat_then_process(
            abs_dat_paths,
            rel_dataset_path,
            image_dataset_path,
            thread_num=args.thread_num,
        )
    elif args.mode == "gen_patch":
        """
        although sparse dataset is no need to generate patches, we still generate the meta patch here for the convenience of comparisions with previous methods.
        """
        image_npz_paths = glob.glob(args.image_npz_paths)
        read_sparse_then_process(
            args.image_dataset_path, image_npz_paths, args.patch_size
        )
    elif args.mode == "split":
        patch_list = None
        with open(args.patch_json_path, "r") as f:
            patch_list = json.load(f)

        patches_list_with_split = {}
        for axis in ["x", "y", "z"]:
            generator = torch.Generator().manual_seed(2024)
            train_patches, eval_patches, test_patches = random_split(
                patch_list[axis],
                lengths=(0.7, 0.1, 0.2),
            )
            patches_list_with_split[axis] = {
                "train": [patch_list[axis][i] for i in train_patches.indices],
                "eval": [patch_list[axis][i] for i in eval_patches.indices],
                "test": [patch_list[axis][i] for i in test_patches.indices],
            }

            print(f"Total patches num: {len(patch_list[axis])}")
            print(f"Train patches num: {len(patches_list_with_split[axis]['train'])}")
            print(f"Eval patches num: {len(patches_list_with_split[axis]['eval'])}")
            print(f"Test patches num: {len(patches_list_with_split[axis]['test'])}")

        file_name = args.patch_json_path.replace(".json", "")
        with open(f"{file_name}_split.json", "w") as f:
            json.dump(patches_list_with_split, f)
