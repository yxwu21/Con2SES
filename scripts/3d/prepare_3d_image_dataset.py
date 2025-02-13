import tyro
import json
import os
import glob
import gzip
import math
import pickle
import numpy as np

from typing import Literal, Tuple
from tqdm import tqdm
from dataclasses import dataclass
from concurrent.futures import as_completed, ProcessPoolExecutor
from filelock import SoftFileLock
from functools import partial
from zipfile import BadZipFile
from src.preprocess.file_helper import (
    read_dat,
    read_bounding_box_info,
    parse_dat_path,
)


def make_3d_image(
    abs_grid_feats: np.ndarray,
    rel_grid_feats: np.ndarray,
    grid_labels: np.ndarray,
    abs_grid_origin: np.ndarray,
    abs_grid_dims: np.ndarray,
    h_space: float,
    precision: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate input image, level-set image, and a mask based on the absolute location of grid point and relative grid features

    x -> d
    y -> h
    z -> w

    :param abs_grid_feats: absolute location (x, y, z) of grid point [N, 3]
    :param rel_grid_feats: accoding relative grid features of grid point [N, feat_dim]
    :param grid_labels: level-set value at each grid point [N,]
    :return: a tuple of input image, level-set image, and a mask
    """
    d, h, w = abs_grid_dims
    image = np.zeros((d, h, w, rel_grid_feats.shape[-1]), dtype=np.float32)
    label = np.zeros((d, h, w, 1), dtype=np.float32)
    mask = np.zeros((d, h, w, 1), dtype=np.float32)

    grid_locs = np.round(
        (
            np.round(abs_grid_feats, precision)
            - np.round(abs_grid_origin[None, :], precision)
        )
        / h_space
    ).astype(np.int32)

    # the grid point feature should be the same
    unique_grid_locs, unique_grid_locs_counts = np.unique(
        grid_locs, axis=0, return_counts=True
    )
    if len(unique_grid_locs) != len(grid_locs):
        for loc, count in zip(unique_grid_locs, unique_grid_locs_counts):
            if count > 1:
                mask = np.all(grid_locs == loc[None, :], axis=1)
                feat = rel_grid_feats[mask]
                if not np.all(feat == feat[0:1]):
                    raise Exception("Duplicate grid points with different features")

    grid_xs = grid_locs[:, 0]
    grid_ys = grid_locs[:, 1]
    grid_zs = grid_locs[:, 2]
    image[grid_xs.tolist(), grid_ys.tolist(), grid_zs.tolist()] = rel_grid_feats
    label[grid_xs.tolist(), grid_ys.tolist(), grid_zs.tolist(), 0] = grid_labels
    mask[grid_xs.tolist(), grid_ys.tolist(), grid_zs.tolist(), 0] = 1
    return image, label, mask


def convert_dat_to_3d_image(
    abs_dat_feats: np.ndarray,
    abs_dat_labels: np.ndarray,
    rel_dat_feats: np.ndarray,
    abs_grid_origin: np.ndarray,
    abs_grid_dims: np.ndarray,
    h_space: float,
    precision: int = 2,
):
    image, label, mask = make_3d_image(
        abs_dat_feats[:, :3],
        rel_dat_feats,
        abs_dat_labels,
        abs_grid_origin,
        abs_grid_dims,
        h_space,
        precision=precision,
    )
    return {"image": image, "label": label, "mask": mask}


def slice_3d_image_to_patches_index(image_npz_path, patch_size: int):
    with gzip.GzipFile(image_npz_path, "rb") as f:
        image_npz = pickle.load(f)

    image = image_npz["image"]
    label = image_npz["label"]
    mask = image_npz["mask"]

    image_depth = image.shape[0]
    image_height = image.shape[1]
    image_width = image.shape[2]

    # make extended img so that it contains integer number of patches
    npatches_depth = math.ceil(image_depth / patch_size)
    npatches_vertical = math.ceil(image_height / patch_size)
    npatches_horizontal = math.ceil(image_width / patch_size)

    # slice patch files
    folder = os.path.dirname(image_npz_path)
    patch_lefttop_corners = []
    for i in range(npatches_depth):
        for j in range(npatches_vertical):
            for k in range(npatches_horizontal):
                x0 = i * patch_size
                y0 = j * patch_size
                z0 = k * patch_size

                # skip thoes empty patches
                if (
                    np.sum(
                        mask[
                            x0 : x0 + patch_size,
                            y0 : y0 + patch_size,
                            z0 : z0 + patch_size,
                        ]
                    )
                    < 1
                ):
                    continue

                patch_ind = len(patch_lefttop_corners)
                patch_file_path = os.path.join(
                    folder,
                    f"{os.path.basename(image_npz_path)[:-7]}_patchSize_{patch_size}_patchInd_{patch_ind}.pkl.gz",
                )
                with gzip.GzipFile(
                    patch_file_path,
                    "wb",
                ) as f:
                    pickle.dump(
                        {
                            "image": image[
                                x0 : x0 + patch_size,
                                y0 : y0 + patch_size,
                                z0 : z0 + patch_size,
                            ],
                            "label": label[
                                x0 : x0 + patch_size,
                                y0 : y0 + patch_size,
                                z0 : z0 + patch_size,
                            ],
                            "mask": mask[
                                x0 : x0 + patch_size,
                                y0 : y0 + patch_size,
                                z0 : z0 + patch_size,
                            ],
                        },
                        f,
                    )
                patch_lefttop_corners.append(
                    {
                        "image_path": image_npz_path,
                        "patch_file_path": patch_file_path,
                        "x0": x0,
                        "y0": y0,
                        "z0": z0,
                    }
                )
    return patch_lefttop_corners


def per_sample_read_dat_then_process(
    abs_dat_path,
    rel_dataset,
    image_dataset_path,
    processing_log_path,
    file_lock: bool = True,
    out_file_suffix: str = ".ipb2.out.bench",
):
    category, case, dat_name = parse_dat_path(abs_dat_path)
    mole_name = dat_name[:6]
    rel_dat_path = glob.glob(
        os.path.join(rel_dataset, category.replace("_abs", ""), case)
        + f"/{mole_name}*.dat"
    )[
        0
    ]  # remove `_abs`
    abs_dat_folder = os.path.dirname(abs_dat_path)
    abs_out_path = glob.glob(
        os.path.join(abs_dat_folder, f"{mole_name}{out_file_suffix}*")
    )[0]

    try:
        # read grid origin in absolute coordinate
        grid_dims, grid_origin, grid_space = read_bounding_box_info(abs_out_path)
        assert grid_dims is not None and grid_origin is not None

        abs_dat_labels, abs_dat_feats = read_dat(abs_dat_path, max_length=3)
        _, rel_dat_feats = read_dat(rel_dat_path, max_length=96, skip_column=True)
        assert len(abs_dat_feats) == len(rel_dat_feats)

        whole_3d_image = convert_dat_to_3d_image(
            abs_dat_feats,
            abs_dat_labels,
            rel_dat_feats,
            grid_origin,
            grid_dims,
            h_space=grid_space,
            precision=2,
        )
    except BaseException as err:
        if not file_lock:
            with open(processing_log_path, "a") as processing_log:
                print(f"\nError: {err}", file=processing_log)
                print(f"abs: {abs_dat_path}", file=processing_log)
                print(f"rel: {rel_dat_path}", file=processing_log)
        else:
            file_lock = SoftFileLock(processing_log_path + ".lock")
            with file_lock:
                with open(processing_log_path, "a") as processing_log:
                    print(f"\nError: {err}", file=processing_log)
                    print(f"abs: {abs_dat_path}", file=processing_log)
                    print(f"rel: {rel_dat_path}", file=processing_log)
        return

    saved_dir = os.path.join(image_dataset_path, category, case, mole_name)
    os.makedirs(saved_dir, exist_ok=True)
    with gzip.GzipFile(os.path.join(saved_dir, f"{mole_name}.pkl.gz"), "wb") as f:
        pickle.dump(whole_3d_image, f)


def read_dat_then_process(
    abs_dat_paths, rel_dataset, image_dataset_path, thread_num: int = 1
):
    if os.path.exists(image_dataset_path):
        raise Exception(f"Path {image_dataset_path} already exists")
    os.makedirs(image_dataset_path)

    processing_log_path = f"{image_dataset_path}/log.txt"
    print(f"Using {thread_num} threads to process data")
    if thread_num == 1:
        for abs_dat_path in tqdm(abs_dat_paths):
            per_sample_read_dat_then_process(
                abs_dat_path,
                rel_dataset,
                image_dataset_path,
                processing_log_path,
                file_lock=False,
            )
    elif thread_num > 1:
        processing_fn = partial(
            per_sample_read_dat_then_process,
            rel_dataset=rel_dataset,
            image_dataset_path=image_dataset_path,
            processing_log_path=processing_log_path,
            file_lock=True,
        )

        # Using ProcessPoolExecutor to process items in parallel
        with ProcessPoolExecutor(max_workers=thread_num) as executor:
            # Submit all the tasks and get their futures
            futures = [
                executor.submit(processing_fn, abs_dat_path)
                for abs_dat_path in abs_dat_paths
            ]

            # Iterate over the futures as they complete (as_completed) and wrap it with tqdm for the progress bar
            for future in tqdm(as_completed(futures), total=len(abs_dat_paths)):
                future.result()


def read_image_then_process(image_dataset_path, image_npz_paths, patch_size):
    patches_list = []
    with tqdm(total=len(image_npz_paths)) as pbar:
        for image_npz_path in image_npz_paths:
            try:
                patches = slice_3d_image_to_patches_index(image_npz_path, patch_size)
                patches_list.extend(patches)
            except (ValueError, BadZipFile):
                print(f"Error: {image_npz_path}")

            pbar.set_description(f"Patches num: {len(patches_list)}")
            pbar.update(1)

    print(f"Total patches num: {len(patches_list)}")
    with open(f"{image_dataset_path}/patches_index_sz{patch_size}.json", "w") as f:
        json.dump(patches_list, f)


@dataclass
class Args:
    mode: Literal["gen_image", "gen_patch"] = "gen_image"
    abs_dat_paths: str = "datasets/benchmark_data_0.9/*/*/*.dat"
    rel_dataset_path: str = "datasets/benchmark_data_0.5_all"
    image_dataset_path: str = "datasets/benchmark_3d_image_0.9"
    image_npz_paths: str = "datasets/benchmark_3d_image_0.9/*/*/*/*.npz"
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
        image_npz_paths = glob.glob(args.image_npz_paths)
        read_image_then_process(
            args.image_dataset_path, image_npz_paths, args.patch_size
        )
