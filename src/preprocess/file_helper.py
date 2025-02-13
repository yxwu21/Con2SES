import os
import re
import numpy as np


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


def find_floatValue(lines, line_prefix, index: int = None):
    """
    find specific data for identifier from given content
    """
    for line in lines:
        if f"{line_prefix}" in line:
            line_strip = line.replace(line_prefix, "")
            datas = line_strip.split(" ")

            valid_data = []
            for data in datas:
                if data:
                    valid_data.append(data)

            if index is None:
                value = [float(i) for i in valid_data]
            else:
                value = float(valid_data[index])
            break

    return value


def find_gridSpace(lines: list[str]):
    """
    find grid space from given content
    """
    prefix = "space="
    for line in lines:
        if prefix in line:
            ind = line.find(prefix) + len(prefix)
            value = float(line[ind : ind + 4])  # grid is always 0.55 or 0.35
            break

    return value


def parse_dat_path(dat_path, has_category=True):
    abs_head, dat_name = os.path.split(dat_path)
    abs_head, case = os.path.split(abs_head)
    category = "all"
    if has_category:
        abs_head, category = os.path.split(abs_head)
    return category, case, dat_name


def read_bounding_box_info(abs_out_path: str):
    grid_origin = None
    grid_dims = None
    grid_space = None

    with open(abs_out_path, "r") as f:
        lines = f.readlines()
        grid_dims = find_floatValue(lines, line_prefix=" Grid dimension at level")[1:]
        grid_dims = np.array(grid_dims, dtype=np.int32)

        grid_origin = find_floatValue(
            lines, line_prefix=" Grid origin corrected at level"
        )[1:]
        grid_origin = np.array(grid_origin, dtype=np.float32)
        grid_space = find_gridSpace(lines)

    return grid_dims, grid_origin, grid_space


def extract_epb_from_final_results(file_path):
    """
    Extracts the EPB value from the "FINAL RESULTS" section in a text file.

    :param file_path: Path to the text file.
    :return: The EPB value as a float, or None if not found or if the "FINAL RESULTS" section is not present.
    """

    try:
        with open(file_path, "r") as file:
            content = file.read()

        # Regular expression to find the "FINAL RESULTS" section and then the EPB value
        final_results_pattern = r"FINAL RESULTS(.*?)\n\s*-+\n"
        epb_pattern = r"EPB\s*=\s*(-?\d+\.\d+)"

        final_results_match = re.search(final_results_pattern, content, re.DOTALL)

        if final_results_match:
            final_results_section = final_results_match.group(1)
            epb_match = re.search(epb_pattern, final_results_section)

            if epb_match:
                return float(epb_match.group(1))
            else:
                return None
        else:
            return None
    except Exception as e:
        return f"Error: {e}"


def read_charge_dat(file):
    """
    Read data from given path
    """
    data = []
    charges = []
    with open(f"{file}", "r") as f:
        reader = f.readlines()
        for line in reader:
            row = line.strip()
            get_col = row.split()
            row_vals = [float(i) for i in get_col]
            charges.append(row_vals[-1])

            row_feats = row_vals[:-1]

            tmp_array = np.array(row_feats)
            data.append(tmp_array)

    feat_matrix = np.stack(data, axis=0)
    return np.array(charges), feat_matrix


def read_pdb_file(
    file, ATOM_TYPE_STATS: dict[str, int], STRC_TYPE_STATS: dict[str, int]
):
    """
    Read pdb file and return the atom coordinates
    """
    xyz = []
    atom_type = []
    atom_struct = []
    charge = []
    with open(file, "r") as f:
        reader = f.readlines()
        for line in reader:
            if line.startswith("ATOM"):
                row = line.strip()
                get_col = row.split()
                xyz.append([float(i) for i in get_col[5:8]])
                charge.append(float(get_col[9]))

                t = get_col[2]
                ATOM_TYPE_STATS[t] = ATOM_TYPE_STATS.get(t, 0) + 1
                atom_type.append(t)

                struct = get_col[3]
                STRC_TYPE_STATS[struct] = STRC_TYPE_STATS.get(struct, 0) + 1
                atom_struct.append(struct)

    return np.array(xyz), np.array(atom_type), np.array(charge), np.array(atom_struct)


def small_mol_read_pdb_file(
    file, ATOM_TYPE_STATS: dict[str, int], STRC_TYPE_STATS: dict[str, int]
):
    """
    Read pdb file and return the atom coordinates
    """
    xyz = []
    atom_type = []
    atom_struct = []
    charge = []
    with open(file, "r") as f:
        reader = f.readlines()
        for line in reader:
            if line.startswith("ATOM"):
                row = line.strip()
                get_col = row.split()
                xyz.append([float(i) for i in get_col[5:8]])
                charge.append(float(get_col[8]))

                t = get_col[2]
                ATOM_TYPE_STATS[t] = ATOM_TYPE_STATS.get(t, 0) + 1
                atom_type.append(t)

                struct = get_col[3]
                STRC_TYPE_STATS[struct] = STRC_TYPE_STATS.get(struct, 0) + 1
                atom_struct.append(struct)

    return np.array(xyz), np.array(atom_type), np.array(charge), np.array(atom_struct)


def read_potential_file(file):
    """
    Read pdb file and return the atom coordinates
    """
    charge = []
    with open(file, "r") as f:
        reader = f.readlines()
        for line in reader:
            row = line.strip()
            if row:
                get_col = row.split()
                charge.append(float(get_col[1]))

    return np.array(charge)


def get_grid_locs(
    abs_grid_feats: np.ndarray,
    abs_grid_origin: np.ndarray,
    h_space: float,
    precision: int = 2,
):
    """convert absolute grid positions to grid locations"""
    grid_locs = np.round(
        (
            np.round(abs_grid_feats, precision)
            - np.round(abs_grid_origin[None, :], precision)
        )
        / h_space
    ).astype(np.int32)
    return grid_locs
