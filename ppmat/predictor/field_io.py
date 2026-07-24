# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gzip
import lzma
import time
from pathlib import Path

import numpy as np
import paddle


def write_cube(fileobj, atom_type, atom_coord, density, info, idx2atom_num=None):
    """Write a Gaussian CUBE file for an electron-density prediction."""

    fileobj.write("Cube file written on " + time.strftime("%c"))
    fileobj.write("\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
    cell = info["cell"]
    shape = info["shape"]
    origin = info.get("origin", np.zeros(3, dtype=np.float32))
    fileobj.write("{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n".format(len(atom_type), *origin))
    for size, vector in zip(shape, cell):
        step = vector / size
        fileobj.write("{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n".format(size, *step))
    for atom, (x_coord, y_coord, z_coord) in zip(atom_type, atom_coord):
        atomic_num = (
            int(idx2atom_num[int(atom)]) if idx2atom_num is not None else int(atom)
        )
        fileobj.write(
            "{0:5}{1:12.6f}{2:12.6f}{3:12.6f}{4:12.6f}\n".format(
                atomic_num,
                float(atomic_num),
                x_coord,
                y_coord,
                z_coord,
            )
        )
    density.tofile(fileobj, sep="\n", format="%e")


def unavailable_cube_writer(*args, **kwargs):
    raise AttributeError("Cube writer not available for this dataset")


def open_text_maybe_compressed(path):
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".lz4"):
        import lz4.frame

        return lz4.frame.open(path, mode="rt")
    if suffixes.endswith(".xz"):
        return lzma.open(path, mode="rt")
    if suffixes.endswith(".gz"):
        return gzip.open(path, mode="rt")
    return path.open(mode="rt")


def read_cube_density(path):
    """Read density values, grid coordinates, and metadata from a CUBE file."""

    with open_text_maybe_compressed(path) as file_obj:
        file_obj.readline()
        file_obj.readline()
        line = file_obj.readline().split()
        if len(line) < 4:
            raise ValueError(f"Invalid CUBE header (line 3) in {path}")
        num_atoms = int(line[0])
        origin = np.array([float(value) for value in line[1:4]], dtype=np.float32)

        shape = []
        cell = np.zeros((3, 3), dtype=np.float32)
        for index in range(3):
            row = file_obj.readline().split()
            if len(row) < 4:
                raise ValueError(f"Invalid CUBE axis line in {path}")
            size, x_coord, y_coord, z_coord = [float(value) for value in row[:4]]
            shape.append(int(size))
            cell[index] = np.array([x_coord, y_coord, z_coord], dtype=np.float32)

        x_coords = np.arange(shape[0], dtype=np.float32)[:, None] * cell[0][None, :]
        y_coords = np.arange(shape[1], dtype=np.float32)[:, None] * cell[1][None, :]
        z_coords = np.arange(shape[2], dtype=np.float32)[:, None] * cell[2][None, :]
        grid_coord = (
            x_coords.reshape(-1, 1, 1, 3)
            + y_coords.reshape(1, -1, 1, 3)
            + z_coords.reshape(1, 1, -1, 3)
        ).reshape(-1, 3)
        grid_coord += origin

        atom_coord_ref = []
        for _ in range(num_atoms):
            row = file_obj.readline().split()
            if len(row) < 5:
                raise ValueError(f"Invalid CUBE atom line in {path}")
            atom_coord_ref.append([float(row[2]), float(row[3]), float(row[4])])

        num_grid_points = shape[0] * shape[1] * shape[2]
        values = [value for line in file_obj for value in line.split()]
        if len(values) < num_grid_points:
            raise ValueError(
                f"CUBE data too short in {path}: expected {num_grid_points}, "
                f"got {len(values)}"
            )
        density = np.asarray(values[:num_grid_points], dtype=np.float32)

    return (
        paddle.to_tensor(density, dtype="float32"),
        paddle.to_tensor(grid_coord, dtype="float32"),
        {
            "shape": shape,
            "cell": paddle.to_tensor(cell, dtype="float32"),
            "origin": paddle.to_tensor(origin, dtype="float32"),
            "atom_coord_ref": np.asarray(atom_coord_ref, dtype=np.float32),
        },
    )


def prepare_cube_info(info, grid_coord):
    """Build CUBE metadata from dataset information and an explicit grid."""

    shape = info.get("shape")
    if shape is None or len(shape) != 3:
        raise ValueError("CUBE output requires a three-dimensional grid shape.")

    shape = [int(size) for size in shape]
    grid = grid_coord.detach().cpu().numpy().reshape(*shape, 3)
    origin = grid[0, 0, 0]
    steps = np.stack(
        [
            grid[1, 0, 0] - origin if shape[0] > 1 else np.zeros(3, dtype=np.float32),
            grid[0, 1, 0] - origin if shape[1] > 1 else np.zeros(3, dtype=np.float32),
            grid[0, 0, 1] - origin if shape[2] > 1 else np.zeros(3, dtype=np.float32),
        ]
    )
    return {
        "shape": shape,
        "cell": steps * np.asarray(shape, dtype=np.float32)[:, None],
        "origin": origin,
    }
