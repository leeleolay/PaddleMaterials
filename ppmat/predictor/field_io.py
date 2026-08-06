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

import numpy as np
import paddle

from ppmat.datasets.build_field import BuildField
from ppmat.utils.crystal import atomic_number_from_symbol
from ppmat.utils.crystal import normalize_coordinate_unit
from ppmat.utils.io import write_cube


def to_numpy(value):
    """Return ``value`` as a NumPy array, detaching Paddle tensors first."""

    if isinstance(value, paddle.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def read_cube_density(
    path,
    field_converter: BuildField,
):
    """Read a CUBE file through the configured field builder."""

    cube_field_converter = BuildField(
        format="cube",
        name=field_converter.name,
        value_unit=field_converter.value_unit,
        coordinate_unit=field_converter.coordinate_unit,
    )
    field = cube_field_converter(path, validate_coordinate_unit=False)
    if field.grid.length_unit != field_converter.coordinate_unit:
        field = field.to_length_unit(
            field_converter.coordinate_unit,
            value_scaling="none",
        )
    grid = field.grid
    if field.structure is None:
        raise ValueError("Reference CUBE does not contain atomic structure data.")

    return (
        np.asarray(field.flat, dtype=np.float32),
        np.asarray(grid.cartesian_coordinates(), dtype=np.float32),
        {
            "shape": list(grid.shape),
            "cell": np.asarray(grid.cell_vectors, dtype=np.float32),
            "origin": np.asarray(grid.origin, dtype=np.float32),
            "atom_numbers": np.asarray(
                [
                    atomic_number_from_symbol(symbol)
                    for symbol in field.structure.symbols
                ],
                dtype=np.int64,
            ),
            "atom_coord_ref": np.asarray(
                field.structure.cartesian_positions(),
                dtype=np.float32,
            ),
            "coordinate_unit": grid.length_unit,
            "density_unit": grid.value_unit,
        },
    )


def write_cube_from_atom_types(
    destination,
    atom_type,
    atom_coord,
    density,
    info,
    idx2atom_num,
):
    """Map internal atom types to atomic numbers and write a CUBE file."""

    if hasattr(atom_type, "detach"):
        atom_type = atom_type.detach().cpu().numpy()
    atom_numbers = np.asarray(
        [idx2atom_num[int(atom)] for atom in np.asarray(atom_type).reshape(-1)],
        dtype=np.int64,
    )
    write_cube(
        destination,
        atom_numbers,
        to_numpy(atom_coord),
        to_numpy(density),
        info,
    )


def prepare_cube_info(
    info,
    grid_coord,
    field_converter: BuildField,
):
    """Return the CUBE geometry metadata for an explicit grid.

    Args:
        info: Sample metadata carrying ``shape``, ``cell``, ``origin`` and the
            coordinate and density units.
        grid_coord: Explicit grid coordinates to validate against the metadata.
        field_converter: Converter whose units the metadata must match.
    """

    shape = info.get("shape")
    if shape is None or len(shape) != 3:
        raise ValueError("CUBE output requires a three-dimensional grid shape.")
    if "cell" not in info:
        raise KeyError("CUBE output metadata must define 'cell'.")
    if "origin" not in info:
        raise KeyError("CUBE output metadata must define 'origin'.")
    if "coordinate_unit" not in info:
        raise KeyError("CUBE output metadata must define 'coordinate_unit'.")

    shape = tuple(int(size) for size in shape)
    cell = to_numpy(info["cell"]).astype(np.float32, copy=False)
    origin = to_numpy(info["origin"]).astype(np.float32, copy=False)
    coordinate_unit = normalize_coordinate_unit(info["coordinate_unit"])
    if coordinate_unit != field_converter.coordinate_unit:
        raise ValueError(
            f"CUBE output uses {coordinate_unit} coordinates, but "
            "Predict.field_converter expects "
            f"{field_converter.coordinate_unit}."
        )
    density_unit = info.get("density_unit")
    if not isinstance(density_unit, str) or not density_unit.strip():
        raise ValueError("CUBE output metadata must define a non-empty 'density_unit'.")
    density_unit = density_unit.strip()
    if density_unit != field_converter.value_unit:
        raise ValueError(
            f"CUBE output uses density unit {density_unit!r}, but "
            "Predict.field_converter expects "
            f"{field_converter.value_unit!r}."
        )
    grid = field_converter.build_grid(
        {
            "shape": shape,
            "voxel_vectors": cell / np.asarray(shape, dtype=np.float32)[:, None],
            "origin": origin,
        }
    )
    actual_points = to_numpy(grid_coord).reshape(-1, 3)
    expected_points = grid.cartesian_coordinates()
    if actual_points.shape != expected_points.shape or not np.allclose(
        actual_points,
        expected_points,
        rtol=1.0e-5,
        atol=1.0e-6,
    ):
        raise ValueError(
            "CUBE output grid coordinates do not match shape, cell, and origin."
        )
    return {
        "shape": list(grid.shape),
        "cell": grid.cell_vectors,
        "origin": grid.origin,
        "coordinate_unit": grid.length_unit,
        "density_unit": field_converter.value_unit,
    }
