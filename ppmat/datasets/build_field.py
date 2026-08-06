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

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace
from itertools import chain
from itertools import islice
from typing import Any

import cvve
import numpy as np
from cvve import GridField
from cvve import GridSpec
from cvve import Structure
from p_tqdm import p_map
from pymatgen.core.periodic_table import Element

from ppmat.utils.crystal import normalize_coordinate_unit
from ppmat.utils.io import materialize_text_path
from ppmat.utils.io import open_text


@dataclass(frozen=True)
class BuildField:
    """Build one real scalar field from arrays or a volumetric field file.

    Args:
        name: Semantic field name, such as ``"density"`` or ``"potential"``.
        value_unit: Physical unit of the values. Use ``"unknown"`` only when
            the source format does not define a unit.
        coordinate_unit: Optional expected coordinate unit. File formats use
            the unit parsed from the source. Array grids default to angstrom
            when no unit is supplied by a dataset-specific workflow.
        format: Input format. ``"array"`` accepts real-space values together
            with a keyword-only :class:`cvve.GridSpec`. ``"fft"`` accepts
            half-space packed FFT coefficients for that same grid and inverts
            them. ``"cube"`` and ``"chgcar"`` parse their own grid and optional
            atom metadata with cvve. ``"json"`` accepts a raw mapping or a
            plain/compressed JSON path from the Materials Project density
            release.
        num_cpus: Number of processes used when building a list or tuple.
    """

    name: str
    value_unit: str
    format: str
    coordinate_unit: str | None = None
    num_cpus: int = 1

    @staticmethod
    def build_grid_one(
        grid_data: Mapping[str, Any],
        coordinate_unit: str,
    ) -> GridSpec:
        """Build one affine grid from an array-style grid mapping."""

        if not isinstance(grid_data, Mapping):
            raise TypeError("grid data must be a mapping.")
        missing = {"shape", "voxel_vectors"} - grid_data.keys()
        if missing:
            raise ValueError(f"grid data is missing required keys {sorted(missing)}.")
        return GridSpec(
            shape=grid_data["shape"],
            origin=grid_data.get("origin", np.zeros(3)),
            vectors=grid_data["voxel_vectors"],
            length_unit=normalize_coordinate_unit(coordinate_unit),
        )

    @staticmethod
    def invert_fft(fft_coeff: Any, shape: tuple[int, ...]) -> np.ndarray:
        """Invert half-space packed FFT coefficients into a real-space field.

        The MD17 electron-density release stores each field as real-valued
        coefficients of a half-space packed transform, so every axis is folded
        back before the inverse transform. ``shape`` must be a cubic grid with
        an even edge length.

        Args:
            fft_coeff: Real coefficients holding ``prod(shape)`` values.
            shape: Grid shape the coefficients decode to.

        Returns:
            The flattened real-space values, C contiguous and float32.
        """

        shape = tuple(int(size) for size in shape)
        if len(shape) != 3 or len(set(shape)) != 1 or shape[0] % 2 != 0:
            raise ValueError(
                "fft coefficients require a cubic grid with an even edge "
                f"length, but got shape {shape}."
            )
        num_values = int(np.prod(shape))
        values = np.asarray(fft_coeff, dtype=np.float32).reshape(-1)
        if values.size != num_values:
            raise ValueError(
                f"fft coefficients must hold {num_values} values for grid "
                f"{shape}, but got {values.size}."
            )

        half = shape[0] // 2
        data = values.astype(np.complex64).reshape(1, *shape, order="C")
        for axis in (1, 2, 3):
            front = [slice(None)] * 4
            back = [slice(None)] * 4
            mirror = [slice(None)] * 4
            front[axis] = slice(None, half)
            back[axis] = slice(half, None)
            mirror[axis] = slice(1, half + 1)
            data[tuple(front)] = (data[tuple(front)] - data[tuple(back)] * 1.0j) / 2
            data[tuple(back)] = np.flip(data[tuple(mirror)], axis=axis).conj()
            data = np.fft.ifft(data, axis=axis).astype(np.complex64)

        return np.ascontiguousarray(
            np.flip(data.real.reshape(num_values, order="C"), axis=-1)
        )

    def build_grid(
        self,
        grid_data: Mapping[str, Any] | list[Mapping[str, Any]] | tuple,
    ) -> GridSpec | list[GridSpec]:
        """Build one grid or a list of grids for array and fft fields."""

        coordinate_unit = self.coordinate_unit or "angstrom"
        if isinstance(grid_data, (list, tuple)):
            if not grid_data:
                return []
            return p_map(
                BuildField.build_grid_one,
                grid_data,
                [coordinate_unit] * len(grid_data),
                num_cpus=self.num_cpus,
                desc="Building grids",
                dynamic_ncols=True,
                mininterval=0.2,
            )
        return self.build_grid_one(
            grid_data,
            coordinate_unit,
        )

    @staticmethod
    def build_one(
        field_data: Any,
        format: str,
        name: str,
        value_unit: str,
        coordinate_unit: str | None,
        grid: GridSpec | None = None,
        validate_coordinate_unit: bool = True,
        atom_numbers: Any = None,
        atom_coord: Any = None,
    ) -> GridField:
        """Build one scalar field.

        Args:
            field_data: Real-space values for ``"array"``; FFT coefficients for
                ``"fft"``; a raw mapping or path for ``"json"``; or a file path
                for another file format.
            format: One of ``"array"``, ``"fft"``, ``"cube"``, ``"chgcar"``, or
                ``"json"``.
            name: Semantic field name.
            value_unit: Expected field-value unit.
            coordinate_unit: Expected grid coordinate unit.
            grid: Grid required for ``"array"`` and ``"fft"`` data. For file
                data, an optional independently built grid to validate and
                reuse.
            validate_coordinate_unit: Whether a parsed file grid must already
                use ``coordinate_unit``.
            atom_numbers: Atomic numbers for ``"array"`` or ``"fft"`` data whose
                atoms are supplied separately. File formats parse their own
                atoms.
            atom_coord: Cartesian atom coordinates matching ``atom_numbers``,
                expressed in ``coordinate_unit``.

        Returns:
            The normalized scalar field.
        """

        configured_coordinate_unit = (
            normalize_coordinate_unit(coordinate_unit)
            if coordinate_unit is not None
            else None
        )
        if format in {"array", "fft"}:
            if grid is None:
                raise ValueError(f"grid is required when format is {format!r}.")
            if not isinstance(grid, GridSpec):
                raise TypeError("grid must be a cvve.GridSpec instance.")
            coordinate_unit = configured_coordinate_unit or grid.length_unit
            if (
                validate_coordinate_unit
                and configured_coordinate_unit is not None
                and grid.length_unit != configured_coordinate_unit
            ):
                raise ValueError(
                    f"{format} grid uses "
                    f"{grid.length_unit!r}, but coordinate_unit is "
                    f"{coordinate_unit!r}."
                )
            if format == "fft":
                field_data = BuildField.invert_fft(field_data, grid.shape)
            structure = None
            if atom_numbers is not None or atom_coord is not None:
                if atom_numbers is None or atom_coord is None:
                    raise ValueError(
                        "atom_numbers and atom_coord must be given together."
                    )
                atom_numbers = np.asarray(atom_numbers).reshape(-1)
                atom_coord = np.asarray(atom_coord, dtype=float)
                if atom_coord.ndim != 2 or atom_coord.shape[1] != 3:
                    raise ValueError(
                        "atom_coord must have shape (num_atoms, 3), but got "
                        f"{atom_coord.shape}."
                    )
                if atom_numbers.shape[0] != atom_coord.shape[0]:
                    raise ValueError(
                        "atom_numbers and atom_coord must describe the same "
                        f"number of atoms, but got {atom_numbers.shape[0]} and "
                        f"{atom_coord.shape[0]}."
                    )
                structure = Structure(
                    symbols=[
                        Element.from_Z(int(number)).symbol for number in atom_numbers
                    ],
                    positions=atom_coord,
                    position_unit=coordinate_unit,
                )
            return GridField(
                grid=replace(grid, value_unit=value_unit),
                data=field_data,
                structure=structure,
                name=name,
                kind="density" if name == "density" else "unknown",
            )
        if format == "json":
            if not isinstance(field_data, Mapping):
                if not isinstance(field_data, (str, os.PathLike)):
                    raise TypeError("json field data must be a mapping or a file path.")
                with open_text(field_data) as file_obj:
                    field_data = json.load(file_obj)
            scale = float(field_data["vector"][0][0])
            cell = np.asarray(field_data["lattice"][0], dtype=float) * scale
            shape = tuple(int(size) for size in field_data["FFTgrid"][0])
            parsed_grid = GridSpec(
                shape=shape,
                origin=np.zeros(3),
                vectors=cell / np.asarray(shape)[:, None],
                length_unit="angstrom",
                value_unit="electron/angstrom^3",
                periodic=(True, True, True),
                cell=cell,
            )
            if (
                validate_coordinate_unit
                and configured_coordinate_unit is not None
                and parsed_grid.length_unit != configured_coordinate_unit
            ):
                raise ValueError(
                    "json grid uses "
                    f"{parsed_grid.length_unit!r}, but coordinate_unit is "
                    f"{coordinate_unit!r}."
                )
            if grid is not None and not grid.same_geometry(parsed_grid):
                raise ValueError(
                    "Provided grid does not match the grid parsed from the "
                    "field source."
                )
            parsed_value_unit = parsed_grid.value_unit
            if parsed_value_unit != value_unit:
                raise ValueError(
                    f"json field uses {parsed_value_unit!r}, but value_unit is "
                    f"{value_unit!r}."
                )

            num_values = int(np.prod(shape))
            raw_values = islice(
                chain.from_iterable(field_data["chargedensity"][0]),
                num_values,
            )
            values = np.fromiter(
                (
                    0.0
                    if isinstance(value, str) and value.startswith("*")
                    else float(value)
                    for value in raw_values
                ),
                dtype=float,
                count=num_values,
            )
            values = values.reshape(shape[::-1]).transpose(2, 1, 0)
            values /= abs(float(np.linalg.det(cell)))

            symbols = [
                str(symbol)
                for symbol, count in zip(
                    field_data["elements"][0],
                    field_data["elements_number"][0],
                )
                for _ in range(int(count))
            ]
            structure = Structure(
                symbols=symbols,
                positions=np.asarray(field_data["coordinates"][0], dtype=float),
                position_unit="angstrom",
                coordinate_mode="fractional",
                lattice=cell,
                periodic=(True, True, True),
            )
            output_grid = parsed_grid if grid is None else grid
            return GridField(
                data=values,
                grid=replace(output_grid, value_unit=value_unit),
                structure=structure,
                name=name,
                kind="density" if name == "density" else "unknown",
                source_format="json",
            )
        with materialize_text_path(field_data) as path:
            field = cvve.read_grid_field(
                path,
                format=format,
                name=name,
                kind="density" if name == "density" else "unknown",
            )
        if (
            validate_coordinate_unit
            and configured_coordinate_unit is not None
            and field.grid.length_unit != configured_coordinate_unit
        ):
            raise ValueError(
                f"{format} grid uses {field.grid.length_unit!r}, but "
                f"coordinate_unit is {coordinate_unit!r}."
            )
        if grid is not None and not grid.same_geometry(field.grid):
            raise ValueError(
                "Provided grid does not match the grid parsed from the field source."
            )

        parsed_value_unit = field.grid.value_unit
        if parsed_value_unit != "unknown" and parsed_value_unit != value_unit:
            raise ValueError(
                f"{format} field uses {parsed_value_unit!r}, but value_unit is "
                f"{value_unit!r}."
            )
        output_grid = field.grid if grid is None else grid
        return replace(
            field,
            grid=replace(output_grid, value_unit=value_unit),
            name=name,
        )

    def __call__(
        self,
        field_data: Any,
        *,
        grid: GridSpec | list[GridSpec] | tuple[GridSpec, ...] | None = None,
        validate_coordinate_unit: bool = True,
        atom_numbers: Any = None,
        atom_coord: Any = None,
    ) -> GridField | list[GridField]:
        """Build one field or a list of fields."""

        if isinstance(field_data, (list, tuple)):
            if not field_data:
                return []
            if atom_numbers is not None or atom_coord is not None:
                raise ValueError(
                    "atom_numbers and atom_coord are only supported when "
                    "building a single field."
                )
            if isinstance(grid, (list, tuple)):
                if len(grid) != len(field_data):
                    raise ValueError(
                        "grid and field_data must contain the same number of items."
                    )
                grids = grid
            else:
                grids = [grid] * len(field_data)
            return p_map(
                BuildField.build_one,
                field_data,
                [self.format] * len(field_data),
                [self.name] * len(field_data),
                [self.value_unit] * len(field_data),
                [self.coordinate_unit] * len(field_data),
                grids,
                [validate_coordinate_unit] * len(field_data),
                num_cpus=self.num_cpus,
                desc="Building fields",
                dynamic_ncols=True,
                mininterval=0.2,
            )

        return self.build_one(
            field_data,
            self.format,
            self.name,
            self.value_unit,
            self.coordinate_unit,
            grid=grid,
            validate_coordinate_unit=validate_coordinate_unit,
            atom_numbers=atom_numbers,
            atom_coord=atom_coord,
        )
