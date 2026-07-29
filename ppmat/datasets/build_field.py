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

from ppmat.utils.io import materialize_text_path
from ppmat.utils.io import open_text


def normalize_coordinate_unit(coordinate_unit: str) -> str:
    """Validate and normalize a supported coordinate unit."""

    if not isinstance(coordinate_unit, str):
        raise TypeError("coordinate_unit must be a string.")
    coordinate_unit = coordinate_unit.strip().lower()
    if coordinate_unit not in {"angstrom", "bohr"}:
        raise ValueError(
            "coordinate_unit must be either 'angstrom' or 'bohr', but got "
            f"{coordinate_unit!r}."
        )
    return coordinate_unit


def normalize_field_format(format: str) -> str:
    """Validate and normalize a field input format."""

    if not isinstance(format, str):
        raise TypeError("format must be a string.")
    format = format.strip().lower()
    supported_formats = {"array", "cube", "chgcar", "json"}
    if format not in supported_formats:
        allowed = ", ".join(sorted(supported_formats))
        raise ValueError(f"format must be one of {allowed}, but got {format!r}.")
    return format


def _normalize_non_empty_string(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


@dataclass(frozen=True)
class BuildField:
    """Build one real scalar field from arrays or a volumetric field file.

    Args:
        name: Semantic field name, such as ``"density"`` or ``"potential"``.
        value_unit: Physical unit of the values. Use ``"unknown"`` only when
            the source format does not define a unit.
        coordinate_unit: Unit shared by the grid origin and voxel vectors.
        format: Input format. ``"array"`` accepts values together with a
            keyword-only :class:`cvve.GridSpec`. ``"cube"`` and ``"chgcar"``
            parse their own grid and optional atom metadata with cvve.
            ``"json"`` accepts a raw mapping or a plain/compressed JSON path
            from the Materials Project density release.
        num_cpus: Number of processes used when building a list or tuple.
    """

    name: str
    value_unit: str
    coordinate_unit: str
    format: str
    num_cpus: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalize_non_empty_string("name", self.name),
        )
        object.__setattr__(
            self,
            "value_unit",
            _normalize_non_empty_string("value_unit", self.value_unit),
        )
        object.__setattr__(
            self,
            "coordinate_unit",
            normalize_coordinate_unit(self.coordinate_unit),
        )
        object.__setattr__(self, "format", normalize_field_format(self.format))
        num_cpus = int(self.num_cpus)
        if num_cpus <= 0:
            raise ValueError("num_cpus must be a positive integer.")
        object.__setattr__(self, "num_cpus", num_cpus)

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

    def build_grid(
        self,
        grid_data: Mapping[str, Any] | list[Mapping[str, Any]] | tuple,
    ) -> GridSpec | list[GridSpec]:
        """Build one grid or a list of grids for array fields."""

        if isinstance(grid_data, (list, tuple)):
            if not grid_data:
                return []
            return p_map(
                BuildField.build_grid_one,
                grid_data,
                [self.coordinate_unit] * len(grid_data),
                num_cpus=self.num_cpus,
                desc="Building grids",
                dynamic_ncols=True,
                mininterval=0.2,
            )
        return self.build_grid_one(
            grid_data,
            self.coordinate_unit,
        )

    @staticmethod
    def build_one(
        field_data: Any,
        format: str,
        name: str,
        value_unit: str,
        coordinate_unit: str,
        grid: GridSpec | None = None,
        validate_coordinate_unit: bool = True,
    ) -> GridField:
        """Build one scalar field.

        Args:
            field_data: Values for ``"array"``; a raw mapping or path for
                ``"json"``; or a path/stream for another file format.
            format: One of ``"array"``, ``"cube"``, ``"chgcar"``, or
                ``"json"``.
            name: Semantic field name.
            value_unit: Expected field-value unit.
            coordinate_unit: Expected grid coordinate unit.
            grid: Grid required for array data. For file data, an optional
                independently built grid to validate and reuse.
            validate_coordinate_unit: Whether a parsed file grid must already
                use ``coordinate_unit``.

        Returns:
            The normalized scalar field.
        """

        format = normalize_field_format(format)
        name = _normalize_non_empty_string("name", name)
        value_unit = _normalize_non_empty_string("value_unit", value_unit)
        coordinate_unit = normalize_coordinate_unit(coordinate_unit)
        if format == "array":
            if grid is None:
                raise ValueError("grid is required when format is 'array'.")
            if not isinstance(grid, GridSpec):
                raise TypeError("grid must be a cvve.GridSpec instance.")
            if validate_coordinate_unit and grid.length_unit != coordinate_unit:
                raise ValueError(
                    "array grid uses "
                    f"{grid.length_unit!r}, but coordinate_unit is "
                    f"{coordinate_unit!r}."
                )
            return GridField(
                grid=replace(grid, value_unit=value_unit),
                data=field_data,
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
            if validate_coordinate_unit and parsed_grid.length_unit != coordinate_unit:
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
        suffix = ".cube" if format == "cube" else ".CHGCAR"
        with materialize_text_path(field_data, suffix=suffix) as path:
            field = cvve.read_grid_field(
                path,
                format=format,
                name=name,
                kind="density" if name == "density" else "unknown",
            )
        if validate_coordinate_unit and field.grid.length_unit != coordinate_unit:
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
    ) -> GridField | list[GridField]:
        """Build one field or a list of fields."""

        if isinstance(field_data, (list, tuple)):
            if not field_data:
                return []
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
        )
