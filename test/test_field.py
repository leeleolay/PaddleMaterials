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

import json
import lzma
import pickle

import numpy as np
import pytest
from cvve import GridField
from cvve import GridSpec

from ppmat.datasets.build_field import BuildField


def _grid_mapping():
    shape = (2, 3, 4)
    origin = np.asarray([0.25, -0.5, 1.25])
    cell = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [0.6, 3.0, 0.0],
            [0.2, 0.8, 4.0],
        ]
    )
    return {
        "shape": shape,
        "origin": origin,
        "voxel_vectors": cell / np.asarray(shape)[:, None],
    }


def test_build_field_build_grid_returns_cvve_grid_spec():
    mapping = _grid_mapping()
    builder = BuildField(
        format="array",
        name="density",
        value_unit="electron/angstrom^3",
        coordinate_unit="angstrom",
    )
    grid = builder.build_grid(mapping)

    assert isinstance(grid, GridSpec)
    assert grid.shape == mapping["shape"]
    assert grid.npts == 24
    assert grid.length_unit == "angstrom"
    np.testing.assert_allclose(grid.origin, mapping["origin"])
    np.testing.assert_allclose(grid.vectors, mapping["voxel_vectors"])
    np.testing.assert_allclose(
        grid.cell_vectors,
        mapping["voxel_vectors"] * np.asarray(mapping["shape"])[:, None],
    )

    expected = np.empty((*mapping["shape"], 3))
    for i in range(mapping["shape"][0]):
        for j in range(mapping["shape"][1]):
            for k in range(mapping["shape"][2]):
                expected[i, j, k] = (
                    mapping["origin"]
                    + i * mapping["voxel_vectors"][0]
                    + j * mapping["voxel_vectors"][1]
                    + k * mapping["voxel_vectors"][2]
                )
    np.testing.assert_allclose(
        grid.cartesian_coordinates(),
        expected.reshape(-1, 3),
    )


def test_build_field_build_grid_defaults_origin_and_normalizes_unit():
    builder = BuildField(
        format="array",
        name="density",
        value_unit="unknown",
        coordinate_unit=" AngStRoM ",
    )
    grid = builder.build_grid(
        {
            "shape": (1, 1, 1),
            "voxel_vectors": np.eye(3),
        }
    )

    assert builder.coordinate_unit == "angstrom"
    assert grid.length_unit == "angstrom"
    np.testing.assert_array_equal(grid.origin, np.zeros(3))


@pytest.mark.parametrize(
    ("mapping", "error"),
    [
        ({"voxel_vectors": np.eye(3)}, ValueError),
        ({"shape": (1, 1, 1)}, ValueError),
        ({"shape": (1, 0, 1), "voxel_vectors": np.eye(3)}, ValueError),
        ({"shape": (1, 1, 1), "voxel_vectors": np.ones((2, 3))}, ValueError),
        (
            {
                "shape": (1, 1, 1),
                "voxel_vectors": np.asarray(
                    [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
                ),
            },
            ValueError,
        ),
    ],
)
def test_build_field_build_grid_uses_cvve_geometry_validation(mapping, error):
    builder = BuildField(
        format="array",
        name="density",
        value_unit="unknown",
        coordinate_unit="angstrom",
    )
    with pytest.raises(error):
        builder.build_grid(mapping)


def test_build_field_returns_cvve_grid_field():
    builder = BuildField(
        format="array",
        name="density",
        value_unit="electron/angstrom^3",
        coordinate_unit="angstrom",
    )
    grid = builder.build_grid(_grid_mapping())
    values = np.arange(grid.npts, dtype=np.float32)
    field = builder(values, grid=grid)

    assert isinstance(field, GridField)
    assert field.grid.same_geometry(grid)
    assert field.grid.value_unit == "electron/angstrom^3"
    assert field.name == "density"
    assert field.kind == "density"
    np.testing.assert_array_equal(field.flat, values)


def test_builders_support_batched_grid_and_field_data():
    mappings = [_grid_mapping(), _grid_mapping()]
    builder = BuildField(
        format="array",
        name="density",
        value_unit="electron/angstrom^3",
        coordinate_unit="angstrom",
        num_cpus=1,
    )
    grids = builder.build_grid(mappings)
    values = [
        np.arange(grid.npts, dtype=np.float32) + index
        for index, grid in enumerate(grids)
    ]
    fields = builder(values, grid=grids)

    assert len(grids) == len(fields) == 2
    for grid, field, expected in zip(grids, fields, values):
        assert field.grid.same_geometry(grid)
        np.testing.assert_array_equal(field.flat, expected)


def test_batched_field_data_requires_matching_grids():
    builder = BuildField(
        format="array",
        name="density",
        value_unit="unknown",
        coordinate_unit="angstrom",
        num_cpus=1,
    )

    with pytest.raises(ValueError, match="same number"):
        builder([np.zeros(1), np.ones(1)], grid=[])


def test_build_field_converts_mp_density_json_mapping():
    cell = np.diag([2.0, 3.0, 4.0])
    volume = np.linalg.det(cell)
    raw_data = {
        "vector": [[1.0]],
        "lattice": [cell.tolist()],
        "elements": [["C"]],
        "elements_number": [[1]],
        "coordinates": [[[0.5, 0.5, 0.5]]],
        "FFTgrid": [[2, 1, 1]],
        "chargedensity": [[[volume, "***********", "***********"]]],
    }

    field = BuildField(
        format="json",
        name="density",
        value_unit="electron/angstrom^3",
        coordinate_unit="angstrom",
    )(raw_data)
    grid = field.grid

    assert grid.shape == (2, 1, 1)
    assert grid.periodic == (True, True, True)
    assert field.structure is not None
    assert field.structure.symbols == ["C"]
    np.testing.assert_allclose(field.flat, [1.0, 0.0])
    np.testing.assert_allclose(
        field.structure.cartesian_positions(),
        [[1.0, 1.5, 2.0]],
    )


@pytest.mark.parametrize("compressed", [False, True])
def test_build_field_converts_mp_density_json_path(tmp_path, compressed):
    cell = np.diag([2.0, 3.0, 4.0])
    volume = np.linalg.det(cell)
    raw_data = {
        "vector": [[1.0]],
        "lattice": [cell.tolist()],
        "elements": [["C"]],
        "elements_number": [[1]],
        "coordinates": [[[0.5, 0.5, 0.5]]],
        "FFTgrid": [[2, 1, 1]],
        "chargedensity": [[[volume, "***********", "***********"]]],
    }
    path = tmp_path / ("density.json.xz" if compressed else "density.json")
    if compressed:
        with lzma.open(path, "wt") as file_obj:
            json.dump(raw_data, file_obj)
    else:
        path.write_text(json.dumps(raw_data))

    field = BuildField(
        format="json",
        name="density",
        value_unit="electron/angstrom^3",
        coordinate_unit="angstrom",
    )(path)

    assert field.grid.shape == (2, 1, 1)
    assert field.structure.symbols == ["C"]
    np.testing.assert_allclose(field.flat, [1.0, 0.0])
    np.testing.assert_allclose(
        field.structure.cartesian_positions(),
        [[1.0, 1.5, 2.0]],
    )


def test_build_field_requires_cvve_grid_for_array_data():
    builder = BuildField(
        format="array",
        name="density",
        value_unit="unknown",
        coordinate_unit="angstrom",
    )

    with pytest.raises(ValueError, match="grid is required"):
        builder(np.zeros(1))
    with pytest.raises(TypeError, match="cvve.GridSpec"):
        builder(np.zeros(1), grid=np.zeros((1, 3)))


@pytest.mark.parametrize(
    ("builder_cls", "kwargs", "error"),
    [
        (
            BuildField,
            {
                "name": "",
                "value_unit": "unknown",
                "coordinate_unit": "angstrom",
                "format": "array",
            },
            ValueError,
        ),
        (
            BuildField,
            {
                "name": 1,
                "value_unit": "unknown",
                "coordinate_unit": "angstrom",
                "format": "array",
            },
            TypeError,
        ),
        (
            BuildField,
            {
                "name": "density",
                "value_unit": "",
                "coordinate_unit": "angstrom",
                "format": "array",
            },
            ValueError,
        ),
        (
            BuildField,
            {
                "name": "density",
                "value_unit": None,
                "coordinate_unit": "angstrom",
                "format": "array",
            },
            TypeError,
        ),
        (
            BuildField,
            {
                "name": "density",
                "value_unit": "unknown",
                "coordinate_unit": "angstrom",
                "format": "yaml",
            },
            ValueError,
        ),
        (
            BuildField,
            {
                "name": "density",
                "value_unit": "unknown",
                "coordinate_unit": None,
                "format": "array",
            },
            TypeError,
        ),
        (
            BuildField,
            {
                "name": "density",
                "value_unit": "unknown",
                "coordinate_unit": "meter",
                "format": "array",
            },
            ValueError,
        ),
    ],
)
def test_builders_reject_invalid_configuration(builder_cls, kwargs, error):
    with pytest.raises(error):
        builder_cls(**kwargs)


def test_builder_and_cvve_objects_pickle_round_trip():
    field_builder = BuildField(
        format="array",
        name="density",
        value_unit="electron/angstrom^3",
        coordinate_unit="angstrom",
    )
    grid = field_builder.build_grid(_grid_mapping())
    field = field_builder(np.arange(grid.npts), grid=grid)

    restored_field_builder = pickle.loads(pickle.dumps(field_builder))
    restored_grid = pickle.loads(pickle.dumps(grid))
    restored_field = pickle.loads(pickle.dumps(field))

    assert restored_field_builder == field_builder
    assert restored_grid.same_geometry(grid)
    assert restored_field.same_grid(field)
    np.testing.assert_array_equal(restored_field.data, field.data)
