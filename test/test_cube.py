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

import shutil
from io import StringIO

import numpy as np
import pytest
from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS
from ase.units import Bohr
from cvve import Structure

from ppmat.datasets.build_field import BuildField
from ppmat.utils.io import open_text
from ppmat.utils.io import write_cube


def _cube_values(coordinate_unit="angstrom"):
    return {
        "atom_numbers": np.asarray([6, 1], dtype=np.int64),
        "atom_coord": np.asarray(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            dtype=np.float32,
        ),
        "density": np.arange(8, dtype=np.float32),
        "info": {
            "shape": [2, 2, 2],
            "cell": np.asarray(
                [[2.0, 0.0, 0.0], [0.5, 2.0, 0.0], [0.0, 0.5, 2.0]],
                dtype=np.float32,
            ),
            "origin": np.asarray([0.25, 0.5, 0.75], dtype=np.float32),
            "coordinate_unit": coordinate_unit,
        },
    }


def _read_cube_with_builder(source, coordinate_unit):
    field = BuildField(
        format="cube",
        name="density",
        value_unit="unknown",
        coordinate_unit=coordinate_unit,
    )(source)
    return field


@pytest.mark.parametrize("coordinate_unit", ["angstrom", "bohr"])
def test_cube_builder_parses_stream_without_taking_ownership(
    tmp_path,
    coordinate_unit,
):
    values = _cube_values(coordinate_unit)
    path = tmp_path / f"density_{coordinate_unit}.cube"
    write_cube(path, **values)
    stream = StringIO(path.read_text())
    assert not stream.closed

    field = _read_cube_with_builder(stream, "bohr")
    grid = field.grid
    assert not stream.closed
    assert field.structure is not None
    expected_scale = 1.0 / Bohr if coordinate_unit == "angstrom" else 1.0

    np.testing.assert_array_equal(
        [ASE_ATOMIC_NUMBERS[symbol] for symbol in field.structure.symbols],
        values["atom_numbers"],
    )
    np.testing.assert_allclose(
        field.structure.cartesian_positions(),
        values["atom_coord"] * expected_scale,
    )
    np.testing.assert_allclose(field.flat, values["density"])
    np.testing.assert_allclose(
        grid.origin,
        values["info"]["origin"] * expected_scale,
        rtol=2e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        grid.cell_vectors,
        values["info"]["cell"] * expected_scale,
        rtol=2e-6,
        atol=1e-6,
    )
    points = grid.cartesian_coordinates()
    np.testing.assert_allclose(
        points[0],
        values["info"]["origin"] * expected_scale,
        rtol=2e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        points[-1],
        (values["info"]["origin"] + values["info"]["cell"].sum(axis=0) / 2)
        * expected_scale,
        rtol=2e-6,
        atol=1e-6,
    )
    assert grid.shape == tuple(values["info"]["shape"])
    assert grid.length_unit == "bohr"


@pytest.mark.parametrize("suffix", [".cube", ".cube.gz", ".cube.xz", ".cube.lz4"])
def test_cube_builder_parses_compressed_paths(tmp_path, suffix):
    if suffix.endswith(".lz4"):
        pytest.importorskip("lz4.frame")

    plain_path = tmp_path / "density.cube"
    path = tmp_path / f"density{suffix}"
    values = _cube_values("bohr")
    write_cube(plain_path, **values)
    if path != plain_path:
        with plain_path.open() as source, open_text(path, mode="wt") as target:
            shutil.copyfileobj(source, target)

    field = _read_cube_with_builder(path, "bohr")
    grid = field.grid

    assert field.structure is not None
    np.testing.assert_array_equal(
        [ASE_ATOMIC_NUMBERS[symbol] for symbol in field.structure.symbols],
        values["atom_numbers"],
    )
    np.testing.assert_allclose(field.flat, values["density"])
    assert grid.length_unit == "bohr"


@pytest.mark.parametrize(
    ("coordinate_unit", "expected_axis_size"),
    [("angstrom", "2"), ("bohr", "2")],
)
def test_cube_builder_parses_units_structure_and_source_once(
    tmp_path,
    monkeypatch,
    coordinate_unit,
    expected_axis_size,
):
    cvve = pytest.importorskip("cvve")
    path = tmp_path / "density.cube"
    values = _cube_values(coordinate_unit)
    write_cube(path, **values)
    cube_lines = path.read_text(encoding="utf-8").splitlines()
    assert cube_lines[0].startswith("Cube file from ASE")
    assert cube_lines[3].split()[0] == expected_axis_size

    read_calls = []
    original_read_grid_field = cvve.read_grid_field

    def record_read(*args, **kwargs):
        read_calls.append(args[0])
        return original_read_grid_field(*args, **kwargs)

    monkeypatch.setattr(cvve, "read_grid_field", record_read)
    field = BuildField(
        format="cube",
        name="density",
        value_unit="unknown",
        coordinate_unit="bohr",
    )(path)
    grid = field.grid

    assert len(read_calls) == 1
    expected_scale = 1.0 / Bohr if coordinate_unit == "angstrom" else 1.0
    np.testing.assert_allclose(
        grid.origin,
        values["info"]["origin"] * expected_scale,
        rtol=2e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        grid.cell_vectors,
        values["info"]["cell"] * expected_scale,
        rtol=2e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(field.flat, values["density"])
    assert isinstance(field.structure, Structure)
    np.testing.assert_array_equal(
        [ASE_ATOMIC_NUMBERS[symbol] for symbol in field.structure.symbols],
        values["atom_numbers"],
    )
    np.testing.assert_allclose(
        field.structure.cartesian_positions(),
        values["atom_coord"] * expected_scale,
    )
    assert field.structure.position_unit == "bohr"


def test_cube_builder_parses_lz4_path(tmp_path):
    pytest.importorskip("cvve")
    pytest.importorskip("lz4.frame")
    plain_path = tmp_path / "density.cube"
    path = tmp_path / "density.cube.lz4"
    values = _cube_values("bohr")
    write_cube(plain_path, **values)
    with plain_path.open() as source, open_text(path, mode="wt") as target:
        shutil.copyfileobj(source, target)

    field = BuildField(
        format="cube",
        name="density",
        value_unit="unknown",
        coordinate_unit="bohr",
    )(path)

    np.testing.assert_array_equal(field.flat, values["density"])
    assert field.structure is not None
    np.testing.assert_array_equal(
        [ASE_ATOMIC_NUMBERS[symbol] for symbol in field.structure.symbols],
        values["atom_numbers"],
    )


def test_cube_builder_preserves_atomic_numbers_beyond_cvve_symbol_table(tmp_path):
    path = tmp_path / "metal.cube"
    values = _cube_values("angstrom")
    values["atom_numbers"] = np.asarray([42, 78], dtype=np.int64)
    write_cube(path, **values)

    field = BuildField(
        format="cube",
        name="density",
        value_unit="unknown",
        coordinate_unit="bohr",
    )(path)

    assert field.structure is not None
    assert field.structure.symbols == ["X42", "X78"]


def test_cube_writer_validates_density_size(tmp_path):
    values = _cube_values()
    values["density"] = np.arange(7, dtype=np.float32)

    with pytest.raises(ValueError, match="density size must be 8"):
        write_cube(tmp_path / "invalid.cube", **values)


def test_cube_writer_rejects_compressed_output(tmp_path):
    with pytest.raises(ValueError, match="uncompressed path"):
        write_cube(tmp_path / "density.cube.lz4", **_cube_values())


def test_density_dataset_reads_shared_cube_from_lz4(tmp_path):
    pytest.importorskip("lz4.frame")

    from ppmat.datasets import DensityDataset

    root = tmp_path / "density"
    root.mkdir()
    (root / "split.json").write_text('{"test": ["000001.cube.lz4"]}')
    values = _cube_values()
    values["atom_numbers"] = np.asarray([6], dtype=np.int64)
    values["atom_coord"] = np.zeros((1, 3), dtype=np.float32)
    values["info"]["coordinate_unit"] = "bohr"
    plain_path = root / "000001.cube"
    compressed_path = root / "000001.cube.lz4"
    write_cube(plain_path, **values)
    with plain_path.open() as source, open_text(
        compressed_path,
        mode="wt",
    ) as target:
        shutil.copyfileobj(source, target)
    plain_path.unlink()

    dataset = DensityDataset(
        path=root / "split.json",
        split="test",
        vocab={
            "atom": {
                "type": "element",
                "tokens": ["C"],
                "num_embeddings": 1,
                "token_to_id": {"C": 0},
                "id_to_token": {0: "C"},
                "atomic_number_to_id": {6: 0},
                "id_to_atomic_number": {0: 6},
            }
        },
        build_field_cfg={
            "format": "cube",
            "name": "density",
            "value_unit": "unknown",
            "coordinate_unit": "bohr",
            "num_cpus": 1,
        },
        build_graph_cfg={
            "__class_name__": "RadiusGraphConverter",
            "__init_params__": {
                "cutoff": 3.0,
                "coordinate_unit": "bohr",
                "inclusive_cutoff": True,
                "atom_vocab": {},
                "include_distance": False,
            },
        },
    )
    sample = dataset[0]
    graph = sample["graph"]
    density = sample["density"]
    grid_coord = sample["grid_coord"]
    info = sample["info"]

    assert isinstance(graph.node_feat["x"], np.ndarray)
    assert isinstance(graph.node_feat["pos"], np.ndarray)
    assert isinstance(density, np.ndarray)
    assert isinstance(grid_coord, np.ndarray)
    np.testing.assert_array_equal(graph.node_feat["x"], [0])
    np.testing.assert_allclose(density, values["density"])
    np.testing.assert_allclose(grid_coord[0], values["info"]["origin"])
    assert info["coordinate_unit"] == "bohr"
