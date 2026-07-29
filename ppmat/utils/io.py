# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import ast
import gzip
import hashlib
import json
import lzma
import os
import os.path as osp
import shutil
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator
from typing import List
from typing import TextIO

import numpy as np


def open_text(path: str | os.PathLike[str], mode: str = "rt") -> TextIO:
    """Open a plain, gzip, xz, or lz4 text file based on its suffix."""

    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".lz4"):
        import lz4.frame

        return lz4.frame.open(path, mode=mode)
    if suffixes.endswith(".xz"):
        return lzma.open(path, mode=mode)
    if suffixes.endswith(".gz"):
        return gzip.open(path, mode=mode)
    return path.open(mode=mode)


@contextmanager
def materialize_text_path(
    source: str | os.PathLike[str] | TextIO,
    *,
    suffix: str = "",
) -> Iterator[Path]:
    """Yield a plain-text path for a path or text stream.

    Plain files are yielded directly. Compressed files and text streams are
    copied to a temporary file so path-only third-party parsers can consume
    them. Temporary files are removed when the context exits.

    Args:
        source: Plain or compressed path, or an open text stream.
        suffix: Suffix for a temporary file, such as ``".cube"``.

    Yields:
        A path suitable for path-only text parsers.
    """

    if isinstance(source, (str, os.PathLike)):
        path = Path(source).expanduser()
        suffixes = "".join(path.suffixes).lower()
        if not suffixes.endswith((".gz", ".xz", ".lz4")):
            yield path
            return
        file_context = open_text(path)
        restore_position = None
    else:
        file_context = nullcontext(source)
        restore_position = None
        if source.seekable():
            restore_position = source.tell()
            source.seek(0)

    temporary_path: Path | None = None
    try:
        with file_context as file_obj:
            with tempfile.NamedTemporaryFile(
                mode="wt",
                encoding="utf-8",
                suffix=suffix,
                delete=False,
            ) as temporary_file:
                shutil.copyfileobj(file_obj, temporary_file)
                temporary_path = Path(temporary_file.name)
        assert temporary_path is not None
        yield temporary_path
    finally:
        if restore_position is not None:
            source.seek(restore_position)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_cube(
    destination: str | os.PathLike[str],
    atom_numbers,
    atom_coord,
    density,
    info: Mapping,
) -> None:
    """Write one scalar Gaussian CUBE file with ASE.

    Input geometry follows ``info["coordinate_unit"]``. ASE serializes CUBE
    geometry in Bohr, while field values are written without unit scaling.

    Args:
        destination: Plain CUBE output path.
        atom_numbers: Atomic numbers with shape ``[num_atoms]``.
        atom_coord: Atomic coordinates with shape ``[num_atoms, 3]``.
        density: Flattened scalar values matching ``info["shape"]``.
        info: Grid metadata containing ``shape``, ``cell``, and
            ``coordinate_unit``; ``origin`` defaults to zero.
    """

    from ase import Atoms
    from ase.io import write as ase_write
    from ase.units import Bohr

    shape = tuple(int(size) for size in info["shape"])
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError(f"CUBE shape must contain three positive sizes: {shape}.")

    destination = Path(destination)
    if "".join(destination.suffixes).lower().endswith((".gz", ".xz", ".lz4")):
        raise ValueError("ASE CUBE output requires an uncompressed path.")

    cell = np.asarray(info["cell"], dtype=float)
    if cell.shape != (3, 3):
        raise ValueError(f"CUBE cell must have shape [3, 3], but got {cell.shape}.")
    origin = np.asarray(info.get("origin", np.zeros(3)), dtype=float)
    if origin.shape != (3,):
        raise ValueError(f"CUBE origin must have shape [3], but got {origin.shape}.")

    if "coordinate_unit" not in info:
        raise KeyError("CUBE metadata must define 'coordinate_unit'.")
    coordinate_unit = info["coordinate_unit"]
    if not isinstance(coordinate_unit, str):
        raise TypeError("CUBE coordinate_unit must be a string.")
    coordinate_unit = coordinate_unit.strip().lower()
    if coordinate_unit not in {"angstrom", "bohr"}:
        raise ValueError("CUBE coordinate_unit must be either 'angstrom' or 'bohr'.")
    to_angstrom = 1.0 if coordinate_unit == "angstrom" else Bohr

    atom_numbers = np.asarray(atom_numbers, dtype=np.int64).reshape(-1)
    atom_coord = np.asarray(atom_coord, dtype=float)
    if atom_coord.shape != (atom_numbers.size, 3):
        raise ValueError(
            "CUBE atom coordinates must have shape [num_atoms, 3], but got "
            f"{atom_coord.shape} for {atom_numbers.size} atoms."
        )

    density = np.asarray(density, dtype=float).reshape(-1)
    expected_density_size = int(np.prod(shape))
    if density.size != expected_density_size:
        raise ValueError(
            f"CUBE density size must be {expected_density_size} for shape "
            f"{shape}, but got {density.size}."
        )

    atoms = Atoms(
        numbers=atom_numbers,
        positions=atom_coord * to_angstrom,
        cell=cell * to_angstrom,
    )
    ase_write(
        destination,
        atoms,
        format="cube",
        data=density.reshape(shape),
        origin=origin * to_angstrom,
    )


def count_samples_json_lines(path: str):
    """Fast count of samples in a line-delimited JSON file."""
    with open(path, "r") as f:
        return sum(1 for _ in f)


def read_json_lines(path):
    """
    Read all lines from a line-delimited JSON file,
    extracting all properties into a dictionary of lists.
    """
    property_data = {}

    with open(path, "r") as f:
        for idx, line in enumerate(f):
            content = ast.literal_eval(line.strip())
            # if idx == 301:
            #     break
            if idx == 0:
                all_property_names = list(content.keys())
                # print("all_property_names:", all_property_names)
                property_data = {name: [] for name in all_property_names}

            for property_name in all_property_names:
                if property_name not in content:
                    raise ValueError(
                        f"'{property_name}' not found in line {idx + 1} of file"
                    )
                property_data[property_name].append(content[property_name])
    return property_data


def read_json(path):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")
    with open(path, "r") as f:
        content = json.load(f)
    return content


def list_files_by_suffix(path: str, suffix: str) -> List[str]:
    """List files under path with the given suffix."""
    if not osp.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")
    file_names = sorted(
        file_name for file_name in os.listdir(path) if file_name.endswith(suffix)
    )
    if not file_names:
        raise FileNotFoundError(f"No files ending with {suffix} found under {path}.")
    return file_names


def update_json(path, data):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")
    content = read_json(path)
    content.update(data)
    write_json(path, content)


def write_json(path, data):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    def handler(obj: object) -> (int | object):
        """Convert numpy int64 to int.

        Fixes TypeError: Object of type int64 is not JSON serializable
        reported in https://github.com/CederGroupHub/chgnet/issues/168.

        Returns:
            int | object: object for serialization
        """
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=handler)


def read_value_json(path, key):
    """ """
    content = read_json(path)
    if key in content.keys():
        return content[key]
    else:
        return None


def calc_md5(fullname):
    md5 = hashlib.md5()
    fullname = os.path.expanduser(fullname)
    with open(fullname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    return calc_md5sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MD5 hash of a file")
    parser.add_argument("filename", help="Path to the file to hash")
    args = parser.parse_args()

    md5 = calc_md5(args.filename)
    print(md5)
