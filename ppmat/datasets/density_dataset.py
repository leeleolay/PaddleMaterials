# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import multiprocessing as mp
import os
import os.path as osp
import pickle
import shutil
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
import paddle.distributed as dist

from ppmat.datasets.build_field import BuildField
from ppmat.datasets.grid_sampler import DensityGridSampler
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils import misc
from ppmat.utils.misc import is_equal

_DENSITY_CACHE_FIELD_BUILDER = None
_DENSITY_CACHE_GRAPH_CONVERTER = None
_DENSITY_CACHE_FIELDS_PATH = None
_DENSITY_CACHE_GRAPHS_PATH = None


def _init_density_cache_worker(
    build_field_cfg,
    build_graph_cfg,
    vocab,
    fields_cache_path,
    graph_cache_path,
):
    """Initialize CPU-only converters once in each spawned cache worker."""
    global _DENSITY_CACHE_FIELD_BUILDER
    global _DENSITY_CACHE_GRAPH_CONVERTER
    global _DENSITY_CACHE_FIELDS_PATH
    global _DENSITY_CACHE_GRAPHS_PATH

    _DENSITY_CACHE_FIELD_BUILDER = BuildField(**build_field_cfg)
    _DENSITY_CACHE_GRAPH_CONVERTER = (
        build_graph_converter(build_graph_cfg, vocab=vocab)
        if build_graph_cfg is not None
        else None
    )
    _DENSITY_CACHE_FIELDS_PATH = fields_cache_path
    _DENSITY_CACHE_GRAPHS_PATH = graph_cache_path


def _build_density_cache_sample(index_and_source):
    """Build and serialize one field and its graph in a cache worker."""
    index, field_source = index_and_source
    field = _DENSITY_CACHE_FIELD_BUILDER(field_source)
    with open(osp.join(_DENSITY_CACHE_FIELDS_PATH, f"{index:010d}.pkl"), "wb") as f:
        pickle.dump(field, f)

    if _DENSITY_CACHE_GRAPH_CONVERTER is not None:
        graph = _DENSITY_CACHE_GRAPH_CONVERTER.from_structures([field.structure])[0]
        with open(osp.join(_DENSITY_CACHE_GRAPHS_PATH, f"{index:010d}.pkl"), "wb") as f:
            pickle.dump(graph, f)
    return index


class DensityDataset(paddle.io.Dataset):
    """Density Dataset Handler."""

    # Optional registry fields; can be overridden per config or subclass
    name: str | None = None
    url: str | None = None
    md5: str | None = None
    split_url: str | None = None
    split_md5: str | None = None

    def __init__(
        self,
        path: str | os.PathLike[str],
        split: str,
        vocab,
        build_graph_cfg: Dict[str, Any],
        build_field_cfg: Dict[str, Any] | None = None,
        transforms: Callable | None = None,
        grid_sampler_cfg: Dict[str, Any] | None = None,
        cache_path: str | os.PathLike[str] | None = None,
        cache_num_workers: int = 1,
        overwrite: bool = False,
    ) -> None:
        super().__init__()
        if not osp.exists(path):
            logger.message("The dataset is not found. Will download it now.")
            root_path = download.get_datasets_path_from_url(self.url, self.md5)
            path = osp.join(root_path, self.name, osp.basename(path))

        self.path = path
        self.split = split

        if build_field_cfg is None:
            build_field_cfg = {
                "format": "cube",
                "name": "density",
                "value_unit": "unknown",
                "coordinate_unit": "bohr",
                "num_cpus": 1,
            }
            logger.message(
                "The build_field_cfg is not set, will use the default "
                f"configs: {build_field_cfg}"
            )

        self.build_graph_cfg = build_graph_cfg
        self.build_field_cfg = build_field_cfg
        self.transforms = transforms
        if (
            isinstance(cache_num_workers, bool)
            or not isinstance(cache_num_workers, int)
            or cache_num_workers <= 0
        ):
            raise ValueError("cache_num_workers must be a positive integer.")
        self.cache_num_workers = cache_num_workers
        self.grid_sampler = (
            DensityGridSampler(**grid_sampler_cfg)
            if grid_sampler_cfg is not None
            else None
        )

        if cache_path is not None:
            self.cache_path = cache_path
        else:
            self.cache_path = osp.join(
                osp.split(path)[0] + "_cache",
                f"{osp.splitext(osp.basename(path))[0]}_{split}",
            )
        logger.info(f"Cache path: {self.cache_path}")

        # prepare vocab
        self.vocab = vocab
        atom_vocab = vocab["atom"]
        graph_vocab = {"atom": atom_vocab}

        self.overwrite = overwrite
        self.cache_exists = True if osp.exists(self.cache_path) else False
        self.row_data, self.num_samples = self.read_data(path)
        logger.info(f"Load {self.num_samples} samples from {path}")

        if self.cache_exists and not overwrite:
            logger.warning(
                "Cache enabled. If a cache file exists, it will be automatically "
                "read and current settings will be ignored. Please ensure that the "
                "settings used in match your current settings."
            )
            try:
                build_field_cfg_cache = self.load_from_cache(
                    osp.join(self.cache_path, "build_field_cfg.pkl")
                )
                if is_equal(build_field_cfg_cache, build_field_cfg):
                    logger.info(
                        "The cached build_field_cfg configuration matches "
                        "the current settings. Reusing previously generated "
                        "field and graph data to optimize performance."
                    )
                else:
                    logger.warning(
                        "build_field_cfg is different from build_field_cfg_cache. "
                        "Will rebuild the fields and graphs."
                    )
                    logger.warning(
                        "If you want to use the cached fields and graphs, please "
                        "ensure that the settings used in match your current settings."
                    )
                    overwrite = True
            except Exception as e:
                logger.warning(e)
                logger.warning(
                    "Failed to load build_field_cfg.pkl from cache. "
                    "Will rebuild the fields and graphs."
                )
                overwrite = True

            if build_graph_cfg is not None and not overwrite:
                try:
                    build_graph_cfg_cache = self.load_from_cache(
                        osp.join(self.cache_path, "build_graph_cfg.pkl")
                    )
                    graph_vocab_cache = self.load_from_cache(
                        osp.join(self.cache_path, "graph_vocab.pkl")
                    )
                    if is_equal(build_graph_cfg_cache, build_graph_cfg) and is_equal(
                        graph_vocab_cache, graph_vocab
                    ):
                        logger.info(
                            "The cached graph configuration and vocabulary match "
                            "the current settings. Reusing previously generated "
                            "graph data to optimize performance."
                        )
                    else:
                        logger.warning(
                            "Graph configuration or vocabulary differs from the "
                            "cache. Will rebuild the fields and graphs."
                        )
                        logger.warning(
                            "If you want to use the cached fields and graphs, "
                            "please ensure that the settings used in match your "
                            "current settings."
                        )
                        overwrite = True
                except Exception as e:
                    logger.warning(e)
                    logger.warning(
                        "Failed to load build_graph_cfg.pkl from cache. "
                        "Will rebuild the fields and graphs."
                    )
                    overwrite = True
        fields_cache_path = osp.join(self.cache_path, "fields")
        graph_cache_path = osp.join(self.cache_path, "graphs")
        if overwrite or not self.cache_exists:
            # convert fields and graphs
            # only rank 0 process do the conversion
            if dist.get_rank() == 0:
                # save build_field_cfg and build_graph_cfg to cache file
                os.makedirs(self.cache_path, exist_ok=True)
                self.save_to_cache(
                    osp.join(self.cache_path, "build_field_cfg.pkl"),
                    build_field_cfg,
                )
                self.save_to_cache(
                    osp.join(self.cache_path, "build_graph_cfg.pkl"),
                    build_graph_cfg,
                )
                self.save_to_cache(
                    osp.join(self.cache_path, "graph_vocab.pkl"),
                    graph_vocab,
                )
                # Save fields and graphs to cache. Spawned workers avoid
                # inheriting any CUDA state initialized before Dataset setup.
                os.makedirs(fields_cache_path, exist_ok=True)
                if build_graph_cfg is not None:
                    os.makedirs(graph_cache_path, exist_ok=True)
                if self.cache_num_workers == 1:
                    field_builder = BuildField(**build_field_cfg)
                    converter = (
                        build_graph_converter(build_graph_cfg, vocab=vocab)
                        if build_graph_cfg is not None
                        else None
                    )
                    for i, field_source in enumerate(self.row_data["field_source"]):
                        field = field_builder(field_source)
                        self.save_to_cache(
                            osp.join(fields_cache_path, f"{i:010d}.pkl"), field
                        )
                        if converter is not None:
                            graph = converter.from_structures([field.structure])[0]
                            self.save_to_cache(
                                osp.join(graph_cache_path, f"{i:010d}.pkl"), graph
                            )
                else:
                    worker_context = mp.get_context("spawn")
                    with ProcessPoolExecutor(
                        max_workers=self.cache_num_workers,
                        mp_context=worker_context,
                        initializer=_init_density_cache_worker,
                        initargs=(
                            build_field_cfg,
                            build_graph_cfg,
                            vocab,
                            fields_cache_path,
                            graph_cache_path,
                        ),
                    ) as executor:
                        for _ in executor.map(
                            _build_density_cache_sample,
                            enumerate(self.row_data["field_source"]),
                            chunksize=1,
                        ):
                            pass
                logger.info(f"Save {self.num_samples} fields to {fields_cache_path}")
                if build_graph_cfg is not None:
                    logger.info(f"Save {self.num_samples} graphs to {graph_cache_path}")

            # sync all processes
            if dist.is_initialized():
                dist.barrier()
        self.fields = [
            osp.join(fields_cache_path, f"{i:010d}.pkl")
            for i in range(self.num_samples)
        ]
        if build_graph_cfg is not None:
            self.graphs = [
                osp.join(graph_cache_path, f"{i:010d}.pkl")
                for i in range(self.num_samples)
            ]
        else:
            self.graphs = None

        assert (
            len(self.fields) == self.num_samples
        ), "The number of fields must be equal to the number of samples."
        assert (
            self.graphs is None or len(self.graphs) == self.num_samples
        ), "The number of graphs must be equal to the number of samples."

    def read_data(self, path: str):
        """Read the requested split index and resolve its field sources.

        Args:
            path (str): Path to the data.
        """
        with open(path) as file_obj:
            split_data = json.load(file_obj)
        entries = list(split_data[self.split])
        file_names = [str(entry) for entry in entries]
        field_source = [
            osp.join(osp.dirname(path), file_name) for file_name in file_names
        ]
        data = {
            "id": entries,
            "file_name": file_names,
            "field_source": field_source,
        }
        num_samples = len(entries)
        return data, num_samples

    def save_to_cache(self, cache_path: str, data: Any):
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_from_cache(self, cache_path: str):
        if osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data
        else:
            raise FileNotFoundError(f"No such file or directory: {cache_path}")

    @staticmethod
    def _read_property_data(field):
        return {
            "density": np.asarray(field.flat, dtype=np.float32),
        }

    def __getitem__(self, idx: int):
        """Get item at index idx."""
        data = {}

        if self.graphs is not None:
            graph = self.graphs[idx]
            if isinstance(graph, str):
                graph = self.load_from_cache(graph)
            data["graph"] = graph

        field = self.fields[idx]
        if isinstance(field, str):
            field = self.load_from_cache(field)
        grid = field.grid
        data.update(self._read_property_data(field))
        data["grid_coord"] = np.asarray(field.coordinates(), dtype=np.float32)
        data["info"] = {
            "shape": list(grid.shape),
            "cell": np.asarray(grid.cell_vectors, dtype=np.float32),
            "origin": np.asarray(grid.origin, dtype=np.float32),
            "coordinate_unit": grid.length_unit,
            "density_unit": grid.value_unit,
            "file_name": self.row_data["file_name"][idx],
        }
        data["id"] = self.row_data["id"][idx]
        if self.grid_sampler is not None:
            identity = (
                data["info"].get("source_split", self.split),
                data["id"],
                data["info"].get("file_name"),
            )
            data = self.grid_sampler(data, identity)
        data = self.transforms(data) if self.transforms is not None else data
        return data

    def __len__(self):
        return self.num_samples


MD17_DENSITY_CACHE_SCHEMA_VERSION = 10
MD17_COORDINATE_UNIT = "bohr"  # Published MD17 FFT coordinates use bohr.
_MD17_CACHE_BATCH_SIZE = 16

MD17_ATOMIC_NUMBERS: Dict[str, np.ndarray] = {
    "benzene": np.array([6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1], dtype="int64"),
    "ethanol": np.array([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype="int64"),
    "phenol": np.array([6, 6, 6, 6, 6, 6, 8, 1, 1, 1, 1, 1, 1], dtype="int64"),
    "resorcinol": np.array([6, 6, 6, 6, 6, 6, 8, 1, 8, 1, 1, 1, 1, 1], dtype="int64"),
    "ethane": np.array([6, 6, 1, 1, 1, 1, 1, 1], dtype="int64"),
    "malonaldehyde": np.array([8, 6, 6, 6, 8, 1, 1, 1, 1], dtype="int64"),
}


class MD17DensityDataset(paddle.io.Dataset):
    """MD17 small-molecule electron-density dataset with caching and download.

    The dataset stores FFT-domain electron-density coefficients for six MD17
    molecules (benzene, ethanol, phenol, resorcinol, ethane, malonaldehyde).
    A missing local path triggers the repository-native download flow. Converted
    samples are cached to avoid recomputing FFT inversions on every run. The
    source release only provides train and test directories; following the
    protocol, ``validation`` reads the same source samples as ``test``.

    Raw layout (after extracting ``md17_es.tar.gz``):
        root/
            <mol_name>/
                <mol_name>_train/{structures.npy,dft_densities.npy,...}
                <mol_name>_test/{structures.npy,dft_densities.npy,...}

    Args:
        path: Path to one molecule/split source directory, such as
            ``./data/data_md/ethanol/ethanol_train``. The molecule and
            physical source split are inferred from the directory name. If the
            path is missing, the MD17 archive is downloaded.
        split: ``train``, ``validation``, or ``test``.
        n_grid: Cube grid resolution per axis.
        grid_size: Physical box size (Bohr) for the grid.
        build_field_cfg: Parameters passed to :class:`BuildField`. Defaults to
            the MD17 ``fft``/bohr source protocol.
        build_graph_cfg: Registered graph converter configuration. The
            converter must support construction from atomic numbers and
            Cartesian coordinate arrays.
        cache_path: Optional cache directory. Defaults beside ``path`` and
            includes the grid geometry in its name.
        overwrite: Force rebuilding cache even if it exists.
        transforms: Optional callable applied to the MP20-style sample mapping.
    """

    name = "md17_es"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MD17_ES/md17_es.tar.gz"
    # BCE does not currently publish a content MD5 for this archive.
    md5 = None

    def __init__(
        self,
        path: str | os.PathLike[str] = "./data/data_md/ethanol/ethanol_train",
        split: str = "train",
        n_grid: int = 50,
        grid_size: float = 20.0,
        *,
        vocab,
        build_graph_cfg: Dict[str, Any],
        build_field_cfg: Dict[str, Any] | None = None,
        cache_path: str | os.PathLike[str] | None = None,
        overwrite: bool = False,
        transforms: Callable | None = None,
        grid_sampler_cfg: Dict[str, Any] | None = None,
    ) -> None:
        # TODO: Refactor this initializer.
        super().__init__()

        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be one of ['train', 'validation', 'test']")
        if isinstance(n_grid, bool) or not isinstance(n_grid, (int, np.integer)):
            raise TypeError("n_grid must be an integer.")
        if transforms is not None and not callable(transforms):
            raise TypeError("transforms must be callable or None.")
        if not isinstance(build_graph_cfg, dict):
            raise TypeError("build_graph_cfg must be a dictionary.")
        if build_field_cfg is None:
            build_field_cfg = {
                "format": "fft",
                "name": "density",
                "value_unit": "unknown",
                "coordinate_unit": "bohr",
                "num_cpus": 1,
            }
        elif not isinstance(build_field_cfg, dict):
            raise TypeError("build_field_cfg must be a dictionary or None.")

        path = osp.abspath(osp.normpath(osp.expanduser(os.fspath(path))))
        requested_tail = path.split(osp.sep)[-2:]
        if not osp.exists(path):
            logger.message(
                f"Dataset path {path} not found. Downloading {self.name} "
                f"from {self.url}."
            )
            root_path = download.get_datasets_path_from_url(self.url, self.md5)
            path = osp.abspath(osp.join(root_path, *requested_tail))
            logger.message(f"Downloaded and resolved dataset path to {path}")
        if not osp.isdir(path):
            raise NotADirectoryError(f"Dataset path {path} is not a directory.")

        source_dir = osp.basename(path)
        try:
            mol_name, source_split = source_dir.rsplit("_", 1)
        except ValueError as error:
            raise ValueError(
                "MD17 path must end in '<molecule>_<train|test>', "
                f"but got {source_dir!r}."
            ) from error
        if source_split not in {"train", "test"}:
            raise ValueError(
                "MD17 path must identify a train or test source directory, "
                f"but got {source_dir!r}."
            )
        expected_source_split = "test" if split == "validation" else split
        if source_split != expected_source_split:
            raise ValueError(
                f"Dataset split {split!r} requires an MD17 "
                f"{expected_source_split!r} source path, but got {source_dir!r}."
            )
        if mol_name not in MD17_ATOMIC_NUMBERS:
            raise ValueError(
                f"Unsupported molecule {mol_name}. "
                f"Options: {list(MD17_ATOMIC_NUMBERS)}"
            )

        self.path = path
        self.data_path = path
        self.mol_name = mol_name
        self.split = split
        self.source_split = source_split
        self.vocab = vocab
        atom_vocab = vocab["atom"]
        self.atom_name2idx = atom_vocab["token_to_id"]
        self.atom_num2idx = atom_vocab["atomic_number_to_id"]
        self.idx2atom_num = atom_vocab["id_to_atomic_number"]
        self.atom_types = np.asarray(
            [
                self.atom_num2idx[int(atomic_number)]
                for atomic_number in MD17_ATOMIC_NUMBERS[mol_name]
            ],
            dtype=np.int64,
        )
        self.n_grid = int(n_grid)
        self.grid_size = float(grid_size)
        build_field_cfg = dict(build_field_cfg)
        build_field_cfg.setdefault("coordinate_unit", MD17_COORDINATE_UNIT)
        self.build_graph_cfg = dict(build_graph_cfg)
        self.field_converter = BuildField(**build_field_cfg)
        if self.field_converter.format != "fft":
            raise ValueError(
                "MD17DensityDataset requires build_field_cfg.format to be 'fft'."
            )
        if self.field_converter.coordinate_unit != "bohr":
            raise ValueError(
                "MD17DensityDataset requires build_field_cfg.coordinate_unit "
                "to be 'bohr'."
            )
        if self.field_converter.name != "density":
            raise ValueError(
                "MD17DensityDataset requires build_field_cfg.name to be 'density'."
            )
        self.build_field_cfg = {
            "format": self.field_converter.format,
            "name": self.field_converter.name,
            "value_unit": self.field_converter.value_unit,
            "coordinate_unit": self.field_converter.coordinate_unit,
            "num_cpus": self.field_converter.num_cpus,
        }
        self.transforms = transforms
        self.grid_sampler = (
            DensityGridSampler(**grid_sampler_cfg)
            if grid_sampler_cfg is not None
            else None
        )
        self._validate_init_params()

        if cache_path is None:
            cache_path = f"{self.path}_n{self.n_grid}_g{self.grid_size:g}_cache"
        self.cache_path = osp.abspath(osp.expanduser(os.fspath(cache_path)))

        grid_step = self.grid_size / self.n_grid
        self.grid_data = self.field_converter.build_grid(
            {
                "shape": (self.n_grid, self.n_grid, self.n_grid),
                "voxel_vectors": np.eye(3, dtype=np.float32) * grid_step,
                "origin": np.full(3, grid_step, dtype=np.float32),
            }
        )
        self.cell = np.asarray(self.grid_data.cell_vectors, dtype=np.float32)
        self.origin = np.asarray(self.grid_data.origin, dtype=np.float32)

        self.samples: List[str] = []
        self.sample_ids: List[int] = []
        self.grid_coord: np.ndarray
        self._prepare_cache(overwrite)
        self.num_samples = len(self.samples)

    def _validate_init_params(self) -> None:
        if self.n_grid < 2 or self.n_grid % 2 != 0:
            raise ValueError("n_grid must be a positive even integer greater than 1.")
        if not np.isfinite(self.grid_size) or self.grid_size <= 0:
            raise ValueError("grid_size must be a positive finite number.")
        atom_types = self.atom_types
        if atom_types.ndim != 1 or atom_types.size == 0:
            raise ValueError(f"Invalid atom types for molecule {self.mol_name}.")
        if not set(atom_types.tolist()).issubset(self.idx2atom_num):
            raise ValueError(
                f"Atom types for {self.mol_name} are not covered by idx2atom_num."
            )

    def _prepare_cache(self, overwrite: bool) -> None:
        expected_cfg = self._cache_config()
        cache_state = None if overwrite else self._load_valid_cache(expected_cfg)
        rebuild_cache = cache_state is None

        with misc.RankZeroOnly() as is_master:
            if rebuild_cache and is_master:
                if osp.exists(self.cache_path) and not overwrite:
                    logger.warning(
                        "Cached data is incomplete or does not match the source data. "
                        "Rebuilding it."
                    )
                self._build_cache(expected_cfg)

        if rebuild_cache:
            cache_state = self._load_valid_cache(expected_cfg)
        if cache_state is None:
            raise RuntimeError(f"No complete cache found under {self.cache_path}.")

        manifest, grid = cache_state
        sample_dir = osp.join(self.cache_path, "samples")
        self.samples = [
            osp.join(sample_dir, sample_file)
            for sample_file in manifest["sample_files"]
        ]
        self.sample_ids = manifest["source_indices"]

        self.grid_coord = grid
        self.grid_coord.setflags(write=False)

    def _raw_paths(self) -> Tuple[str, str]:
        structures_path = osp.join(self.data_path, "structures.npy")
        density_path = osp.join(self.data_path, "dft_densities.npy")
        if not osp.exists(structures_path) or not osp.exists(density_path):
            raise FileNotFoundError(
                f"Cannot locate expected files under {self.data_path}. "
                "Expected structures.npy and dft_densities.npy."
            )
        return structures_path, density_path

    def _array_fingerprint(self, path: str) -> Dict[str, Any]:
        array = np.load(path, mmap_mode="r")
        stat = os.stat(path)
        return {
            "path": osp.abspath(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

    def _cache_config(self) -> Dict[str, Any]:
        structures_path, density_path = self._raw_paths()
        return {
            "schema_version": MD17_DENSITY_CACHE_SCHEMA_VERSION,
            "mol_name": self.mol_name,
            "source_split": self.source_split,
            "n_grid": self.n_grid,
            "grid_size": self.grid_size,
            "build_field_cfg": self.build_field_cfg,
            "build_graph_cfg": self.build_graph_cfg,
            "atom_types": self.atom_types.tolist(),
            "structures": self._array_fingerprint(structures_path),
            "densities": self._array_fingerprint(density_path),
        }

    def _load_valid_cache(
        self,
        expected_cfg: Dict[str, Any],
    ) -> Optional[Tuple[Dict[str, Any], np.ndarray]]:
        sample_dir = osp.join(self.cache_path, "samples")
        required_paths = (
            osp.join(self.cache_path, "completed.flag"),
            osp.join(self.cache_path, "dataset_cfg.pkl"),
            osp.join(self.cache_path, "manifest.pkl"),
            osp.join(self.cache_path, "grid.pkl"),
            sample_dir,
        )
        if not all(osp.exists(path) for path in required_paths):
            return None

        try:
            cached_cfg = self.load_from_cache(required_paths[1])
            manifest = self.load_from_cache(required_paths[2])
            grid = self.load_from_cache(required_paths[3])
        except Exception:
            return None

        if not is_equal(cached_cfg, expected_cfg):
            return None
        if not isinstance(manifest, dict):
            return None

        num_samples = manifest.get("num_samples")
        sample_files = manifest.get("sample_files")
        sample_sizes = manifest.get("sample_sizes")
        source_indices = manifest.get("source_indices")
        if not isinstance(num_samples, int) or num_samples <= 0:
            return None
        if (
            not isinstance(sample_files, list)
            or not isinstance(sample_sizes, list)
            or not isinstance(source_indices, list)
        ):
            return None
        if not all(isinstance(sample_file, str) for sample_file in sample_files):
            return None
        if not all(
            isinstance(source_idx, int) and source_idx >= 0
            for source_idx in source_indices
        ):
            return None
        expected_files = [f"{idx:010d}.pkl" for idx in range(num_samples)]
        if (
            sample_files != expected_files
            or len(sample_sizes) != num_samples
            or len(source_indices) != num_samples
        ):
            return None
        if any(not isinstance(size, int) or size <= 0 for size in sample_sizes):
            return None
        if len(set(source_indices)) != num_samples:
            return None

        source_samples = expected_cfg["structures"]["shape"][0]
        expected_indices = self._select_sample_indices(source_samples).tolist()
        if source_indices != expected_indices:
            return None

        try:
            actual_files = sorted(
                file_name
                for file_name in os.listdir(sample_dir)
                if file_name.endswith(".pkl")
            )
            actual_sizes = [
                osp.getsize(osp.join(sample_dir, sample_file))
                for sample_file in sample_files
            ]
            grid_is_valid = (
                isinstance(grid, np.ndarray)
                and grid.dtype == np.float32
                and grid.shape == (self.n_grid**3, 3)
            )
        except (OSError, TypeError, ValueError):
            return None
        if (
            actual_files != expected_files
            or actual_sizes != sample_sizes
            or not grid_is_valid
        ):
            return None
        return manifest, grid

    def _build_cache(self, expected_cfg: Dict[str, Any]) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return

        cache_path = osp.abspath(self.cache_path)
        sample_dir = osp.join(cache_path, "samples")
        if osp.isdir(sample_dir):
            if not osp.isfile(osp.join(cache_path, "dataset_cfg.pkl")):
                raise ValueError(
                    f"Refusing to replace unrecognized cache directory {cache_path}."
                )
            shutil.rmtree(sample_dir)
        os.makedirs(cache_path, exist_ok=True)
        logger.message(f"Building cache at {self.cache_path}")
        self._write_cache(cache_path, expected_cfg)

    def read_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Read the MD17 structures and FFT-density arrays."""

        structures_path, density_path = self._raw_paths()
        return (
            np.load(structures_path, mmap_mode="r"),
            np.load(density_path, mmap_mode="r"),
        )

    def _write_cache(self, cache_path: str, expected_cfg: Dict[str, Any]) -> None:
        sample_dir = osp.join(cache_path, "samples")
        os.makedirs(sample_dir)
        self.save_to_cache(osp.join(cache_path, "dataset_cfg.pkl"), expected_cfg)

        structures, densities_fft = self.read_data()
        self._validate_raw_layout(structures, densities_fft)

        source_indices = self._select_sample_indices(structures.shape[0])
        atom_type = self.atom_types
        atomic_numbers = MD17_ATOMIC_NUMBERS[self.mol_name]
        graph_converter = build_graph_converter(self.build_graph_cfg)

        grid_coord = self._generate_grid()
        self.save_to_cache(osp.join(cache_path, "grid.pkl"), grid_coord)

        logger.message(
            f"Precomputing {len(source_indices)} {self.split} density samples "
            "from FFT coefficients with "
            f"n_grid={self.n_grid}, batch_size={_MD17_CACHE_BATCH_SIZE} ..."
        )
        sample_files = []
        sample_sizes = []
        for batch_start in range(0, len(source_indices), _MD17_CACHE_BATCH_SIZE):
            batch_indices = source_indices[
                batch_start : batch_start + _MD17_CACHE_BATCH_SIZE
            ]
            structure_batch = np.asarray(structures[batch_indices], dtype=np.float32)
            density_fft_batch = np.asarray(
                densities_fft[batch_indices],
                dtype=np.float32,
            )
            self._check_finite("structures", structure_batch)
            self._check_finite("dft_densities", density_fft_batch)
            for batch_offset, source_idx in enumerate(batch_indices.tolist()):
                cache_index = batch_start + batch_offset
                sample_file = f"{cache_index:010d}.pkl"
                field = self.field_converter(
                    density_fft_batch[batch_offset],
                    grid=self.grid_data,
                    atom_numbers=atomic_numbers,
                    atom_coord=structure_batch[batch_offset],
                )
                graph = graph_converter.from_arrays(
                    atomic_numbers,
                    structure_batch[batch_offset],
                    coordinate_unit=self.field_converter.coordinate_unit,
                    node_features={"x": atom_type.copy()},
                )
                if graph is None:
                    raise ValueError(
                        "The graph converter failed to build a graph for "
                        f"{self.mol_name}_{self.source_split}_{source_idx:06d}."
                    )
                sample = {
                    "graph": graph,
                    "density": np.asarray(field.flat, dtype=np.float32),
                    "shape": list(field.grid.shape),
                    "file_name": (
                        f"{self.mol_name}_{self.source_split}_{source_idx:06d}"
                    ),
                    "source_index": source_idx,
                    "source_split": self.source_split,
                }
                sample_path = osp.join(sample_dir, sample_file)
                self.save_to_cache(sample_path, sample)
                sample_files.append(sample_file)
                sample_sizes.append(osp.getsize(sample_path))

        manifest = {
            "num_samples": len(source_indices),
            "sample_files": sample_files,
            "sample_sizes": sample_sizes,
            "source_indices": source_indices.tolist(),
        }
        self.save_to_cache(osp.join(cache_path, "manifest.pkl"), manifest)
        if not is_equal(self._cache_config(), expected_cfg):
            raise RuntimeError("Source data changed while the cache was being built.")
        with open(osp.join(cache_path, "completed.flag"), "w") as file:
            file.write("done\n")
        logger.info(f"Cached {len(source_indices)} samples to {self.cache_path}")

    def _validate_raw_layout(
        self, structures: np.ndarray, densities_fft: np.ndarray
    ) -> None:
        atom_count = len(self.atom_types)
        if structures.ndim != 3 or structures.shape[1:] != (atom_count, 3):
            raise ValueError(
                "structures.npy must have shape "
                f"(num_samples, {atom_count}, 3), got {structures.shape}."
            )
        if not np.issubdtype(structures.dtype, np.number) or np.iscomplexobj(
            structures
        ):
            raise TypeError("structures.npy must contain real numeric values.")

        if densities_fft.ndim < 2:
            raise ValueError("dft_densities.npy must have a leading sample dimension.")
        if densities_fft.shape[0] != structures.shape[0]:
            raise ValueError(
                "structures.npy and dft_densities.npy contain different sample "
                f"counts: {structures.shape[0]} and {densities_fft.shape[0]}."
            )
        values_per_sample = int(np.prod(densities_fft.shape[1:]))
        if values_per_sample != self.n_grid**3:
            raise ValueError(
                "Each FFT density must contain "
                f"{self.n_grid**3} values, got {values_per_sample}."
            )
        if not np.issubdtype(densities_fft.dtype, np.number) or np.iscomplexobj(
            densities_fft
        ):
            raise TypeError(
                "dft_densities.npy must contain real encoded FFT coefficients."
            )

    def _select_sample_indices(self, num_samples: int) -> np.ndarray:
        if num_samples <= 0:
            raise ValueError(f"The {self.source_split} source split is empty.")
        return np.arange(num_samples, dtype="int64")

    @staticmethod
    def _check_finite(name: str, values: np.ndarray) -> None:
        batch_size = 64
        for start in range(0, values.shape[0], batch_size):
            if not np.isfinite(values[start : start + batch_size]).all():
                raise ValueError(f"{name} contains non-finite values.")

    def _generate_grid(self) -> np.ndarray:
        return np.asarray(
            self.grid_data.cartesian_coordinates(),
            dtype=np.float32,
        )

    def save_to_cache(self, cache_path: str, data: Any) -> None:
        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_from_cache(self, cache_path: str) -> Any:
        if not osp.exists(cache_path):
            raise FileNotFoundError(f"No such file or directory: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.load_from_cache(self.samples[idx])
        density = np.asarray(sample["density"], dtype=np.float32)

        info = {
            "cell": self.cell,
            "shape": sample["shape"],
            "origin": self.origin,
            "file_name": sample["file_name"],
            "coordinate_unit": self.field_converter.coordinate_unit,
            "density_unit": self.field_converter.value_unit,
            "source_split": sample["source_split"],
        }

        data = {
            "graph": sample["graph"],
            "density": density,
            "grid_coord": self.grid_coord,
            "info": info,
            "id": sample["source_index"],
        }
        if self.grid_sampler is not None:
            identity = (
                data["info"].get("source_split", self.source_split),
                data["id"],
                data["info"].get("file_name"),
            )
            data = self.grid_sampler(data, identity)
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self) -> int:
        return self.num_samples


class QM9DensityDataset(DensityDataset):
    """QM9 electron-density release used by field prediction models.

    Args:
        path: Path to the split manifest. The published dataset is downloaded
            automatically when this path is missing.
        split: One of ``train``, ``validation``, or ``test``.
        vocab: Vocabularies used to encode atom types.
        **kwargs: Cache and transform options accepted by
            :class:`DensityDataset`.
    """

    name = "qm9_es"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/QM9_ES/qm9_es.tar"
    # BCE does not currently publish a content MD5 for this archive.
    md5 = None
    split_url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/QM9_ES/qm9_data_split.json"
    split_md5 = "a7547fe43cc1b36348bbd4c8d1a18b2a"
    split_filename = "qm9_data_split.json"

    def read_data(self, path: str | os.PathLike[str]):
        with open(path) as file_obj:
            split_data = json.load(file_obj)
        entries = list(split_data[self.split])
        file_names = [f"{int(entry) + 1:06d}.CHGCAR.lz4" for entry in entries]
        data = {
            "id": entries,
            "file_name": file_names,
            "field_source": [
                osp.join(osp.dirname(path), file_name) for file_name in file_names
            ],
        }
        return data, len(entries)

    def __init__(
        self,
        path: str | os.PathLike[str] = "./data/qm9_data_split.json",
        split: str = "train",
        *,
        vocab,
        build_field_cfg: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if build_field_cfg is None:
            build_field_cfg = {
                "format": "chgcar",
                "name": "density",
                "value_unit": "electron/angstrom^3",
                "coordinate_unit": "angstrom",
                "num_cpus": 1,
            }
        super().__init__(
            path=path,
            split=split,
            vocab=vocab,
            build_field_cfg=build_field_cfg,
            **kwargs,
        )


class MPCubicDensityDataset(DensityDataset):
    """Materials Project cubic electron-density release.

    Args:
        path: Path to the split manifest. The published dataset is downloaded
            automatically when this path is missing.
        split: One of ``train``, ``validation``, or ``test``.
        vocab: Vocabularies used to encode atom types.
        **kwargs: Cache and transform options accepted by
            :class:`DensityDataset`.
    """

    name = "mp_es_cubic"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/mp_es.tar"
    # BCE does not currently publish a content MD5 for this archive.
    md5 = None
    split_url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/crystal_data_split.json"
    split_md5 = "3f19bd6bce7d4f10ace1f80f202b1aa5"
    split_filename = "crystal_data_split.json"

    def read_data(self, path: str | os.PathLike[str]):
        with open(path) as file_obj:
            split_data = json.load(file_obj)
        entries = list(split_data[self.split])
        file_names = [f"{entry}.json.xz" for entry in entries]
        data = {
            "id": entries,
            "file_name": file_names,
            "field_source": [
                osp.join(osp.dirname(path), file_name) for file_name in file_names
            ],
        }
        return data, len(entries)

    def __init__(
        self,
        path: str | os.PathLike[str] = "./data/crystal_data_split.json",
        split: str = "train",
        *,
        vocab,
        build_field_cfg: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if build_field_cfg is None:
            build_field_cfg = {
                "format": "json",
                "name": "density",
                "value_unit": "electron/angstrom^3",
                "coordinate_unit": "angstrom",
                "num_cpus": 1,
            }
        super().__init__(
            path=path,
            split=split,
            vocab=vocab,
            build_field_cfg=build_field_cfg,
            **kwargs,
        )


class OMol25MC5kDensityDataset(DensityDataset):
    """OMol25 metal-complex electron-density release with 4,929 samples.

    Args:
        path: Path to the split manifest. The published dataset is downloaded
            automatically when this path is missing.
        split: One of ``train``, ``validation``, or ``test``.
        vocab: Vocabularies used to encode atom types.
        **kwargs: Cache and transform options accepted by
            :class:`DensityDataset`.
    """

    name = "dataset_OMol25_MC_5k"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OMol25_ES/MC_5k/omol25_mc_5k.tar"
    # BCE does not currently publish a content MD5 for this archive.
    md5 = None
    split_md5 = "4fbc298ab48e34ee44c112567b69a927"
    split_filename = "omol25_data_split.json"

    def read_data(self, path: str | os.PathLike[str]):
        with open(path) as file_obj:
            split_data = json.load(file_obj)
        entries = list(split_data[self.split])
        file_names = [f"{int(entry):06d}.cube.lz4" for entry in entries]
        data = {
            "id": entries,
            "file_name": file_names,
            "field_source": [
                osp.join(osp.dirname(path), file_name) for file_name in file_names
            ],
        }
        return data, len(entries)

    def __init__(
        self,
        path: str | os.PathLike[str] = "./data/omol25_data_split.json",
        split: str = "train",
        *,
        vocab,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            path=path,
            split=split,
            vocab=vocab,
            **kwargs,
        )


class OMol25MC5kTrimmedDensityDataset(OMol25MC5kDensityDataset):
    """Filtered OMol25 MC 5k split used by the InfGCN configuration.

    The raw archive is shared with :class:`OMol25MC5kDensityDataset`; only the
    split manifest differs.
    """

    name = "dataset_OMol25_MC_5k"
    split_filename = "omol25_mc_5k_trimmed_split.json"
    split_url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OMol25/omol25_mc_5k_trimmed_split.json"
    split_md5 = "03f7c71cf9ed448ee476c6ce47042bea"

    def __init__(
        self,
        path: str | os.PathLike[str] = ("./data/omol25_mc_5k_trimmed_split.json"),
        split: str = "train",
        *,
        vocab,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            path=path,
            split=split,
            vocab=vocab,
            **kwargs,
        )
