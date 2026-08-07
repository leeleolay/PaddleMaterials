import inspect
import json
import lzma
import pickle
from pathlib import Path

import numpy as np
import pytest

import ppmat.datasets.density_dataset as density_dataset_module
from ppmat.datasets.build_field import BuildField
from ppmat.datasets.density_dataset import MD17_ATOMIC_NUMBERS
from ppmat.datasets.density_dataset import DensityDataset
from ppmat.datasets.density_dataset import MD17DensityDataset
from ppmat.datasets.density_dataset import MPCubicDensityDataset
from ppmat.datasets.density_dataset import OMol25MC5kDensityDataset
from ppmat.datasets.density_dataset import OMol25MC5kTrimmedDensityDataset
from ppmat.datasets.density_dataset import QM9DensityDataset
from ppmat.utils import download
from ppmat.utils.io import write_cube


def _atom_vocab(
    symbols=("C", "H", "O"),
    atomic_numbers=(6, 1, 8),
):
    token_to_id = {symbol: index for index, symbol in enumerate(symbols)}
    atomic_number_to_id = {
        atomic_number: index for index, atomic_number in enumerate(atomic_numbers)
    }
    return {
        "atom": {
            "type": "element",
            "tokens": list(symbols),
            "num_embeddings": len(symbols),
            "token_to_id": token_to_id,
            "id_to_token": {index: symbol for symbol, index in token_to_id.items()},
            "atomic_number_to_id": atomic_number_to_id,
            "id_to_atomic_number": {
                index: atomic_number
                for atomic_number, index in atomic_number_to_id.items()
            },
        }
    }


def _field_converter_cfg(
    format,
    value_unit,
    coordinate_unit,
    name="density",
    num_cpus=1,
):
    return {
        "format": format,
        "name": name,
        "value_unit": value_unit,
        "coordinate_unit": coordinate_unit,
        "num_cpus": num_cpus,
    }


def _graph_converter_cfg(coordinate_unit, cutoff=3.0, num_cpus=1):
    return {
        "__class_name__": "RadiusGraphConverter",
        "__init_params__": {
            "cutoff": cutoff,
            "coordinate_unit": coordinate_unit,
            "inclusive_cutoff": True,
            "atom_vocab": {},
            "include_distance": False,
            "include_bond_vec": False,
            "return_triplet_indices": False,
            "num_cpus": num_cpus,
        },
    }


def _published_converter_kwargs(dataset_class):
    if issubclass(dataset_class, QM9DensityDataset):
        source_format = "chgcar"
        coordinate_unit = "angstrom"
        value_unit = "electron/angstrom^3"
    elif issubclass(dataset_class, MPCubicDensityDataset):
        source_format = "json"
        coordinate_unit = "angstrom"
        value_unit = "electron/angstrom^3"
    else:
        source_format = "cube"
        coordinate_unit = "bohr"
        value_unit = "unknown"
    return {
        "vocab": _atom_vocab(),
        "build_graph_cfg": _graph_converter_cfg(coordinate_unit),
        "build_field_cfg": _field_converter_cfg(
            source_format,
            value_unit,
            coordinate_unit,
        ),
    }


def _write_published_cache(
    root,
    split,
    dataset_class,
    source_id=0,
    manifest_path=None,
):
    params = _published_converter_kwargs(dataset_class)
    if manifest_path is None:
        manifest_path = Path(root) / dataset_class.split_filename
    manifest_path = Path(manifest_path)
    cache_path = Path(f"{manifest_path.parent}_cache") / f"{manifest_path.stem}_{split}"
    field_path = cache_path / "fields" / "0000000000.pkl"
    graph_path = cache_path / "graphs" / "0000000000.pkl"
    field_path.parent.mkdir(parents=True)
    graph_path.parent.mkdir(parents=True)
    with open(cache_path / "build_graph_cfg.pkl", "wb") as file_obj:
        pickle.dump(params["build_graph_cfg"], file_obj)
    with open(cache_path / "graph_vocab.pkl", "wb") as file_obj:
        pickle.dump({"atom": params["vocab"]["atom"]}, file_obj)
    with open(cache_path / "build_field_cfg.pkl", "wb") as file_obj:
        pickle.dump(params["build_field_cfg"], file_obj)
    field_builder = BuildField(
        format="array",
        name=params["build_field_cfg"]["name"],
        value_unit=params["build_field_cfg"]["value_unit"],
        coordinate_unit=params["build_field_cfg"]["coordinate_unit"],
    )
    grid = field_builder.build_grid(
        {
            "shape": (1, 1, 1),
            "voxel_vectors": np.eye(3),
            "origin": np.zeros(3),
        }
    )
    with open(field_path, "wb") as file_obj:
        pickle.dump(field_builder(np.zeros(1), grid=grid), file_obj)
    with open(graph_path, "wb") as file_obj:
        pickle.dump(None, file_obj)


def _write_cube_fixture(root, split_data):
    import lz4.frame

    root.mkdir()
    (root / "split.json").write_text(json.dumps(split_data))
    for source_id in {item for values in split_data.values() for item in values}:
        if not isinstance(source_id, (int, float)):
            continue
        plain_path = root / f"{source_id + 1:06d}.cube"
        compressed_path = root / f"{source_id + 1:06d}.cube.lz4"
        write_cube(
            plain_path,
            atom_numbers=np.asarray([6], dtype=np.int64),
            atom_coord=np.zeros([1, 3], dtype=np.float32),
            density=np.arange(8, dtype=np.float32) + source_id,
            info={
                "shape": [2, 2, 2],
                "cell": np.eye(3, dtype=np.float32) * 2,
                "origin": np.asarray([0.25, 0.5, 0.75], dtype=np.float32),
                "coordinate_unit": "bohr",
            },
        )
        compressed_path.write_bytes(lz4.frame.compress(plain_path.read_bytes()))
        plain_path.unlink()


class _CubeDensityDataset(DensityDataset):
    def read_data(self, path):
        with open(path) as file_obj:
            split_data = json.load(file_obj)
        entries = list(split_data[self.split])
        file_names = [f"{int(entry) + 1:06d}.cube.lz4" for entry in entries]
        data = {
            "id": entries,
            "file_name": file_names,
            "field_source": [
                str(Path(path).parent / file_name) for file_name in file_names
            ],
        }
        return data, len(entries)


def _cube_dataset(root, vocab=None, **kwargs):
    params = dict(
        path=root / "split.json",
        split="test",
        vocab=vocab or _atom_vocab(),
        build_graph_cfg=_graph_converter_cfg("bohr"),
        build_field_cfg=_field_converter_cfg("cube", "unknown", "bohr"),
    )
    params.update(kwargs)
    return _CubeDensityDataset(**params)


def _unpack_density_sample(sample):
    assert set(sample) == {"graph", "density", "grid_coord", "info", "id"}
    return (
        sample["graph"],
        sample["density"],
        sample["grid_coord"],
        sample["info"],
    )


def _graph_x(graph):
    return np.asarray(graph.node_feat["x"])


def _graph_pos(graph):
    return np.asarray(graph.node_feat["cart_coords"])


def test_density_dataset_reads_compressed_cube_through_converters(tmp_path):
    root = tmp_path / "density"
    _write_cube_fixture(root, {"test": [0]})
    sample = _cube_dataset(root)[0]
    graph, density, grid_coord, info = _unpack_density_sample(sample)

    np.testing.assert_array_equal(_graph_x(graph), [0])
    np.testing.assert_allclose(_graph_pos(graph), np.zeros([1, 3]))
    np.testing.assert_array_equal(graph.node_feat["atom_types"], [6])
    assert np.asarray(graph.edges).shape == (0, 2)
    np.testing.assert_allclose(density, np.arange(8))
    np.testing.assert_allclose(grid_coord[0], [0.25, 0.5, 0.75])
    np.testing.assert_allclose(info["cell"], np.eye(3) * 2)
    assert info["coordinate_unit"] == "bohr"
    assert info["density_unit"] == "unknown"
    assert sample["id"] == 0


def test_density_dataset_parallel_cache_matches_serial_cache(tmp_path):
    root = tmp_path / "density"
    _write_cube_fixture(root, {"test": [0, 1]})

    serial = _cube_dataset(
        root,
        cache_path=tmp_path / "serial_cache",
        cache_num_workers=1,
    )
    parallel = _cube_dataset(
        root,
        cache_path=tmp_path / "parallel_cache",
        cache_num_workers=2,
    )

    assert len(serial) == len(parallel) == 2
    for index in range(2):
        serial_sample = serial[index]
        parallel_sample = parallel[index]
        assert serial_sample["id"] == parallel_sample["id"]
        np.testing.assert_array_equal(
            np.asarray(serial_sample["graph"].edges),
            np.asarray(parallel_sample["graph"].edges),
        )
        np.testing.assert_array_equal(
            _graph_x(serial_sample["graph"]),
            _graph_x(parallel_sample["graph"]),
        )
        np.testing.assert_allclose(serial_sample["density"], parallel_sample["density"])
        np.testing.assert_allclose(
            serial_sample["grid_coord"], parallel_sample["grid_coord"]
        )


@pytest.mark.parametrize("cache_num_workers", [0, -1, True, 1.5])
def test_density_dataset_rejects_invalid_cache_num_workers(tmp_path, cache_num_workers):
    root = tmp_path / "density"
    _write_cube_fixture(root, {"test": [0]})

    with pytest.raises(ValueError, match="cache_num_workers"):
        _cube_dataset(root, cache_num_workers=cache_num_workers)


def test_density_dataset_passes_atom_vocabulary_to_graph_converter(tmp_path):
    root = tmp_path / "density"
    _write_cube_fixture(root, {"test": [0]})
    vocab = _atom_vocab(
        symbols=("C", "Cl"),
        atomic_numbers=(6, 17),
    )

    dataset = _cube_dataset(root, vocab=vocab)

    assert dataset.vocab is vocab
    np.testing.assert_array_equal(_graph_x(dataset[0]["graph"]), [0])


def test_density_dataset_rebuilds_graph_cache_when_vocab_changes(tmp_path):
    root = tmp_path / "density"
    cache_path = tmp_path / "cache"
    _write_cube_fixture(root, {"test": [0]})
    _cube_dataset(
        root,
        cache_path=cache_path,
        vocab=_atom_vocab(
            symbols=("C", "Cl"),
            atomic_numbers=(6, 17),
        ),
    )

    changed_vocab = _atom_vocab(
        symbols=("Cl", "C"),
        atomic_numbers=(17, 6),
    )
    changed = _cube_dataset(
        root,
        cache_path=cache_path,
        vocab=changed_vocab,
    )

    np.testing.assert_array_equal(_graph_x(changed[0]["graph"]), [1])
    with open(cache_path / "graph_vocab.pkl", "rb") as file_obj:
        assert pickle.load(file_obj) == {"atom": changed_vocab["atom"]}


def test_density_dataset_rebuilds_legacy_cache_without_graph_vocab(tmp_path):
    root = tmp_path / "density"
    cache_path = tmp_path / "cache"
    _write_cube_fixture(root, {"test": [0]})
    _cube_dataset(root, cache_path=cache_path)
    (cache_path / "graph_vocab.pkl").unlink()

    reloaded = _cube_dataset(root, cache_path=cache_path)

    np.testing.assert_array_equal(_graph_x(reloaded[0]["graph"]), [0])
    assert (cache_path / "graph_vocab.pkl").exists()


def test_density_dataset_uses_default_field_config(tmp_path):
    root = tmp_path / "density"
    _write_cube_fixture(root, {"test": [0]})

    dataset = _cube_dataset(root, build_field_cfg=None)

    assert dataset.build_field_cfg == _field_converter_cfg(
        "cube",
        "unknown",
        "bohr",
    )


def test_density_dataset_accepts_configured_field_name(tmp_path):
    root = tmp_path / "density"
    _write_cube_fixture(root, {"test": [0]})

    dataset = _cube_dataset(
        root,
        build_field_cfg=_field_converter_cfg(
            "cube",
            "unknown",
            "bohr",
            name="charge_density",
        ),
    )

    assert dataset[0]["density"].shape == (8,)


@pytest.mark.parametrize(
    ("config_name", "invalid_config", "error", "message"),
    [
        (
            "build_field_cfg",
            {
                "name": "density",
                "value_unit": "unknown",
                "coordinate_unit": "bohr",
                "num_cpus": 1,
            },
            TypeError,
            "format",
        ),
        (
            "build_field_cfg",
            _field_converter_cfg("cube", "unknown", "meter"),
            ValueError,
            "coordinate_unit",
        ),
        (
            "build_field_cfg",
            {
                "coordinate_unit": "bohr",
                "__class_name__": "BuildField",
                "__init_params__": {
                    "format": "cube",
                    "name": "density",
                    "value_unit": "unknown",
                    "coordinate_unit": "bohr",
                },
            },
            TypeError,
            "__class_name__",
        ),
    ],
)
def test_density_dataset_rejects_invalid_converter_configs(
    tmp_path,
    config_name,
    invalid_config,
    error,
    message,
):
    root = tmp_path / "density"
    _write_cube_fixture(root, {"test": [0]})

    with pytest.raises(error, match=message):
        _cube_dataset(root, **{config_name: invalid_config})


def test_density_dataset_no_longer_accepts_density_unit(tmp_path):
    root = tmp_path / "density"
    _write_cube_fixture(root, {"test": [0]})

    with pytest.raises(TypeError, match="density_unit"):
        _cube_dataset(root, density_unit="unknown")


@pytest.mark.parametrize(
    "parameter",
    [
        "auto_download",
        "url",
        "md5",
        "max_cache_samples",
        "mol_name",
        "pbc",
        "root",
        "rotate",
    ],
)
def test_density_dataset_uses_mp20_style_init_contract(parameter):
    assert parameter not in inspect.signature(DensityDataset.__init__).parameters
    assert parameter not in inspect.signature(MD17DensityDataset.__init__).parameters


def test_density_dataset_rejects_grid_unit_mismatching_cube_source(tmp_path):
    root = tmp_path / "density"
    _write_cube_fixture(root, {"test": [0]})

    with pytest.raises(ValueError, match="coordinate_unit"):
        _cube_dataset(
            root,
            build_field_cfg=_field_converter_cfg(
                "cube",
                "unknown",
                "angstrom",
            ),
        )


def test_density_dataset_passes_field_sources_to_builder(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "density"
    _write_cube_fixture(root, {"test": [0, 1]})
    field_sources = []

    class TrackingBuildField(BuildField):
        def __call__(self, *args, **kwargs):
            assert "grid" not in kwargs
            field_sources.append(args[0])
            return super().__call__(*args, **kwargs)

    monkeypatch.setattr(density_dataset_module, "BuildField", TrackingBuildField)

    dataset = _cube_dataset(root)
    dataset[0]
    dataset[1]

    assert field_sources == dataset.row_data["field_source"]
    assert not hasattr(DensityDataset, "read_density_data")


@pytest.mark.parametrize(
    ("dataset_class", "split_entry", "expected_file"),
    [
        (
            QM9DensityDataset,
            0,
            "000001.CHGCAR.lz4",
        ),
        (
            MPCubicDensityDataset,
            "mp-1",
            "mp-1.json.xz",
        ),
        (
            OMol25MC5kDensityDataset,
            1,
            "000001.cube.lz4",
        ),
        (
            OMol25MC5kTrimmedDensityDataset,
            1,
            "000001.cube.lz4",
        ),
    ],
)
def test_published_density_dataset_contracts(
    tmp_path,
    dataset_class,
    split_entry,
    expected_file,
):
    pytest.importorskip("lz4.frame")
    root = tmp_path / dataset_class.name
    root.mkdir()
    (root / "qm9_data_split.json").write_text(json.dumps({"train": [split_entry]}))
    (root / "crystal_data_split.json").write_text(json.dumps({"train": [split_entry]}))
    (root / "omol25_data_split.json").write_text(json.dumps({"train": [split_entry]}))
    (root / "omol25_mc_5k_trimmed_split.json").write_text(
        json.dumps({"train": [split_entry]})
    )
    (root / expected_file).touch()

    dataset = dataset_class.__new__(dataset_class)
    dataset.split = "train"
    row_data, num_samples = dataset.read_data(root / dataset_class.split_filename)

    assert row_data["file_name"] == [expected_file]
    assert row_data["field_source"] == [str(root / expected_file)]
    assert num_samples == 1


def test_qm9_download_uses_shared_download_utils(tmp_path, monkeypatch):
    pytest.importorskip("lz4.frame")
    datasets_home = tmp_path / "datasets"
    extraction_root = datasets_home / "qm9_es"
    canonical_root = extraction_root / "qm9_es"
    requested_root = tmp_path / "missing_qm9"
    raw_downloads = 0
    metadata_downloads = 0

    monkeypatch.setattr(download, "DATASETS_HOME", str(datasets_home))
    monkeypatch.setattr(QM9DensityDataset, "split_md5", None)

    def download_archive(url, md5):
        nonlocal raw_downloads
        raw_downloads += 1
        canonical_root.mkdir(parents=True)
        (canonical_root / "qm9_data_split.json").write_text(
            json.dumps({"train": [0], "validation": [1], "test": [2]})
        )
        (canonical_root / "000001.CHGCAR.lz4").touch()
        _write_published_cache(canonical_root, "train", QM9DensityDataset)
        return str(extraction_root)

    def download_metadata(url, root_dir, md5sum, decompress):
        nonlocal metadata_downloads
        metadata_downloads += 1
        assert decompress is False
        path = canonical_root / url.rsplit("/", 1)[-1]
        path.write_text(
            json.dumps(
                {
                    "train": [0],
                    "validation": [1],
                    "test": [2],
                }
            )
        )
        (canonical_root / "000001.CHGCAR.lz4").touch()
        (canonical_root / "000002.CHGCAR.lz4").touch()
        _write_published_cache(
            canonical_root,
            "train",
            QM9DensityDataset,
        )
        return str(path)

    monkeypatch.setattr(download, "get_datasets_path_from_url", download_archive)
    monkeypatch.setattr(download, "get_path_from_url", download_metadata)

    train = QM9DensityDataset(
        path=requested_root / "qm9_data_split.json",
        split="train",
        **_published_converter_kwargs(QM9DensityDataset),
    )

    assert Path(train.path).parent == canonical_root
    assert raw_downloads == 1
    assert metadata_downloads == 0


def test_qm9_download_passes_metadata_checksums(tmp_path, monkeypatch):
    pytest.importorskip("lz4.frame")
    datasets_home = tmp_path / "datasets"
    extraction_root = datasets_home / "qm9_es"
    canonical_root = extraction_root / "qm9_es"
    received_checksums = {}

    monkeypatch.setattr(download, "DATASETS_HOME", str(datasets_home))

    def download_archive(url, md5):
        canonical_root.mkdir(parents=True)
        assert md5 is QM9DensityDataset.md5
        (canonical_root / "qm9_data_split.json").write_text(json.dumps({"train": [0]}))
        (canonical_root / "000001.CHGCAR.lz4").touch()
        _write_published_cache(canonical_root, "train", QM9DensityDataset)
        return str(extraction_root)

    def download_metadata(url, root_dir, md5sum, decompress):
        assert decompress is False
        path = canonical_root / url.rsplit("/", 1)[-1]
        received_checksums[path.name] = md5sum
        path.write_text(json.dumps({"train": [0]}))
        (canonical_root / "000001.CHGCAR.lz4").touch()
        _write_published_cache(
            canonical_root,
            "train",
            QM9DensityDataset,
        )
        return str(path)

    monkeypatch.setattr(download, "get_datasets_path_from_url", download_archive)
    monkeypatch.setattr(download, "get_path_from_url", download_metadata)

    QM9DensityDataset(
        path=tmp_path / "missing_qm9" / "qm9_data_split.json",
        split="train",
        **_published_converter_kwargs(QM9DensityDataset),
    )

    assert received_checksums == {}


def test_qm9_uses_existing_published_metadata_without_download(tmp_path, monkeypatch):
    pytest.importorskip("lz4.frame")
    root = tmp_path / "qm9"
    root.mkdir()
    (root / "qm9_data_split.json").write_text(json.dumps({"train": [0]}))
    (root / "000001.CHGCAR.lz4").touch()
    _write_published_cache(root, "train", QM9DensityDataset)

    def fail_download(*args, **kwargs):
        raise AssertionError("Existing dataset metadata must be used directly.")

    monkeypatch.setattr(download, "get_path_from_url", fail_download)

    dataset = QM9DensityDataset(
        path=root / "qm9_data_split.json",
        split="train",
        **_published_converter_kwargs(QM9DensityDataset),
    )

    assert len(dataset) == 1


def test_missing_downloaded_density_files_raise(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("lz4.frame")
    datasets_home = tmp_path / "datasets"
    canonical_root = datasets_home / "qm9_es" / "qm9_es"
    monkeypatch.setattr(download, "DATASETS_HOME", str(datasets_home))
    monkeypatch.setattr(QM9DensityDataset, "split_md5", None)

    def download_archive(url, md5):
        canonical_root.mkdir(parents=True)
        (canonical_root / "qm9_data_split.json").write_text(json.dumps({"train": [0]}))
        return str(datasets_home / "qm9_es")

    def download_metadata(url, root_dir, md5sum, decompress):
        path = canonical_root / url.rsplit("/", 1)[-1]
        path.write_text(json.dumps({"train": [0]}))
        return str(path)

    monkeypatch.setattr(download, "get_datasets_path_from_url", download_archive)
    monkeypatch.setattr(download, "get_path_from_url", download_metadata)

    with pytest.raises(FileNotFoundError, match="No such file"):
        QM9DensityDataset(
            path=tmp_path / "missing_qm9" / "qm9_data_split.json",
            split="train",
            **_published_converter_kwargs(QM9DensityDataset),
        )


def test_omol25_full_and_trimmed_splits_share_downloaded_archive(tmp_path, monkeypatch):
    pytest.importorskip("lz4.frame")
    datasets_home = tmp_path / "datasets"
    extraction_root = datasets_home / "omol25_mc_5k"
    canonical_root = extraction_root / "dataset_OMol25_MC_5k"
    requested_root = tmp_path / "missing_omol25"
    raw_downloads = 0
    metadata_downloads = 0

    monkeypatch.setattr(download, "DATASETS_HOME", str(datasets_home))
    monkeypatch.setattr(OMol25MC5kDensityDataset, "split_md5", None)
    monkeypatch.setattr(OMol25MC5kTrimmedDensityDataset, "split_md5", None)

    def download_archive(url, md5):
        nonlocal raw_downloads
        raw_downloads += 1
        canonical_root.mkdir(parents=True)
        (canonical_root / "omol25_data_split.json").write_text(
            json.dumps({"train": [1]})
        )
        (canonical_root / "omol25_mc_5k_trimmed_split.json").write_text(
            json.dumps({"train": [1]})
        )
        (canonical_root / "000001.cube.lz4").touch()
        _write_published_cache(
            canonical_root,
            "train",
            OMol25MC5kDensityDataset,
            source_id=1,
        )
        _write_published_cache(
            canonical_root,
            "train",
            OMol25MC5kTrimmedDensityDataset,
            source_id=1,
        )
        return str(extraction_root)

    def download_metadata(url, root_dir, md5sum, decompress):
        nonlocal metadata_downloads
        metadata_downloads += 1
        assert decompress is False
        assert root_dir == str(canonical_root)
        assert url == OMol25MC5kTrimmedDensityDataset.split_url
        path = canonical_root / "omol25_mc_5k_trimmed_split.json"
        path.write_text(json.dumps({"train": [1]}))
        return str(path)

    monkeypatch.setattr(download, "get_datasets_path_from_url", download_archive)
    monkeypatch.setattr(download, "get_path_from_url", download_metadata)

    full = OMol25MC5kDensityDataset(
        path=requested_root / "omol25_data_split.json",
        split="train",
        **_published_converter_kwargs(OMol25MC5kDensityDataset),
    )
    trimmed = OMol25MC5kTrimmedDensityDataset(
        path=Path(full.path).parent / "omol25_mc_5k_trimmed_split.json",
        split="train",
        **_published_converter_kwargs(OMol25MC5kTrimmedDensityDataset),
    )

    assert Path(full.path).parent == Path(trimmed.path).parent == canonical_root
    assert raw_downloads == 1
    assert metadata_downloads == 0


def test_published_density_dataset_builders_use_registered_class(tmp_path):
    pytest.importorskip("lz4.frame")
    from ppmat.datasets import build_dataloader
    from ppmat.predictor import FieldPredictor

    root = tmp_path / "omol25"
    root.mkdir()
    (root / "omol25_data_split.json").write_text(json.dumps({"train": [1]}))
    (root / "000001.cube.lz4").touch()
    _write_published_cache(
        root,
        "train",
        OMol25MC5kDensityDataset,
        source_id=1,
    )
    dataset_params = {
        "path": root / "omol25_data_split.json",
        "split": "train",
        **_published_converter_kwargs(OMol25MC5kDensityDataset),
    }
    loader = build_dataloader(
        {
            "dataset": {
                "__class_name__": "OMol25MC5kDensityDataset",
                "__init_params__": dataset_params,
            },
            "sampler": {
                "__class_name__": "BatchSampler",
                "__init_params__": {
                    "shuffle": False,
                    "drop_last": False,
                    "batch_size": 1,
                },
            },
            "loader": {
                "num_workers": 0,
                "use_shared_memory": False,
                "collate_fn": "DensityCollator",
            },
        }
    )
    predictor_dataset = FieldPredictor._build_dataset(
        {"__class_name__": "OMol25MC5kDensityDataset"},
        dataset_params,
    )

    assert isinstance(loader.dataset, OMol25MC5kDensityDataset)
    assert isinstance(predictor_dataset, OMol25MC5kDensityDataset)


@pytest.mark.parametrize(
    ("dataset_class", "source_id", "density_filename"),
    [
        (QM9DensityDataset, 0, "000001.CHGCAR.lz4"),
        (OMol25MC5kTrimmedDensityDataset, 1, "000001.cube.lz4"),
    ],
)
def test_published_density_datasets_accept_local_metadata_overrides(
    tmp_path,
    dataset_class,
    source_id,
    density_filename,
):
    pytest.importorskip("lz4.frame")
    root = tmp_path / dataset_class.name
    root.mkdir()
    split_file = tmp_path / f"{dataset_class.name}_split.json"
    split_file.write_text(json.dumps({"test": [source_id]}))
    (root / density_filename).touch()
    _write_published_cache(
        root,
        "test",
        dataset_class,
        source_id=source_id,
        manifest_path=split_file,
    )

    dataset = dataset_class(
        path=split_file,
        split="test",
        **_published_converter_kwargs(dataset_class),
    )

    assert dataset.path == split_file


def test_density_dataset_rejects_incomplete_existing_root(tmp_path):
    pytest.importorskip("lz4.frame")
    root = tmp_path / "incomplete_qm9"
    root.mkdir()
    (root / "split.json").write_text(json.dumps({"train": [0]}))

    with pytest.raises(FileNotFoundError, match="No such file"):
        QM9DensityDataset(
            path=root / "split.json",
            split="train",
            **_published_converter_kwargs(QM9DensityDataset),
        )


def test_field_predictor_discovers_new_density_dataset_subclasses(monkeypatch):
    import ppmat.datasets as datasets
    from ppmat.predictor import FieldPredictor

    class FutureDensityDataset(DensityDataset):
        def __init__(self, marker):
            self.marker = marker

    monkeypatch.setattr(
        datasets,
        "FutureDensityDataset",
        FutureDensityDataset,
        raising=False,
    )

    dataset = FieldPredictor._build_dataset(
        {"__class_name__": "FutureDensityDataset"},
        {"marker": "registered-once"},
    )

    assert isinstance(dataset, FutureDensityDataset)
    assert dataset.marker == "registered-once"
    with pytest.raises(ValueError, match="Unsupported field dataset class"):
        FieldPredictor._build_dataset(
            {"__class_name__": "QM9Dataset"},
            {},
        )


def test_density_dataset_cache_reload_preserves_metadata(tmp_path):
    pytest.importorskip("lz4.frame")
    root = tmp_path / "density"
    _write_cube_fixture(root, {"test": [0]})

    raw = _cube_dataset(root)
    cached = _cube_dataset(
        root,
        cache_path=tmp_path / "cache",
    )
    reloaded = _cube_dataset(
        root,
        cache_path=tmp_path / "cache",
    )

    assert (
        cached.fields
        == reloaded.fields
        == [str(tmp_path / "cache" / "fields" / "0000000000.pkl")]
    )
    assert (
        cached.graphs
        == reloaded.graphs
        == [str(tmp_path / "cache" / "graphs" / "0000000000.pkl")]
    )
    cached_field = cached.load_from_cache(cached.fields[0])
    assert cached_field.grid.shape == (2, 2, 2)
    assert cached_field.structure is not None
    assert cached.load_from_cache(cached.graphs[0]).num_nodes == 1

    raw_sample = raw[0]
    for sample in (raw_sample, cached[0], reloaded[0]):
        for value in (
            _graph_x(sample["graph"]),
            _graph_pos(sample["graph"]),
            sample["density"],
            sample["grid_coord"],
            sample["info"]["cell"],
            sample["info"]["origin"],
        ):
            assert isinstance(value, np.ndarray)
        assert _graph_x(sample["graph"]).dtype == np.int64
        assert _graph_pos(sample["graph"]).dtype == np.float32
        assert sample["density"].dtype == np.float32
        assert sample["grid_coord"].dtype == np.float32
        assert sample["info"]["cell"].dtype == np.float32
        np.testing.assert_array_equal(
            _graph_x(sample["graph"]),
            _graph_x(raw_sample["graph"]),
        )
        np.testing.assert_allclose(
            _graph_pos(sample["graph"]),
            _graph_pos(raw_sample["graph"]),
        )
        np.testing.assert_allclose(sample["density"], raw_sample["density"])
        np.testing.assert_allclose(sample["grid_coord"], raw_sample["grid_coord"])
        np.testing.assert_allclose(sample["info"]["origin"], [0.25, 0.5, 0.75])
        assert sample["id"] == 0
        assert sample["info"]["file_name"] == "000001.cube.lz4"
        assert sample["info"]["density_unit"] == "unknown"


def test_density_dataset_cache_hit_skips_converter_rebuild_and_value_scan(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("lz4.frame")
    root = tmp_path / "density"
    cache_path = tmp_path / "cache"
    _write_cube_fixture(root, {"test": [0]})
    _cube_dataset(
        root,
        cache_path=cache_path,
    )

    original_isfinite = np.isfinite
    scanned_sizes = []
    info_messages = []

    def track_isfinite(value, *args, **kwargs):
        scanned_sizes.append(np.asarray(value).size)
        return original_isfinite(value, *args, **kwargs)

    def fail_converter_rebuild(*args, **kwargs):
        raise AssertionError("A cache hit must not rebuild the field.")

    with monkeypatch.context() as patch:
        patch.setattr(BuildField, "__call__", fail_converter_rebuild)
        patch.setattr(
            density_dataset_module,
            "build_graph_converter",
            fail_converter_rebuild,
        )
        patch.setattr(
            density_dataset_module.logger,
            "info",
            lambda message: info_messages.append(message),
        )
        patch.setattr(density_dataset_module.np, "isfinite", track_isfinite)
        reloaded = _cube_dataset(
            root,
            cache_path=cache_path,
        )
        sample = reloaded[0]
        graph, density, grid_coord, info = _unpack_density_sample(sample)

    assert density.size == 8
    assert grid_coord.shape == (8, 3)
    assert sample["id"] == 0
    np.testing.assert_array_equal(_graph_x(graph), [0])
    assert 8 not in scanned_sizes
    assert any(
        "cached build_field_cfg configuration matches" in message
        for message in info_messages
    )
    assert any(
        "cached graph configuration and vocabulary match" in message
        for message in info_messages
    )


def test_density_dataset_rebuilds_cache_when_converter_config_changes(
    tmp_path,
):
    pytest.importorskip("lz4.frame")
    root = tmp_path / "density"
    cache_path = tmp_path / "cache"
    _write_cube_fixture(root, {"test": [0]})
    _cube_dataset(
        root,
        cache_path=cache_path,
    )
    changed = _cube_dataset(
        root,
        cache_path=cache_path,
        build_field_cfg=_field_converter_cfg(
            "cube",
            "electron/bohr^3",
            "bohr",
        ),
    )

    assert changed[0]["info"]["density_unit"] == "electron/bohr^3"


def test_density_dataset_rebuilds_cache_when_graph_config_changes(tmp_path):
    pytest.importorskip("lz4.frame")
    root = tmp_path / "density"
    cache_path = tmp_path / "cache"
    _write_cube_fixture(root, {"test": [0]})
    _cube_dataset(
        root,
        cache_path=cache_path,
    )
    changed_cfg = _graph_converter_cfg("bohr", cutoff=2.0)

    changed = _cube_dataset(
        root,
        cache_path=cache_path,
        build_graph_cfg=changed_cfg,
    )

    assert changed.build_graph_cfg == changed_cfg
    with open(cache_path / "build_graph_cfg.pkl", "rb") as file_obj:
        assert pickle.load(file_obj) == changed_cfg


def test_density_dataset_cache_identity_includes_num_cpus(
    tmp_path,
):
    pytest.importorskip("lz4.frame")
    root = tmp_path / "density"
    cache_path = tmp_path / "cache"
    _write_cube_fixture(root, {"test": [0]})
    _cube_dataset(
        root,
        cache_path=cache_path,
    )
    changed = _cube_dataset(
        root,
        cache_path=cache_path,
        build_field_cfg=_field_converter_cfg(
            "cube",
            "unknown",
            "bohr",
            num_cpus=2,
        ),
    )

    assert changed.build_field_cfg["num_cpus"] == 2

    reloaded = _cube_dataset(
        root,
        cache_path=cache_path,
        build_field_cfg=_field_converter_cfg(
            "cube",
            "unknown",
            "bohr",
            num_cpus=2,
        ),
    )

    assert reloaded.build_field_cfg["num_cpus"] == 2


def test_density_dataset_preserves_unrecognized_cache_directory(tmp_path):
    pytest.importorskip("lz4.frame")
    root = tmp_path / "density"
    _write_cube_fixture(root, {"test": [0]})
    cache_path = tmp_path / "unrelated"
    sample_dir = cache_path / "samples"
    sample_dir.mkdir(parents=True)
    sentinel = sample_dir / "keep.txt"
    sentinel.write_text("user data")

    dataset = _cube_dataset(
        root,
        cache_path=cache_path,
    )

    assert sentinel.read_text() == "user data"
    assert dataset[0]["id"] == 0


def test_density_dataset_preserves_split_order(tmp_path):
    root = tmp_path / "density"
    _write_cube_fixture(root, {"test": [2, 0, 1]})

    dataset = _cube_dataset(root)

    assert dataset.row_data["id"] == [2, 0, 1]
    assert dataset.row_data["file_name"] == [
        "000003.cube.lz4",
        "000001.cube.lz4",
        "000002.cube.lz4",
    ]


def test_density_dataset_reads_only_requested_split(tmp_path):
    root = tmp_path / "density"
    _write_cube_fixture(
        root,
        {"train": [0], "validation": [1], "test": [1]},
    )

    dataset = _cube_dataset(root)

    assert dataset.row_data["id"] == [1]
    assert dataset.row_data["file_name"] == ["000002.cube.lz4"]


def _write_md17_density_source(root, molecule="ethane", train_size=10, test_size=3):
    atom_count = len(MD17_ATOMIC_NUMBERS[molecule])
    for split, num_samples in (("train", train_size), ("test", test_size)):
        data_path = root / molecule / f"{molecule}_{split}"
        data_path.mkdir(parents=True)
        structures = np.arange(num_samples * atom_count * 3, dtype=np.float32).reshape(
            num_samples, atom_count, 3
        )
        if split == "test":
            structures += 10_000
        densities = np.arange(num_samples * 8, dtype=np.float32).reshape(num_samples, 8)
        np.save(data_path / "structures.npy", structures)
        np.save(data_path / "dft_densities.npy", densities)


def _md17_density_dataset(root, split, **kwargs):
    source_split = "test" if split == "validation" else split
    molecule = kwargs.pop("molecule", "ethane")
    params = dict(
        path=Path(root) / molecule / f"{molecule}_{source_split}",
        split=split,
        n_grid=2,
        grid_size=4.0,
        vocab=_atom_vocab(),
        build_graph_cfg=_graph_converter_cfg("bohr"),
    )
    params.update(kwargs)
    return MD17DensityDataset(**params)


def test_md17_density_download_uses_shared_download_utils(tmp_path, monkeypatch):
    datasets_home = tmp_path / "datasets"
    extraction_root = datasets_home / "md17_es.tar"
    canonical_root = extraction_root / "md17_es.tar"
    requested_root = tmp_path / "missing_md17"
    raw_downloads = 0

    monkeypatch.setattr(download, "DATASETS_HOME", str(datasets_home))

    def download_archive(url, md5):
        nonlocal raw_downloads
        raw_downloads += 1
        _write_md17_density_source(canonical_root)
        return str(canonical_root)

    monkeypatch.setattr(download, "get_datasets_path_from_url", download_archive)

    train = MD17DensityDataset(
        path=requested_root / "ethane" / "ethane_train",
        split="train",
        n_grid=2,
        grid_size=4.0,
        vocab=_atom_vocab(),
        build_graph_cfg=_graph_converter_cfg("bohr"),
    )
    validation = MD17DensityDataset(
        path=canonical_root / "ethane" / "ethane_test",
        split="validation",
        n_grid=2,
        grid_size=4.0,
        vocab=_atom_vocab(),
        build_graph_cfg=_graph_converter_cfg("bohr"),
    )

    assert train.path == str(canonical_root / "ethane" / "ethane_train")
    assert validation.path == str(canonical_root / "ethane" / "ethane_test")
    assert raw_downloads == 1


def test_md17_density_dataset_defaults_to_fft_field_config(tmp_path):
    root = tmp_path / "small"
    _write_md17_density_source(root)

    dataset = _md17_density_dataset(root, "train", build_field_cfg=None)

    assert dataset.build_field_cfg == {
        "format": "fft",
        "name": "density",
        "value_unit": "unknown",
        "coordinate_unit": "bohr",
        "num_cpus": 1,
    }


def test_md17_density_dataset_requires_graph_config(tmp_path):
    root = tmp_path / "small"
    _write_md17_density_source(root)

    with pytest.raises(TypeError, match="build_graph_cfg"):
        _md17_density_dataset(root, "train", build_graph_cfg=None)


def test_md17_converter_configs_are_required_keyword_only():
    signature = inspect.signature(MD17DensityDataset)

    for parameter_name in ("vocab", "build_graph_cfg"):
        parameter = signature.parameters[parameter_name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty
    build_field_param = signature.parameters["build_field_cfg"]
    assert build_field_param.kind is inspect.Parameter.KEYWORD_ONLY
    assert build_field_param.default is None


def test_md17_density_dataset_requires_explicit_converter_format(
    tmp_path,
):
    root = tmp_path / "small"
    _write_md17_density_source(root)

    with pytest.raises(TypeError, match="format"):
        _md17_density_dataset(
            root,
            "train",
            build_field_cfg={
                "name": "density",
                "value_unit": "unknown",
                "coordinate_unit": "bohr",
                "num_cpus": 1,
            },
        )


def test_md17_density_dataset_no_longer_accepts_density_unit(tmp_path):
    root = tmp_path / "small"
    _write_md17_density_source(root)

    with pytest.raises(TypeError, match="density_unit"):
        _md17_density_dataset(root, "train", density_unit="unknown")


def test_md17_density_dataset_rejects_grid_unit_mismatch(tmp_path):
    root = tmp_path / "small"
    _write_md17_density_source(root)

    with pytest.raises(ValueError, match="coordinate.unit"):
        _md17_density_dataset(
            root,
            "train",
            build_field_cfg=_field_converter_cfg(
                "fft",
                "unknown",
                "angstrom",
            ),
        )


def test_md17_density_dataset_routes_samples_through_field_builder(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "small"
    _write_md17_density_source(root, train_size=2)
    read_calls = []
    grid_calls = []
    field_calls = []
    original_read_data = MD17DensityDataset.read_data

    def tracking_read_data(self):
        read_calls.append(self.data_path)
        return original_read_data(self)

    class TrackingBuildField(BuildField):
        def build_grid(self, *args, **kwargs):
            grid_calls.append((args, kwargs))
            return super().build_grid(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            field_calls.append((args, kwargs))
            return super().__call__(*args, **kwargs)

    monkeypatch.setattr(density_dataset_module, "BuildField", TrackingBuildField)
    monkeypatch.setattr(MD17DensityDataset, "read_data", tracking_read_data)

    dataset = _md17_density_dataset(root, "train")

    assert len(dataset) == 2
    assert len(read_calls) == 1
    assert len(grid_calls) == 1
    assert len(field_calls) == 2


def test_md17_density_dataset_matches_reference_split_protocol(tmp_path):
    root = tmp_path / "small"
    _write_md17_density_source(root)

    train = _md17_density_dataset(root, "train")
    validation = _md17_density_dataset(root, "validation")
    test = _md17_density_dataset(root, "test")
    train_reloaded = _md17_density_dataset(root, "train")
    validation_rebuilt = _md17_density_dataset(
        root,
        "validation",
        cache_path=tmp_path / "independent_validation_cache",
    )

    assert len(train) == 10
    assert len(validation) == 3
    assert train.sample_ids == list(range(10))
    assert validation.sample_ids == test.sample_ids
    assert test.sample_ids == list(range(3))
    assert train_reloaded.sample_ids == train.sample_ids
    assert validation_rebuilt.sample_ids == validation.sample_ids

    sample = train[0]
    reloaded_sample = train_reloaded[0]
    for value in (
        _graph_x(sample["graph"]),
        _graph_pos(sample["graph"]),
        sample["density"],
        sample["grid_coord"],
        sample["info"]["cell"],
        sample["info"]["origin"],
    ):
        assert isinstance(value, np.ndarray)
    assert _graph_x(sample["graph"]).dtype == np.int64
    assert _graph_pos(sample["graph"]).dtype == np.float32
    np.testing.assert_array_equal(
        sample["graph"].node_feat["atom_types"].reshape(-1),
        MD17_ATOMIC_NUMBERS["ethane"],
    )
    np.testing.assert_array_equal(
        np.asarray(sample["graph"].edges),
        np.asarray(reloaded_sample["graph"].edges),
    )
    assert sample["density"].dtype == np.float32
    assert sample["grid_coord"].dtype == np.float32
    assert sample["info"]["cell"].dtype == np.float32
    np.testing.assert_allclose(sample["density"], reloaded_sample["density"])
    expected_grid = np.stack(
        np.meshgrid([2.0, 4.0], [2.0, 4.0], [2.0, 4.0], indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    np.testing.assert_allclose(sample["grid_coord"], expected_grid)
    np.testing.assert_allclose(sample["info"]["cell"], np.eye(3) * 4.0)
    np.testing.assert_allclose(sample["info"]["origin"], [2.0, 2.0, 2.0])
    assert sample["id"] == train.sample_ids[0]
    assert sample["info"]["source_split"] == "train"
    assert sample["info"]["coordinate_unit"] == "bohr"
    assert sample["info"]["density_unit"] == "unknown"
    np.testing.assert_allclose(validation[0]["density"], test[0]["density"])
    np.testing.assert_allclose(
        _graph_pos(validation[0]["graph"]),
        _graph_pos(test[0]["graph"]),
    )
    assert validation[0]["info"]["source_split"] == "test"
    assert _graph_pos(test[0]["graph"]).min().item() >= 10_000
    assert test[0]["info"]["source_split"] == "test"


def test_md17_validation_and_test_share_converted_cache(tmp_path, monkeypatch):
    root = tmp_path / "small"
    _write_md17_density_source(root)
    validation = _md17_density_dataset(root, "validation")

    def fail_if_rebuilt(*args, **kwargs):
        raise AssertionError("validation and test should share the source-test cache")

    monkeypatch.setattr(MD17DensityDataset, "_build_cache", fail_if_rebuilt)
    test = _md17_density_dataset(root, "test")

    assert validation.cache_path == test.cache_path
    assert validation.sample_ids == test.sample_ids


def test_md17_cache_hit_loads_grid_once_without_full_scan(tmp_path, monkeypatch):
    root = tmp_path / "small"
    _write_md17_density_source(root, train_size=2)
    cached = _md17_density_dataset(root, "train")
    grid_size = cached.grid_coord.size
    original_load = MD17DensityDataset.load_from_cache
    original_isfinite = np.isfinite
    grid_loads = []
    scanned_sizes = []

    def tracking_load(self, cache_path):
        if cache_path.endswith("grid.pkl"):
            grid_loads.append(cache_path)
        return original_load(self, cache_path)

    def tracking_isfinite(value, *args, **kwargs):
        scanned_sizes.append(np.asarray(value).size)
        return original_isfinite(value, *args, **kwargs)

    monkeypatch.setattr(MD17DensityDataset, "load_from_cache", tracking_load)
    monkeypatch.setattr(density_dataset_module.np, "isfinite", tracking_isfinite)

    reloaded = _md17_density_dataset(root, "train")
    sample = reloaded[0]

    assert len(grid_loads) == 1
    assert grid_size not in scanned_sizes
    assert sample["grid_coord"] is reloaded.grid_coord


def test_md17_cache_build_converts_bounded_batches(tmp_path, monkeypatch):
    root = tmp_path / "small"
    _write_md17_density_source(root, train_size=7)
    original_check_finite = MD17DensityDataset._check_finite
    batch_sizes = []
    log_messages = []

    def tracking_check_finite(name, values):
        if name == "dft_densities":
            batch_sizes.append(values.shape[0])
        return original_check_finite(name, values)

    monkeypatch.setattr(density_dataset_module, "_MD17_CACHE_BATCH_SIZE", 3)
    monkeypatch.setattr(
        MD17DensityDataset, "_check_finite", staticmethod(tracking_check_finite)
    )
    monkeypatch.setattr(
        density_dataset_module.logger,
        "message",
        lambda message: log_messages.append(message),
    )

    dataset = _md17_density_dataset(root, "train")

    assert len(dataset) == 7
    assert batch_sizes == [3, 3, 1]
    assert len(dataset.samples) == 7
    conversion_messages = [
        message for message in log_messages if message.startswith("Precomputing")
    ]
    assert conversion_messages == [
        "Precomputing 7 train density samples from FFT coefficients with "
        "n_grid=2, batch_size=3 ..."
    ]


def test_md17_density_dataset_rebuilds_cache_when_converter_config_changes(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "small"
    _write_md17_density_source(root, train_size=2)
    _md17_density_dataset(root, "train")
    original_build_cache = MD17DensityDataset._build_cache
    build_calls = []

    def tracking_build_cache(self, *args, **kwargs):
        build_calls.append(self)
        return original_build_cache(self, *args, **kwargs)

    monkeypatch.setattr(MD17DensityDataset, "_build_cache", tracking_build_cache)

    changed = _md17_density_dataset(
        root,
        "train",
        build_field_cfg=_field_converter_cfg(
            "fft",
            "electron/bohr^3",
            "bohr",
        ),
    )

    assert len(build_calls) == 1
    assert len(changed) == 2
    assert changed[0]["info"]["density_unit"] == "electron/bohr^3"


def test_md17_density_dataset_rebuilds_cache_when_graph_config_changes(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "small"
    _write_md17_density_source(root, train_size=2)
    _md17_density_dataset(root, "train")
    original_build_cache = MD17DensityDataset._build_cache
    build_calls = []

    def tracking_build_cache(self, *args, **kwargs):
        build_calls.append(self)
        return original_build_cache(self, *args, **kwargs)

    monkeypatch.setattr(MD17DensityDataset, "_build_cache", tracking_build_cache)
    changed_cfg = _graph_converter_cfg("bohr", cutoff=2.0)

    changed = _md17_density_dataset(
        root,
        "train",
        build_graph_cfg=changed_cfg,
    )

    assert len(build_calls) == 1
    assert changed.build_graph_cfg == changed_cfg
    assert len(changed) == 2


def test_fft_field_format_inverts_coefficients_as_numpy():
    density = BuildField.invert_fft(np.arange(8, dtype=np.float32), (2, 2, 2))

    assert isinstance(density, np.ndarray)
    assert density.dtype == np.float32
    assert density.flags.c_contiguous
    np.testing.assert_allclose(
        density,
        [
            -0.21875,
            0.15625,
            0.96875,
            -1.40625,
            -0.03125,
            -1.90625,
            -0.84375,
            1.53125,
        ],
    )


def test_md17_density_dataset_rebuilds_stale_samples(tmp_path):
    root = tmp_path / "small"
    _write_md17_density_source(root)
    dataset = _md17_density_dataset(root, "validation")
    sample_dir = root / "ethane" / "ethane_test_n2_g4_cache" / "samples"

    (sample_dir / "0000000000.pkl").unlink()
    (sample_dir / "9999999999.pkl").write_bytes(b"stale")
    reloaded = _md17_density_dataset(root, "validation")

    assert reloaded.sample_ids == dataset.sample_ids
    assert sorted(path.name for path in sample_dir.glob("*.pkl")) == [
        f"{index:010d}.pkl" for index in range(len(dataset))
    ]


def test_md17_density_dataset_rebuilds_truncated_sample(tmp_path):
    root = tmp_path / "small"
    _write_md17_density_source(root)
    _md17_density_dataset(root, "validation")
    sample_path = (
        root / "ethane" / "ethane_test_n2_g4_cache" / "samples" / "0000000000.pkl"
    )
    original_size = sample_path.stat().st_size
    sample_path.write_bytes(sample_path.read_bytes()[:16])

    reloaded = _md17_density_dataset(root, "validation")

    assert sample_path.stat().st_size == original_size
    assert len(reloaded) == 3


def test_md17_density_dataset_rejects_mismatched_sample_counts(tmp_path):
    root = tmp_path / "small"
    _write_md17_density_source(root, train_size=3)
    density_path = root / "ethane" / "ethane_train" / "dft_densities.npy"
    np.save(density_path, np.zeros([2, 8], dtype=np.float32))

    with pytest.raises(ValueError, match="different sample counts"):
        _md17_density_dataset(root, "train")


def test_md17_density_dataset_rejects_unknown_config_key(tmp_path):
    root = tmp_path / "small"
    _write_md17_density_source(root)

    with pytest.raises(TypeError, match="unexpected keyword"):
        _md17_density_dataset(root, "train", misspelled_option=True)


def test_md17_density_dataset_preserves_unrecognized_cache_directory(tmp_path):
    root = tmp_path / "small"
    _write_md17_density_source(root)
    cache_path = tmp_path / "unrelated"
    sample_dir = cache_path / "samples"
    sample_dir.mkdir(parents=True)
    sentinel = sample_dir / "keep.txt"
    sentinel.write_text("user data")

    with pytest.raises(ValueError, match="unrecognized cache"):
        _md17_density_dataset(
            root,
            "train",
            cache_path=cache_path,
        )

    assert sentinel.read_text() == "user data"


def test_density_dataset_transform_is_applied(tmp_path):
    pytest.importorskip("lz4.frame")
    root = tmp_path / "density"
    _write_cube_fixture(root, {"test": [0]})

    dataset = _cube_dataset(
        root,
        transforms=lambda sample: {
            **sample,
            "info": {**sample["info"], "transformed": True},
        },
    )

    assert dataset[0]["info"]["transformed"] is True
    assert isinstance(dataset[0]["density"], np.ndarray)


def _text_density_dataset(tmp_path, format, payload, **kwargs):
    root = tmp_path / format
    root.mkdir()
    extension = "CHGCAR" if format == "chgcar" else format
    file_name = "sample.json.xz" if format == "json" else f"sample.{extension}"
    (root / "split.json").write_text(json.dumps({"test": [file_name]}))
    if format == "json":
        with lzma.open(root / "sample.json.xz", "wt") as file_obj:
            file_obj.write(payload)
    else:
        (root / f"sample.{extension}").write_text(payload)
    params = {
        "path": root / "split.json",
        "split": "test",
        "vocab": _atom_vocab(),
        "build_graph_cfg": _graph_converter_cfg("angstrom"),
        "build_field_cfg": _field_converter_cfg(
            format,
            "electron/angstrom^3",
            "angstrom",
        ),
    }
    params.update(kwargs)
    return DensityDataset(**params)


def _vasp_grid_values():
    shape = (2, 3, 4)
    cell = np.asarray(
        [[2.0, 0.0, 0.0], [0.6, 3.0, 0.0], [0.2, 0.9, 4.0]],
        dtype=np.float32,
    )
    density = np.fromfunction(
        lambda i, j, k: 100 * i + 10 * j + k,
        shape,
        dtype=np.float32,
    ).astype(np.float32)
    raw_values = density.transpose(2, 1, 0).reshape(-1) * abs(np.linalg.det(cell))
    return shape, cell, density, raw_values


def _expected_grid(shape, cell):
    axes = [
        np.arange(size, dtype=np.float32)[:, None] * cell[axis] / size
        for axis, size in enumerate(shape)
    ]
    return (
        axes[0].reshape(-1, 1, 1, 3)
        + axes[1].reshape(1, -1, 1, 3)
        + axes[2].reshape(1, 1, -1, 3)
    ).reshape(-1, 3)


def _chgcar_payload(shape, cell, fractional_position, raw_values):
    lines = [
        "synthetic CHGCAR",
        "1.0",
        *((" ".join(str(value) for value in vector)) for vector in cell),
        "C",
        "1",
        "Direct",
        " ".join(str(value) for value in fractional_position),
        "",
        " ".join(str(size) for size in shape),
    ]
    lines.extend(
        " ".join(str(value) for value in raw_values[start : start + 6])
        for start in range(0, len(raw_values), 6)
    )
    return "\n".join(lines)


def test_density_dataset_reads_chgcar_through_converters_in_c_order(tmp_path):
    shape, cell, expected_density, raw_values = _vasp_grid_values()
    fractional_position = np.asarray([0.25, 0.5, 0.75], dtype=np.float32)
    dataset = _text_density_dataset(
        tmp_path,
        "chgcar",
        _chgcar_payload(shape, cell, fractional_position, raw_values),
    )

    graph, density, grid_coord, info = _unpack_density_sample(dataset[0])

    assert isinstance(_graph_x(graph), np.ndarray)
    assert isinstance(_graph_pos(graph), np.ndarray)
    assert isinstance(density, np.ndarray)
    assert isinstance(grid_coord, np.ndarray)
    np.testing.assert_array_equal(_graph_x(graph), [0])
    np.testing.assert_allclose(_graph_pos(graph)[0], fractional_position @ cell)
    np.testing.assert_allclose(density.reshape(shape), expected_density, rtol=1e-5)
    np.testing.assert_allclose(grid_coord, _expected_grid(shape, cell))
    np.testing.assert_allclose(info["cell"], cell)
    np.testing.assert_allclose(info["origin"], np.zeros(3))
    assert info["coordinate_unit"] == "angstrom"
    assert info["density_unit"] == "electron/angstrom^3"


def test_density_dataset_rejects_known_value_unit_mismatch(tmp_path):
    shape, cell, _, raw_values = _vasp_grid_values()
    fractional_position = np.asarray([0.25, 0.5, 0.75], dtype=np.float32)
    with pytest.raises(ValueError, match="value_unit"):
        _text_density_dataset(
            tmp_path,
            "chgcar",
            _chgcar_payload(shape, cell, fractional_position, raw_values),
            build_field_cfg=_field_converter_cfg(
                "chgcar",
                "unknown",
                "angstrom",
            ),
        )


def test_density_dataset_reads_compressed_json_via_field_builder_in_c_order(tmp_path):
    shape, cell, expected_density, raw_values = _vasp_grid_values()
    fractional_position = np.asarray([0.25, 0.5, 0.75], dtype=np.float32)
    density_lines = [
        raw_values[start : start + 10].tolist()
        for start in range(0, len(raw_values), 10)
    ]
    payload = {
        "vector": [[1.0]],
        "lattice": [cell.tolist()],
        "elements": [["C"]],
        "elements_number": [[1]],
        "coordinates": [[fractional_position.tolist()]],
        "FFTgrid": [list(shape)],
        "chargedensity": [density_lines],
    }

    dataset = _text_density_dataset(tmp_path, "json", json.dumps(payload))

    graph, density, grid_coord, info = _unpack_density_sample(dataset[0])

    assert dataset.build_field_cfg["format"] == "json"
    assert dataset.row_data["field_source"] == [
        str(Path(dataset.path).parent / "sample.json.xz")
    ]
    assert isinstance(_graph_x(graph), np.ndarray)
    assert isinstance(_graph_pos(graph), np.ndarray)
    assert isinstance(density, np.ndarray)
    assert isinstance(grid_coord, np.ndarray)
    np.testing.assert_array_equal(_graph_x(graph), [0])
    np.testing.assert_allclose(_graph_pos(graph)[0], fractional_position @ cell)
    np.testing.assert_allclose(density.reshape(shape), expected_density, rtol=1e-5)
    np.testing.assert_allclose(grid_coord, _expected_grid(shape, cell))
    np.testing.assert_allclose(info["cell"], cell)
    np.testing.assert_allclose(info["origin"], np.zeros(3))
    assert info["coordinate_unit"] == "angstrom"
    assert info["density_unit"] == "electron/angstrom^3"
