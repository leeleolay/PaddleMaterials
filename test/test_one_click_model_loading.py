from __future__ import annotations

import ast
import importlib.util
import inspect
import re
import sys
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
INFGCN_CONFIG_DIR = ROOT / "electronic_structure/configs/infgcn"
INFGCN_MODEL_NAMES = [
    "infgcn_md17_benzene",
    "infgcn_md17_ethane",
    "infgcn_md17_ethanol",
    "infgcn_md17_malonaldehyde",
    "infgcn_md17_phenol",
    "infgcn_md17_resorcinol",
    "infgcn_mp",
    "infgcn_omol25_mc_5k_trimmed",
    "infgcn_qm9",
]
QM9_ATOM_NAME2IDX = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}


def _build_field_cfg(format, value_unit, name="density"):
    return {
        "format": format,
        "name": name,
        "value_unit": value_unit,
        "num_cpus": 1,
    }


def _build_graph_cfg(cutoff):
    return {
        "__class_name__": "RadiusGraphConverter",
        "__init_params__": {
            "cutoff": cutoff,
            "inclusive_cutoff": True,
            "include_distance": False,
        },
    }


def _models_module_ast():
    return ast.parse((ROOT / "ppmat/models/__init__.py").read_text())


def _literal_assign(name: str):
    for node in _models_module_ast().body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} assignment not found")


def test_infgcn_and_diffnmr_are_registered_for_one_click_loading():
    registry = _literal_assign("MODEL_REGISTRY")

    for model_name in [*INFGCN_MODEL_NAMES, "diffnmr_msdnmr_nless15"]:
        assert model_name in registry
        assert registry[model_name].startswith("https://paddle-org.bj.bcebos.com/")
        assert registry[model_name].endswith(".zip")
        assert registry[model_name].endswith(f"{model_name}.zip")


def test_models_init_keeps_one_click_surface_minimal():
    source = (ROOT / "ppmat/models/__init__.py").read_text()

    forbidden_names = [
        "MODEL_CONFIG_REGISTRY",
        "MODEL_SUPPORT_REGISTRY",
        "_repo_root",
        "_resolve_repo_path",
        "get_model_config_path_from_name",
        "get_model_package_path_from_name",
        "get_model_file_path_from_package",
        "get_model_config_path_from_package",
    ]
    for name in forbidden_names:
        assert name not in source


def test_model_package_helpers_resolve_standard_zip_layout(tmp_path):
    from ppmat.utils.model_package import get_model_config_path
    from ppmat.utils.model_package import resolve_model_package_dir

    cache_dir = tmp_path / "infgcn_qm9"
    package_dir = cache_dir / "infgcn_qm9"
    checkpoints_dir = package_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    config_path = package_dir / "infgcn_qm9.yaml"
    config_path.write_text("Model: {}\n")

    resolved_dir = resolve_model_package_dir("infgcn_qm9", str(cache_dir))
    assert Path(resolved_dir) == package_dir
    assert Path(get_model_config_path("infgcn_qm9", resolved_dir)) == config_path


def test_model_package_helpers_reject_nonstandard_recursive_layout(tmp_path):
    import pytest

    from ppmat.utils.model_package import resolve_model_package_dir

    nested_dir = tmp_path / "download" / "arbitrary" / "nested"
    nested_dir.mkdir(parents=True)
    (nested_dir / "infgcn_qm9.yaml").write_text("Model: {}\n")

    with pytest.raises(FileNotFoundError, match="Invalid package"):
        resolve_model_package_dir("infgcn_qm9", str(tmp_path / "download"))


def test_model_package_helpers_reject_unrelated_single_yaml(tmp_path):
    import pytest

    from ppmat.utils.model_package import get_model_config_path

    package_dir = tmp_path / "infgcn_qm9"
    package_dir.mkdir()
    (package_dir / "unrelated.yaml").write_text("Model: {}\n")

    with pytest.raises(FileNotFoundError, match="infgcn_qm9.yaml"):
        get_model_config_path("infgcn_qm9", str(package_dir))


def test_model_package_helpers_have_focused_module():
    package_source = (ROOT / "ppmat/utils/model_package.py").read_text()
    generic_io_source = (ROOT / "ppmat/utils/io.py").read_text()

    assert "def resolve_model_package_dir" in package_source
    assert "def get_model_config_path" in package_source
    assert "model_package" not in generic_io_source
    assert "find_config_file_in_package" not in generic_io_source


def test_diffnmr_assets_do_not_expand_shared_download_api():
    download_source = (ROOT / "ppmat/utils/download.py").read_text()
    sampler_source = (ROOT / "ppmat/sampler/diffnmr.py").read_text()

    assert "get_assets_path_from_url" not in download_source
    assert "DIFFNMR_ASSETS_HOME" not in sampler_source
    assert "_download_diffnmr_asset" not in sampler_source


def test_build_model_from_name_requires_standard_package():
    import ppmat.models as models

    source = inspect.getsource(models.build_model_from_name)

    assert "resolve_model_package_dir(model_name, extracted_path)" in source
    assert "get_model_config_path(model_name, path)" in source
    assert "os.walk" not in source


def test_diffnmr_train_smiles_uses_existing_datadir_cache(tmp_path, monkeypatch):
    import numpy as np

    from ppmat.datasets import msd_nmr_dataset

    global_home = tmp_path / "global_datasets"
    global_subset = (
        global_home / "msd_nmr" / msd_nmr_dataset.MSDnmrDataset.name / "msd_nmr_nless15"
    )
    global_subset.mkdir(parents=True)
    (global_subset / "train.csv").write_text(
        "smiles,tokenized_input,atom_count\nCC,{},2\n", encoding="utf-8"
    )
    monkeypatch.setattr(msd_nmr_dataset.download, "DATASETS_HOME", str(global_home))

    cache_dir = tmp_path / "msd_nmr_nless15_cache" / "train"
    cache_dir.mkdir(parents=True)
    smiles_path = cache_dir / "train_smiles_no_h.npy"
    expected_smiles = np.array(["CCO", "CO"])
    np.save(smiles_path, expected_smiles)

    cfg = {
        "datadir": str(tmp_path),
        "data_flag": "n<15",
        "build_graph_cfg": {"__init_params__": {"remove_h": True}},
    }
    dataset_infos = SimpleNamespace(atom_decoder=["C", "N", "O", "F"])

    train_smiles = msd_nmr_dataset.get_train_smiles(
        cfg, dataloader=[], dataset_infos=dataset_infos
    )

    np.testing.assert_array_equal(train_smiles, expected_smiles)


def test_diffnmr_train_smiles_downloads_registered_asset(monkeypatch, tmp_path):
    import numpy as np

    from ppmat.datasets import msd_nmr_dataset

    asset_path = tmp_path / "msd_nmr_nless15_train_smiles_no_h.npy"
    expected_smiles = np.array(["CCO", "CO"])
    np.save(asset_path, expected_smiles)
    calls = []

    def fake_download(url, md5):
        calls.append((url, md5))
        return str(asset_path)

    monkeypatch.setattr(
        msd_nmr_dataset.download,
        "get_datasets_path_from_url",
        fake_download,
    )
    cfg = {
        "datadir": str(tmp_path / "missing_dataset"),
        "data_flag": "n<15",
        "build_graph_cfg": {"__init_params__": {"remove_h": True}},
    }
    dataset_infos = SimpleNamespace(atom_decoder=["C", "N", "O", "F"])

    train_smiles = msd_nmr_dataset.get_train_smiles(
        cfg,
        dataloader=None,
        dataset_infos=dataset_infos,
    )

    resource = msd_nmr_dataset.TRAIN_SMILES_REGISTRY[("n<15", True)]
    assert calls == [(resource["url"], resource["md5"])]
    np.testing.assert_array_equal(train_smiles, expected_smiles)


def test_diffnmr_dataset_infos_can_skip_train_smiles_without_local_dataset():
    import pytest

    from ppmat.datasets.msd_nmr_dataset import MSDnmrinfos

    atom_tokens = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    vocab = {
        "atom": {
            "token_to_id": {token: index for index, token in enumerate(atom_tokens)},
            "id_to_token": dict(enumerate(atom_tokens)),
            "num_embeddings": len(atom_tokens),
        },
        "bond": {
            "token_to_id": {
                token: index
                for index, token in enumerate(
                    ["NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
                )
            },
            "num_embeddings": 5,
        },
    }
    cfg = {
        "data_flag": "n<15",
        "build_graph_cfg": {"__init_params__": {"remove_h": True}},
        "build_spectrum_cfg": {
            "seq_len_H1": 20,
            "seq_len_C13": 75,
        },
        "load_train_smiles": False,
    }

    dataset_infos = MSDnmrinfos(
        dataloaders=SimpleNamespace(train_dataloader=None),
        cfg=cfg,
        vocab=vocab,
    )

    assert not hasattr(dataset_infos, "train_smiles")
    assert callable(dataset_infos.load_train_smiles)
    assert dataset_infos.atom_decoder == atom_tokens
    assert dataset_infos.seq_len_H1 == 20
    assert dataset_infos.seq_len_C13 == 75
    assert dataset_infos.max_n_nodes == 15
    assert float(dataset_infos.n_nodes.sum()) == pytest.approx(1.0)
    assert float(dataset_infos.valency_distribution.sum()) == pytest.approx(1.0)


def test_diffnmr_sampling_loads_train_smiles_only_for_novelty(monkeypatch):
    import ppmat.metrics.diffnmr_streaming_adapter as adapter_module

    loaded = []
    metric_inputs = []

    class FakeSamplingMetric:
        def __init__(self, *, train_smiles, **kwargs):
            metric_inputs.append(train_smiles)

        def __call__(self, **kwargs):
            return {"Accuracy": 1.0, "Right Number": 1, "Total Number": 1}

        def reset(self):
            pass

    dataset_infos = SimpleNamespace(
        load_train_smiles=lambda: loaded.append(True) or ["CCO"],
    )
    monkeypatch.setattr(
        adapter_module,
        "SamplingMolecularMetrics",
        FakeSamplingMetric,
    )
    result = {"samples": {}, "output_dir": "."}

    accuracy_adapter = adapter_module.DiffNMRStreamingAdapter(
        sample_metrics={"Accuracy"},
    )
    accuracy_adapter.dataset_infos = dataset_infos
    accuracy_adapter._update_sample(result, batch=None)

    novelty_adapter = adapter_module.DiffNMRStreamingAdapter(
        sample_metrics={"Novelty"},
    )
    novelty_adapter.dataset_infos = dataset_infos
    novelty_adapter._update_sample(result, batch=None)

    assert loaded == [True]
    assert metric_inputs == [None, ["CCO"]]


def test_infgcn_predict_config_uses_standard_field_builders():
    source = (ROOT / "electronic_structure/predict.py").read_text()
    field_source = (ROOT / "ppmat/predictor/field_predictor.py").read_text()
    cfg = OmegaConf.to_container(
        OmegaConf.load(INFGCN_CONFIG_DIR / "infgcn_qm9.yaml"),
        resolve=True,
    )

    assert "FieldPredictor" in source
    assert "def apply_predict_config" not in field_source
    assert "def from_data" in field_source
    assert "def from_dataset" in field_source
    assert "def from_mol_file" in field_source
    assert cfg["Predict"] == {
        "grid_batch_size": 20000,
        "build_graph_cfg": _build_graph_cfg(3.0),
        "build_field_cfg": _build_field_cfg(
            "chgcar",
            "electron/angstrom^3",
        ),
    }


def test_infgcn_predict_cli_uses_explicit_runtime_options():
    from electronic_structure.predict import parse_args

    args, config_overrides = parse_args(
        [
            "--model_name",
            "infgcn_qm9",
            "--mol_file_path",
            "methane.mol",
            "--grid_batch_size",
            "123",
        ]
    )

    assert config_overrides == []
    assert args.split == "test"
    assert args.index == 0
    assert args.save_path == "./results"
    assert args.grid_batch_size == 123
    assert args.grid_shape == "80,80,80"
    assert args.grid_padding == 6.0
    assert args.visualize is False
    assert not hasattr(args, "mol_input")
    assert not hasattr(args, "save_pred_cube")


def test_infgcn_predict_cli_accepts_one_click_model_arguments():
    source = (ROOT / "electronic_structure/predict.py").read_text()

    assert '"--model_name"' in source
    assert '"--weights_name"' in source
    assert "FieldPredictor(" in source


def test_field_predictor_is_shared_predictor_entrypoint():
    import ppmat.predictor as predictor

    field_source = (ROOT / "ppmat/predictor/field_predictor.py").read_text()
    entry_source = (ROOT / "electronic_structure/predict.py").read_text()

    assert not (ROOT / "ppmat/predictor/field.py").exists()
    assert hasattr(predictor, "FieldPredictor")
    assert "class FieldPredictor" in field_source
    assert "from ppmat.predictor import FieldPredictor" in entry_source
    assert "apply_predict_config" not in entry_source
    assert "predictor.predict(args)" not in entry_source
    assert "from ppmat.models import MODEL_REGISTRY" not in entry_source
    assert "from ppmat.datasets import DensityDataset" not in entry_source


def test_field_predictor_reuses_base_and_keeps_helpers_outside_predictor():
    field_source = (ROOT / "ppmat/predictor/field_predictor.py").read_text()
    io_source = (ROOT / "ppmat/utils/io.py").read_text()
    build_field_source = (ROOT / "ppmat/datasets/build_field.py").read_text()
    dataset_source = (ROOT / "ppmat/datasets/density_dataset.py").read_text()
    field_io_source = (ROOT / "ppmat/predictor/field_io.py").read_text()
    visualization_source = (ROOT / "ppmat/utils/visualization.py").read_text()

    assert "from ppmat.predictor.base import BasePredictor" in field_source
    assert "class FieldPredictor(BasePredictor):" in field_source
    assert "self.load_inference_model()" in field_source
    assert "def _load_model(self):" not in field_source
    assert "def predict(self, args)" not in field_source
    assert not (ROOT / "ppmat/utils/field_io.py").exists()
    assert not (ROOT / "ppmat/utils/field.py").exists()
    assert not (ROOT / "ppmat/utils/cube.py").exists()
    assert not (ROOT / "ppmat/datasets/_field_source.py").exists()
    assert (ROOT / "ppmat/predictor/field_io.py").exists()
    assert not (ROOT / "ppmat/utils/field_visualization.py").exists()

    for helper_name in [
        "draw_volume",
        "safe_write_image",
        "maybe_downsample_volume",
        "read_cube_density",
        "write_cube_from_atom_types",
        "prepare_cube_info",
    ]:
        assert f"def {helper_name}" not in field_source

    for helper_name in [
        "read_cube_density",
        "write_cube_from_atom_types",
        "prepare_cube_info",
    ]:
        assert f"def {helper_name}" in field_io_source
        assert f"def {helper_name}" not in io_source

    assert "def write_cube(" in io_source
    assert "def write_cube(" not in field_io_source
    assert "def read_cube(" not in io_source
    assert "ase_write(" in io_source
    assert (
        "from ppmat.utils.crystal import normalize_coordinate_unit"
        in build_field_source
    )
    assert "class BuildField:" in build_field_source
    assert "def build_grid(" in build_field_source
    assert "from cvve import GridSpec" in build_field_source
    assert "from cvve import GridField" in build_field_source

    assert "OUTER LOOP" not in io_source
    assert "OUTER LOOP" not in field_io_source
    assert "OUTER LOOP" not in dataset_source
    assert "cvve.read_grid_field" in build_field_source
    assert "from cvve" not in dataset_source
    assert "import cvve" not in dataset_source
    assert "def read_data(" in dataset_source
    assert "def read_cube(" not in dataset_source
    assert "def read_chgcar(" not in dataset_source
    assert "def read_json(" not in dataset_source
    assert "def write_cube(" not in dataset_source
    assert "write_cube_file" not in dataset_source

    for helper_name in ["draw_volume", "safe_write_image", "maybe_downsample_volume"]:
        assert f"def {helper_name}" in visualization_source

    top_level_imports = "\n".join(
        line
        for line in visualization_source.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    )
    assert "import imageio" not in top_level_imports
    assert "import plotly.graph_objects as go" not in top_level_imports
    assert "import imageio" in visualization_source
    assert "import plotly.graph_objects as go" in visualization_source
    assert "import matplotlib.pyplot as plt" not in top_level_imports
    assert "import networkx as nx" not in top_level_imports
    assert "import matplotlib.pyplot as plt" not in visualization_source
    assert "import networkx as nx" not in visualization_source
    assert "import rdkit" in top_level_imports
    assert "def _rdkit_modules" not in visualization_source
    assert "def _matplotlib_pyplot" not in visualization_source
    assert "def _networkx" not in visualization_source
    assert "def _imageio" not in visualization_source

    assert "def _save_cubes" in field_source
    assert "def _save_visualizations" in field_source
    assert "self._save_cubes(" in field_source
    assert "self._save_visualizations(" in field_source


def test_field_predictor_from_data_uses_standard_batch_contract():
    import numpy as np
    import paddle
    import pgl
    import pytest

    from ppmat.datasets.build_field import BuildField
    from ppmat.predictor import FieldPredictor

    class FieldModel(paddle.nn.Layer):
        target_name = "density"
        loss_eps = 1e-8

        def forward(self, batch, return_loss=True, return_prediction=True):
            assert return_loss is False
            assert return_prediction is True
            assert "density" not in batch
            assert batch["density_mask"] is None
            assert isinstance(batch["graph"], pgl.Graph)
            np.testing.assert_array_equal(batch["graph"].graph_node_id, [0, 0])
            assert set(batch["info"]) == {"cell"}
            assert list(batch["info"]["cell"].shape) == [1, 3, 3]
            prediction = paddle.sum(batch["grid_coord"], axis=-1)
            return {"loss_dict": {}, "pred_dict": {"density": prediction}}

    predictor = FieldPredictor.__new__(FieldPredictor)
    predictor.model = FieldModel()
    predictor.device = "cpu"
    predictor.predict_config = {"grid_batch_size": 2}
    predictor.target_name = "density"
    predictor.field_converter = BuildField(
        format="array",
        name="density",
        value_unit="electron/angstrom^3",
        coordinate_unit="angstrom",
    )
    predictor.post_transforms = None

    graph = pgl.Graph(
        num_nodes=2,
        edges=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        node_feat={
            "x": np.asarray([0, 1], dtype=np.int64),
            "pos": np.zeros([2, 3], dtype=np.float32),
        },
    )
    grid_coord = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]],
        dtype="float32",
    )
    info = {
        "cell": np.eye(3, dtype=np.float32),
        "shape": [2, 2, 1],
        "coordinate_unit": "angstrom",
        "density_unit": "electron/angstrom^3",
    }
    reference = np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float32)

    without_reference = predictor.from_data(graph, grid_coord, info)
    with_reference = predictor.from_data(
        graph,
        grid_coord,
        info,
        density=reference,
    )

    np.testing.assert_allclose(
        without_reference["density"].numpy(),
        with_reference["density"].numpy(),
    )
    np.testing.assert_allclose(with_reference["density"].numpy(), [0, 1, 2, 3])
    np.testing.assert_allclose(with_reference["grid_coord"].numpy(), grid_coord)
    assert isinstance(graph.node_feat["x"], np.ndarray)
    assert isinstance(graph.node_feat["pos"], np.ndarray)
    np.testing.assert_array_equal(graph.graph_node_id, [0, 0])
    assert with_reference["info"]["shape"] == [2, 2, 1]
    assert with_reference["loss"] == pytest.approx(0.375)
    assert with_reference["nmae"] == pytest.approx(0.4)

    with pytest.raises(KeyError, match="coordinate_unit"):
        predictor.from_data(
            graph,
            grid_coord,
            {
                "cell": np.eye(3),
                "shape": [2, 2, 1],
                "density_unit": "electron/angstrom^3",
            },
        )
    with pytest.raises(ValueError, match="density unit"):
        predictor.from_data(
            graph,
            grid_coord,
            {
                "cell": np.eye(3),
                "shape": [2, 2, 1],
                "coordinate_unit": "angstrom",
                "density_unit": "unknown",
            },
        )


def test_field_cube_io_round_trip(tmp_path):
    import numpy as np
    from ase.units import Bohr

    from ppmat.datasets.build_field import BuildField
    from ppmat.predictor.field_io import read_cube_density
    from ppmat.utils.io import write_cube

    cube_path = tmp_path / "density.cube"
    density = np.arange(8, dtype=np.float32)
    info = {
        "shape": [2, 2, 2],
        "cell": np.eye(3, dtype=np.float32) * 2,
        "origin": np.asarray([0.5, 1.0, 1.5], dtype=np.float32),
        "coordinate_unit": "angstrom",
    }
    write_cube(
        cube_path,
        atom_numbers=np.asarray([6]),
        atom_coord=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        density=density,
        info=info,
    )

    loaded_density, grid_coord, loaded_info = read_cube_density(
        cube_path,
        BuildField(
            format="array",
            name="density",
            value_unit="electron/angstrom^3",
            coordinate_unit="angstrom",
        ),
    )

    assert isinstance(loaded_density, np.ndarray)
    assert isinstance(grid_coord, np.ndarray)
    assert isinstance(loaded_info["cell"], np.ndarray)
    assert isinstance(loaded_info["origin"], np.ndarray)
    np.testing.assert_allclose(loaded_density, density)
    np.testing.assert_allclose(loaded_info["origin"], info["origin"] / Bohr)
    np.testing.assert_allclose(grid_coord[0], info["origin"] / Bohr)
    assert loaded_info["shape"] == info["shape"]
    assert loaded_info["coordinate_unit"] == "bohr"
    assert loaded_info["density_unit"] == "electron/angstrom^3"
    np.testing.assert_array_equal(loaded_info["atom_numbers"], [6])


def test_field_cube_io_preserves_bohr_axis_sign(tmp_path):
    import numpy as np

    from ppmat.datasets.build_field import BuildField
    from ppmat.predictor.field_io import read_cube_density
    from ppmat.utils.io import write_cube

    cube_path = tmp_path / "density_bohr.cube"
    write_cube(
        cube_path,
        atom_numbers=np.asarray([6]),
        atom_coord=np.zeros([1, 3], dtype=np.float32),
        density=np.arange(8, dtype=np.float32),
        info={
            "shape": [2, 2, 2],
            "cell": np.eye(3, dtype=np.float32) * 2,
            "origin": np.zeros(3, dtype=np.float32),
            "coordinate_unit": "bohr",
        },
    )

    axis_counts = [
        int(line.split()[0]) for line in cube_path.read_text().splitlines()[3:6]
    ]
    _, _, loaded_info = read_cube_density(
        cube_path,
        BuildField(
            format="array",
            name="density",
            value_unit="unknown",
            coordinate_unit="bohr",
        ),
    )

    assert axis_counts == [2, 2, 2]
    assert loaded_info["coordinate_unit"] == "bohr"


def test_field_cube_metadata_requires_explicit_coordinate_unit(tmp_path):
    import numpy as np
    import pytest

    from ppmat.utils.io import write_cube

    with pytest.raises(KeyError, match="coordinate_unit"):
        write_cube(
            tmp_path / "missing_unit.cube",
            atom_numbers=np.asarray([6]),
            atom_coord=np.zeros([1, 3], dtype=np.float32),
            density=np.zeros(8, dtype=np.float32),
            info={"shape": [2, 2, 2], "cell": np.eye(3) * 2},
        )


def test_field_dataset_output_uses_shared_cube_writer(tmp_path):
    import numpy as np

    from ppmat.predictor import FieldPredictor

    class Dataset:
        def write_cube(self, *args, **kwargs):
            raise AssertionError("dataset writer must not be used")

    predictor = FieldPredictor.__new__(FieldPredictor)
    predictor.vocab = {"atom": {"id_to_atomic_number": {0: 17}}}
    writer = predictor._get_cube_writer(Dataset())
    cube_path = tmp_path / "dataset_output.cube"
    writer(
        cube_path,
        atom_type=np.asarray([0]),
        atom_coord=np.zeros([1, 3], dtype=np.float32),
        density=np.zeros(8, dtype=np.float32),
        info={
            "shape": [2, 2, 2],
            "cell": np.eye(3, dtype=np.float32) * 2,
            "origin": np.zeros(3, dtype=np.float32),
            "coordinate_unit": "angstrom",
        },
    )

    cube_lines = cube_path.read_text().splitlines()
    axis_counts = [int(line.split()[0]) for line in cube_lines[3:6]]
    assert axis_counts == [2, 2, 2]
    assert int(cube_lines[6].split()[0]) == 17


def test_reference_cube_coordinates_require_matching_atom_order():
    import numpy as np
    import pgl
    import pytest

    from ppmat.models.common.graph_converter import RadiusGraphConverter
    from ppmat.predictor.field_predictor import use_reference_atom_coordinates

    graph = pgl.Graph(
        num_nodes=2,
        edges=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        node_feat={
            "x": np.asarray([0, 1], dtype=np.int64),
            "pos": np.zeros([2, 3], dtype=np.float32),
        },
    )
    graph_converter = RadiusGraphConverter(
        cutoff=10.0,
        coordinate_unit="angstrom",
        atom_vocab={},
    )
    reference_coord = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    reference_info = {
        "atom_numbers": np.asarray([6, 1]),
        "atom_coord_ref": reference_coord,
        "coordinate_unit": "angstrom",
    }

    graph = use_reference_atom_coordinates(
        graph,
        reference_info,
        {0: 6, 1: 1},
        "sample.mol",
        graph_converter,
    )
    assert isinstance(graph.node_feat["pos"], np.ndarray)
    np.testing.assert_allclose(graph.node_feat["pos"], reference_coord)

    reference_info["atom_numbers"] = np.asarray([1, 6])
    with pytest.raises(ValueError, match="Atom order"):
        use_reference_atom_coordinates(
            graph,
            reference_info,
            {0: 6, 1: 1},
            "sample.mol",
            graph_converter,
        )


def test_prepare_cube_info_preserves_explicit_geometry_with_singleton_axis():
    import numpy as np
    import paddle
    import pytest

    from ppmat.datasets.build_field import BuildField
    from ppmat.predictor.field_io import prepare_cube_info

    origin = np.asarray([0.5, 1.0, 1.5], dtype=np.float32)
    shape = (2, 3, 1)
    cell = np.asarray(
        [[2.0, 0.0, 0.0], [0.6, 3.0, 0.0], [0.2, 0.4, 1.0]],
        dtype=np.float32,
    )
    steps = cell / np.asarray(shape, dtype=np.float32)[:, None]
    expected_grid = np.empty((*shape, 3), dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                expected_grid[i, j, k] = (
                    origin + i * steps[0] + j * steps[1] + k * steps[2]
                )
    grid = paddle.to_tensor(expected_grid.reshape(1, -1, 3))

    info = prepare_cube_info(
        {
            "shape": list(shape),
            "cell": cell,
            "origin": origin,
            "coordinate_unit": "angstrom",
            "density_unit": "electron/angstrom^3",
        },
        grid,
        BuildField(
            format="array",
            name="density",
            value_unit="electron/angstrom^3",
            coordinate_unit="angstrom",
        ),
    )

    np.testing.assert_allclose(info["origin"], origin)
    np.testing.assert_allclose(info["cell"], cell)
    assert info["coordinate_unit"] == "angstrom"

    mismatched_grid = grid.clone()
    mismatched_grid[..., 0] += 0.25
    with pytest.raises(ValueError, match="grid coordinates do not match"):
        prepare_cube_info(
            {
                "shape": list(shape),
                "cell": cell,
                "origin": origin,
                "coordinate_unit": "angstrom",
                "density_unit": "electron/angstrom^3",
            },
            mismatched_grid,
            BuildField(
                format="array",
                name="density",
                value_unit="electron/angstrom^3",
                coordinate_unit="angstrom",
            ),
        )


def test_electronic_structure_models_use_builtin_scatter():
    for relative_path in [
        "ppmat/models/infgcn/infgcn.py",
        "ppmat/models/mateno/mateno.py",
    ]:
        source = (ROOT / relative_path).read_text()
        assert "from paddle_scatter import scatter" not in source
        assert "from ppmat.utils.scatter import scatter" in source


def test_all_infgcn_configs_are_parseable_and_complete():
    config_paths = sorted(INFGCN_CONFIG_DIR.glob("*.yaml"))
    assert [path.name for path in config_paths] == [
        "infgcn_md17_benzene.yaml",
        "infgcn_md17_ethane.yaml",
        "infgcn_md17_ethanol.yaml",
        "infgcn_md17_malonaldehyde.yaml",
        "infgcn_md17_phenol.yaml",
        "infgcn_md17_resorcinol.yaml",
        "infgcn_mp.yaml",
        "infgcn_omol25_MC_5k_trimmed.yaml",
        "infgcn_qm9.yaml",
    ]

    required_model_params = {
        "num_radial",
        "num_spherical",
        "radial_embed_size",
        "radial_hidden_size",
        "cutoff",
        "grid_cutoff",
    }
    expected_dataset_classes = {
        "infgcn_qm9": "QM9DensityDataset",
        "infgcn_mp": "MPCubicDensityDataset",
        "infgcn_omol25_MC_5k_trimmed": "OMol25MC5kTrimmedDensityDataset",
    }
    expected_vocab_names = {
        "infgcn_qm9": "infgcn_qm9",
        "infgcn_mp": "infgcn_mp",
        "infgcn_omol25_MC_5k_trimmed": "infgcn_omol25",
    }
    expected_dataset_formats = {
        "infgcn_qm9": "chgcar",
        "infgcn_mp": "json",
        "infgcn_omol25_MC_5k_trimmed": "cube",
    }

    for config_path in config_paths:
        loaded_cfg = OmegaConf.load(config_path)
        unresolved_cfg = OmegaConf.to_container(loaded_cfg, resolve=False)
        cfg = OmegaConf.to_container(loaded_cfg, resolve=True)
        assert cfg["Model"]["__class_name__"] == "InfGCN", config_path.name
        assert required_model_params.issubset(
            cfg["Model"]["__init_params__"]
        ), config_path.name
        assert "n_atom_type" not in cfg["Model"]["__init_params__"], config_path.name
        assert cfg["Vocabulary"] == expected_vocab_names.get(
            config_path.stem, "infgcn_md17"
        ), config_path.name
        uses_angstrom = config_path.name in {
            "infgcn_qm9.yaml",
            "infgcn_mp.yaml",
        }
        dataset_format = expected_dataset_formats.get(config_path.stem, "fft")
        expected_field_cfg = _build_field_cfg(
            dataset_format,
            "electron/angstrom^3" if uses_angstrom else "unknown",
        )
        assert cfg["Global"]["build_field_cfg"] == expected_field_cfg
        graph_cutoff = 6.0 if config_path.stem == "infgcn_omol25_MC_5k_trimmed" else 3.0
        expected_graph_cfg = _build_graph_cfg(graph_cutoff)
        assert cfg["Global"]["build_graph_cfg"] == expected_graph_cfg
        assert cfg["Model"]["__init_params__"]["cutoff"] == graph_cutoff

        dataset_cfg = cfg["Dataset"]
        for split in ["train", "val", "test"]:
            split_cfg = dataset_cfg[split]
            dataset = split_cfg["dataset"]
            expected_dataset_class = expected_dataset_classes.get(
                config_path.stem, "MD17DensityDataset"
            )
            assert dataset["__class_name__"] == expected_dataset_class, config_path.name
            assert "path" in dataset["__init_params__"], config_path.name
            assert "sampler" in split_cfg, config_path.name
            assert "loader" in split_cfg, config_path.name
            assert isinstance(
                split_cfg["loader"]["use_shared_memory"], bool
            ), config_path.name
            assert (
                split_cfg["loader"]["collate_fn"] == "DensityCollator"
            ), config_path.name
            if split in {"val", "test"}:
                assert split_cfg["sampler"]["__init_params__"]["shuffle"] is False
            init_params = dataset["__init_params__"]
            unresolved_init_params = unresolved_cfg["Dataset"][split]["dataset"][
                "__init_params__"
            ]
            assert init_params["build_graph_cfg"] == expected_graph_cfg
            assert (
                unresolved_init_params["build_graph_cfg"] == "${Global.build_graph_cfg}"
            )
            if dataset["__class_name__"] == "MD17DensityDataset":
                # The MD17 dataset supplies its own array/bohr field defaults.
                assert "build_field_cfg" not in init_params
                assert "validation_ratio" not in init_params
                assert "split_seed" not in init_params
                assert init_params["n_grid"] == 50
                assert init_params["grid_size"] == 20.0
            else:
                assert init_params["build_field_cfg"] == expected_field_cfg
                assert (
                    unresolved_init_params["build_field_cfg"]
                    == "${Global.build_field_cfg}"
                )
                assert set(init_params) <= {
                    "path",
                    "split",
                    "build_field_cfg",
                    "build_graph_cfg",
                    "grid_sampler_cfg",
                    "overwrite",
                }
                assert init_params["overwrite"] is False
            assert init_params["grid_sampler_cfg"]["n_samples"] > 0, config_path.name
            if config_path.stem == "infgcn_omol25_MC_5k_trimmed" and split in {
                "val",
                "test",
            }:
                assert init_params["grid_sampler_cfg"]["resample_each_epoch"] is False

        assert set(cfg["Predict"]) == {
            "grid_batch_size",
            "build_field_cfg",
            "build_graph_cfg",
        }, config_path.name
        assert cfg["Predict"]["build_graph_cfg"] == expected_graph_cfg
        assert (
            unresolved_cfg["Predict"]["build_graph_cfg"] == "${Global.build_graph_cfg}"
        )
        assert cfg["Predict"]["build_field_cfg"] == expected_field_cfg
        assert (
            unresolved_cfg["Predict"]["build_field_cfg"]
            == "${Global.build_field_cfg}"
        )


def test_electronic_structure_train_wires_vocabulary_into_runtime():
    source = (ROOT / "electronic_structure/train.py").read_text()

    assert "electronic_structure/configs/infgcn/infgcn_md17_benzene.yaml" in source
    assert 'vocab = build_vocab(config.get("Vocabulary"))' in source
    assert "model = build_model(model_cfg, vocab=vocab)" in source
    assert "build_dataloader(train_data_cfg, vocab=vocab)" in source


def test_infgcn_readme_commands_and_config_links_are_clean():
    readme_path = INFGCN_CONFIG_DIR / "README.md"
    readme = readme_path.read_text()
    example_mol = INFGCN_CONFIG_DIR / "example/methane.mol"

    assert "--model_name infgcn_qm9" in readme
    assert "--weights_name best.pdparams" in readme
    assert (
        "--mol_file_path electronic_structure/configs/infgcn/example/methane.mol"
        in (readme)
    )
    assert example_mol.is_file()
    assert "conda run" not in readme
    assert "/home/" not in readme
    assert ".pt" not in readme
    assert "_t_2026" not in readme
    assert "_s_42.zip" not in readme
    dataset_section = readme.split("### Datasets", 1)[1].split("---", 1)[0]
    assert "http://" not in dataset_section
    assert "https://" not in dataset_section

    hrefs = re.findall(r'href="([^"]*configs/infgcn/[^"]+\.yaml)"', readme)
    assert hrefs
    for href in hrefs:
        assert (readme_path.parent / href).resolve().exists(), href


def test_infgcn_bundled_molecule_builds_inference_grid():
    import numpy as np

    from ppmat.datasets.build_field import BuildField
    from ppmat.models.common.graph_converter import RadiusGraphConverter
    from ppmat.predictor.field_predictor import build_mol_sample

    example_mol = INFGCN_CONFIG_DIR / "example/methane.mol"

    graph, density, grid_coord, info = build_mol_sample(
        example_mol,
        QM9_ATOM_NAME2IDX,
        grid_shape=[8, 8, 8],
        grid_padding=6.0,
        field_converter=BuildField(
            format="array",
            name="density",
            value_unit="unknown",
            coordinate_unit="angstrom",
        ),
        graph_converter=RadiusGraphConverter(
            cutoff=3.0,
            coordinate_unit="angstrom",
            atom_vocab={},
        ),
    )

    assert isinstance(graph.node_feat["x"], np.ndarray)
    assert isinstance(graph.node_feat["pos"], np.ndarray)
    assert isinstance(grid_coord, np.ndarray)
    assert isinstance(info["cell"], np.ndarray)
    assert isinstance(info["origin"], np.ndarray)
    assert graph.node_feat["x"].shape == (5,)
    assert graph.node_feat["pos"].shape == (5, 3)
    assert density is None
    assert grid_coord.shape == (512, 3)
    assert info["shape"] == [8, 8, 8]
    assert info["coordinate_unit"] == "angstrom"
    assert info["density_unit"] == "unknown"


def test_infgcn_molecule_grid_always_uses_angstrom():
    import numpy as np

    from ppmat.datasets.build_field import BuildField
    from ppmat.models.common.graph_converter import RadiusGraphConverter
    from ppmat.predictor.field_predictor import build_mol_sample

    example_mol = INFGCN_CONFIG_DIR / "example/methane.mol"
    field_converter = BuildField(
        format="array",
        name="density",
        value_unit="unknown",
        coordinate_unit="angstrom",
    )

    graph_ang, _, grid_ang, info_ang = build_mol_sample(
        example_mol,
        QM9_ATOM_NAME2IDX,
        grid_shape=[8, 8, 8],
        grid_padding=6.0,
        field_converter=field_converter,
        graph_converter=RadiusGraphConverter(
            cutoff=3.0,
            atom_vocab={},
        ),
    )
    field_converter_bohr = BuildField(
        format="array",
        name="density",
        value_unit="unknown",
        coordinate_unit="bohr",
    )
    graph_bohr, _, grid_bohr, info_bohr = build_mol_sample(
        example_mol,
        QM9_ATOM_NAME2IDX,
        grid_shape=[8, 8, 8],
        grid_padding=6.0,
        field_converter=field_converter_bohr,
        graph_converter=RadiusGraphConverter(
            cutoff=3.0,
            atom_vocab={},
        ),
    )

    np.testing.assert_allclose(
        graph_bohr.node_feat["pos"],
        graph_ang.node_feat["pos"],
    )
    np.testing.assert_allclose(grid_bohr, grid_ang, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(info_bohr["cell"], info_ang["cell"])
    assert info_bohr["coordinate_unit"] == "angstrom"


def test_cube_field_builder_preserves_geometry(tmp_path):
    import numpy as np

    from ppmat.datasets.build_field import BuildField

    cube_path = tmp_path / "density.cube"
    cube_path.write_text(
        "comment\n"
        "comment\n"
        "1 0.5 1.0 1.5\n"
        "-2 1.0 0.0 0.0\n"
        "-2 0.0 1.0 0.0\n"
        "-2 0.0 0.0 1.0\n"
        "6 6.0 0.0 0.0 0.0\n"
        "0 1 2 3 4 5 6 7\n",
        encoding="utf-8",
    )
    field = BuildField(
        format="cube",
        name="density",
        value_unit="unknown",
        coordinate_unit="angstrom",
    )(cube_path)

    assert field.structure is not None
    np.testing.assert_array_equal(
        field.structure.symbols,
        ["C"],
    )

    assert isinstance(field.flat, np.ndarray)
    assert isinstance(field.grid.cartesian_coordinates(), np.ndarray)
    assert isinstance(field.grid.cell_vectors, np.ndarray)
    np.testing.assert_allclose(field.flat, np.arange(8))
    np.testing.assert_allclose(
        field.grid.cartesian_coordinates()[0],
        [0.5, 1.0, 1.5],
    )
    np.testing.assert_allclose(field.grid.cell_vectors, np.eye(3) * 2)
    assert field.grid.shape == (2, 2, 2)
    assert field.grid.length_unit == "angstrom"


def test_diffnmr_sample_readme_documents_one_click_sample_command():
    readme = (ROOT / "spectrum_elucidation/configs/diffnmr/README.md").read_text()
    sample_csv = ROOT / "spectrum_elucidation/configs/diffnmr/example/sample.csv"

    assert "--model_name='diffnmr_msdnmr_nless15'" in readme
    assert "--weights_name='best.pdparams'" in readme
    assert "bundled one-row validation example" in readme
    assert "Sampler.data.dataset.__init_params__.path" in readme
    assert (
        "Sampler.data.dataset.__init_params__.path='./data/MSD_nmr/test.csv'" in readme
    )
    assert "### Sampling Sample" not in readme
    assert "--checkpoint_path='./checkpoints'" in readme
    assert sample_csv.exists()
    assert sample_csv.read_text().splitlines()[0] == "smiles,tokenized_input,atom_count"
    assert (
        sample_csv.read_text()
        .splitlines()[1]
        .startswith('CSc1ccc(C(C)C(=O)O)cc1F,"{""1HNMR"":')
    )


def test_diffnmr_package_sample_defaults_to_bundled_example():
    config = OmegaConf.to_container(
        OmegaConf.load(ROOT / "spectrum_elucidation/configs/diffnmr/DiffNMR.yaml"),
        resolve=False,
    )

    assert config["Sampler"]["name"] == "diffnmr"
    assert config["Sampler"]["retrieval_database_path"] is None
    assert config["Vocabulary"] == "diffnmr_msdnmr_nless15"
    sampler_params = config["Sampler"]["data"]["dataset"]["__init_params__"]
    assert sampler_params["path"] == "./example/sample.csv"
    assert sampler_params["cache_path"] == "./output/diffnmr_example_cache"
    assert sampler_params["overwrite"] is True
    assert config["Sampler"]["data"]["sampler"]["__init_params__"]["batch_size"] == 1
    assert config["Sampler"]["sample_batch_iters"] == 1
    assert config["Sampler"]["visual_num"] == 1
    assert config["Sampler"]["chains_to_save"] == 0


def test_molecular_sampler_entrypoint_keeps_diffnmr_imports_lazy():
    source = (ROOT / "ppmat/sampler/molecular_sampler.py").read_text()

    forbidden_snippets = [
        "import paddle",
        "from ppmat.datasets",
        "from ppmat.metrics",
        "from ppmat.models.diffnmr",
        "from ppmat.schedulers",
        "DiffNMRStreamingAdapter",
        "ExtraMolecularFeatures",
        "MolecularVisualization",
        "scheduling_diffnmr",
        "graphs_from_mol",
    ]
    for snippet in forbidden_snippets:
        assert snippet not in source
    assert "importlib.import_module" in source
    assert "ppmat.sampler.diffnmr:DiffNMRSampler" in source
    assert "MODEL_NAME_TO_SAMPLER" in source


def test_molecular_sampler_module_source_load_does_not_load_diffnmr():
    sys.modules.pop("ppmat.models.diffnmr.diffnmr", None)
    sys.modules.pop("ppmat.sampler.diffnmr", None)

    module_path = ROOT / "ppmat/sampler/molecular_sampler.py"
    spec = importlib.util.spec_from_file_location(
        "molecular_sampler_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.SAMPLER_REGISTRY["diffnmr"].endswith("diffnmr:DiffNMRSampler")
    assert "ppmat.models.diffnmr.diffnmr" not in sys.modules
    assert "ppmat.sampler.diffnmr" not in sys.modules


def test_molecular_sampler_dispatches_diffnmr_config(tmp_path, monkeypatch):
    import ppmat.sampler.molecular_sampler as molecular_sampler

    class FakeSampler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_import_module(module_name):
        assert module_name == "fake_sampler_module"
        return SimpleNamespace(FakeSampler=FakeSampler)

    monkeypatch.setattr(
        molecular_sampler,
        "SAMPLER_REGISTRY",
        {"diffnmr": "fake_sampler_module:FakeSampler"},
    )
    monkeypatch.setattr(
        molecular_sampler.importlib, "import_module", fake_import_module
    )

    config_path = tmp_path / "DiffNMR.yaml"
    checkpoint_path = tmp_path / "checkpoints"
    checkpoint_path.mkdir()
    config_path.write_text(
        "Model:\n" "  __class_name__: DiffNMR\n" "Sampler:\n" "  name: diffnmr\n"
    )

    sampler = molecular_sampler.MolecularSampler(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        config_overrides=["Sampler.name=diffnmr"],
    )

    assert isinstance(sampler, FakeSampler)
    assert sampler.kwargs["config_path"] == str(config_path)
    assert sampler.kwargs["checkpoint_path"] == str(checkpoint_path)
    assert sampler.kwargs["config_overrides"] == ["Sampler.name=diffnmr"]


def test_molecular_sampler_dispatches_registered_diffnmr_name(monkeypatch):
    import ppmat.sampler.molecular_sampler as molecular_sampler

    class FakeSampler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        molecular_sampler,
        "SAMPLER_REGISTRY",
        {"diffnmr": "fake_sampler_module:FakeSampler"},
    )
    monkeypatch.setattr(
        molecular_sampler.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(FakeSampler=FakeSampler),
    )

    sampler = molecular_sampler.MolecularSampler(model_name="diffnmr_msdnmr_nless15")

    assert isinstance(sampler, FakeSampler)
    assert sampler.kwargs["model_name"] == "diffnmr_msdnmr_nless15"


def test_molecular_sampler_infers_diffnmr_for_graphformer_config(tmp_path):
    import ppmat.sampler.molecular_sampler as molecular_sampler

    config_path = tmp_path / "DiffNMR_DiffGraphFormer.yaml"
    config_path.write_text("Model:\n" "  __class_name__: MolecularGraphFormer\n")

    assert (
        molecular_sampler._infer_sampler_name(
            model_name=None,
            config_path=str(config_path),
            config_overrides=None,
        )
        == "diffnmr"
    )


def test_molecular_sampler_updates_visualization_output_dir_for_save_path(tmp_path):
    from ppmat.sampler.diffnmr import DiffNMRSampler

    sampler = object.__new__(DiffNMRSampler)
    sampler.sample_config = {"data": {}}
    sampler.output_dir = "old_output"
    sampler.visualization_tools = SimpleNamespace(result_path="old_output/graph/")
    sampler.model = SimpleNamespace(eval=lambda: None)
    sampler.flag_retrieval_sampling = False
    sampler.num_candidates = 1
    sampler.metric_dict_sample = {"Accuracy"}

    save_path = tmp_path / "sample_output"
    sampler.sample_epoch = lambda *args, **kwargs: {"Accuracy": 1.0}

    result = sampler.sample_by_dataloader(
        save_path=str(save_path),
        data_loader=[],
    )

    assert result == {"Accuracy": 1.0}
    assert sampler.output_dir == str(save_path)
    assert sampler.visualization_tools.result_path == str(save_path / "graph")


def test_molecular_sampler_compute_metric_reuses_sample_metrics(tmp_path):
    from ppmat.sampler.diffnmr import DiffNMRSampler

    sampler = object.__new__(DiffNMRSampler)
    sampler.sample_config = {}
    sampler.output_dir = "old_output"
    sampler.visualization_tools = None
    sampler.sample_by_dataloader = lambda save_path: {"Accuracy": 0.5}

    assert sampler.compute_metric(save_path=str(tmp_path)) == {"Accuracy": 0.5}
    assert sampler.output_dir == str(tmp_path)


def test_diffnmr_config_uses_standard_checkpoint_paths():
    config_path = ROOT / "spectrum_elucidation/configs/diffnmr/DiffNMR.yaml"
    source = config_path.read_text()
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    assert "./pretrained/" not in source
    assert cfg["Sampler"]["pretrained_model_path"] == "./checkpoints/best.pdparams"
    assert cfg["Model"]["__init_params__"]["encoder_cfg"]["pretrained_path"].startswith(
        "./checkpoints/"
    )
    assert cfg["Model"]["__init_params__"]["decoder_cfg"]["pretrained_path"].startswith(
        "./checkpoints/"
    )
    assert cfg["CLIP"]["__init_params__"]["spectrum_encoder"][
        "pretrained_model_path"
    ].startswith("./checkpoints/")
    assert cfg["CLIP"]["__init_params__"]["graph_encoder"][
        "pretrained_model_path"
    ].startswith("./checkpoints/")
    for config_name in ("DiffNMR.yaml", "PP-DiffNMR.yaml"):
        final_cfg = OmegaConf.load(
            ROOT / "spectrum_elucidation/configs/diffnmr" / config_name
        )
        assert "vocab_dim" not in final_cfg["Model"]["__init_params__"]["decoder_cfg"]


def test_molecular_sampler_resolves_diffnmr_checkpoint_paths(tmp_path):
    from ppmat.sampler.diffnmr import DiffNMRSampler

    package_dir = tmp_path / "diffnmr_msdnmr_nless15"
    package_ckpt_dir = package_dir / "checkpoints"
    package_assets_dir = package_dir / "assets"
    package_retrieval_dir = package_assets_dir / "retrieval_database"
    package_ckpt_dir.mkdir(parents=True)
    package_retrieval_dir.mkdir(parents=True)
    package_weight = package_ckpt_dir / "DiffNMR_NMRNet_nless15_best.pdparams"
    package_retrieval = (
        package_retrieval_dir
        / "msd_nmr_nless15_retrieval_molecular_representations.csv"
    )
    package_weight.write_bytes(b"fake")
    package_retrieval.write_text("smiles,mol_rep\n")

    package_config = {
        "Vocabulary": "diffnmr_msdnmr_nless15",
        "Model": {
            "__init_params__": {
                "encoder_cfg": {
                    "pretrained_path": (
                        "./checkpoints/DiffNMR_NMRNet_nless15_best.pdparams"
                    )
                }
            }
        },
        "Sampler": {
            "retrieval_database_path": (
                "./assets/retrieval_database/"
                "msd_nmr_nless15_retrieval_molecular_representations.csv"
            )
        },
    }

    DiffNMRSampler._resolve_package_paths(
        package_config,
        config_base_dir=str(package_dir),
        checkpoint_dir=None,
    )

    assert package_config["Model"]["__init_params__"]["encoder_cfg"][
        "pretrained_path"
    ] == str(package_weight)
    assert package_config["Vocabulary"] == "diffnmr_msdnmr_nless15"
    assert package_config["Sampler"]["retrieval_database_path"] == str(
        package_retrieval
    )

    custom_ckpt_dir = tmp_path / "custom_checkpoints"
    custom_ckpt_dir.mkdir()
    custom_weight = custom_ckpt_dir / "DiffNMR_DiffGraphFormer_nless15_best.pdparams"
    custom_weight.write_bytes(b"fake")
    custom_config = {
        "CLIP": {
            "__init_params__": {
                "graph_encoder": {
                    "pretrained_model_path": (
                        "./checkpoints/DiffNMR_DiffGraphFormer_nless15_best.pdparams"
                    )
                }
            }
        }
    }

    DiffNMRSampler._resolve_package_paths(
        custom_config,
        config_base_dir=str(tmp_path / "config_dir"),
        checkpoint_dir=str(custom_ckpt_dir),
    )

    assert custom_config["CLIP"]["__init_params__"]["graph_encoder"][
        "pretrained_model_path"
    ] == str(custom_weight)


def test_molecular_sampler_does_not_require_packaged_diffnmr_vocab(tmp_path):
    from ppmat.sampler.diffnmr import DiffNMRSampler

    package_dir = tmp_path / "diffnmr_msdnmr_nless15"
    package_dir.mkdir()
    config = {"Vocabulary": "diffnmr_msdnmr_nless15"}

    DiffNMRSampler._resolve_package_paths(
        config,
        config_base_dir=str(package_dir),
        checkpoint_dir=None,
    )

    assert config["Vocabulary"] == "diffnmr_msdnmr_nless15"


def test_molecular_sampler_requires_retrieval_database_only_when_enabled(tmp_path):
    import pytest

    from ppmat.sampler.diffnmr import DiffNMRSampler

    package_dir = tmp_path / "diffnmr_msdnmr_nless15"
    package_dir.mkdir()
    config = {
        "Sampler": {
            "flag_retrieval_sampling": True,
            "retrieval_database_path": "./missing/retrieval.csv",
        }
    }

    with pytest.raises(FileNotFoundError, match="retrieval_database_path"):
        DiffNMRSampler._resolve_package_paths(
            config,
            config_base_dir=str(package_dir),
            checkpoint_dir=None,
        )

    config["Sampler"]["flag_retrieval_sampling"] = False
    DiffNMRSampler._resolve_package_paths(
        config,
        config_base_dir=str(package_dir),
        checkpoint_dir=None,
    )


def test_diffnmr_sample_entrypoint_supports_config_overrides():
    source = (ROOT / "spectrum_elucidation/sample.py").read_text()
    sampler_source = (ROOT / "ppmat/sampler/diffnmr.py").read_text()

    assert "parse_known_args(argv)" in source
    assert "config_overrides=config_overrides" in source
    assert "config_overrides: Optional[List[str]] = None" in sampler_source
    assert "OmegaConf.merge(config, cli_config)" in sampler_source
    assert "_apply_package_support_files" not in sampler_source
    assert "_replace_with_package_file" not in sampler_source


def test_diffnmr_train_entrypoint_builds_training_statistics_for_eval_and_test():
    source = (ROOT / "spectrum_elucidation/train.py").read_text()
    dataloader_helper = source.split(
        "def read_independent_dataloader_config", maxsplit=1
    )[1].split("def parse_args", maxsplit=1)[0]

    assert 'train_data_cfg = config["Dataset"].get("train")' in dataloader_helper
    assert (
        "train_loader = build_dataloader(train_data_cfg, vocab=vocab)"
        in dataloader_helper
    )
    assert "At least one of Global.do_train" in dataloader_helper
    assert "val_data_cfg must be defined when do_eval is true." in dataloader_helper
    assert "spectrum_elucidation/configs/diffnmr/DiffNMR.yaml" in source
    assert "build_metric(metric_cfg)" not in source
    assert (
        "dataloaders.train_dataloader"
        in (ROOT / "ppmat/datasets/msd_nmr_dataset.py").read_text()
    )


def test_diffnmr_uses_molecular_sampler_from_sampler_package():
    source = (ROOT / "spectrum_elucidation/sample.py").read_text()
    sampler_path = ROOT / "ppmat/sampler/molecular_sampler.py"
    diffnmr_sampler_path = ROOT / "ppmat/sampler/diffnmr.py"
    diffnmr_sampler_source = diffnmr_sampler_path.read_text()
    legacy_sample_dir = ROOT / "ppmat/sample"

    assert sampler_path.exists()
    assert diffnmr_sampler_path.exists()
    assert not legacy_sample_dir.exists()
    assert "from ppmat.sampler import MolecularSampler" in source
    assert "class MolecularSampler" in sampler_path.read_text()
    assert "class DiffNMRSampler" in diffnmr_sampler_source
    assert 'setattr(self.model, "clip"' not in diffnmr_sampler_source
    assert 'setattr(self.model, "streaming_adapter"' not in diffnmr_sampler_source
