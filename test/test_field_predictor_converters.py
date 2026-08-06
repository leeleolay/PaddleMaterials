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
import pytest

from ppmat.datasets.build_field import BuildField
from ppmat.models.common.graph_converter import RadiusGraphConverter
from ppmat.predictor import FieldPredictor
from ppmat.predictor.field_io import read_cube_density
from ppmat.utils.io import write_cube


class _FieldModel:
    target_name = "density"
    cutoff = 3.0

    def __call__(self, batch, return_loss=True, return_prediction=True):
        self.forward_flags = (return_loss, return_prediction)
        prediction = paddle.sum(batch["grid_coord"], axis=-1)
        return {"loss_dict": {}, "pred_dict": {"density": prediction}}

    def eval(self):
        return self


def _atom_vocab(symbols=("C",), atomic_numbers=(6,)):
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


def _fake_model_loader(predict_config, vocab=None):
    def load_inference_model(predictor):
        predictor.model = _FieldModel()
        predictor.config = {
            "Model": {"__init_params__": {"target_name": "density"}},
        }
        predictor.predict_config = predict_config
        predictor.device = "cpu"
        predictor.vocab = vocab or _atom_vocab()

    return load_inference_model


def _build_field_cfg(
    name="density",
    value_unit="electron/angstrom^3",
    coordinate_unit="angstrom",
):
    return {
        "format": "array",
        "name": name,
        "value_unit": value_unit,
        "coordinate_unit": coordinate_unit,
        "num_cpus": 1,
    }


def _build_graph_cfg(coordinate_unit="angstrom"):
    return {
        "__class_name__": "RadiusGraphConverter",
        "__init_params__": {
            "cutoff": 3.0,
            "coordinate_unit": coordinate_unit,
            "inclusive_cutoff": True,
            "atom_vocab": {},
            "include_distance": False,
        },
    }


def test_field_predictor_builds_configured_fields(monkeypatch):
    predict_config = {
        "build_field_cfg": _build_field_cfg(),
        "build_graph_cfg": _build_graph_cfg(),
    }
    monkeypatch.setattr(
        FieldPredictor,
        "load_inference_model",
        _fake_model_loader(predict_config),
    )

    predictor = FieldPredictor(config_path="unused", checkpoint_path="unused")

    assert predictor.field_converter == BuildField(
        format="array",
        name="density",
        value_unit="electron/angstrom^3",
        coordinate_unit="angstrom",
    )
    assert isinstance(predictor.graph_converter_fn, RadiusGraphConverter)
    assert predictor.graph_converter_fn.coordinate_unit == "angstrom"
    assert not hasattr(predictor, "grid_converter")


def test_field_predictor_requests_prediction_without_loss():
    predictor = FieldPredictor.__new__(FieldPredictor)
    predictor.model = _FieldModel()
    predictor.target_name = "density"
    predictor.post_transforms = None
    grid_coord = paddle.zeros([1, 2, 3], dtype="float32")

    prediction = predictor._predict_grid(
        graph=object(),
        grid_coord=grid_coord,
        info={"cell": paddle.eye(3, dtype="float32")},
        grid_batch_size=2,
    )

    assert predictor.model.forward_flags == (False, True)
    assert list(prediction.shape) == [2]


def test_field_predictor_requires_coordinate_unit_in_field_config(monkeypatch):
    monkeypatch.setattr(
        FieldPredictor,
        "load_inference_model",
        _fake_model_loader(
            {
                "coordinate_unit": "angstrom",
                "build_graph_cfg": _build_graph_cfg(),
                "build_field_cfg": {
                    "format": "array",
                    "name": "density",
                    "value_unit": "unknown",
                    "num_cpus": 1,
                },
            }
        ),
    )

    with pytest.raises(TypeError, match="coordinate_unit"):
        FieldPredictor(config_path="unused", checkpoint_path="unused")


def test_field_predictor_rejects_registry_config(monkeypatch):
    monkeypatch.setattr(
        FieldPredictor,
        "load_inference_model",
        _fake_model_loader(
            {
                "build_graph_cfg": _build_graph_cfg(),
                "build_field_cfg": {
                    "__class_name__": "BuildField",
                    "__init_params__": {
                        "format": "array",
                        "name": "density",
                        "value_unit": "unknown",
                        "coordinate_unit": "angstrom",
                    },
                },
            }
        ),
    )

    with pytest.raises(TypeError, match="__class_name__"):
        FieldPredictor(config_path="unused", checkpoint_path="unused")


def test_field_predictor_requires_field_name_to_match_model(monkeypatch):
    monkeypatch.setattr(
        FieldPredictor,
        "load_inference_model",
        _fake_model_loader(
            {
                "build_graph_cfg": _build_graph_cfg(),
                "build_field_cfg": _build_field_cfg(
                    name="potential",
                    value_unit="unknown",
                ),
            }
        ),
    )

    with pytest.raises(ValueError, match="target_name"):
        FieldPredictor(config_path="unused", checkpoint_path="unused")


def test_field_predictor_requires_graph_cutoff_to_match_model(monkeypatch):
    graph_cfg = _build_graph_cfg()
    graph_cfg["__init_params__"]["cutoff"] = 4.0
    monkeypatch.setattr(
        FieldPredictor,
        "load_inference_model",
        _fake_model_loader(
            {
                "build_graph_cfg": graph_cfg,
                "build_field_cfg": _build_field_cfg(),
            }
        ),
    )

    with pytest.raises(ValueError, match="model cutoff"):
        FieldPredictor(config_path="unused", checkpoint_path="unused")


def test_reference_cube_converts_to_field_coordinate_unit(tmp_path):
    cube_path = tmp_path / "density.cube"
    write_cube(
        cube_path,
        atom_numbers=np.asarray([6]),
        atom_coord=np.zeros((1, 3), dtype=np.float32),
        density=np.arange(8, dtype=np.float32),
        info={
            "shape": [2, 2, 2],
            "cell": np.eye(3, dtype=np.float32) * 2,
            "coordinate_unit": "angstrom",
        },
    )

    density, _, info = read_cube_density(
        cube_path,
        BuildField(
            format="array",
            name="density",
            value_unit="unknown",
            coordinate_unit="angstrom",
        ),
    )

    np.testing.assert_allclose(density, np.arange(8))
    np.testing.assert_allclose(info["cell"], np.eye(3) * 2)
    assert info["coordinate_unit"] == "angstrom"


def test_field_predictor_cube_writer_consumes_plain_vocab_mapping(monkeypatch):
    predict_config = {
        "build_field_cfg": _build_field_cfg(),
        "build_graph_cfg": _build_graph_cfg(),
    }
    vocab = _atom_vocab(
        symbols=("H", "Cl"),
        atomic_numbers=(1, 17),
    )
    monkeypatch.setattr(
        FieldPredictor,
        "load_inference_model",
        _fake_model_loader(predict_config, vocab=vocab),
    )
    predictor = FieldPredictor(config_path="unused", checkpoint_path="unused")

    writer = predictor._get_cube_writer(object())

    assert predictor.vocab is vocab
    assert writer.keywords["idx2atom_num"] == {0: 1, 1: 17}


def test_from_dataset_uses_full_grid_and_standard_path_overrides(tmp_path):
    predictor = FieldPredictor.__new__(FieldPredictor)
    predictor.vocab = _atom_vocab()
    predictor.target_name = "density"
    captured = {}
    configured_path = tmp_path / "configured" / "ethane" / "ethane_test"

    predictor._dataset_config = lambda split: (
        "test",
        {
            "__class_name__": "MD17DensityDataset",
            "__init_params__": {
                "path": str(configured_path),
                "split": "test",
                "grid_sampler_cfg": {"n_samples": 2},
            },
        },
    )

    class _Dataset:
        idx2atom_num = {0: 6}

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return {
                "graph": object(),
                "density": np.arange(8, dtype=np.float32),
                "grid_coord": np.zeros((8, 3), dtype=np.float32),
                "info": {
                    "shape": [2, 2, 2],
                    "cell": np.eye(3, dtype=np.float32),
                    "origin": np.zeros(3, dtype=np.float32),
                    "file_name": "sample",
                },
            }

    def build_dataset(config, params):
        captured.update(params)
        return _Dataset()

    predictor._build_dataset = build_dataset
    predictor.from_data = lambda *args, **kwargs: {
        "density": np.zeros(8, dtype=np.float32)
    }
    predictor._save_outputs = lambda **kwargs: {}
    predictor._log_metrics = lambda *args, **kwargs: None

    predictor.from_dataset(
        data_root=str(tmp_path / "override"),
        grid_batch_size=4,
    )

    assert captured["path"] == str(tmp_path / "override" / "ethane" / "ethane_test")
    assert "grid_sampler_cfg" not in captured
