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

import os
from pathlib import Path

import numpy as np
import paddle
import pytest
from omegaconf import OmegaConf

from ppmat.datasets.collate_fn import RadiusGraphCollator
from ppmat.models.common.graph_converter import RadiusGraphConverter
from ppmat.models.spherenet.spherenet import SphereNet
from ppmat.models.spherenet.spherenet_cinn import SphereNetTensorCore
from ppmat.predictor import PropertyPredictor
from ppmat.trainer.base_trainer import BaseTrainer


@pytest.fixture(autouse=True)
def _cpu_device():
    original_device = paddle.get_device()
    paddle.set_device("cpu")
    yield
    paddle.set_device(original_device)


@pytest.fixture
def tensor_runtime_proxy(monkeypatch):
    """Exercise public workflow hooks without requiring a GPU compiler."""

    cores = {}

    def get_runtime(model):
        mode = "train" if model.training else "eval"
        key = (id(model), mode)
        core = cores.get(key)
        if core is None:
            core = SphereNetTensorCore(model)
            core.training = model.training
            cores[key] = core
        return core

    monkeypatch.setattr(SphereNet, "validate_execution_backend", lambda self: None)
    monkeypatch.setattr(SphereNet, "_validate_cinn_environment", lambda self: None)
    monkeypatch.setattr(SphereNet, "_get_cinn_runtime", get_runtime)
    return cores


def _make_graphs():
    converter = RadiusGraphConverter(
        cutoff=5.0,
        return_triplet_indices=True,
        num_cpus=1,
    )
    graph_a = converter.from_arrays(
        np.asarray([6, 1, 7, 8], dtype=np.int64),
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.2, 1.1, 0.1],
                [0.1, 0.3, 1.2],
            ],
            dtype=np.float32,
        ),
    )
    graph_b = converter.from_arrays(
        np.asarray([6, 1, 7, 8, 9], dtype=np.int64),
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [0.0, 1.3, 0.0],
                [0.0, 0.0, 1.4],
                [1.0, 1.0, 0.2],
            ],
            dtype=np.float32,
        ),
    )
    return graph_a, graph_b


def _make_samples():
    graph_a, graph_b = _make_graphs()
    return [
        {"graph": graph_a, "mu": np.asarray([0.25], dtype=np.float32)},
        {"graph": graph_b, "mu": np.asarray([-0.75], dtype=np.float32)},
    ]


def _make_loader():
    return paddle.io.DataLoader(
        _make_samples(),
        batch_size=2,
        shuffle=False,
        collate_fn=RadiusGraphCollator(),
        return_list=True,
    )


def _make_model(execution_backend="cinn"):
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        return SphereNet(
            num_layers=1,
            hidden_channels=8,
            out_channels=1,
            int_emb_size=4,
            basis_emb_size_dist=4,
            basis_emb_size_angle=4,
            basis_emb_size_torsion=4,
            out_emb_channels=8,
            num_spherical=2,
            num_radial=3,
            num_before_skip=1,
            num_after_skip=1,
            num_output_layers=1,
            property_name="mu",
            execution_backend=execution_backend,
        )


def _trainer_config(output_dir, max_epochs):
    return {
        "max_epochs": max_epochs,
        "output_dir": str(output_dir),
        "save_freq": 1,
        "log_freq": 100,
        "start_eval_epoch": 1,
        "eval_freq": 1,
        "seed": 2026,
        "pretrained_model_path": None,
        "resume_from_checkpoint": None,
        "compute_metric_during_train": False,
        "use_amp": False,
        "eval_with_no_grad": True,
        "gradient_accumulation_steps": 1,
        "best_metric_indicator": "eval_loss",
        "name_for_best_metric": "loss",
        "greater_is_better": False,
        "execution_backend": "cinn",
    }


def _build_trainer(model, output_dir, max_epochs):
    optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        parameters=model.parameters(),
    )
    trainer = BaseTrainer(
        _trainer_config(output_dir, max_epochs),
        model,
        train_dataloader=_make_loader(),
        val_dataloader=_make_loader(),
        optimizer=optimizer,
    )
    return trainer, optimizer


def _assert_state_dict_close(actual, expected, atol=3e-6):
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(
            actual[key].numpy(),
            expected[key].numpy(),
            atol=atol,
            rtol=atol,
            err_msg=key,
        )


def test_cinn_trainer_checkpoint_resume_matches_uninterrupted_training(
    tmp_path, tensor_runtime_proxy
):
    initial_model = _make_model()
    initial_state = initial_model.state_dict()

    first_model = _make_model()
    first_model.set_state_dict(initial_state)
    first_trainer, _ = _build_trainer(first_model, tmp_path / "first", 1)
    with paddle.utils.unique_name.guard():
        first_trainer.train()

    checkpoint_dir = tmp_path / "first" / "checkpoints"
    for prefix in ("epoch_1", "latest", "best"):
        for suffix in ("pdparams", "pdopt", "pdstates"):
            assert (checkpoint_dir / f"{prefix}.{suffix}").is_file()
    checkpoint_state = paddle.load(str(checkpoint_dir / "latest.pdparams"))
    assert checkpoint_state.keys() == first_model.state_dict().keys()
    assert all(not key.startswith("model.") for key in checkpoint_state)

    resumed_model = _make_model()
    resumed_trainer, resumed_optimizer = _build_trainer(
        resumed_model, tmp_path / "resumed", 2
    )
    with paddle.utils.unique_name.guard():
        resumed_trainer.train(resume_from_checkpoint=str(checkpoint_dir / "latest"))

    uninterrupted_model = _make_model()
    uninterrupted_model.set_state_dict(initial_state)
    uninterrupted_trainer, uninterrupted_optimizer = _build_trainer(
        uninterrupted_model, tmp_path / "uninterrupted", 2
    )
    with paddle.utils.unique_name.guard():
        uninterrupted_trainer.train()

    assert first_trainer.state.global_step == 1
    assert resumed_trainer.state.global_step == 2
    assert resumed_trainer.state.epoch == 2
    assert resumed_optimizer.get_lr() == uninterrupted_optimizer.get_lr()
    _assert_state_dict_close(
        resumed_model.state_dict(), uninterrupted_model.state_dict()
    )
    _assert_state_dict_close(
        resumed_optimizer.state_dict(), uninterrupted_optimizer.state_dict()
    )
    assert resumed_model._cinn_warmed_modes == {"train", "eval"}


def _predictor_config(checkpoint_path):
    return {
        "Model": {
            "__class_name__": "SphereNet",
            "__init_params__": {
                "num_layers": 1,
                "hidden_channels": 8,
                "out_channels": 1,
                "int_emb_size": 4,
                "basis_emb_size_dist": 4,
                "basis_emb_size_angle": 4,
                "basis_emb_size_torsion": 4,
                "out_emb_channels": 8,
                "num_spherical": 2,
                "num_radial": 3,
                "num_before_skip": 1,
                "num_after_skip": 1,
                "num_output_layers": 1,
                "property_name": "mu",
            },
        },
        "Predict": {
            "checkpoint_path": str(checkpoint_path),
            "execution_backend": "cinn",
            "eval_with_no_grad": True,
            "graph_converter": {
                "__class_name__": "RadiusGraphConverter",
                "__init_params__": {
                    "cutoff": 5.0,
                    "return_triplet_indices": True,
                    "num_cpus": 1,
                },
            },
        },
    }


def _example_xyz_path():
    return (
        Path(__file__).resolve().parents[4]
        / "property_prediction"
        / "example_data"
        / "molecules"
        / "isoguvacine.xyz"
    )


def test_cinn_property_predictor_loads_checkpoint_and_predicts_xyz(
    tmp_path, tensor_runtime_proxy
):
    model = _make_model(execution_backend="eager")
    checkpoint_path = tmp_path / "best.pdparams"
    paddle.save(model.state_dict(), str(checkpoint_path))
    config_path = tmp_path / "spherenet.yaml"
    OmegaConf.save(OmegaConf.create(_predictor_config(checkpoint_path)), config_path)

    predictor = PropertyPredictor(config_path=config_path, device="cpu")
    result = predictor.from_xyz_file(str(_example_xyz_path()))

    assert predictor.execution_backend == "cinn"
    assert predictor.model._cinn_warmed_modes == {"eval"}
    assert result.keys() == {"mu"}
    assert np.asarray(result["mu"]).shape == (1, 1)
    assert predictor.model.state_dict().keys() == model.state_dict().keys()
    assert all(not key.startswith("model.") for key in predictor.model.state_dict())


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_SPHERENET_CINN_WORKFLOW_TESTS") != "1",
    reason="Set PPMAT_RUN_SPHERENET_CINN_WORKFLOW_TESTS=1 for the GPU workflow.",
)
def test_gpu_cinn_trainer_checkpoint_and_property_predictor(tmp_path):
    if not paddle.is_compiled_with_cuda():
        pytest.skip("Paddle was not compiled with CUDA.")
    if not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle was not compiled with CINN.")

    paddle.set_device("gpu:0")
    model = _make_model()
    trainer, _ = _build_trainer(model, tmp_path / "trainer", 1)
    with paddle.utils.unique_name.guard():
        trainer.train()

    checkpoint_path = tmp_path / "trainer" / "checkpoints" / "best.pdparams"
    config_path = tmp_path / "spherenet.yaml"
    OmegaConf.save(OmegaConf.create(_predictor_config(checkpoint_path)), config_path)
    predictor = PropertyPredictor(config_path=config_path, device="gpu:0")
    result = predictor.from_xyz_file(str(_example_xyz_path()))

    assert trainer.state.global_step == 1
    assert result.keys() == {"mu"}
    assert predictor.model._cinn_warmed_modes == {"eval"}
    assert predictor.model.state_dict().keys() == model.state_dict().keys()
