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
import pgl
import pytest
from omegaconf import OmegaConf
from pymatgen.core import Structure

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.models.dimenetpp.dimenetpp import DimeNetPlusPlus
from ppmat.models.dimenetpp.dimenetpp_cinn import DimeNetPPTensorCore
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
    """Exercise public workflow hooks without paying GPU compilation cost."""

    cores = {}

    def get_runtime(model):
        mode = "train" if model.training else "eval"
        key = (id(model), mode)
        core = cores.get(key)
        if core is None:
            core = DimeNetPPTensorCore(model)
            core.training = model.training
            cores[key] = core
        return core

    monkeypatch.setattr(
        DimeNetPlusPlus, "validate_execution_backend", lambda self: None
    )
    monkeypatch.setattr(
        DimeNetPlusPlus, "_validate_cinn_environment", lambda self: None
    )
    monkeypatch.setattr(DimeNetPlusPlus, "_get_cinn_runtime", get_runtime)
    return cores


def _make_graph(atom_types, coordinate_shift=0.0):
    num_nodes = len(atom_types)
    coordinates = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )[:num_nodes]
    coordinates = coordinates + coordinate_shift
    edges = np.asarray(
        [
            [source, destination]
            for source in range(num_nodes)
            for destination in range(num_nodes)
            if source != destination
        ],
        dtype=np.int64,
    )
    lattice = np.eye(3, dtype=np.float32) * 10.0
    return pgl.Graph(
        edges=edges,
        num_nodes=num_nodes,
        node_feat={
            "atom_types": np.asarray(atom_types, dtype=np.int64),
            "frac_coords": coordinates / 10.0,
            "cart_coords": coordinates,
            "lattice": lattice.reshape([1, 3, 3]),
            "num_atoms": np.asarray([num_nodes], dtype=np.int64),
        },
        edge_feat={
            "pbc_offset": np.zeros([edges.shape[0], 3], dtype=np.int64),
            "num_edges": np.asarray([edges.shape[0]], dtype=np.int64),
        },
    )


def _make_samples():
    return [
        {
            "graph": _make_graph([1, 8, 14]),
            "formation_energy_per_atom": np.asarray([0.25], dtype=np.float32),
        },
        {
            "graph": _make_graph([6, 7, 8], coordinate_shift=0.125),
            "formation_energy_per_atom": np.asarray([-0.75], dtype=np.float32),
        },
    ]


def _make_loader():
    return paddle.io.DataLoader(
        _make_samples(),
        batch_size=2,
        shuffle=False,
        collate_fn=DefaultCollator(),
        return_list=True,
    )


def _make_model(execution_backend="cinn"):
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        model = DimeNetPlusPlus(
            out_channels=1,
            hidden_channels=8,
            num_blocks=1,
            int_emb_size=4,
            basis_emb_size=2,
            out_emb_channels=8,
            num_spherical=2,
            num_embeddings=95,
            num_radial=2,
            cutoff=7.0,
            num_before_skip=1,
            num_after_skip=1,
            num_output_layers=1,
            readout="mean",
            loss_type="mse_loss",
            execution_backend=execution_backend,
        )
    model.rbf.freq.set_value(
        paddle.arange(1, model.rbf.freq.shape[0] + 1, dtype="float32") * np.pi
    )
    return model


def _trainer_config(output_dir, max_epochs, execution_backend="cinn"):
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
        "execution_backend": execution_backend,
    }


def _build_trainer(model, output_dir, max_epochs, execution_backend="cinn"):
    optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
    trainer = BaseTrainer(
        _trainer_config(output_dir, max_epochs, execution_backend),
        model,
        train_dataloader=_make_loader(),
        val_dataloader=_make_loader(),
        optimizer=optimizer,
    )
    return trainer, optimizer


def _assert_state_dict_close(actual, expected, atol=2e-6):
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(
            actual[key].numpy(), expected[key].numpy(), atol=atol, rtol=atol
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


def _predictor_config(checkpoint_path, execution_backend="cinn"):
    return {
        "Model": {
            "__class_name__": "DimeNetPlusPlus",
            "__init_params__": {
                "out_channels": 1,
                "hidden_channels": 8,
                "num_blocks": 1,
                "int_emb_size": 4,
                "basis_emb_size": 2,
                "out_emb_channels": 8,
                "num_spherical": 2,
                "num_embeddings": 95,
                "num_radial": 2,
                "cutoff": 7.0,
                "num_before_skip": 1,
                "num_after_skip": 1,
                "num_output_layers": 1,
                "readout": "mean",
                "loss_type": "mse_loss",
                "property_names": "formation_energy_per_atom",
            },
        },
        "Predict": {
            "checkpoint_path": str(checkpoint_path),
            "execution_backend": execution_backend,
            "eval_with_no_grad": True,
            "graph_converter": {
                "__class_name__": "FindPointsInSpheres",
                "__init_params__": {"cutoff": 4.0, "num_cpus": 1},
            },
        },
    }


def test_cinn_property_predictor_loads_checkpoint_and_batches_cifs(
    tmp_path, tensor_runtime_proxy
):
    model = _make_model(execution_backend="eager")
    checkpoint_path = tmp_path / "best.pdparams"
    paddle.save(model.state_dict(), str(checkpoint_path))
    config_path = tmp_path / "dimenetpp.yaml"
    OmegaConf.save(OmegaConf.create(_predictor_config(checkpoint_path)), config_path)

    predictor = PropertyPredictor(config_path=config_path, device="cpu")
    cif_path = (
        Path(__file__).resolve().parents[1]
        / "property_prediction"
        / "example_data"
        / "cifs"
        / "mp-18767-LiMnO2.cif"
    )
    single = predictor.from_cif_file(str(cif_path))
    structure = Structure.from_file(cif_path)
    batch = predictor.from_structures([structure, structure])

    assert predictor.execution_backend == "cinn"
    assert predictor.model._cinn_warmed_modes == {"eval"}
    assert len(batch) == 2
    assert single.keys() == {"formation_energy_per_atom"}
    np.testing.assert_allclose(
        [item["formation_energy_per_atom"] for item in batch],
        [single["formation_energy_per_atom"]] * 2,
        atol=1e-6,
        rtol=1e-6,
    )
    assert predictor.model.state_dict().keys() == model.state_dict().keys()
    assert all(not key.startswith("model.") for key in predictor.model.state_dict())


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") != "1",
    reason="Set PPMAT_RUN_CINN_WORKFLOW_TESTS=1 for the GPU workflow smoke.",
)
def test_gpu_cinn_trainer_checkpoint_and_property_predictor(
    tmp_path,
):
    if not paddle.is_compiled_with_cuda():
        pytest.skip("Paddle was not compiled with CUDA.")
    if not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle was not compiled with CINN.")

    paddle.set_device("gpu:0")
    initial_model = _make_model(execution_backend="eager")
    initial_state = {
        key: value.clone() for key, value in initial_model.state_dict().items()
    }
    reference_model = _make_model(execution_backend="eager")
    reference_model.set_state_dict(initial_state)
    reference_trainer, reference_optimizer = _build_trainer(
        reference_model, tmp_path / "eager_trainer", 1, execution_backend="eager"
    )
    model = _make_model(execution_backend="cinn")
    model.set_state_dict(initial_state)
    trainer, optimizer = _build_trainer(model, tmp_path / "cinn_trainer", 1)
    with paddle.utils.unique_name.guard():
        reference_trainer.train()
    with paddle.utils.unique_name.guard():
        trainer.train()
    _assert_state_dict_close(
        model.state_dict(), reference_model.state_dict(), atol=2e-5
    )
    _assert_state_dict_close(
        optimizer.state_dict(), reference_optimizer.state_dict(), atol=2e-5
    )

    resume_checkpoint = tmp_path / "cinn_trainer" / "checkpoints" / "latest"
    reference_resumed_model = _make_model(execution_backend="eager")
    reference_resumed_trainer, reference_resumed_optimizer = _build_trainer(
        reference_resumed_model,
        tmp_path / "eager_resumed",
        2,
        execution_backend="eager",
    )
    resumed_model = _make_model(execution_backend="cinn")
    resumed_trainer, resumed_optimizer = _build_trainer(
        resumed_model, tmp_path / "cinn_resumed", 2
    )
    with paddle.utils.unique_name.guard():
        reference_resumed_trainer.train(resume_from_checkpoint=str(resume_checkpoint))
    with paddle.utils.unique_name.guard():
        resumed_trainer.train(resume_from_checkpoint=str(resume_checkpoint))
    _assert_state_dict_close(
        resumed_model.state_dict(),
        reference_resumed_model.state_dict(),
        atol=2e-5,
    )
    _assert_state_dict_close(
        resumed_optimizer.state_dict(),
        reference_resumed_optimizer.state_dict(),
        atol=2e-5,
    )

    checkpoint_path = tmp_path / "cinn_resumed" / "checkpoints" / "latest.pdparams"
    config_path = tmp_path / "dimenetpp_cinn.yaml"
    OmegaConf.save(OmegaConf.create(_predictor_config(checkpoint_path)), config_path)
    predictor = PropertyPredictor(config_path=config_path, device="gpu:0")
    reference_config_path = tmp_path / "dimenetpp_eager.yaml"
    OmegaConf.save(
        OmegaConf.create(_predictor_config(checkpoint_path, "eager")),
        reference_config_path,
    )
    reference_predictor = PropertyPredictor(
        config_path=reference_config_path, device="gpu:0"
    )
    cif_path = (
        Path(__file__).resolve().parents[1]
        / "property_prediction"
        / "example_data"
        / "cifs"
        / "mp-18767-LiMnO2.cif"
    )
    result = predictor.from_cif_file(str(cif_path))
    reference_result = reference_predictor.from_cif_file(str(cif_path))

    assert trainer.state.global_step == 1
    assert reference_trainer.state.global_step == 1
    assert resumed_trainer.state.global_step == 2
    assert reference_resumed_trainer.state.global_step == 2
    assert resumed_model._cinn_warmed_modes == {"train", "eval"}
    assert result.keys() == {"formation_energy_per_atom"}
    np.testing.assert_allclose(
        result["formation_energy_per_atom"],
        reference_result["formation_energy_per_atom"],
        atol=2e-5,
        rtol=2e-5,
    )
    assert predictor.model._cinn_warmed_modes == {"eval"}
    assert predictor.model.state_dict().keys() == model.state_dict().keys()
