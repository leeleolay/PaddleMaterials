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
from ppmat.models.comformer.comformer import iComformer
from ppmat.models.comformer.comformer_cinn import ComformerTensorCore
from ppmat.models.comformer.comformer_cinn import compile_comformer
from ppmat.models.comformer.comformer_cinn import compile_comformer_cinn
from ppmat.models.comformer.comformer_cinn import graph_to_tensor_batch
from ppmat.predictor import PropertyPredictor
from ppmat.trainer.base_trainer import BaseTrainer


@pytest.fixture(autouse=True)
def _cpu_device():
    original_device = paddle.get_device()
    paddle.set_device("cpu")
    yield
    paddle.set_device(original_device)


def _make_graph(num_nodes: int, offset: float = 0.0) -> pgl.Graph:
    edges = np.asarray(
        [
            [source, target]
            for source in range(num_nodes)
            for target in range(num_nodes)
            if source != target
        ],
        dtype=np.int64,
    )
    num_edges = edges.shape[0]
    edge_r = (
        np.arange(num_edges * 3, dtype=np.float32).reshape(num_edges, 3) / 20.0
        + np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        + offset
    )
    edge_nei = np.tile(
        np.eye(3, dtype=np.float32)[None, :, :],
        (num_edges, 1, 1),
    ) * (1.0 + offset)
    node_feat = np.zeros([num_nodes, 92], dtype=np.float32)
    node_feat[np.arange(num_nodes), np.arange(num_nodes) + 1] = 1.0
    return pgl.Graph(
        edges,
        num_nodes=num_nodes,
        node_feat={"node_feat": node_feat},
        edge_feat={"r": edge_r, "nei": edge_nei},
    )


def _make_graphs() -> tuple[pgl.Graph, pgl.Graph]:
    return _make_graph(3), _make_graph(4, offset=0.1)


def _make_model(execution_backend: str = "eager") -> iComformer:
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        return iComformer(
            conv_layers=1,
            edge_layers=1,
            atom_input_features=92,
            edge_features=8,
            triplet_input_features=6,
            node_features=8,
            fc_features=8,
            output_features=1,
            execution_backend=execution_backend,
        )


def _assert_allclose(actual, expected, atol=2e-6, rtol=2e-6):
    np.testing.assert_allclose(
        actual.numpy(),
        expected.numpy(),
        atol=atol,
        rtol=rtol,
    )


def _assert_state_dict_close(actual, expected, atol=2e-6, rtol=2e-6):
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(
            actual[key].numpy(),
            expected[key].numpy(),
            atol=atol,
            rtol=rtol,
            err_msg=key,
        )


def test_graph_to_tensor_batch_packs_single_and_dynamic_batches():
    graph_a, graph_b = _make_graphs()
    single = graph_to_tensor_batch(graph_a)
    batch = graph_to_tensor_batch([graph_a, graph_b])

    assert single.node_feat.shape == [3, 92]
    assert single.edge_r.shape == [6, 3]
    assert single.edge_nei.shape == [6, 3, 3]
    assert single.graph_node_count.numpy().tolist() == [3]
    assert batch.node_feat.shape == [7, 92]
    assert batch.edge_r.shape == [18, 3]
    assert batch.graph_node_count.numpy().tolist() == [3, 4]
    assert batch.node_graph_id.numpy().tolist() == [0, 0, 0, 1, 1, 1, 1]


def test_graph_to_tensor_batch_validates_required_features():
    graph = pgl.Graph(
        np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        num_nodes=2,
        node_feat={"node_feat": np.zeros([2, 92], dtype=np.float32)},
        edge_feat={"r": np.ones([2, 3], dtype=np.float32)},
    )
    with pytest.raises(ValueError, match=r"graph\.edge_feat\['nei'\] is required"):
        graph_to_tensor_batch(graph)


def test_tensor_core_matches_pgl_eager_for_dynamic_batches():
    model = _make_model()
    model.eval()
    core = ComformerTensorCore(model)
    core.training = False
    graph_a, graph_b = _make_graphs()

    for graph in (graph_a, pgl.Graph.batch([graph_a, graph_b]), _make_graph(5, 0.2)):
        expected = model._forward_eager({"graph": graph})
        actual = core(*graph_to_tensor_batch(graph))
        _assert_allclose(actual, expected)


def test_pir_static_core_supports_dynamic_node_edge_and_batch_sizes():
    model = _make_model()
    model.eval()
    core = ComformerTensorCore(model)
    core.training = False
    static_core = compile_comformer(core, backend=None, full_graph=True)
    static_core.training = False
    graph_a, graph_b = _make_graphs()

    for graph in (graph_a, pgl.Graph.batch([graph_a, graph_b]), _make_graph(5, 0.2)):
        packed = graph_to_tensor_batch(graph)
        _assert_allclose(static_core(*packed), core(*packed))


def test_public_backend_preserves_forward_predict_and_checkpoint_contract(monkeypatch):
    reference = _make_model()
    model = _make_model()
    model.set_state_dict(reference.state_dict())
    reference.eval()
    model.eval()
    expected_keys = list(model.state_dict())
    core = ComformerTensorCore(model)
    core.training = False
    monkeypatch.setattr(model, "_validate_cinn_environment", lambda: None)
    monkeypatch.setattr(model, "_get_cinn_runtime", lambda: core)
    model.set_execution_backend("cinn")

    graph_a, graph_b = _make_graphs()
    expected_batch = {
        "graph": pgl.Graph.batch([graph_a, graph_b]),
        "formation_energy_per_atom": paddle.to_tensor(
            [[0.25], [-0.75]], dtype="float32"
        ),
    }
    runtime_graphs = _make_graphs()
    runtime_batch = {
        "graph": pgl.Graph.batch(list(runtime_graphs)),
        "formation_energy_per_atom": paddle.to_tensor(
            [[0.25], [-0.75]], dtype="float32"
        ),
    }
    expected = reference(expected_batch)
    actual = model(runtime_batch)

    _assert_allclose(actual["loss_dict"]["loss"], expected["loss_dict"]["loss"])
    _assert_allclose(
        actual["pred_dict"]["formation_energy_per_atom"],
        expected["pred_dict"]["formation_energy_per_atom"],
    )
    expected_prediction = reference.predict([graph_a, graph_b])
    actual_prediction = model.predict(list(runtime_graphs))
    np.testing.assert_allclose(
        [item["formation_energy_per_atom"] for item in actual_prediction],
        [item["formation_energy_per_atom"] for item in expected_prediction],
        atol=2e-6,
        rtol=2e-6,
    )
    single_list_prediction = model.predict([_make_graphs()[0]])
    assert isinstance(single_list_prediction, list)
    assert len(single_list_prediction) == 1
    assert list(model.state_dict()) == expected_keys
    assert all(not key.startswith("model.") for key in model.state_dict())


def test_public_cinn_backend_fails_fast_on_cpu():
    model = _make_model(execution_backend="cinn")
    with pytest.raises(RuntimeError, match="requires a GPU device"):
        model({"graph": _make_graph(3)}, return_loss=False)


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_CINN_TESTS") != "1",
    reason="Set PPMAT_RUN_CINN_TESTS=1 to run the GPU CINN inference smoke.",
)
def test_gpu_cinn_inference_matches_tensor_eager_with_dynamic_batches():
    if not paddle.is_compiled_with_cuda():
        pytest.skip("Paddle was not compiled with CUDA.")
    if not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle was not compiled with CINN.")

    paddle.set_device("gpu:0")
    model = _make_model()
    model.eval()
    core = ComformerTensorCore(model)
    core.training = False
    compiled = compile_comformer_cinn(core)
    compiled.training = False
    graph_a, graph_b = _make_graphs()
    for graph in (graph_a, pgl.Graph.batch([graph_a, graph_b]), _make_graph(5, 0.2)):
        packed = graph_to_tensor_batch(graph)
        _assert_allclose(compiled(*packed), core(*packed))


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_CINN_TRAINING_TESTS") != "1",
    reason="Set PPMAT_RUN_CINN_TRAINING_TESTS=1 to run the GPU training canary.",
)
def test_gpu_cinn_backward_adam_and_dynamic_batch_parity():
    if not paddle.is_compiled_with_cuda():
        pytest.skip("Paddle was not compiled with CUDA.")
    if not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle was not compiled with CINN.")

    paddle.set_device("gpu:0")
    reference_model = _make_model()
    cinn_model = _make_model()
    cinn_model.set_state_dict(reference_model.state_dict())
    initial_state = {
        key: value.clone() for key, value in reference_model.state_dict().items()
    }
    reference_model.train()
    cinn_model.train()
    reference_core = ComformerTensorCore(reference_model)
    cinn_core = ComformerTensorCore(cinn_model)
    reference_core.training = True
    cinn_core.training = True
    compiled = compile_comformer_cinn(cinn_core)
    compiled.training = True

    graph_a, graph_b = _make_graphs()
    graph_sequence = (
        pgl.Graph.batch([graph_a, graph_b]),
        _make_graph(5, 0.2),
        pgl.Graph.batch([_make_graph(4, 0.3), _make_graph(3, 0.4)]),
    )
    target_sequence = (
        paddle.to_tensor([[0.25], [-0.75]], dtype="float32"),
        paddle.to_tensor([[0.5]], dtype="float32"),
        paddle.to_tensor([[-0.25], [0.75]], dtype="float32"),
    )

    # Trigger compilation, then restore mutable BatchNorm buffers just as the
    # public execution lifecycle does before the first optimizer update.
    compiled(*graph_to_tensor_batch(graph_sequence[0]))
    cinn_model.set_state_dict(initial_state)
    reference_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=reference_model.parameters()
    )
    cinn_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=cinn_model.parameters()
    )

    for graph, target in zip(graph_sequence, target_sequence):
        packed = graph_to_tensor_batch(graph)
        reference_output = reference_core(*packed)
        cinn_output = compiled(*packed)
        reference_loss = paddle.nn.functional.mse_loss(reference_output, target)
        cinn_loss = paddle.nn.functional.mse_loss(cinn_output, target)
        reference_loss.backward()
        cinn_loss.backward()
        _assert_allclose(cinn_output, reference_output, atol=3e-6, rtol=3e-6)
        _assert_allclose(cinn_loss, reference_loss, atol=5e-6, rtol=5e-6)

        reference_parameters = dict(reference_model.named_parameters())
        cinn_parameters = dict(cinn_model.named_parameters())
        assert reference_parameters.keys() == cinn_parameters.keys()
        for name, reference_parameter in reference_parameters.items():
            cinn_parameter = cinn_parameters[name]
            if reference_parameter.grad is None:
                assert cinn_parameter.grad is None, name
                continue
            _assert_allclose(
                cinn_parameter.grad,
                reference_parameter.grad,
                atol=2e-4,
                rtol=2e-4,
            )

        reference_optimizer.step()
        cinn_optimizer.step()
        for name, reference_parameter in reference_parameters.items():
            _assert_allclose(
                cinn_parameters[name],
                reference_parameter,
                atol=2e-5,
                rtol=2e-5,
            )
        reference_optimizer.clear_grad()
        cinn_optimizer.clear_grad()

        _assert_state_dict_close(
            cinn_model.state_dict(),
            reference_model.state_dict(),
            atol=2e-5,
            rtol=2e-5,
        )


@pytest.fixture
def tensor_runtime_proxy(monkeypatch):
    cores = {}

    def get_runtime(model):
        core = cores.get(id(model))
        if core is None:
            core = ComformerTensorCore(model)
            cores[id(model)] = core
        core.training = model.training
        return core

    monkeypatch.setattr(iComformer, "validate_execution_backend", lambda self: None)
    monkeypatch.setattr(iComformer, "_validate_cinn_environment", lambda self: None)
    monkeypatch.setattr(iComformer, "_get_cinn_runtime", get_runtime)
    return cores


def _make_samples():
    graph_a, graph_b = _make_graphs()
    return [
        {
            "graph": graph_a,
            "formation_energy_per_atom": np.asarray([0.25], dtype=np.float32),
        },
        {
            "graph": graph_b,
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
        resumed_model.state_dict(), uninterrupted_model.state_dict(), atol=3e-6
    )
    _assert_state_dict_close(
        resumed_optimizer.state_dict(),
        uninterrupted_optimizer.state_dict(),
        atol=3e-6,
    )
    assert resumed_model._cinn_warmed_modes == {"train", "eval"}


def _predictor_config(checkpoint_path):
    return {
        "Model": {
            "__class_name__": "iComformer",
            "__init_params__": {
                "conv_layers": 1,
                "edge_layers": 1,
                "atom_input_features": 92,
                "edge_features": 8,
                "triplet_input_features": 6,
                "node_features": 8,
                "fc_features": 8,
                "output_features": 1,
                "property_name": "formation_energy_per_atom",
            },
        },
        "Predict": {
            "checkpoint_path": str(checkpoint_path),
            "execution_backend": "cinn",
            "eval_with_no_grad": True,
            "graph_converter": {
                "__class_name__": "ComformerGraphConverter",
                "__init_params__": {
                    "cutoff": 4.0,
                    "max_neighbors": 4,
                    "num_cpus": 1,
                },
            },
        },
    }


def test_cinn_property_predictor_loads_checkpoint_and_batches_cifs(
    tmp_path, tensor_runtime_proxy
):
    model = _make_model()
    checkpoint_path = tmp_path / "best.pdparams"
    paddle.save(model.state_dict(), str(checkpoint_path))
    config_path = tmp_path / "comformer.yaml"
    OmegaConf.save(OmegaConf.create(_predictor_config(checkpoint_path)), config_path)

    predictor = PropertyPredictor(config_path=config_path, device="cpu")
    cif_path = (
        Path(__file__).resolve().parents[4]
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
        atol=2e-6,
        rtol=2e-6,
    )
    assert predictor.model.state_dict().keys() == model.state_dict().keys()


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") != "1",
    reason="Set PPMAT_RUN_CINN_WORKFLOW_TESTS=1 for the GPU workflow smoke.",
)
def test_gpu_cinn_trainer_checkpoint_and_property_predictor(tmp_path):
    if not paddle.is_compiled_with_cuda():
        pytest.skip("Paddle was not compiled with CUDA.")
    if not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle was not compiled with CINN.")

    paddle.set_device("gpu:0")
    model = _make_model(execution_backend="cinn")
    trainer, _ = _build_trainer(model, tmp_path / "trainer", 1)
    with paddle.utils.unique_name.guard():
        trainer.train()

    checkpoint_path = tmp_path / "trainer" / "checkpoints" / "best.pdparams"
    config_path = tmp_path / "comformer.yaml"
    OmegaConf.save(OmegaConf.create(_predictor_config(checkpoint_path)), config_path)
    predictor = PropertyPredictor(config_path=config_path, device="gpu:0")
    cif_path = (
        Path(__file__).resolve().parents[4]
        / "property_prediction"
        / "example_data"
        / "cifs"
        / "mp-18767-LiMnO2.cif"
    )
    result = predictor.from_cif_file(str(cif_path))

    assert trainer.state.global_step == 1
    assert result.keys() == {"formation_energy_per_atom"}
    assert predictor.model._cinn_warmed_modes == {"eval"}
    assert predictor.model.state_dict().keys() == model.state_dict().keys()
