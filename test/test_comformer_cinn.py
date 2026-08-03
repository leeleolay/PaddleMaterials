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


def _make_single_node_graph() -> pgl.Graph:
    node_feat = np.zeros([1, 92], dtype=np.float32)
    node_feat[0, 1] = 1.0
    return pgl.Graph(
        np.zeros([3, 2], dtype=np.int64),
        num_nodes=1,
        node_feat={"node_feat": node_feat},
        edge_feat={
            "r": np.eye(3, dtype=np.float32),
            "nei": np.tile(np.eye(3, dtype=np.float32)[None], [3, 1, 1]),
        },
    )


def _make_model(
    execution_backend: str = "eager",
    conv_layers: int = 1,
) -> iComformer:
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        return iComformer(
            conv_layers=conv_layers,
            edge_layers=1,
            atom_input_features=92,
            edge_features=8,
            triplet_input_features=6,
            node_features=8,
            fc_features=8,
            output_features=1,
            execution_backend=execution_backend,
        )


def _assert_allclose(actual, expected, atol=2e-6, rtol=2e-6, err_msg=""):
    np.testing.assert_allclose(
        actual.numpy(),
        expected.numpy(),
        atol=atol,
        rtol=rtol,
        err_msg=err_msg,
    )


def _assert_state_dict_close(
    actual,
    expected,
    atol=2e-6,
    rtol=2e-6,
    excluded_keys=frozenset(),
):
    assert actual.keys() == expected.keys()
    for key in actual:
        if key in excluded_keys:
            continue
        np.testing.assert_allclose(
            actual[key].numpy(),
            expected[key].numpy(),
            atol=atol,
            rtol=rtol,
            err_msg=key,
        )


def _cancelled_batch_norm_biases(parameters):
    return {name for name in parameters if name.endswith("lin_concate.bias")}


def _cancelled_optimizer_moment_keys(model, optimizer_state):
    parameters = dict(model.named_parameters())
    cancelled_biases = _cancelled_batch_norm_biases(parameters)
    parameter_names = {
        parameter.name
        for name, parameter in parameters.items()
        if name in cancelled_biases
    }
    return {
        key
        for key in optimizer_state
        if any(key.startswith(f"{name}_moment") for name in parameter_names)
    }


def test_graph_to_tensor_batch_packs_single_and_dynamic_batches():
    graph_a, graph_b = _make_graphs()
    graph_a.node_feat["unused_metadata"] = np.asarray(["ignored"] * 3)
    graph_b.node_feat["unused_metadata"] = np.asarray(["ignored"] * 4)
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

    tensor_graph = _make_graph(3).tensor(inplace=False)
    tensor_packed = graph_to_tensor_batch(tensor_graph)
    assert tensor_packed.node_feat is tensor_graph.node_feat["node_feat"]
    assert tensor_packed.edge_r is tensor_graph.edge_feat["r"]


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


def test_tensor_core_matches_single_node_training_batch_norm():
    reference = _make_model()
    model = _make_model()
    model.set_state_dict(reference.state_dict())
    reference.train()
    model.train()
    core = ComformerTensorCore(model)
    core.training = True

    expected = reference._forward_eager({"graph": _make_single_node_graph()})
    actual = core(*graph_to_tensor_batch(_make_single_node_graph()))

    _assert_allclose(actual, expected)
    _assert_allclose(model.att_layers[0].bn._mean, reference.att_layers[0].bn._mean)
    _assert_allclose(
        model.att_layers[0].bn._variance,
        reference.att_layers[0].bn._variance,
    )

    expected.sum().backward()
    actual.sum().backward()
    reference_parameters = dict(reference.named_parameters())
    actual_parameters = dict(model.named_parameters())
    assert reference_parameters.keys() == actual_parameters.keys()
    for name, reference_parameter in reference_parameters.items():
        actual_parameter = actual_parameters[name]
        if reference_parameter.grad is None:
            assert actual_parameter.grad is None, name
            continue
        _assert_allclose(
            actual_parameter.grad,
            reference_parameter.grad,
            atol=2e-5,
            rtol=2e-5,
        )


def test_tensor_core_matches_eager_gradients_and_adam_step():
    reference_model = _make_model()
    tensor_model = _make_model()
    tensor_model.set_state_dict(reference_model.state_dict())
    reference_model.train()
    tensor_model.train()
    tensor_core = ComformerTensorCore(tensor_model)
    tensor_core.training = True
    graph_a, graph_b = _make_graphs()
    target = paddle.to_tensor([[0.25], [-0.75]], dtype="float32")

    reference_output = reference_model._forward_eager(
        {"graph": pgl.Graph.batch([graph_a, graph_b])}
    )
    tensor_output = tensor_core(*graph_to_tensor_batch(pgl.Graph.batch(_make_graphs())))
    reference_loss = paddle.nn.functional.mse_loss(reference_output, target)
    tensor_loss = paddle.nn.functional.mse_loss(tensor_output, target)
    reference_loss.backward()
    tensor_loss.backward()

    _assert_allclose(tensor_output, reference_output)
    _assert_allclose(tensor_loss, reference_loss)
    reference_parameters = dict(reference_model.named_parameters())
    tensor_parameters = dict(tensor_model.named_parameters())
    assert reference_parameters.keys() == tensor_parameters.keys()
    cancelled_biases = _cancelled_batch_norm_biases(reference_parameters)
    tensor_biases_before_step = {
        name: tensor_parameters[name].clone() for name in cancelled_biases
    }
    for name, reference_parameter in reference_parameters.items():
        tensor_parameter = tensor_parameters[name]
        if reference_parameter.grad is None:
            assert tensor_parameter.grad is None, name
        elif name in cancelled_biases:
            assert float(paddle.max(paddle.abs(reference_parameter.grad))) < 2e-5
            assert float(paddle.max(paddle.abs(tensor_parameter.grad))) == 0.0
        else:
            _assert_allclose(
                tensor_parameter.grad,
                reference_parameter.grad,
                atol=2e-4,
                rtol=2e-4,
                err_msg=name,
            )

    reference_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=reference_model.parameters()
    )
    tensor_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=tensor_model.parameters()
    )
    reference_optimizer.step()
    tensor_optimizer.step()
    reference_parameters = dict(reference_model.named_parameters())
    tensor_parameters = dict(tensor_model.named_parameters())
    assert reference_parameters.keys() == tensor_parameters.keys()
    for name, reference_parameter in reference_parameters.items():
        if name in cancelled_biases:
            _assert_allclose(
                tensor_parameters[name],
                tensor_biases_before_step[name],
                atol=0.0,
                rtol=0.0,
                err_msg=name,
            )
            continue
        if reference_parameter.stop_gradient:
            continue
        _assert_allclose(
            tensor_parameters[name],
            reference_parameter,
            atol=2e-5,
            rtol=2e-5,
            err_msg=name,
        )


def test_two_layer_tensor_core_propagates_edge_update_gradients():
    reference_model = _make_model(conv_layers=2)
    tensor_model = _make_model(conv_layers=2)
    tensor_model.set_state_dict(reference_model.state_dict())
    reference_model.train()
    tensor_model.train()
    tensor_core = ComformerTensorCore(tensor_model)
    tensor_core.training = True
    graph = pgl.Graph.batch(_make_graphs())
    target = paddle.to_tensor([[0.25], [-0.75]], dtype="float32")

    reference_output = reference_model._forward_eager({"graph": graph})
    tensor_output = tensor_core(*graph_to_tensor_batch(graph))
    paddle.nn.functional.mse_loss(reference_output, target).backward()
    paddle.nn.functional.mse_loss(tensor_output, target).backward()

    _assert_allclose(tensor_output, reference_output, atol=1e-5, rtol=1e-5)
    reference_parameters = dict(reference_model.named_parameters())
    tensor_parameters = dict(tensor_model.named_parameters())
    edge_parameter_names = [
        name
        for name, parameter in reference_parameters.items()
        if name.startswith("edge_update_layer.")
        and not name.endswith("lin_concate.bias")
        and not parameter.stop_gradient
    ]
    assert edge_parameter_names
    for name in edge_parameter_names:
        reference_gradient = reference_parameters[name].grad
        tensor_gradient = tensor_parameters[name].grad
        if reference_gradient is None:
            assert tensor_gradient is None, name
            continue
        assert tensor_gradient is not None, name
        _assert_allclose(
            tensor_gradient,
            reference_gradient,
            atol=1e-3,
            rtol=1e-3,
            err_msg=name,
        )
    assert reference_parameters["edge_update_layer.lin_concate.weight"].grad is not None
    assert tensor_parameters["edge_update_layer.lin_concate.weight"].grad is not None


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
    batched_prediction = model.predict(pgl.Graph.batch(list(_make_graphs())))
    assert isinstance(batched_prediction, dict)
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
def test_gpu_cinn_inference_matches_eager_with_dynamic_batches():
    if not paddle.is_compiled_with_cuda():
        pytest.skip("Paddle was not compiled with CUDA.")
    if not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle was not compiled with CINN.")

    paddle.set_device("gpu:0")
    reference_model = _make_model(conv_layers=2)
    cinn_model = _make_model(conv_layers=2)
    cinn_model.set_state_dict(reference_model.state_dict())
    reference_model.eval()
    cinn_model.eval()
    cinn_core = ComformerTensorCore(cinn_model)
    cinn_core.training = False
    compiled = compile_comformer_cinn(cinn_core)
    compiled.training = False
    graph_a, graph_b = _make_graphs()
    for graph in (graph_a, pgl.Graph.batch([graph_a, graph_b]), _make_graph(5, 0.2)):
        expected = reference_model._forward_eager({"graph": graph})
        packed = graph_to_tensor_batch(graph)
        _assert_allclose(compiled(*packed), expected)


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_CINN_TRAINING_TESTS") != "1",
    reason="Set PPMAT_RUN_CINN_TRAINING_TESTS=1 to run the GPU training canary.",
)
def test_gpu_cinn_training_matches_eager_except_cancelled_batch_norm_biases():
    if not paddle.is_compiled_with_cuda():
        pytest.skip("Paddle was not compiled with CUDA.")
    if not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle was not compiled with CINN.")

    paddle.set_device("gpu:0")
    reference_model = _make_model(conv_layers=2)
    cinn_model = _make_model(conv_layers=2)
    cinn_model.set_state_dict(reference_model.state_dict())
    reference_model.train()
    cinn_model.train()
    cinn_core = ComformerTensorCore(cinn_model)
    cinn_core.training = True
    compiled = compile_comformer_cinn(cinn_core)
    compiled.training = True

    graph = pgl.Graph.batch(_make_graphs())
    target = paddle.to_tensor([[0.25], [-0.75]], dtype="float32")

    # Trigger compilation, then restore mutable BatchNorm buffers just as the
    # public execution lifecycle does before the first optimizer update.
    warmup_state = cinn_model._snapshot_warmup_state()
    compiled(*graph_to_tensor_batch(graph))
    cinn_model._restore_warmup_state(warmup_state)
    reference_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=reference_model.parameters()
    )
    cinn_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=cinn_model.parameters()
    )

    reference_output = reference_model._forward_eager({"graph": graph})
    cinn_output = compiled(*graph_to_tensor_batch(graph))
    reference_loss = paddle.nn.functional.mse_loss(reference_output, target)
    cinn_loss = paddle.nn.functional.mse_loss(cinn_output, target)
    reference_loss.backward()
    cinn_loss.backward()
    _assert_allclose(cinn_output, reference_output, atol=3e-6, rtol=3e-6)
    _assert_allclose(cinn_loss, reference_loss, atol=5e-6, rtol=5e-6)

    reference_parameters = dict(reference_model.named_parameters())
    cinn_parameters = dict(cinn_model.named_parameters())
    assert reference_parameters.keys() == cinn_parameters.keys()
    # Training BatchNorm cancels these biases; only eager roundoff reaches Adam.
    cancelled_biases = _cancelled_batch_norm_biases(reference_parameters)
    assert cancelled_biases
    cinn_biases_before_step = {
        name: cinn_parameters[name].clone() for name in cancelled_biases
    }
    for name, reference_parameter in reference_parameters.items():
        cinn_parameter = cinn_parameters[name]
        if reference_parameter.grad is None:
            assert cinn_parameter.grad is None, name
            continue
        if name in cancelled_biases:
            assert float(paddle.max(paddle.abs(reference_parameter.grad))) < 2e-5
            assert float(paddle.max(paddle.abs(cinn_parameter.grad))) == 0.0
            continue
        _assert_allclose(
            cinn_parameter.grad,
            reference_parameter.grad,
            atol=2e-5,
            rtol=2e-5,
            err_msg=name,
        )

    reference_optimizer.step()
    cinn_optimizer.step()
    for name, reference_parameter in reference_parameters.items():
        if name in cancelled_biases:
            _assert_allclose(
                cinn_parameters[name],
                cinn_biases_before_step[name],
                atol=0.0,
                rtol=0.0,
                err_msg=name,
            )
            continue
        _assert_allclose(
            cinn_parameters[name],
            reference_parameter,
            atol=3e-6,
            rtol=3e-6,
            err_msg=name,
        )
    _assert_state_dict_close(
        cinn_model.state_dict(),
        reference_model.state_dict(),
        atol=3e-6,
        rtol=3e-6,
        excluded_keys=cancelled_biases,
    )

    reference_model.eval()
    cinn_model.eval()
    reference_output = reference_model._forward_eager({"graph": graph})
    cinn_output = cinn_model._forward_eager({"graph": graph})
    _assert_allclose(cinn_output, reference_output, atol=3e-3, rtol=0.0)


@pytest.fixture
def tensor_runtime_proxy(monkeypatch):
    cores = {}

    def get_runtime(model):
        mode = "train" if model.training else "eval"
        key = (id(model), mode)
        core = cores.get(key)
        if core is None:
            core = ComformerTensorCore(model)
            core.training = model.training
            cores[key] = core
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
    optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        parameters=model.parameters(),
    )
    trainer = BaseTrainer(
        _trainer_config(output_dir, max_epochs, execution_backend),
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


def _predictor_config(
    checkpoint_path,
    execution_backend="cinn",
    conv_layers=1,
):
    return {
        "Model": {
            "__class_name__": "iComformer",
            "__init_params__": {
                "conv_layers": conv_layers,
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
            "execution_backend": execution_backend,
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
    initial_model = _make_model(execution_backend="eager", conv_layers=2)
    initial_state = {
        key: value.clone() for key, value in initial_model.state_dict().items()
    }
    reference_model = _make_model(execution_backend="eager", conv_layers=2)
    reference_model.set_state_dict(initial_state)
    reference_trainer, reference_optimizer = _build_trainer(
        reference_model, tmp_path / "eager_trainer", 1, execution_backend="eager"
    )
    model = _make_model(execution_backend="cinn", conv_layers=2)
    model.set_state_dict(initial_state)
    trainer, optimizer = _build_trainer(model, tmp_path / "cinn_trainer", 1)
    with paddle.utils.unique_name.guard():
        reference_trainer.train()
    with paddle.utils.unique_name.guard():
        trainer.train()
    cancelled_biases = _cancelled_batch_norm_biases(initial_state)
    _assert_state_dict_close(
        model.state_dict(),
        reference_model.state_dict(),
        atol=2e-5,
        rtol=2e-5,
        excluded_keys=cancelled_biases,
    )
    for key in cancelled_biases:
        _assert_allclose(
            model.state_dict()[key],
            initial_state[key],
            atol=0.0,
            rtol=0.0,
            err_msg=key,
        )
    cancelled_optimizer_keys = _cancelled_optimizer_moment_keys(
        model, optimizer.state_dict()
    )
    assert cancelled_optimizer_keys
    for key in cancelled_optimizer_keys:
        assert float(paddle.max(paddle.abs(optimizer.state_dict()[key]))) == 0.0
    _assert_state_dict_close(
        optimizer.state_dict(),
        reference_optimizer.state_dict(),
        atol=2e-5,
        rtol=2e-5,
        excluded_keys=cancelled_optimizer_keys,
    )

    resume_checkpoint = tmp_path / "cinn_trainer" / "checkpoints" / "latest"
    reference_resumed_model = _make_model(execution_backend="eager", conv_layers=2)
    reference_resumed_trainer, reference_resumed_optimizer = _build_trainer(
        reference_resumed_model,
        tmp_path / "eager_resumed",
        2,
        execution_backend="eager",
    )
    resumed_model = _make_model(execution_backend="cinn", conv_layers=2)
    resumed_trainer, resumed_optimizer = _build_trainer(
        resumed_model, tmp_path / "cinn_resumed", 2
    )
    with paddle.utils.unique_name.guard():
        reference_resumed_trainer.train(resume_from_checkpoint=str(resume_checkpoint))
    with paddle.utils.unique_name.guard():
        resumed_trainer.train(resume_from_checkpoint=str(resume_checkpoint))
    resumed_cancelled_biases = _cancelled_batch_norm_biases(
        dict(resumed_model.named_parameters())
    )
    _assert_state_dict_close(
        resumed_model.state_dict(),
        reference_resumed_model.state_dict(),
        atol=2e-5,
        rtol=2e-5,
        excluded_keys=resumed_cancelled_biases,
    )
    resumed_cancelled_optimizer_keys = _cancelled_optimizer_moment_keys(
        resumed_model, resumed_optimizer.state_dict()
    )
    assert resumed_cancelled_optimizer_keys
    for key in resumed_cancelled_optimizer_keys:
        assert float(paddle.max(paddle.abs(resumed_optimizer.state_dict()[key]))) == 0.0
    _assert_state_dict_close(
        resumed_optimizer.state_dict(),
        reference_resumed_optimizer.state_dict(),
        atol=2e-5,
        rtol=2e-5,
        excluded_keys=resumed_cancelled_optimizer_keys,
    )

    checkpoint_path = tmp_path / "cinn_resumed" / "checkpoints" / "latest.pdparams"
    config_path = tmp_path / "comformer_cinn.yaml"
    OmegaConf.save(
        OmegaConf.create(_predictor_config(checkpoint_path, conv_layers=2)),
        config_path,
    )
    predictor = PropertyPredictor(config_path=config_path, device="gpu:0")
    reference_config_path = tmp_path / "comformer_eager.yaml"
    OmegaConf.save(
        OmegaConf.create(_predictor_config(checkpoint_path, "eager", conv_layers=2)),
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
        atol=2e-6,
        rtol=2e-6,
    )
    assert predictor.model._cinn_warmed_modes == {"eval"}
    assert predictor.model.state_dict().keys() == model.state_dict().keys()
