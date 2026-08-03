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

import numpy as np
import paddle
import pytest

from ppmat.datasets.collate_fn import RadiusGraphCollator
from ppmat.models.common.graph_converter import RadiusGraphConverter
from ppmat.models.spherenet.spherenet import SphereNet
from ppmat.models.spherenet.spherenet_cinn import SphereNetTensorCore
from ppmat.models.spherenet.spherenet_cinn import compile_spherenet
from ppmat.models.spherenet.spherenet_cinn import compile_spherenet_cinn
from ppmat.models.spherenet.spherenet_cinn import graph_to_tensor_batch
from ppmat.models.spherenet.spherenet_cinn import pack_pgl_graph


@pytest.fixture(autouse=True)
def _cpu_device():
    original_device = paddle.get_device()
    paddle.set_device("cpu")
    yield
    paddle.set_device(original_device)


def _make_model(execution_backend="eager", **kwargs):
    params = {
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
        "execution_backend": execution_backend,
    }
    params.update(kwargs)
    with paddle.utils.unique_name.guard():
        paddle.seed(2026)
        return SphereNet(**params)


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
    graph_no_triplets = converter.from_arrays(
        np.asarray([1, 8], dtype=np.int64),
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    )
    return graph_a, graph_b, graph_no_triplets


def _collate(graphs, targets=None):
    if targets is None:
        targets = np.arange(len(graphs), dtype=np.float32)
    return RadiusGraphCollator()(
        [
            {"graph": graph, "mu": np.asarray([target], dtype=np.float32)}
            for graph, target in zip(graphs, targets)
        ]
    )


def _assert_allclose(actual, expected, atol=3e-5, rtol=3e-5):
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=atol, rtol=rtol)


def _assert_gradients_close(reference_model, actual_model):
    reference_parameters = dict(reference_model.named_parameters())
    actual_parameters = dict(actual_model.named_parameters())
    assert reference_parameters.keys() == actual_parameters.keys()
    for name, reference_parameter in reference_parameters.items():
        actual_parameter = actual_parameters[name]
        if reference_parameter.grad is None:
            assert actual_parameter.grad is None, name
            continue
        _assert_allclose(actual_parameter.grad, reference_parameter.grad)


def test_pack_pgl_graph_offsets_sequence_and_preserves_sources():
    graph_a, graph_b, _ = _make_graphs()
    graph_a.node_feat["unused_metadata"] = np.asarray(["ignored"] * 4)
    graph_b.node_feat["unused_metadata"] = np.asarray(["ignored"] * 5)
    original_a = np.asarray(graph_a.edge_feat["ti_idx_kj"]).copy()
    original_b = np.asarray(graph_b.edge_feat["ti_idx_kj"]).copy()

    packed = pack_pgl_graph([graph_a, graph_b])

    assert packed.atomic_number.shape == [9]
    assert packed.pos.shape == [9, 3]
    assert packed.edge_src.shape == [32]
    assert packed.idx_kj.shape == [84]
    assert packed.idx_qj.shape == [84]
    assert packed.triplet_mask.shape == [84, 1]
    assert packed.graph_template.shape == [2, 1]
    np.testing.assert_array_equal(packed.node_graph_id.numpy(), [0] * 4 + [1] * 5)
    np.testing.assert_array_equal(
        packed.idx_kj.numpy()[: original_a.shape[0]], original_a
    )
    np.testing.assert_array_equal(
        packed.idx_kj.numpy()[original_a.shape[0] :],
        original_b + np.asarray(graph_a.edges).shape[0],
    )
    np.testing.assert_array_equal(graph_a.edge_feat["ti_idx_kj"], original_a)
    np.testing.assert_array_equal(graph_b.edge_feat["ti_idx_kj"], original_b)

    tensor_graph = _collate(_make_graphs()[:2])["graph"].tensor(inplace=False)
    tensor_packed = graph_to_tensor_batch(tensor_graph)
    assert tensor_packed.pos is tensor_graph.node_feat["pos"]


def test_pack_pgl_graph_masks_empty_triplet_sentinel():
    _, _, graph = _make_graphs()

    packed = graph_to_tensor_batch(graph)

    assert packed.idx_kj.shape == [1]
    assert packed.idx_ji.shape == [1]
    assert packed.idx_qj.shape == [1]
    np.testing.assert_array_equal(packed.triplet_mask.numpy(), [[0.0]])


@pytest.mark.parametrize("batch_size", [1, 2])
def test_tensor_core_matches_eager_forward(batch_size):
    graph_a, graph_b, _ = _make_graphs()
    data = (
        {"graph": graph_a}
        if batch_size == 1
        else _collate([graph_a, graph_b], [0.25, -0.75])
    )
    model = _make_model()
    model.eval()
    core = SphereNetTensorCore(model)
    core.eval()

    expected = model._forward_eager(data)[0]
    actual = core(*graph_to_tensor_batch(data["graph"]))

    assert actual.shape == [batch_size, 1]
    _assert_allclose(actual, expected)


def test_tensor_core_matches_empty_triplet_eager_forward():
    _, _, graph = _make_graphs()
    model = _make_model()
    model.eval()
    core = SphereNetTensorCore(model)
    core.eval()

    expected = model._forward_eager({"graph": graph})[0]
    actual = core(*graph_to_tensor_batch(graph))

    _assert_allclose(actual, expected)


def test_public_cinn_training_preserves_empty_triplet_adam_semantics(monkeypatch):
    graph, _, graph_no_triplets = _make_graphs()
    reference_model = _make_model()
    cinn_model = _make_model(execution_backend="cinn")
    cinn_model.set_state_dict(reference_model.state_dict())
    reference_model.train()
    cinn_model.train()
    runtime = SphereNetTensorCore(cinn_model)
    runtime.train()
    runtime_calls = []

    def get_runtime():
        runtime_calls.append(True)
        return runtime

    monkeypatch.setattr(cinn_model, "_get_cinn_runtime", get_runtime)
    reference_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=reference_model.parameters()
    )
    cinn_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=cinn_model.parameters()
    )
    graph_sequence = (graph_no_triplets, graph, graph_no_triplets, graph)
    target_sequence = (0.0, 0.25, -0.5, 0.75)

    for step_graph, target in zip(graph_sequence, target_sequence):
        data = {
            "graph": step_graph,
            "mu": paddle.to_tensor([[target]], dtype="float32"),
        }
        reference = reference_model(data)
        actual = cinn_model(data)
        reference["loss_dict"]["loss"].backward()
        actual["loss_dict"]["loss"].backward()

        _assert_allclose(actual["pred_dict"]["mu"], reference["pred_dict"]["mu"])
        _assert_gradients_close(reference_model, cinn_model)
        reference_optimizer.step()
        cinn_optimizer.step()
        for name, reference_parameter in reference_model.named_parameters():
            _assert_allclose(
                dict(cinn_model.named_parameters())[name], reference_parameter
            )
        reference_optimizer.clear_grad(set_to_zero=False)
        cinn_optimizer.clear_grad(set_to_zero=False)

    assert len(runtime_calls) == len(graph_sequence)


def test_tensor_core_matches_eager_gradients_and_adam_step():
    graph_a, graph_b, _ = _make_graphs()
    reference_model = _make_model()
    tensor_model = _make_model()
    tensor_model.set_state_dict(reference_model.state_dict())
    reference_model.train()
    tensor_model.train()
    tensor_core = SphereNetTensorCore(tensor_model)
    tensor_core.train()
    batch = _collate([graph_a, graph_b], [0.25, -0.75])
    target = paddle.to_tensor([[0.25], [-0.75]], dtype="float32")

    reference_output = reference_model._forward_eager(batch)[0]
    tensor_output = tensor_core(*graph_to_tensor_batch(batch["graph"]))
    reference_loss = paddle.nn.functional.l1_loss(reference_output, target)
    tensor_loss = paddle.nn.functional.l1_loss(tensor_output, target)
    reference_loss.backward()
    tensor_loss.backward()

    _assert_allclose(tensor_output, reference_output)
    _assert_allclose(tensor_loss, reference_loss)
    _assert_gradients_close(reference_model, tensor_model)

    reference_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=reference_model.parameters()
    )
    tensor_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=tensor_model.parameters()
    )
    reference_optimizer.step()
    tensor_optimizer.step()
    for name, reference_parameter in reference_model.named_parameters():
        _assert_allclose(
            dict(tensor_model.named_parameters())[name], reference_parameter
        )


def test_dynamic_shape_pir_matches_tensor_eager():
    graph_a, graph_b, graph_no_triplets = _make_graphs()
    model = _make_model()
    model.eval()
    core = SphereNetTensorCore(model)
    core.eval()
    static_core = compile_spherenet(core, backend=None, full_graph=True)
    static_core.eval()

    for graph in (
        graph_a,
        _collate([graph_a, graph_b])["graph"],
        graph_no_triplets,
    ):
        packed = graph_to_tensor_batch(graph)
        _assert_allclose(static_core(*packed), core(*packed))


def test_pir_training_matches_eager_gradients_and_adam_step():
    graph_a, graph_b, _ = _make_graphs()
    reference_model = _make_model()
    static_model = _make_model()
    static_model.set_state_dict(reference_model.state_dict())
    reference_model.train()
    static_model.train()
    static_core = SphereNetTensorCore(static_model)
    static_core.train()
    compiled = compile_spherenet(static_core, backend=None, full_graph=True)
    compiled.train()
    graph = _collate([graph_a, graph_b])["graph"]
    target = paddle.to_tensor([[0.25], [-0.75]], dtype="float32")

    reference_output = reference_model._forward_eager({"graph": graph})[0]
    static_output = compiled(*graph_to_tensor_batch(graph))
    reference_loss = paddle.nn.functional.l1_loss(reference_output, target)
    static_loss = paddle.nn.functional.l1_loss(static_output, target)
    reference_loss.backward()
    static_loss.backward()

    _assert_allclose(static_output, reference_output)
    _assert_allclose(static_loss, reference_loss)
    _assert_gradients_close(reference_model, static_model)

    reference_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=reference_model.parameters()
    )
    static_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=static_model.parameters()
    )
    reference_optimizer.step()
    static_optimizer.step()
    for name, reference_parameter in reference_model.named_parameters():
        _assert_allclose(
            dict(static_model.named_parameters())[name], reference_parameter
        )


def test_public_backend_preserves_dict_and_checkpoint_contract(monkeypatch):
    graph_a, graph_b, _ = _make_graphs()
    reference_model = _make_model()
    model = _make_model()
    model.set_state_dict(reference_model.state_dict())
    reference_model.eval()
    model.eval()
    expected_keys = list(model.state_dict())
    runtime = SphereNetTensorCore(model)
    runtime.eval()
    monkeypatch.setattr(model, "_validate_cinn_environment", lambda: None)
    monkeypatch.setattr(model, "_get_cinn_runtime", lambda: runtime)
    model.set_execution_backend("cinn")
    reference_batch = _collate([graph_a, graph_b], [0.25, -0.75])
    runtime_batch = _collate(_make_graphs()[:2], [0.25, -0.75])

    expected = reference_model(reference_batch)
    actual = model(runtime_batch)

    _assert_allclose(actual["loss_dict"]["loss"], expected["loss_dict"]["loss"])
    _assert_allclose(actual["pred_dict"]["mu"], expected["pred_dict"]["mu"])
    assert list(model.state_dict()) == expected_keys
    assert all(not key.startswith("model.") for key in model.state_dict())


def test_public_cinn_backend_fails_fast_on_cpu():
    graph, _, graph_no_triplets = _make_graphs()
    for test_graph in (graph, graph_no_triplets):
        model = _make_model(execution_backend="cinn")
        with pytest.raises(RuntimeError, match="requires"):
            model({"graph": test_graph}, return_loss=False)


def test_public_cinn_backend_rejects_force_training():
    model = _make_model(energy_and_force=True, execution_backend="cinn")

    with pytest.raises(ValueError, match="property prediction only"):
        model.validate_execution_backend()


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_SPHERENET_CINN_TESTS") != "1",
    reason="Set PPMAT_RUN_SPHERENET_CINN_TESTS=1 for GPU CINN inference.",
)
def test_dynamic_shape_cinn_inference_matches_eager():
    if not paddle.is_compiled_with_cuda():
        pytest.skip("Paddle was not compiled with CUDA.")
    if not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle was not compiled with CINN.")
    paddle.set_device("gpu:0")
    graph_a, graph_b, _ = _make_graphs()
    model = _make_model()
    model.eval()
    core = SphereNetTensorCore(model)
    core.eval()
    compiled = compile_spherenet_cinn(core, full_graph=True)
    compiled.eval()

    for graph in (graph_a, _collate([graph_a, graph_b])["graph"]):
        packed = graph_to_tensor_batch(graph)
        _assert_allclose(compiled(*packed), core(*packed))


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_SPHERENET_CINN_TRAINING_TESTS") != "1",
    reason="Set PPMAT_RUN_SPHERENET_CINN_TRAINING_TESTS=1 for GPU training.",
)
def test_dynamic_shape_cinn_training_matches_eager_adam_steps():
    if not paddle.is_compiled_with_cuda():
        pytest.skip("Paddle was not compiled with CUDA.")
    if not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle was not compiled with CINN.")
    paddle.set_device("gpu:0")
    graph_a, graph_b, graph_no_triplets = _make_graphs()
    reference_model = _make_model()
    cinn_model = _make_model(execution_backend="cinn")
    cinn_model.set_state_dict(reference_model.state_dict())
    reference_model.train()
    cinn_model.train()
    reference_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=reference_model.parameters()
    )
    cinn_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=cinn_model.parameters()
    )
    graph_sequence = (
        graph_no_triplets,
        _collate([graph_a, graph_b])["graph"],
        graph_a,
    )
    target_sequence = (
        paddle.to_tensor([[0.0]], dtype="float32"),
        paddle.to_tensor([[0.25], [-0.75]], dtype="float32"),
        paddle.to_tensor([[0.5]], dtype="float32"),
    )

    for graph, target in zip(graph_sequence, target_sequence):
        data = {"graph": graph, "mu": target}
        reference = reference_model(data)
        actual = cinn_model(data)
        reference_loss = reference["loss_dict"]["loss"]
        cinn_loss = actual["loss_dict"]["loss"]
        reference_loss.backward()
        cinn_loss.backward()

        _assert_allclose(actual["pred_dict"]["mu"], reference["pred_dict"]["mu"])
        _assert_allclose(cinn_loss, reference_loss)
        _assert_gradients_close(reference_model, cinn_model)
        reference_optimizer.step()
        cinn_optimizer.step()
        for name, reference_parameter in reference_model.named_parameters():
            _assert_allclose(
                dict(cinn_model.named_parameters())[name], reference_parameter
            )
        reference_optimizer.clear_grad(set_to_zero=False)
        cinn_optimizer.clear_grad(set_to_zero=False)
