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
import pgl
import pytest

from ppmat.models.megnet.megnet import MEGNetPlus
from ppmat.models.megnet.megnet_cinn import MEGNetTensorCore
from ppmat.models.megnet.megnet_cinn import _segment_softmax
from ppmat.models.megnet.megnet_cinn import compile_megnet
from ppmat.models.megnet.megnet_cinn import compile_megnet_cinn
from ppmat.models.megnet.megnet_cinn import graph_to_tensor_batch
from ppmat.models.megnet.megnet_cinn import pack_pgl_graph


@pytest.fixture(autouse=True)
def _cpu_device():
    original_device = paddle.get_device()
    paddle.set_device("cpu")
    yield
    paddle.set_device(original_device)


def _make_graph(atom_types, edges, bond_dist):
    return pgl.Graph(
        np.asarray(edges, dtype=np.int64),
        num_nodes=len(atom_types),
        node_feat={
            "atom_types": np.asarray(atom_types, dtype=np.int64),
        },
        edge_feat={
            "bond_dist": np.asarray(bond_dist, dtype=np.float32),
        },
    )


def _make_graphs():
    graph_a = _make_graph(
        atom_types=[1, 8, 14],
        edges=[
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 1],
            [0, 2],
            [2, 0],
        ],
        bond_dist=[1.0, 1.0, 1.5, 1.5, 2.0, 2.0],
    )
    graph_b = _make_graph(
        atom_types=[6, 7],
        edges=[[0, 1], [1, 0]],
        bond_dist=[1.2, 1.2],
    )
    return graph_a, graph_b


def _make_model(dim_state_embedding=2):
    paddle.seed(2026)
    return MEGNetPlus(
        dim_node_embedding=4,
        dim_edge_embedding=6,
        dim_state_embedding=dim_state_embedding,
        nblocks=2,
        hidden_layer_sizes_input=(8, 4),
        hidden_layer_sizes_conv=(8, 8, 4),
        hidden_layer_sizes_output=(8, 4),
        nlayers_set2set=1,
        niters_set2set=2,
        bond_expansion_cfg={
            "rbf_type": "Gaussian",
            "initial": 0.0,
            "final": 5.0,
            "num_centers": 6,
            "width": 0.5,
        },
    )


def _assert_allclose(actual, expected, atol=1e-6, rtol=1e-6):
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=atol, rtol=rtol)


def test_graph_to_tensor_batch_preserves_pgl_batch_contract():
    graph_a, graph_b = _make_graphs()
    graph = pgl.Graph.batch([graph_a, graph_b])
    graph.node_feat["unused_metadata"] = np.asarray(["ignored"] * 5)

    packed = graph_to_tensor_batch(graph)

    assert not graph.is_tensor()
    np.testing.assert_array_equal(packed.atom_types.numpy(), [1, 8, 14, 6, 7])
    np.testing.assert_array_equal(packed.node_graph_id.numpy(), [0, 0, 0, 1, 1])
    np.testing.assert_array_equal(
        packed.edge_graph_id.numpy(), [0, 0, 0, 0, 0, 0, 1, 1]
    )
    np.testing.assert_array_equal(packed.edge_src.numpy(), [0, 1, 1, 2, 0, 2, 3, 4])
    np.testing.assert_array_equal(packed.edge_dst.numpy(), [1, 0, 2, 1, 2, 0, 4, 3])
    assert packed.state_attr.shape == [2, 2]
    np.testing.assert_array_equal(packed.state_attr.numpy(), np.zeros([2, 2]))

    tensor_graph = pgl.Graph.batch(_make_graphs()).tensor(inplace=False)
    tensor_packed = graph_to_tensor_batch(tensor_graph)
    assert tensor_packed.atom_types is tensor_graph.node_feat["atom_types"]
    assert tensor_packed.bond_dist is tensor_graph.edge_feat["bond_dist"]


@pytest.mark.parametrize("batch_size", [1, 2])
def test_tensor_core_matches_eager_forward(batch_size):
    graph_a, graph_b = _make_graphs()
    graph = graph_a if batch_size == 1 else pgl.Graph.batch([graph_a, graph_b])
    model = _make_model()
    model.eval()
    core = MEGNetTensorCore(model)
    core.eval()

    packed = graph_to_tensor_batch(graph)
    expected = model._forward_eager({"graph": graph})
    actual = core(*packed)

    assert actual.shape == [batch_size, 1]
    _assert_allclose(actual, expected)


def test_tensor_core_matches_eager_gradients_and_optimizer_step():
    graph_a, graph_b = _make_graphs()
    reference_graph = pgl.Graph.batch([graph_a, graph_b])
    tensor_graph = pgl.Graph.batch(_make_graphs())

    reference_model = _make_model()
    tensor_model = _make_model()
    tensor_model.set_state_dict(reference_model.state_dict())
    reference_model.train()
    tensor_model.train()
    tensor_core = MEGNetTensorCore(tensor_model)
    tensor_core.train()

    target = paddle.to_tensor([[0.25], [-0.75]], dtype="float32")
    reference_output = reference_model._forward_eager({"graph": reference_graph})
    tensor_output = tensor_core(*graph_to_tensor_batch(tensor_graph))
    reference_loss = paddle.nn.functional.mse_loss(reference_output, target)
    tensor_loss = paddle.nn.functional.mse_loss(tensor_output, target)
    reference_loss.backward()
    tensor_loss.backward()

    _assert_allclose(tensor_output, reference_output)
    _assert_allclose(tensor_loss, reference_loss)
    reference_parameters = dict(reference_model.named_parameters())
    tensor_parameters = dict(tensor_model.named_parameters())
    assert reference_parameters.keys() == tensor_parameters.keys()
    for name, reference_parameter in reference_parameters.items():
        tensor_parameter = tensor_parameters[name]
        if reference_parameter.grad is None:
            assert tensor_parameter.grad is None
            continue
        _assert_allclose(
            tensor_parameter.grad,
            reference_parameter.grad,
            atol=2e-6,
            rtol=2e-6,
        )

    reference_optimizer = paddle.optimizer.SGD(
        learning_rate=1e-3, parameters=reference_model.parameters()
    )
    tensor_optimizer = paddle.optimizer.SGD(
        learning_rate=1e-3, parameters=tensor_model.parameters()
    )
    reference_optimizer.step()
    tensor_optimizer.step()
    for name, reference_parameter in reference_parameters.items():
        _assert_allclose(
            tensor_parameters[name],
            reference_parameter,
            atol=2e-6,
            rtol=2e-6,
        )


def test_segment_softmax_is_stable_across_graph_score_ranges():
    values = paddle.to_tensor(
        [[1000.0], [999.0], [-1000.0], [-1001.0]], dtype="float32"
    )
    segment_ids = paddle.to_tensor([0, 0, 1, 1], dtype="int64")

    actual = _segment_softmax(
        values,
        segment_ids,
        out_size=paddle.to_tensor(2, dtype="int64"),
    )
    expected = np.asarray(
        [[0.7310586], [0.2689414], [0.7310586], [0.2689414]],
        dtype=np.float32,
    )

    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6, rtol=1e-6)


def test_tensor_convenience_apis():
    graph_a, graph_b = _make_graphs()
    packed = pack_pgl_graph([graph_a, graph_b])
    model = _make_model()
    model.eval()
    core = MEGNetTensorCore(model)
    core.eval()

    assert int(packed.node_count) == 5
    assert int(packed.edge_count) == 8
    assert int(packed.graph_count) == 2

    expected = core(*packed)
    _assert_allclose(core.forward_graph([graph_a, graph_b]), expected)
    _assert_allclose(core.forward_tensor(packed), expected)
    _assert_allclose(core.predict_tensor(packed), expected)


def test_compile_accepts_original_megnet_model():
    graph_a, _ = _make_graphs()
    model = _make_model()
    model.eval()
    compiled = compile_megnet(model, backend=None, full_graph=True)
    compiled.eval()

    packed = graph_to_tensor_batch(graph_a)
    expected = model._forward_eager({"graph": graph_a})
    actual = compiled(*packed)
    _assert_allclose(actual, expected)


def test_non_default_state_dimension_is_explicitly_packed():
    graph_a, _ = _make_graphs()
    model = _make_model(dim_state_embedding=4)
    model.eval()
    core = MEGNetTensorCore(model)
    core.eval()

    packed = graph_to_tensor_batch(graph_a, state_dim=4)
    assert packed.state_attr.shape == [1, 4]
    expected = core.forward_tensor(packed)
    assert expected.shape == [1, 1]
    assert core.forward_graph(graph_a).shape == [1, 1]

    with pytest.raises(ValueError, match="expected 4"):
        core.forward_tensor(graph_to_tensor_batch(graph_a))

    static_core = compile_megnet(core, backend=None, full_graph=True)
    static_core.eval()
    _assert_allclose(static_core(*packed), expected)


def test_dynamic_shape_pir_static_matches_tensor_eager():
    model = _make_model()
    model.eval()
    core = MEGNetTensorCore(model)
    core.eval()
    static_core = compile_megnet(core, backend=None, full_graph=True)
    static_core.eval()

    graph_a, graph_b = _make_graphs()
    for graph in (graph_a, pgl.Graph.batch([graph_a, graph_b])):
        packed = graph_to_tensor_batch(graph)
        expected = core(*packed)
        actual = static_core(*packed)
        _assert_allclose(actual, expected)


def test_static_core_can_be_exported_and_reloaded(tmp_path):
    model = _make_model()
    model.eval()
    core = MEGNetTensorCore(model)
    core.eval()
    static_core = compile_megnet(core, backend=None, full_graph=True)
    static_core.eval()

    graph_a, _ = _make_graphs()
    packed = graph_to_tensor_batch(graph_a)
    expected = static_core(*packed)
    export_path = str(tmp_path / "megnet_tensor_core")
    paddle.jit.save(static_core, export_path)
    loaded = paddle.jit.load(export_path)
    actual = loaded(*packed)

    _assert_allclose(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_CINN_TESTS") != "1",
    reason="Set PPMAT_RUN_CINN_TESTS=1 to run the GPU CINN smoke test.",
)
def test_dynamic_shape_cinn_inference_matches_eager():
    if not paddle.is_compiled_with_cuda():
        pytest.skip("Paddle was not compiled with CUDA.")
    if not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle was not compiled with CINN.")

    paddle.set_device("gpu:0")
    reference_model = _make_model()
    cinn_model = _make_model()
    cinn_model.set_state_dict(reference_model.state_dict())
    reference_model.eval()
    cinn_model.eval()
    cinn_core = MEGNetTensorCore(cinn_model)
    cinn_core.eval()
    compiled = compile_megnet_cinn(cinn_core, full_graph=True)
    compiled.eval()

    graph_a, graph_b = _make_graphs()
    baseline = None
    for graph in (graph_a, pgl.Graph.batch([graph_a, graph_b])):
        expected = reference_model._forward_eager({"graph": graph})
        packed = graph_to_tensor_batch(graph)
        actual = compiled(*packed)
        _assert_allclose(actual, expected)
        baseline = actual

    state_dict = cinn_model.state_dict()
    output_bias = "fc_out.layers.4.bias"
    state_dict[output_bias] = state_dict[output_bias] + 1.0
    cinn_model.set_state_dict(state_dict)
    reference_model.set_state_dict(state_dict)
    expected = reference_model._forward_eager({"graph": graph})
    actual = compiled(*packed)
    _assert_allclose(actual, expected)
    assert not np.allclose(actual.numpy(), baseline.numpy())


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_CINN_TRAINING_TESTS") != "1",
    reason=(
        "Set PPMAT_RUN_CINN_TRAINING_TESTS=1 to run the GPU CINN training " "canary."
    ),
)
def test_dynamic_shape_cinn_training_matches_eager():
    if not paddle.is_compiled_with_cuda():
        pytest.skip("Paddle was not compiled with CUDA.")
    if not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle was not compiled with CINN.")

    paddle.set_device("gpu:0")
    reference_model = _make_model()
    cinn_model = _make_model()
    cinn_model.set_state_dict(reference_model.state_dict())
    cinn_core = MEGNetTensorCore(cinn_model)
    cinn_core.train()
    compiled = compile_megnet_cinn(cinn_core, full_graph=True)
    compiled.train()

    reference_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        parameters=reference_model.parameters(),
    )
    cinn_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        parameters=cinn_model.parameters(),
    )
    graph_a, graph_b = _make_graphs()
    graph_sequence = (
        pgl.Graph.batch([graph_a, graph_b]),
        graph_a,
        pgl.Graph.batch(_make_graphs()),
    )
    target_sequence = (
        paddle.to_tensor([[0.25], [-0.75]], dtype="float32"),
        paddle.to_tensor([[0.5]], dtype="float32"),
        paddle.to_tensor([[-0.25], [0.75]], dtype="float32"),
    )
    reference_losses = []
    cinn_losses = []

    for graph, target in zip(graph_sequence, target_sequence):
        reference_output = reference_model._forward_eager({"graph": graph})
        cinn_batch = graph_to_tensor_batch(graph)
        cinn_output = compiled(*cinn_batch)
        reference_loss = paddle.nn.functional.mse_loss(reference_output, target)
        cinn_loss = paddle.nn.functional.mse_loss(cinn_output, target)
        reference_loss.backward()
        cinn_loss.backward()

        _assert_allclose(cinn_output, reference_output, atol=2e-6, rtol=2e-6)
        _assert_allclose(cinn_loss, reference_loss, atol=2e-6, rtol=2e-6)
        reference_losses.append(float(reference_loss))
        cinn_losses.append(float(cinn_loss))

        reference_parameters = dict(reference_model.named_parameters())
        cinn_parameters = dict(cinn_model.named_parameters())
        assert reference_parameters.keys() == cinn_parameters.keys()
        for name, reference_parameter in reference_parameters.items():
            cinn_parameter = cinn_parameters[name]
            if reference_parameter.grad is None:
                assert cinn_parameter.grad is None
                continue
            _assert_allclose(
                cinn_parameter.grad,
                reference_parameter.grad,
                atol=2e-6,
                rtol=2e-6,
            )

        reference_optimizer.step()
        cinn_optimizer.step()
        for name, reference_parameter in reference_parameters.items():
            _assert_allclose(
                cinn_parameters[name],
                reference_parameter,
                atol=2e-6,
                rtol=2e-6,
            )
        reference_optimizer.clear_grad()
        cinn_optimizer.clear_grad()

    np.testing.assert_allclose(
        cinn_losses,
        reference_losses,
        atol=2e-6,
        rtol=2e-6,
    )


def test_graph_to_tensor_batch_validates_required_features():
    graph = pgl.Graph(
        np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        num_nodes=2,
        node_feat={"atom_types": np.asarray([1, 8], dtype=np.int64)},
    )

    with pytest.raises(
        ValueError, match=r"graph\.edge_feat\['bond_dist'\] is required"
    ):
        graph_to_tensor_batch(graph)


def test_public_megnet_backend_preserves_trainer_and_predictor_contract(monkeypatch):
    reference_model = _make_model()
    model = _make_model()
    model.set_state_dict(reference_model.state_dict())
    reference_model.eval()
    model.eval()

    reference_graphs = _make_graphs()
    runtime_graphs = _make_graphs()
    reference_batch = {
        "graph": pgl.Graph.batch(reference_graphs),
        "state_attr": np.ones([2, 2], dtype=np.float32),
        "formation_energy_per_atom": paddle.to_tensor(
            [[0.25], [-0.75]], dtype="float32"
        ),
    }
    runtime_batch = {
        "graph": pgl.Graph.batch(runtime_graphs),
        "state_attr": np.ones([2, 2], dtype=np.float32),
        "formation_energy_per_atom": paddle.to_tensor(
            [[0.25], [-0.75]], dtype="float32"
        ),
    }
    expected_keys = list(model.state_dict())
    core = MEGNetTensorCore(model)
    monkeypatch.setattr(model, "_validate_cinn_environment", lambda: None)
    monkeypatch.setattr(model, "_get_cinn_runtime", lambda: core)
    model.set_execution_backend("cinn")

    expected = reference_model(reference_batch)
    actual = model(runtime_batch)
    _assert_allclose(actual["loss_dict"]["loss"], expected["loss_dict"]["loss"])
    _assert_allclose(
        actual["pred_dict"]["formation_energy_per_atom"],
        expected["pred_dict"]["formation_energy_per_atom"],
    )

    expected_prediction = reference_model.predict(reference_graphs[0])
    actual_prediction = model.predict(runtime_graphs[0])
    np.testing.assert_allclose(
        actual_prediction["formation_energy_per_atom"],
        expected_prediction["formation_energy_per_atom"],
        atol=1e-6,
        rtol=1e-6,
    )
    single_list_prediction = model.predict([_make_graphs()[0]])
    assert isinstance(single_list_prediction, list)
    assert len(single_list_prediction) == 1
    batched_prediction = model.predict(pgl.Graph.batch(list(_make_graphs())))
    assert isinstance(batched_prediction, dict)
    assert list(model.state_dict()) == expected_keys
    assert all(not key.startswith("model.") for key in model.state_dict())


def test_public_cinn_backend_fails_fast_on_cpu():
    graph, _ = _make_graphs()
    model = _make_model()
    model.set_execution_backend("cinn")

    with pytest.raises(RuntimeError, match="requires"):
        model({"graph": graph}, return_loss=False)


def test_eager_backend_supports_non_default_state_dimension():
    graph, _ = _make_graphs()
    model = _make_model(dim_state_embedding=4)
    model.eval()

    result = model({"graph": graph}, return_loss=False)

    assert model.execution_backend == "eager"
    assert result["pred_dict"]["formation_energy_per_atom"].shape == [1, 1]


def test_eager_backend_ignores_unconsumed_state_attr():
    graph, _ = _make_graphs()
    model = _make_model()
    model.eval()

    expected = model({"graph": graph}, return_loss=False)
    actual = model(
        {
            "graph": graph,
            "state_attr": np.ones((1, 2), dtype=np.float64),
        },
        return_loss=False,
    )

    _assert_allclose(
        actual["pred_dict"]["formation_energy_per_atom"],
        expected["pred_dict"]["formation_energy_per_atom"],
    )
