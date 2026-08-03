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

from ppmat.models.dimenetpp.dimenetpp import DimeNetPlusPlus
from ppmat.models.dimenetpp.dimenetpp_cinn import DimeNetPPTensorCore
from ppmat.models.dimenetpp.dimenetpp_cinn import _triplet_indices
from ppmat.models.dimenetpp.dimenetpp_cinn import graph_to_tensor_batch
from ppmat.models.dimenetpp.dimenetpp_cinn import pack_pgl_graph


@pytest.fixture(autouse=True)
def _cpu_device():
    original_device = paddle.get_device()
    paddle.set_device("cpu")
    yield
    paddle.set_device(original_device)


def _make_graph(num_nodes: int, coordinate_shift: float = 0.0) -> pgl.Graph:
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
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
            "atom_types": np.arange(1, num_nodes + 1, dtype=np.int64),
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


def _make_graphs() -> tuple[pgl.Graph, pgl.Graph]:
    return _make_graph(3), _make_graph(4, coordinate_shift=0.125)


def _make_model(execution_backend: str = "eager") -> DimeNetPlusPlus:
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
            num_embeddings=20,
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


def _assert_allclose(actual, expected, atol=1e-6, rtol=1e-6):
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=atol, rtol=rtol)


def test_graph_to_tensor_batch_preserves_graph_and_triplet_contract():
    graph_a, graph_b = _make_graphs()
    graph = pgl.Graph.batch([graph_a, graph_b])
    graph.node_feat["unused_metadata"] = np.asarray(["ignored"] * 7)
    packed = graph_to_tensor_batch(graph)

    assert not graph.is_tensor()
    assert packed is not graph_to_tensor_batch(graph)
    assert pack_pgl_graph(graph)._fields == packed._fields
    np.testing.assert_array_equal(packed.node_graph_id.numpy(), [0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(packed.edge_graph_id.numpy(), [0] * 6 + [1] * 12)

    del graph.node_feat["unused_metadata"]
    tensor_graph = graph.tensor(inplace=False)
    tensor_packed = graph_to_tensor_batch(tensor_graph)
    assert tensor_packed.atom_types is tensor_graph.node_feat["atom_types"]
    assert tensor_packed.frac_coords is tensor_graph.node_feat["frac_coords"]
    eager_triplets = _make_model().triplets(
        tensor_graph.edges.T, num_nodes=tensor_graph.graph_node_id.shape[0]
    )
    np.testing.assert_array_equal(packed.idx_kj.numpy(), eager_triplets[-2].numpy())
    np.testing.assert_array_equal(packed.idx_ji.numpy(), eager_triplets[-1].numpy())


def test_triplet_indices_preserve_eager_edge_order():
    edges = np.asarray(
        [[0, 1], [2, 0], [3, 0], [1, 2], [0, 2], [2, 1]],
        dtype=np.int64,
    )

    idx_kj, idx_ji = _triplet_indices(edges)

    np.testing.assert_array_equal(idx_kj, [1, 2, 3, 0, 2, 4])
    np.testing.assert_array_equal(idx_ji, [0, 0, 1, 3, 4, 5])
    empty_kj, empty_ji = _triplet_indices(np.empty([0, 2], dtype=np.int64))
    assert empty_kj.shape == empty_ji.shape == (0,)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_tensor_core_matches_eager_forward(batch_size):
    graph_a, graph_b = _make_graphs()
    graph = graph_a if batch_size == 1 else pgl.Graph.batch([graph_a, graph_b])
    model = _make_model()
    core = DimeNetPPTensorCore(model)

    expected = model._forward_eager({"graph": graph})
    actual = core(*graph_to_tensor_batch(graph))

    assert actual.shape == [batch_size, 1]
    _assert_allclose(actual, expected)


def test_tensor_core_matches_eager_without_triplets():
    graph = _make_graph(2)
    model = _make_model()
    packed = graph_to_tensor_batch(graph)

    assert packed.idx_kj.shape == [0]
    assert packed.idx_ji.shape == [0]
    expected = model._forward_eager({"graph": graph})
    actual = DimeNetPPTensorCore(model)(*packed)

    _assert_allclose(actual, expected)


def test_tensor_core_matches_eager_gradients_and_adam_step():
    graph_a, graph_b = _make_graphs()
    reference_model = _make_model()
    tensor_model = _make_model()
    tensor_model.set_state_dict(reference_model.state_dict())
    tensor_core = DimeNetPPTensorCore(tensor_model)
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
    for name, reference_parameter in reference_parameters.items():
        tensor_parameter = tensor_parameters[name]
        if reference_parameter.grad is None:
            assert tensor_parameter.grad is None
        else:
            _assert_allclose(
                tensor_parameter.grad,
                reference_parameter.grad,
                atol=2e-6,
                rtol=2e-6,
            )

    reference_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=reference_model.parameters()
    )
    tensor_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=tensor_model.parameters()
    )
    reference_optimizer.step()
    tensor_optimizer.step()
    for name, reference_parameter in reference_parameters.items():
        _assert_allclose(
            tensor_parameters[name], reference_parameter, atol=2e-6, rtol=2e-6
        )


def test_public_backend_preserves_forward_predict_and_checkpoint_contract(monkeypatch):
    reference_model = _make_model()
    model = _make_model()
    model.set_state_dict(reference_model.state_dict())
    reference_model.eval()
    model.eval()
    runtime = DimeNetPPTensorCore(model)
    monkeypatch.setattr(model, "_validate_cinn_environment", lambda: None)
    monkeypatch.setattr(model, "_get_cinn_runtime", lambda: runtime)
    expected_keys = list(model.state_dict())
    model.set_execution_backend("cinn")

    graph_a, graph_b = _make_graphs()
    expected = reference_model(
        {
            "graph": pgl.Graph.batch([graph_a, graph_b]),
            "formation_energy_per_atom": paddle.to_tensor(
                [[0.25], [-0.75]], dtype="float32"
            ),
        }
    )
    actual = model(
        {
            "graph": pgl.Graph.batch(_make_graphs()),
            "formation_energy_per_atom": paddle.to_tensor(
                [[0.25], [-0.75]], dtype="float32"
            ),
        }
    )
    _assert_allclose(actual["loss_dict"]["loss"], expected["loss_dict"]["loss"])
    _assert_allclose(
        actual["pred_dict"]["formation_energy_per_atom"],
        expected["pred_dict"]["formation_energy_per_atom"],
    )
    expected_prediction = reference_model.predict(graph_a)
    actual_prediction = model.predict(_make_graphs()[0])
    np.testing.assert_allclose(
        actual_prediction["formation_energy_per_atom"],
        expected_prediction["formation_energy_per_atom"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert list(model.state_dict()) == expected_keys
    assert all(not key.startswith("model.") for key in model.state_dict())


def test_set_state_dict_reuses_cached_runtime():
    model = _make_model()
    runtime = object()
    object.__setattr__(model, "_cinn_runtimes", {"train": runtime})
    object.__setattr__(model, "_cinn_warmed_modes", {"train"})

    model.set_state_dict(model.state_dict())

    assert model._cinn_runtimes == {"train": runtime}
    assert model._cinn_warmed_modes == {"train"}


def test_public_cinn_backend_fails_fast_on_cpu():
    model = _make_model(execution_backend="cinn")
    with pytest.raises(RuntimeError, match="requires a GPU device"):
        model.validate_execution_backend()


def test_graph_to_tensor_batch_validates_required_features():
    graph = pgl.Graph(
        edges=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        num_nodes=2,
        node_feat={"atom_types": np.asarray([1, 8], dtype=np.int64)},
    )
    with pytest.raises(ValueError, match=r"node_feat\['frac_coords'\]"):
        graph_to_tensor_batch(graph)


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_CINN_TRAINING_TESTS") != "1",
    reason="Set PPMAT_RUN_CINN_TRAINING_TESTS=1 for the GPU CINN canary.",
)
def test_public_dynamic_cinn_forward_backward_and_adam_step():
    if not paddle.is_compiled_with_cuda():
        pytest.skip("Paddle was not compiled with CUDA.")
    if not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle was not compiled with CINN.")
    paddle.set_device("gpu:0")

    reference_model = _make_model()
    cinn_model = _make_model(execution_backend="cinn")
    cinn_model.set_state_dict(reference_model.state_dict())
    expected_keys = list(cinn_model.state_dict())
    reference_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=reference_model.parameters()
    )
    cinn_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=cinn_model.parameters()
    )
    graph_a, graph_b = _make_graphs()
    graph_sequence = (
        graph_a,
        pgl.Graph.batch([graph_a, graph_b]),
    )
    target_sequence = (
        paddle.to_tensor([[0.25]], dtype="float32"),
        paddle.to_tensor([[0.5], [-0.75]], dtype="float32"),
    )
    no_triplet_graph = _make_graph(2)
    no_triplet_target = paddle.to_tensor([[0.0]], dtype="float32")
    no_triplet_reference = reference_model(
        {
            "graph": no_triplet_graph,
            "formation_energy_per_atom": no_triplet_target,
        }
    )
    no_triplet_actual = cinn_model(
        {
            "graph": no_triplet_graph,
            "formation_energy_per_atom": no_triplet_target,
        }
    )
    _assert_allclose(
        no_triplet_actual["pred_dict"]["formation_energy_per_atom"],
        no_triplet_reference["pred_dict"]["formation_energy_per_atom"],
        atol=2e-5,
        rtol=2e-5,
    )
    runtime_id = id(cinn_model._cinn_runtimes["train"])

    for graph, target in zip(graph_sequence, target_sequence):
        reference = reference_model(
            {"graph": graph, "formation_energy_per_atom": target}
        )
        actual = cinn_model({"graph": graph, "formation_energy_per_atom": target})
        reference_loss = reference["loss_dict"]["loss"]
        actual_loss = actual["loss_dict"]["loss"]
        reference_loss.backward()
        actual_loss.backward()

        _assert_allclose(
            actual["pred_dict"]["formation_energy_per_atom"],
            reference["pred_dict"]["formation_energy_per_atom"],
            atol=2e-5,
            rtol=2e-5,
        )
        _assert_allclose(actual_loss, reference_loss, atol=2e-5, rtol=2e-5)
        reference_parameters = dict(reference_model.named_parameters())
        cinn_parameters = dict(cinn_model.named_parameters())
        for name, reference_parameter in reference_parameters.items():
            cinn_parameter = cinn_parameters[name]
            if reference_parameter.grad is None:
                assert cinn_parameter.grad is None
            else:
                _assert_allclose(
                    cinn_parameter.grad,
                    reference_parameter.grad,
                    atol=2e-5,
                    rtol=2e-5,
                )

        reference_optimizer.step()
        cinn_optimizer.step()
        reference_optimizer.clear_grad()
        cinn_optimizer.clear_grad()
        for name, reference_parameter in reference_parameters.items():
            _assert_allclose(
                cinn_parameters[name],
                reference_parameter,
                atol=2e-5,
                rtol=2e-5,
            )

        current_runtime_id = id(cinn_model._cinn_runtimes["train"])
        assert current_runtime_id == runtime_id

    assert list(cinn_model.state_dict()) == expected_keys
    assert all(not key.startswith("model.") for key in cinn_model.state_dict())
