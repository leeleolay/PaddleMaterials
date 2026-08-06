"""Contract tests for MatENO's standard density batch interface."""

import numpy as np
import paddle
import pgl
import pytest

from ppmat.datasets.collate_fn import DensityCollator
from ppmat.models.mateno.mateno import MatENO


def setup_module():
    # The contract is intentionally exercised on the CPU path used by workers.
    paddle.set_device("cpu")


def _vocab():
    return {
        "atom": {
            "type": "element",
            "tokens": ["C"],
            "num_embeddings": 1,
            "token_to_id": {"C": 0},
            "id_to_token": {0: "C"},
            "atomic_number_to_id": {6: 0},
            "id_to_atomic_number": {0: 6},
        }
    }


def _graph(offset=0.0):
    return pgl.Graph(
        edges=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        num_nodes=2,
        node_feat={
            "x": np.asarray([0, 0], dtype=np.int64),
            "pos": np.asarray(
                [[offset, 0.0, 0.0], [offset + 0.8, 0.0, 0.0]],
                dtype=np.float32,
            ),
        },
    )


def _sample(index):
    offset = float(index) * 4.0
    return {
        "graph": _graph(offset),
        "density": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "grid_coord": np.asarray(
            [
                [offset, 0.0, 0.0],
                [offset + 0.2, 0.0, 0.0],
                [offset + 0.7, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "info": {"cell": np.eye(3, dtype=np.float32) * 4.0},
        "id": f"sample-{index}",
    }


def _model():
    return MatENO(
        vocab=_vocab(),
        num_radial=2,
        num_spherical=1,
        radial_embed_size=3,
        radial_hidden_size=4,
        num_radial_layer=1,
        num_gcn_layer=1,
        cutoff=2.0,
        grid_cutoff=2.0,
        embedding_dim=4,
        max_num_neighbors=8,
    )


def _batch(*samples):
    loader = paddle.io.DataLoader(
        list(samples),
        batch_size=len(samples),
        collate_fn=DensityCollator(),
        return_list=True,
    )
    return next(iter(loader))


def test_mateno_standard_dict_batch_contract_and_backward():
    """A density DataLoader batch must run through all model contracts."""
    batch = _batch(_sample(0), _sample(1))

    assert isinstance(batch["grid_coord"], paddle.Tensor)
    assert isinstance(batch["density"], paddle.Tensor)
    assert isinstance(batch["info"]["cell"], paddle.Tensor)
    assert isinstance(batch["graph"].node_feat["x"], np.ndarray)

    model = _model()
    output = model(batch)

    assert set(output) == {"loss_dict", "pred_dict"}
    assert set(output["loss_dict"]) == {"loss", "mae"}
    assert set(output["pred_dict"]) == {"density"}
    prediction = output["pred_dict"]["density"]
    assert list(prediction.shape) == [2, 3]
    for value in (output["loss_dict"]["loss"], output["loss_dict"]["mae"]):
        assert bool(paddle.isfinite(value).item())

    output["loss_dict"]["loss"].backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert any(gradient is not None for gradient in gradients)
    assert all(
        bool(paddle.isfinite(gradient).all().item())
        for gradient in gradients
        if gradient is not None
    )


def test_mateno_cpu_graph_tensor_path(monkeypatch):
    """Dict batches materialize NumPy PGL fields through ``Graph.tensor``."""
    model = _model()
    batch = _batch(_sample(0))
    graph = batch["graph"]
    tensor_calls = []
    original_tensor = pgl.Graph.tensor

    def tensor(self, *args, **kwargs):
        tensor_calls.append(self)
        return original_tensor(self, *args, **kwargs)

    monkeypatch.setattr(pgl.Graph, "tensor", tensor)
    seen = {}

    def fake_forward(atom_types, atom_coord, grid, graph_batch, cell):
        seen.update(
            atom_types=atom_types,
            atom_coord=atom_coord,
            grid=grid,
            graph_batch=graph_batch,
            cell=cell,
        )
        return paddle.zeros(grid.shape[:2], dtype="float32")

    monkeypatch.setattr(model, "_forward_density", fake_forward)
    assert isinstance(batch["grid_coord"], paddle.Tensor)
    assert isinstance(graph.node_feat["x"], np.ndarray)
    output = model(batch, return_loss=False, return_prediction=True)

    assert output["loss_dict"] == {}
    assert len(tensor_calls) == 1
    assert tensor_calls[0] is graph
    assert list(output["pred_dict"]["density"].shape) == [1, 3]
    assert isinstance(seen["atom_types"], paddle.Tensor)
    assert isinstance(seen["atom_coord"], paddle.Tensor)
    assert isinstance(seen["graph_batch"], paddle.Tensor)
    assert isinstance(seen["grid"], paddle.Tensor)
    assert seen["atom_types"].place.is_cpu_place()
    assert seen["atom_coord"].place.is_cpu_place()
    assert seen["graph_batch"].place.is_cpu_place()
    assert seen["grid"].place.is_cpu_place()
    assert isinstance(graph.node_feat["x"], paddle.Tensor)
    assert isinstance(graph.node_feat["pos"], paddle.Tensor)


def test_mateno_honors_standard_forward_flags(monkeypatch):
    model = _model()
    batch = _batch(_sample(0))

    monkeypatch.setattr(
        model,
        "_forward_density",
        lambda *args, **kwargs: paddle.zeros([1, 3], dtype="float32"),
    )

    prediction_batch = {key: value for key, value in batch.items() if key != "density"}
    prediction_only = model(
        prediction_batch,
        return_loss=False,
        return_prediction=True,
    )
    assert prediction_only["loss_dict"] == {}
    assert set(prediction_only["pred_dict"]) == {"density"}

    loss_only = model(
        batch,
        return_loss=True,
        return_prediction=False,
    )
    assert set(loss_only["loss_dict"]) == {"loss", "mae"}
    assert loss_only["pred_dict"] == {}

    with pytest.raises(AssertionError, match="At least one"):
        model(batch, return_loss=False, return_prediction=False)
