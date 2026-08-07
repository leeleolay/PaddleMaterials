import numpy as np
import paddle
import pgl
import pytest

from ppmat.datasets.collate_fn import DensityCollator


def _atom_vocab():
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


def _sample(density, grid_coord=None, sample_id="sample"):
    density = np.asarray(density, dtype=np.float32)
    if grid_coord is None:
        grid_coord = np.arange(density.shape[0] * 3, dtype=np.float32).reshape(-1, 3)
    graph = pgl.Graph(
        edges=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        num_nodes=2,
        node_feat={
            "x": np.asarray([0, 0], dtype=np.int64),
            "cart_coords": np.asarray(
                [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
                dtype=np.float32,
            ),
        },
    )
    info = {
        "cell": np.eye(3, dtype=np.float32) * 4,
        "file_name": f"{sample_id}.cube",
    }
    return {
        "graph": graph,
        "density": density,
        "grid_coord": np.asarray(grid_coord, dtype=np.float32),
        "info": info,
        "id": sample_id,
    }


def _as_model_batch(batch):
    """Convert a collated batch the way the DataLoader or predictor would.

    Models consume tensors; only PGL graph fields are converted inside the
    model because the DataLoader passes graphs through untouched.
    """
    batch = dict(batch)
    for key in ("grid_coord", "density", "density_mask"):
        if isinstance(batch.get(key), np.ndarray):
            batch[key] = paddle.to_tensor(batch[key], dtype="float32")
    info = batch.get("info")
    if isinstance(info, dict) and isinstance(info.get("cell"), np.ndarray):
        batch["info"] = {**info, "cell": paddle.to_tensor(info["cell"], "float32")}
    return batch


def _infgcn(target_name="density", pbc=False):
    from ppmat.models.infgcn.infgcn import InfGCN

    return InfGCN(
        vocab=_atom_vocab(),
        num_radial=2,
        num_spherical=1,
        radial_embed_size=3,
        radial_hidden_size=4,
        num_radial_layer=1,
        num_gcn_layer=1,
        atom_graph_cutoff=2.0,
        atom_grid_cutoff=2.0,
        residual=not pbc,
        periodic_mode=("official" if pbc else "none"),
        target_name=target_name,
    )


def test_density_collator_rejects_length_mismatch():
    sample = _sample(
        [1.0, 2.0],
        grid_coord=np.zeros([1, 3], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="must match"):
        DensityCollator()([sample])


def test_density_collator_rejects_mismatched_channel_shapes():
    first = _sample(np.zeros([4, 2], dtype=np.float32), sample_id="first")
    second = _sample(np.zeros([4, 3], dtype=np.float32), sample_id="second")
    grid = np.zeros([4, 3], dtype=np.float32)
    first["grid_coord"] = grid
    second["grid_coord"] = grid

    with pytest.raises(ValueError, match="channel shapes must match"):
        DensityCollator()([first, second])


def test_density_padding_warns_on_strongly_skewed_lengths(monkeypatch):
    from ppmat.datasets import collate_fn as collate_fn_module

    warnings: list[str] = []
    monkeypatch.setattr(
        collate_fn_module.logger,
        "warning",
        lambda message, *args: warnings.append(message),
    )
    collator = DensityCollator(pad_skew_warn_ratio=4.0)

    collator(
        [
            _sample(np.zeros(10, dtype=np.float32), sample_id="short"),
            _sample(np.zeros(2000, dtype=np.float32), sample_id="long"),
        ]
    )
    assert len(warnings) == 1
    assert "very different" in warnings[0]

    collator(
        [
            _sample(np.zeros(1000, dtype=np.float32), sample_id="a"),
            _sample(np.zeros(2000, dtype=np.float32), sample_id="b"),
        ]
    )
    assert len(warnings) == 1


def test_density_collator_rejects_non_positive_pad_skew_warn_ratio():
    with pytest.raises(ValueError, match="pad_skew_warn_ratio"):
        DensityCollator(pad_skew_warn_ratio=0.0)


def test_density_collator_rejects_empty_density_grid_pair():
    with pytest.raises(ValueError, match="Empty density/grid"):
        DensityCollator()([_sample([])])


def test_density_collator_requires_keyword_arguments():
    with pytest.raises(TypeError, match="positional"):
        DensityCollator(8)


def test_equal_length_batch_leaves_density_mask_unset():
    batch = DensityCollator()([_sample([1.0, 2.0]), _sample([3.0, 4.0])])

    assert batch["density_mask"] is None
    np.testing.assert_array_equal(batch["density"], [[1.0, 2.0], [3.0, 4.0]])


def test_equal_length_batch_matches_all_ones_mask_in_infgcn():
    paddle.seed(0)
    model = _infgcn()
    batch = DensityCollator()([_sample([0.1, 0.2]), _sample([0.3, 0.4])])
    assert batch["density_mask"] is None

    masked_batch = dict(batch)
    masked_batch["density_mask"] = np.ones_like(batch["density"], dtype=np.float32)

    without_mask = model(_as_model_batch(batch))
    with_mask = model(_as_model_batch(masked_batch))

    np.testing.assert_allclose(
        without_mask["pred_dict"]["density"].numpy(),
        with_mask["pred_dict"]["density"].numpy(),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        float(without_mask["loss_dict"]["mae"]),
        float(with_mask["loss_dict"]["mae"]),
        rtol=1e-6,
        atol=0,
    )


def test_density_collator_clips_only_values_above_maximum():
    batch = DensityCollator(clip_max=1.0)([_sample([-2.0, 0.5, 2.0])])

    np.testing.assert_array_equal(
        batch["density"],
        np.asarray([[-2.0, 0.5, 1.0]], dtype=np.float32),
    )


def test_density_collator_mask_uses_lengths_not_density_values():
    batch = DensityCollator()([_sample([-1.0, 2.0]), _sample([3.0])])

    assert isinstance(batch["density"], np.ndarray)
    assert isinstance(batch["density_mask"], np.ndarray)
    assert isinstance(batch["grid_coord"], np.ndarray)
    assert isinstance(batch["graph"].node_feat["x"], np.ndarray)
    assert isinstance(batch["graph"].node_feat["cart_coords"], np.ndarray)
    np.testing.assert_array_equal(batch["graph"].graph_node_id, [0, 0, 1, 1])
    assert batch["graph"].edges.shape == (4, 2)
    np.testing.assert_array_equal(batch["density_mask"], [[1, 1], [1, 0]])
    assert batch["density"][0, 0].item() == -1.0
    assert batch["density"][1, 1].item() == 0.0


def test_pgl_batches_graphs_without_edges():
    graphs = [
        pgl.Graph(
            edges=[],
            num_nodes=num_nodes,
            node_feat={
                "x": np.zeros([num_nodes], dtype=np.int64),
                "cart_coords": np.zeros([num_nodes, 3], dtype=np.float32),
            },
        )
        for num_nodes in [1, 2]
    ]

    graph = pgl.Graph.batch(graphs)

    assert graph.edges.shape == (0, 2)
    np.testing.assert_array_equal(graph.graph_node_id, [0, 1, 1])
    np.testing.assert_array_equal(graph.node_feat["x"], [0, 0, 0])


def test_density_collator_preserves_default_mapping_contract():
    first = _sample([1.0, 2.0], sample_id="first")
    second = _sample([3.0], sample_id="second")
    first["extra"] = np.asarray([1.0], dtype=np.float32)
    second["extra"] = np.asarray([2.0], dtype=np.float32)

    batch = DensityCollator()([first, second])

    assert set(batch) == {
        "graph",
        "density",
        "density_mask",
        "grid_coord",
        "info",
        "id",
        "extra",
    }
    assert batch["id"] == ["first", "second"]
    assert set(batch["info"]) == {"cell"}
    assert batch["info"]["cell"].shape == (2, 3, 3)
    np.testing.assert_array_equal(batch["extra"], [[1.0], [2.0]])


@pytest.mark.parametrize("through_dataloader", [False, True])
def test_density_collator_batch_runs_infgcn_forward_and_backward(through_dataloader):
    from ppmat.models.infgcn.infgcn import InfGCN

    previous_device = paddle.get_device()
    paddle.set_device("cpu")
    try:
        graph = pgl.Graph(
            edges=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
            num_nodes=2,
            node_feat={
                "x": np.asarray([0, 0], dtype=np.int64),
                "cart_coords": np.asarray(
                    [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
                    dtype=np.float32,
                ),
            },
        )
        density = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        grid_coord = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
            ],
            dtype="float32",
        )
        info = {
            "shape": [4, 1, 1],
            "cell": np.eye(3, dtype=np.float32) * 4,
            "coordinate_unit": "angstrom",
            "density_unit": "unknown",
            "file_name": "sample",
        }
        samples = [
            {
                "graph": graph,
                "density": density,
                "grid_coord": grid_coord,
                "info": info,
                "id": "sample",
            }
        ]
        if through_dataloader:
            loader = paddle.io.DataLoader(
                samples, batch_size=1, collate_fn=DensityCollator(), return_list=True
            )
            batch = next(iter(loader))
        else:
            # The collator keeps NumPy; only the DataLoader tensorizes, so the
            # predictor path converts the non-graph fields itself.
            raw = DensityCollator()(samples)
            assert isinstance(raw["density"], np.ndarray)
            assert isinstance(raw["info"]["cell"], np.ndarray)
            batch = _as_model_batch(raw)
        # PGL graphs pass through both paths untouched.
        assert isinstance(batch["graph"].node_feat["x"], np.ndarray)
        assert isinstance(batch["density"], paddle.Tensor)
        assert isinstance(batch["info"]["cell"], paddle.Tensor)
        model = InfGCN(
            vocab=_atom_vocab(),
            num_radial=2,
            num_spherical=1,
            radial_embed_size=3,
            radial_hidden_size=4,
            num_radial_layer=1,
            num_gcn_layer=1,
            atom_graph_cutoff=2.0,
            atom_grid_cutoff=2.0,
            residual=True,
        )

        output = model(batch)
        loss = output["loss_dict"]["loss"]
        loss.backward()

        assert output["pred_dict"]["density"].shape == [1, 4]
        assert paddle.isfinite(loss).item()
        assert any(parameter.grad is not None for parameter in model.parameters())
    finally:
        paddle.set_device(previous_device)


def test_infgcn_forward_uses_target_name_and_output_flags(monkeypatch):
    target_name = "charge_density"
    model = _infgcn(target_name=target_name)
    sample = _sample([0.1, 0.2])
    batch = _as_model_batch(DensityCollator()([sample]))
    batch[target_name] = batch.pop("density")
    prediction = paddle.to_tensor([[0.2, 0.3]], dtype="float32")
    monkeypatch.setattr(
        model,
        "_forward_density",
        lambda *args: prediction,
    )

    output = model(batch)
    assert set(output["loss_dict"]) == {"loss", "mae"}
    assert set(output["pred_dict"]) == {target_name}

    loss_only = model(batch, return_loss=True, return_prediction=False)
    assert set(loss_only["loss_dict"]) == {"loss", "mae"}
    assert loss_only["pred_dict"] == {}

    prediction_batch = {
        key: value for key, value in batch.items() if key != target_name
    }
    prediction_only = model(
        prediction_batch,
        return_loss=False,
        return_prediction=True,
    )
    assert prediction_only["loss_dict"] == {}
    assert set(prediction_only["pred_dict"]) == {target_name}

    with pytest.raises(AssertionError, match="At least one"):
        model(batch, return_loss=False, return_prediction=False)
    with pytest.raises(KeyError, match=target_name):
        model(
            prediction_batch,
            return_loss=True,
            return_prediction=False,
        )


def test_infgcn_uses_batched_cell_for_periodic_inputs(monkeypatch):
    model = _infgcn(pbc=True)
    batch = _as_model_batch(
        DensityCollator()(
            [
                _sample([0.1], sample_id="first"),
                _sample([0.2], sample_id="second"),
            ]
        )
    )

    def fake_forward(*args):
        cell = args[-1]
        assert list(cell.shape) == [2, 3, 3]
        return paddle.zeros([2, 1], dtype="float32")

    monkeypatch.setattr(model, "_forward_density", fake_forward)
    output = model(batch)

    assert list(output["pred_dict"]["density"].shape) == [2, 1]


def test_infgcn_chunks_full_grid_only_during_evaluation(monkeypatch):
    model = _infgcn()
    model.inference_grid_batch_size = 1
    batch = _as_model_batch(DensityCollator()([_sample([0.1, 0.2])]))
    chunk_sizes = []

    def fake_forward(*args):
        grid = args[3]
        chunk_sizes.append(grid.shape[1])
        return paddle.zeros(grid.shape[:2], dtype="float32")

    monkeypatch.setattr(model, "_forward_density", fake_forward)
    model.eval()
    output = model(batch)

    assert chunk_sizes == [1, 1]
    assert list(output["pred_dict"]["density"].shape) == [1, 2]


def test_radius_converter_deterministically_keeps_nearest_32_neighbors():
    from ppmat.models.common.graph_converter import RadiusGraphConverter

    positions = np.zeros([41, 3], dtype=np.float32)
    positions[:, 0] = np.arange(41, dtype=np.float32) * 0.01
    graph = RadiusGraphConverter(
        cutoff=1.0,
        coordinate_unit="angstrom",
        inclusive_cutoff=True,
        atom_vocab={},
        include_distance=False,
        max_num_neighbors=32,
    ).from_arrays(
        np.ones([41], dtype=np.int64),
        positions,
    )

    edges = np.asarray(graph.edges, dtype=np.int64)
    assert edges.shape == (41 * 32, 2)
    for target in range(41):
        sources = edges[edges[:, 1] == target, 0]
        expected = sorted(
            (source for source in range(41) if source != target),
            key=lambda source: (
                abs(positions[source, 0] - positions[target, 0]),
                source,
            ),
        )[:32]
        np.testing.assert_array_equal(sources, expected)
