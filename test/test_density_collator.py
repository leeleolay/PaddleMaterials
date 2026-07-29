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
            "pos": np.asarray(
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
        cutoff=2.0,
        grid_cutoff=2.0,
        residual=True,
        pbc=pbc,
        target_name=target_name,
    )


def test_density_collator_rejects_length_mismatch():
    sample = _sample(
        [1.0, 2.0],
        grid_coord=np.zeros([1, 3], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="must match"):
        DensityCollator(n_samples=1)([sample])


def test_density_collator_rejects_empty_density_grid_pair():
    with pytest.raises(ValueError, match="Empty density/grid"):
        DensityCollator()([_sample([])])


@pytest.mark.parametrize("importance_sampling", [False, True])
def test_density_collator_sampling_seed_is_deterministic(importance_sampling):
    sample = _sample(np.linspace(0.0, 1.0, 20, dtype=np.float32))
    params = {
        "n_samples": 8,
        "sampling_mode": "random",
        "sampling_seed": 2026,
        "importance_sampling": importance_sampling,
    }

    first = DensityCollator(**params)([sample])
    second = DensityCollator(**params)([sample])

    assert first["id"] == ["sample"]
    np.testing.assert_array_equal(first["density"], second["density"])
    np.testing.assert_array_equal(first["grid_coord"], second["grid_coord"])


def test_density_collator_seed_reproduces_sampling_sequence():
    sample = _sample(np.linspace(0.0, 1.0, 20, dtype=np.float32))
    first = DensityCollator(
        n_samples=8,
        sampling_mode="uniform",
        uniform_random_offset=True,
        sampling_seed=2026,
    )
    second = DensityCollator(
        n_samples=8,
        sampling_mode="uniform",
        uniform_random_offset=True,
        sampling_seed=2026,
    )

    for _ in range(2):
        first_batch = first([sample])
        second_batch = second([sample])
        np.testing.assert_array_equal(
            first_batch["grid_coord"],
            second_batch["grid_coord"],
        )


def test_uniform_random_offset_samples_without_replacement_when_possible():
    sample = _sample(np.arange(20, dtype=np.float32))
    batch = DensityCollator(
        n_samples=8,
        sampling_mode="uniform",
        uniform_random_offset=True,
        sampling_seed=4,
    )([sample])

    sampled_density = batch["density"].reshape(-1)
    assert np.unique(sampled_density).size == sampled_density.size


def test_density_collator_can_fix_sampling_per_sample():
    sample = _sample(np.linspace(0.0, 1.0, 20, dtype=np.float32))
    collator = DensityCollator(
        n_samples=8,
        sampling_mode="uniform",
        uniform_random_offset=True,
        sampling_seed=2026,
        resample_each_call=False,
        importance_sampling=True,
    )

    first = collator([sample])
    second = collator([sample])

    np.testing.assert_array_equal(
        first["grid_coord"],
        second["grid_coord"],
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
    assert isinstance(batch["graph"].node_feat["pos"], np.ndarray)
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
                "pos": np.zeros([num_nodes, 3], dtype=np.float32),
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


@pytest.mark.parametrize("n_samples", [0, -1])
def test_density_collator_requires_positive_sample_count(n_samples):
    with pytest.raises(ValueError, match="positive"):
        DensityCollator(n_samples=n_samples)


def test_density_collator_rejects_non_integer_sample_count():
    with pytest.raises(TypeError, match="integer"):
        DensityCollator(n_samples=1.5)


def test_fixed_random_sampling_requires_seed():
    with pytest.raises(ValueError, match="sampling_seed"):
        DensityCollator(
            n_samples=1,
            sampling_mode="random",
            resample_each_call=False,
        )


def test_fixed_uniform_sampling_does_not_require_seed():
    batch = DensityCollator(n_samples=1, resample_each_call=False)(
        [_sample([1.0, 2.0])]
    )

    np.testing.assert_array_equal(batch["density"], [[1.0]])


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
                "pos": np.asarray(
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
            batch = DensityCollator()(samples)
        assert isinstance(batch["graph"].node_feat["x"], np.ndarray)
        if through_dataloader:
            assert isinstance(batch["density"], paddle.Tensor)
            assert isinstance(batch["info"]["cell"], paddle.Tensor)
        else:
            assert isinstance(batch["density"], np.ndarray)
            assert isinstance(batch["info"]["cell"], np.ndarray)
        model = InfGCN(
            vocab=_atom_vocab(),
            num_radial=2,
            num_spherical=1,
            radial_embed_size=3,
            radial_hidden_size=4,
            num_radial_layer=1,
            num_gcn_layer=1,
            cutoff=2.0,
            grid_cutoff=2.0,
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
    batch = DensityCollator()([sample])
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
    batch = DensityCollator()(
        [
            _sample([0.1], sample_id="first"),
            _sample([0.2], sample_id="second"),
        ]
    )

    def fake_forward(*args):
        cell = args[-1]
        assert list(cell.shape) == [2, 3, 3]
        return paddle.zeros([2, 1], dtype="float32")

    monkeypatch.setattr(model, "_forward_density", fake_forward)
    output = model(batch)

    assert list(output["pred_dict"]["density"].shape) == [2, 1]


def test_infgcn_preserves_random_max_32_atom_neighbor_truncation(monkeypatch):
    from ppmat.datasets.graph_utils.infgcn_graph_utils import radius_graph
    from ppmat.models.common.graph_converter import RadiusGraphConverter
    from ppmat.models.infgcn.infgcn import _randomly_truncate_atom_edges

    positions = np.zeros([41, 3], dtype=np.float32)
    positions[:, 0] = np.arange(41, dtype=np.float32) * 0.01
    graph = RadiusGraphConverter(
        cutoff=1.0,
        coordinate_unit="angstrom",
        inclusive_cutoff=True,
        atom_vocab={},
        include_distance=False,
    ).from_arrays(
        np.ones([41], dtype=np.int64),
        positions,
    )

    def deterministic_randperm(count):
        return paddle.arange(count - 1, -1, -1, dtype="int64")

    monkeypatch.setattr(paddle, "randperm", deterministic_randperm)
    expected = radius_graph(
        paddle.to_tensor(positions),
        r=1.0,
        loop=False,
    )
    candidate_edges = paddle.to_tensor(graph.edges).transpose([1, 0])
    actual = _randomly_truncate_atom_edges(candidate_edges)

    assert expected.shape == [2, 41 * 32]
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
