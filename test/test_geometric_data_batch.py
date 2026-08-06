# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import pytest

from ppmat.datasets.geometric_data_type.batch import Batch
from ppmat.datasets.geometric_data_type.data import Data


def _numpy_graph(num_nodes, edge_index, face, value):
    return Data(
        x=np.arange(num_nodes, dtype=np.int64),
        pos=np.arange(num_nodes * 3, dtype=np.float32).reshape(num_nodes, 3),
        edge_index=np.asarray(edge_index, dtype=np.int64),
        edge_attr=np.full(len(edge_index[0]), value, dtype=np.float32),
        face=np.asarray(face, dtype=np.int64),
        y=np.asarray(value, dtype=np.float32),
        node_index_mask=np.asarray([True] * num_nodes, dtype=np.bool_),
    )


def test_batch_collates_and_restores_numpy_graph_fields():
    first = _numpy_graph(
        3,
        [[0, 1], [1, 2]],
        [[0], [1], [2]],
        1.0,
    )
    second = _numpy_graph(
        2,
        [[0, 1], [1, 0]],
        [[0], [1], [0]],
        2.0,
    )

    first.debug()
    second.debug()
    batch = Batch.from_data_list([first, second])

    assert isinstance(batch.x, np.ndarray)
    assert isinstance(batch.pos, np.ndarray)
    assert isinstance(batch.edge_index, np.ndarray)
    assert isinstance(batch.face, np.ndarray)
    assert isinstance(batch.batch, paddle.Tensor)
    np.testing.assert_array_equal(
        batch.edge_index,
        np.asarray([[0, 1, 3, 4], [1, 2, 4, 3]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        batch.face,
        np.asarray([[0, 3], [1, 4], [2, 3]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        batch.node_index_mask,
        np.asarray([True, True, True, True, True]),
    )
    assert batch.num_node_features == 1
    assert batch.num_edge_features == 1

    restored = batch.to_data_list()
    for actual, expected in zip(restored, (first, second)):
        for key in expected.keys:
            assert isinstance(actual[key], np.ndarray)
            assert actual[key].dtype == expected[key].dtype
            assert actual[key].shape == expected[key].shape
            np.testing.assert_array_equal(actual[key], expected[key])

    selected = batch[np.asarray([False, True], dtype=np.bool_)]
    assert len(selected) == 1
    np.testing.assert_array_equal(selected[0].edge_index, second.edge_index)


def test_batch_supports_empty_numpy_indices():
    graph = _numpy_graph(
        2,
        np.empty((2, 0), dtype=np.int64),
        np.empty((3, 0), dtype=np.int64),
        1.0,
    )

    graph.debug()
    batch = Batch.from_data_list([graph, graph])

    assert batch.edge_index.shape == (2, 0)
    assert batch.face.shape == (3, 0)
    for restored in batch.to_data_list():
        assert restored.edge_index.shape == (2, 0)
        assert restored.face.shape == (3, 0)


def test_batch_uses_declared_numpy_concatenation_axis():
    class AxisData(Data):
        def __cat_dim__(self, key, value):
            if key == "matrix":
                return -1
            return super().__cat_dim__(key, value)

    first = AxisData(
        x=np.zeros(2, dtype=np.float32),
        matrix=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    second = AxisData(
        x=np.zeros(1, dtype=np.float32),
        matrix=np.asarray([[5.0], [6.0]], dtype=np.float32),
    )

    batch = Batch.from_data_list([first, second])

    np.testing.assert_array_equal(
        batch.matrix,
        np.asarray([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]], dtype=np.float32),
    )
    restored = batch.to_data_list()
    np.testing.assert_array_equal(restored[0].matrix, first.matrix)
    np.testing.assert_array_equal(restored[1].matrix, second.matrix)


def test_batch_supports_numpy_vector_index_offsets():
    class BipartiteData(Data):
        def __inc__(self, key, value):
            if key == "edge_index":
                return np.asarray(
                    [[self.num_source_nodes], [self.num_target_nodes]],
                    dtype=np.int64,
                )
            return super().__inc__(key, value)

    first = BipartiteData(
        edge_index=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        num_source_nodes=2,
        num_target_nodes=3,
    )
    second = BipartiteData(
        edge_index=np.asarray([[0], [1]], dtype=np.int64),
        num_source_nodes=1,
        num_target_nodes=2,
    )

    batch = Batch.from_data_list([first, second])

    np.testing.assert_array_equal(
        batch.edge_index,
        np.asarray([[0, 1, 2], [1, 2, 4]], dtype=np.int64),
    )
    restored = batch.to_data_list()
    np.testing.assert_array_equal(restored[0].edge_index, first.edge_index)
    np.testing.assert_array_equal(restored[1].edge_index, second.edge_index)


@pytest.mark.parametrize("numpy_first", [False, True])
def test_batch_rejects_mixed_array_backends(numpy_first):
    numpy_graph = Data(x=np.asarray([1], dtype=np.int64))
    paddle_graph = Data(x=paddle.to_tensor([1], dtype="int64"))
    graphs = [numpy_graph, paddle_graph]
    if not numpy_first:
        graphs.reverse()

    with pytest.raises(TypeError, match="must be"):
        Batch.from_data_list(graphs)


def test_batch_preserves_paddle_index_support():
    graph = Data(
        x=paddle.to_tensor([0, 1], dtype="int64"),
        edge_index=paddle.to_tensor([[0, 1], [1, 0]], dtype="int64"),
        face=paddle.to_tensor([[0], [1], [0]], dtype="int64"),
        node_index_mask=paddle.to_tensor([True, False]),
    )

    graph.debug()
    batch = Batch.from_data_list([graph, graph])

    assert isinstance(batch.edge_index, paddle.Tensor)
    assert isinstance(batch.face, paddle.Tensor)
    np.testing.assert_array_equal(
        batch.node_index_mask.numpy(),
        np.asarray([True, False, True, False]),
    )
    np.testing.assert_array_equal(
        batch.edge_index.numpy(),
        np.asarray([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=np.int64),
    )
    restored = batch.to_data_list()
    np.testing.assert_array_equal(
        restored[1].edge_index.numpy(), graph.edge_index.numpy()
    )
    np.testing.assert_array_equal(
        restored[1].node_index_mask.numpy(), graph.node_index_mask.numpy()
    )
    selected = batch[paddle.to_tensor([False, True])]
    np.testing.assert_array_equal(
        selected[0].edge_index.numpy(), graph.edge_index.numpy()
    )


def test_batch_rejects_unsupported_index_type():
    batch = Batch.from_data_list([Data(x=np.asarray([1], dtype=np.int64))])

    with pytest.raises(
        IndexError,
        match="paddle.Tensor/numpy.ndarray with int64 or bool dtype",
    ):
        batch[1.5]
