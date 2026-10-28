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

import numpy as np
import paddle
import pytest

from ppmat.datasets.graph_utils import infgcn_graph_utils as graph_utils

NO_GPU = not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() == 0
PLACES = ["cpu", pytest.param("gpu:0", marks=pytest.mark.skipif(NO_GPU, reason="GPU"))]


def brute_force_edges(x, y, r, batch_x, batch_y, max_num_neighbors):
    """Same-graph pairs within ``r``, nearest first with ties to the lower index."""
    dist = ((y[:, None] - x[None]) ** 2).sum(-1)
    edges = []
    for dst in range(len(y)):
        src = np.flatnonzero((dist[dst] <= r * r) & (batch_x == batch_y[dst]))
        src = src[np.argsort(dist[dst, src], kind="stable")][:max_num_neighbors]
        edges += [(dst, s) for s in sorted(src.tolist())]
    return edges


def edge_list(dst, src):
    assert dst.dtype == paddle.int64 and src.dtype == paddle.int64
    return list(zip(dst.numpy().tolist(), src.numpy().tolist()))


@pytest.mark.parametrize("place", PLACES)
@pytest.mark.parametrize("chunk_pairs, max_num_neighbors", [(1 << 22, None), (997, 4)])
def test_radius_matches_brute_force(monkeypatch, place, chunk_pairs, max_num_neighbors):
    monkeypatch.setattr(graph_utils, "_RADIUS_CHUNK_PAIRS", chunk_pairs)
    rng = np.random.default_rng(0)
    # Sources and 500 destinations on a 0.25 lattice give exact distance ties and
    # pairs exactly at the cutoff; 1000 more destinations lie within a few float32
    # ulps of the cutoff sphere of one source of their graph.
    x = rng.integers(0, 24, size=(200, 3)) * 0.25
    batch_x = rng.integers(0, 3, size=200)
    owner = rng.integers(0, 200, size=1000)
    direction = rng.normal(size=(1000, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    scale = 1.5 * (1 + 6e-8 * rng.integers(-8, 9, size=(1000, 1)))
    lattice = rng.integers(0, 24, size=(500, 3)) * 0.25
    y = np.concatenate([lattice, x[owner] + scale * direction])
    # Unsorted graph ids; graph 3 has destinations but no sources.
    batch_y = np.concatenate([rng.integers(0, 4, size=500), batch_x[owner]])
    x, y = x.astype("float32"), y.astype("float32")
    tensors = [paddle.to_tensor(a, place=place) for a in (x, y, batch_x, batch_y)]

    dst, src = graph_utils.radius(*tensors[:2], 1.5, *tensors[2:], max_num_neighbors)

    expected = brute_force_edges(x, y, 1.5, batch_x, batch_y, max_num_neighbors)
    assert edge_list(dst, src) == expected


@pytest.mark.parametrize("place", PLACES)
def test_radius_cap_keeps_nearest_then_lowest_index(place):
    # Sources 0-3 lie exactly at r = 1 from the destination, 4 just beyond it
    # and 5 closer.
    x = [[0, 0, 0], [2, 0, 0], [1, 1, 0], [1, 0, 1], [1, -1.001, 0], [1, 0, 0.5]]
    x = paddle.to_tensor(x, dtype="float32", place=place)
    y = paddle.to_tensor([[1, 0, 0]], dtype="float32", place=place)

    all_edges = graph_utils.radius(x, y, 1.0, max_num_neighbors=None)
    capped = graph_utils.radius(x, y, 1.0, max_num_neighbors=2)

    assert edge_list(*all_edges) == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 5)]
    assert edge_list(*capped) == [(0, 0), (0, 5)]


@pytest.mark.parametrize("place", PLACES)
def test_radius_with_empty_input_returns_no_edges(place):
    points = paddle.to_tensor([[0.0, 0.0, 0.0]], place=place)
    empty = paddle.to_tensor(np.zeros((0, 3), dtype="float32"), place=place)

    assert edge_list(*graph_utils.radius(empty, points, 1.0)) == []
    assert edge_list(*graph_utils.radius(points, empty, 1.0)) == []
