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

PLACES = ["cpu"]
if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
    PLACES.append("gpu:0")


def numpy_radius_edges(x, y, r, batch_x, batch_y, max_num_neighbors):
    """Edge set of the NumPy search that ``radius`` used before it ran on device."""
    if x.shape[0] < 1000 and y.shape[0] > 1000:
        search = graph_utils.radius_atoms_to_grids
    elif x.shape[0] > 1000 or y.shape[0] > 1000:
        search = graph_utils.radius_grid
    else:
        search = graph_utils.radius_simple
    src, dst, dist = search(x, y, r, batch_x, batch_y)
    keep = graph_utils._cap_edges_per_destination(src, dst, dist, max_num_neighbors)
    return set(zip(dst[keep].tolist(), src[keep].tolist()))


def lattice_points(rng, num_points, num_graphs):
    """Points on a 0.25 lattice, so that many distances tie or equal the cutoff."""
    coords = rng.integers(0, 24, size=(num_points, 3)) * 0.25
    batch = rng.integers(0, num_graphs, size=num_points)
    return coords.astype("float32"), batch.astype("int64")


def edge_list(dst, src):
    assert dst.dtype == paddle.int64 and src.dtype == paddle.int64
    return list(zip(dst.numpy().tolist(), src.numpy().tolist()))


@pytest.mark.parametrize("place", PLACES)
@pytest.mark.parametrize("chunk_pairs", [1 << 22, 997])
@pytest.mark.parametrize(
    "num_x, num_y, max_num_neighbors",
    [(40, 300, 32), (40, 300, 3), (60, 3000, 4), (60, 3000, None), (1200, 200, 6)],
)
def test_radius_matches_numpy_search(
    monkeypatch, place, chunk_pairs, num_x, num_y, max_num_neighbors
):
    monkeypatch.setattr(graph_utils, "_RADIUS_CHUNK_PAIRS", chunk_pairs)
    rng = np.random.default_rng(num_x + num_y)
    # Unsorted graph ids; graph 3 has destinations but no sources.
    x, batch_x = lattice_points(rng, num_x, 3)
    y, batch_y = lattice_points(rng, num_y, 4)
    expected = numpy_radius_edges(x, y, 1.5, batch_x, batch_y, max_num_neighbors)

    dst, src = graph_utils.radius(
        paddle.to_tensor(x, place=place),
        paddle.to_tensor(y, place=place),
        1.5,
        paddle.to_tensor(batch_x, place=place),
        paddle.to_tensor(batch_y, place=place),
        max_num_neighbors,
    )

    assert edge_list(dst, src) == sorted(expected)


@pytest.mark.parametrize("place", PLACES)
def test_radius_cutoff_is_inclusive_and_cap_prefers_near_then_low_index(place):
    # Sources 0-3 lie exactly at r = 1 from the destination, 4 just beyond it
    # and 5 closer.
    x = [[0, 0, 0], [2, 0, 0], [1, 1, 0], [1, 0, 1], [1, -1.001, 0], [1, 0, 0.5]]
    x = paddle.to_tensor(x, dtype="float32", place=place)
    y = paddle.to_tensor([[1, 0, 0]], dtype="float32", place=place)

    all_edges = graph_utils.radius(x, y, 1.0, max_num_neighbors=None)
    capped = graph_utils.radius(x, y, 1.0, max_num_neighbors=2)

    assert edge_list(*all_edges) == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 5)]
    assert edge_list(*capped) == [(0, 0), (0, 5)]
    assert edge_list(*graph_utils.radius(x, y + 10, 1.0)) == []
