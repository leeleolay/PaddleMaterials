# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle

from ppmat.datasets.graph_utils.infgcn_graph_utils import radius


class AtomGridRadiusGraphConverter:
    """Build a runtime bipartite radius graph from atoms to grid points.

    Unlike the static atom graph converter, this converter consumes the current
    sampled or chunked grid tensor. Its edge contract follows PGL ordering:
    ``edge_index[0]`` is the atom source and ``edge_index[1]`` is the grid
    destination.
    """

    def __init__(self, cutoff: float, max_num_neighbors: int = 32) -> None:
        if cutoff <= 0:
            raise ValueError("cutoff must be positive.")
        if max_num_neighbors <= 0:
            raise ValueError("max_num_neighbors must be positive.")
        self.cutoff = float(cutoff)
        self.max_num_neighbors = int(max_num_neighbors)

    def __call__(
        self,
        atom_coord: paddle.Tensor,
        grid_coord: paddle.Tensor,
        atom_batch: paddle.Tensor,
        grid_batch: paddle.Tensor,
    ) -> paddle.Tensor:
        grid_dst, atom_src = radius(
            atom_coord,
            grid_coord,
            self.cutoff,
            atom_batch,
            grid_batch,
            self.max_num_neighbors,
        )
        if atom_src.shape[0] == 0:
            return paddle.zeros([2, 0], dtype="int64")
        return paddle.stack([atom_src, grid_dst], axis=0)
