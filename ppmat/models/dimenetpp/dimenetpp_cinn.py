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

"""Tensor-only DimeNet++ path for dynamic-shape CINN execution.

PGL graph batching and discrete triplet construction remain at the Python
boundary. The compiled core owns differentiable geometry, basis expansion,
message passing, and graph readout while reusing the eager model's parameters.
"""

from __future__ import annotations

from typing import NamedTuple
from typing import Optional
from typing import Sequence

import numpy as np
import paddle
import pgl

from ppmat.models.dimenetpp.dimenetpp import DimeNetPlusPlus

__all__ = [
    "DimeNetPPTensorBatch",
    "DimeNetPPTensorCore",
    "compile_dimenetpp",
    "compile_dimenetpp_cinn",
    "graph_to_tensor_batch",
    "make_dimenetpp_input_spec",
    "pack_pgl_graph",
]


class DimeNetPPTensorBatch(NamedTuple):
    """Flat tensor representation of a possibly batched periodic PGL graph."""

    atom_types: paddle.Tensor
    frac_coords: paddle.Tensor
    lattice: paddle.Tensor
    edge_src: paddle.Tensor
    edge_dst: paddle.Tensor
    pbc_offset: paddle.Tensor
    node_graph_id: paddle.Tensor
    edge_graph_id: paddle.Tensor
    idx_kj: paddle.Tensor
    idx_ji: paddle.Tensor


def _to_tensor(value, dtype: str) -> paddle.Tensor:
    if isinstance(value, paddle.Tensor):
        return (
            value
            if value.dtype == getattr(paddle, dtype)
            else paddle.cast(value, dtype)
        )
    return paddle.to_tensor(np.asarray(value), dtype=dtype)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, paddle.Tensor):
        return value.numpy()
    return np.asarray(value)


def _triplet_indices(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match :meth:`DimeNetPlusPlus.triplets` without an E-by-E tensor."""

    num_edges = edges.shape[0]
    if num_edges == 0:
        empty = np.empty([0], dtype=np.int64)
        return empty, empty
    num_nodes = int(edges.max()) + 1
    incoming_order = np.argsort(edges[:, 1], kind="stable")
    incoming_count = np.bincount(edges[:, 1], minlength=num_nodes)
    incoming_start = np.cumsum(incoming_count) - incoming_count

    candidate_count = incoming_count[edges[:, 0]]
    idx_ji = np.repeat(np.arange(num_edges, dtype=np.int64), candidate_count)
    candidate_start = np.cumsum(candidate_count) - candidate_count
    candidate_offset = np.arange(idx_ji.shape[0], dtype=np.int64) - np.repeat(
        candidate_start, candidate_count
    )
    candidate_position = (
        np.repeat(incoming_start[edges[:, 0]], candidate_count) + candidate_offset
    )
    idx_kj = incoming_order[candidate_position]

    mask = edges[idx_kj, 0] != edges[idx_ji, 1]
    return idx_kj[mask], idx_ji[mask]


def graph_to_tensor_batch(
    graph: pgl.Graph | Sequence[pgl.Graph],
) -> DimeNetPPTensorBatch:
    """Pack a public PGL graph into the dynamic tensor contract used by CINN."""

    if isinstance(graph, (list, tuple)):
        if not graph:
            raise ValueError("graph must contain at least one PGL graph.")
        graph = pgl.Graph.batch(list(graph))
    if not isinstance(graph, pgl.Graph):
        raise TypeError(f"graph must be a pgl.Graph, got {type(graph)!r}.")

    edges_numpy = _to_numpy(graph.edges).astype(np.int64, copy=False)
    edges = _to_tensor(graph.edges, "int64")
    if len(edges.shape) != 2 or edges.shape[1] != 2:
        raise ValueError(f"graph.edges must have shape [E, 2], got {edges.shape}.")
    if edges.shape[0] == 0:
        raise ValueError("DimeNet++ requires at least one edge in the graph batch.")

    required_node_features = (
        "atom_types",
        "frac_coords",
        "lattice",
    )
    for name in required_node_features:
        if name not in graph.node_feat:
            raise ValueError(f"graph.node_feat[{name!r}] is required.")
    if "pbc_offset" not in graph.edge_feat:
        raise ValueError("graph.edge_feat['pbc_offset'] is required.")

    idx_kj, idx_ji = _triplet_indices(edges_numpy)
    return DimeNetPPTensorBatch(
        atom_types=_to_tensor(graph.node_feat["atom_types"], "int64"),
        frac_coords=_to_tensor(graph.node_feat["frac_coords"], "float32"),
        lattice=_to_tensor(graph.node_feat["lattice"], "float32"),
        edge_src=_to_tensor(edges[:, 0], "int64"),
        edge_dst=_to_tensor(edges[:, 1], "int64"),
        pbc_offset=_to_tensor(graph.edge_feat["pbc_offset"], "float32"),
        node_graph_id=_to_tensor(graph.graph_node_id, "int64"),
        edge_graph_id=_to_tensor(graph.graph_edge_id, "int64"),
        idx_kj=_to_tensor(idx_kj, "int64"),
        idx_ji=_to_tensor(idx_ji, "int64"),
    )


def pack_pgl_graph(
    graph: pgl.Graph | Sequence[pgl.Graph],
) -> DimeNetPPTensorBatch:
    """Alias that names the PGL-to-tensor boundary explicitly."""

    return graph_to_tensor_batch(graph)


def _sum_by_id(
    values: paddle.Tensor,
    segment_ids: paddle.Tensor,
    out_size: paddle.Tensor,
) -> paddle.Tensor:
    """CINN-compatible dynamic scatter sum."""

    # Paddle 3.3 implements an empty scatter as ``x + updates``, which fails
    # when a graph has no triplets. A zero-valued sentinel keeps the update
    # non-empty without changing either the result or gradients.
    safe_segment_ids = paddle.concat(
        [segment_ids, paddle.zeros([1], dtype="int64")], axis=0
    )
    safe_values = paddle.concat(
        [
            values,
            paddle.zeros([1, values.shape[1]], dtype=values.dtype),
        ],
        axis=0,
    )
    return paddle.scatter_nd_add(
        paddle.zeros([out_size, values.shape[1]], dtype=values.dtype),
        paddle.unsqueeze(safe_segment_ids, axis=1),
        safe_values,
    )


def _mean_by_id(
    values: paddle.Tensor,
    segment_ids: paddle.Tensor,
    out_size: paddle.Tensor,
) -> paddle.Tensor:
    """CINN-compatible dynamic scatter mean with zero empty segments."""

    sums = _sum_by_id(values, segment_ids, out_size)
    counts = _sum_by_id(paddle.ones_like(values[:, :1]), segment_ids, out_size)
    return sums / paddle.clip(counts, min=1.0)


class DimeNetPPTensorCore(paddle.nn.Layer):
    """DimeNet++ numerical core with a Tensor-only input boundary."""

    def __init__(self, model: DimeNetPlusPlus):
        super().__init__()
        if not isinstance(model, DimeNetPlusPlus):
            raise TypeError(f"model must be DimeNetPlusPlus, got {type(model)!r}.")
        if model.readout not in {"sum", "add", "mean"}:
            raise ValueError(
                "DimeNet++ CINN supports readout='sum' or 'mean', "
                f"got {model.readout!r}."
            )
        self.model = model

    @staticmethod
    def _output_block(block, x, rbf, atom_index, num_nodes):
        x = block.lin_rbf(rbf) * x
        x = _sum_by_id(x, atom_index, num_nodes)
        x = block.lin_up(x)
        for lin in block.lins:
            x = block.act(lin(x))
        return block.lin(x)

    @staticmethod
    def _interaction_block(block, x, rbf, sbf, idx_kj, idx_ji, num_edges):
        x_ji = block.act(block.lin_ji(x))
        x_kj = block.act(block.lin_kj(x))
        rbf_features = block.lin_rbf2(block.lin_rbf1(rbf))
        x_kj = block.act(block.lin_down(x_kj * rbf_features))
        sbf_features = block.lin_sbf2(block.lin_sbf1(sbf))
        x_kj = paddle.gather(x_kj, idx_kj, axis=0) * sbf_features
        x_kj = _sum_by_id(x_kj, idx_ji, num_edges)
        x_kj = block.act(block.lin_up(x_kj))
        h = x_ji + x_kj
        for layer in block.layers_before_skip:
            h = layer(h)
        h = block.act(block.lin(h)) + x
        for layer in block.layers_after_skip:
            h = layer(h)
        return h

    def forward(
        self,
        atom_types: paddle.Tensor,
        frac_coords: paddle.Tensor,
        lattice: paddle.Tensor,
        edge_src: paddle.Tensor,
        edge_dst: paddle.Tensor,
        pbc_offset: paddle.Tensor,
        node_graph_id: paddle.Tensor,
        edge_graph_id: paddle.Tensor,
        idx_kj: paddle.Tensor,
        idx_ji: paddle.Tensor,
    ) -> paddle.Tensor:
        model = self.model
        num_nodes = paddle.shape(atom_types)[0]
        num_edges = paddle.shape(edge_src)[0]
        num_graphs = paddle.shape(lattice)[0]

        node_lattice = paddle.gather(lattice, node_graph_id, axis=0)
        pos = paddle.bmm(frac_coords.unsqueeze(1), node_lattice).squeeze(1)
        edge_lattice = paddle.gather(lattice, edge_graph_id, axis=0)
        offsets = paddle.bmm(pbc_offset.unsqueeze(1), edge_lattice).squeeze(1)
        distance_vectors = (
            paddle.gather(pos, edge_src, axis=0)
            - paddle.gather(pos, edge_dst, axis=0)
            + offsets
        )
        dist = paddle.linalg.norm(distance_vectors, axis=-1)

        idx_i = paddle.gather(edge_dst, idx_ji, axis=0)
        idx_j = paddle.gather(edge_src, idx_ji, axis=0)
        idx_k = paddle.gather(edge_src, idx_kj, axis=0)
        pos_ji = (
            paddle.gather(pos, idx_j, axis=0)
            - paddle.gather(pos, idx_i, axis=0)
            + paddle.gather(offsets, idx_ji, axis=0)
        )
        pos_kj = (
            paddle.gather(pos, idx_k, axis=0)
            - paddle.gather(pos, idx_j, axis=0)
            + paddle.gather(offsets, idx_kj, axis=0)
        )
        angle = paddle.atan2(
            paddle.linalg.norm(paddle.cross(pos_ji, pos_kj), axis=-1),
            paddle.sum(pos_ji * pos_kj, axis=-1),
        )

        rbf = model.rbf(dist)
        sbf = model.sbf(dist, angle, idx_kj)
        x = model.emb(atom_types, rbf, edge_src, edge_dst)
        prediction = self._output_block(
            model.output_blocks[0], x, rbf, edge_src, num_nodes
        )
        for interaction, output in zip(
            model.interaction_blocks, model.output_blocks[1:]
        ):
            x = self._interaction_block(
                interaction, x, rbf, sbf, idx_kj, idx_ji, num_edges
            )
            prediction = prediction + self._output_block(
                output, x, rbf, edge_src, num_nodes
            )

        if model.readout == "mean":
            return _mean_by_id(prediction, node_graph_id, num_graphs)
        return _sum_by_id(prediction, node_graph_id, num_graphs)

    def forward_graph(self, graph: pgl.Graph | Sequence[pgl.Graph]) -> paddle.Tensor:
        """Convenience eager call for a public PGL graph."""

        return self(*graph_to_tensor_batch(graph))


def make_dimenetpp_input_spec() -> list[paddle.static.InputSpec]:
    """Return the dynamic-shape tensor contract used by dy2static/CINN."""

    return [
        paddle.static.InputSpec([None], dtype="int64", name="atom_types"),
        paddle.static.InputSpec([None, 3], dtype="float32", name="frac_coords"),
        paddle.static.InputSpec([None, 3, 3], dtype="float32", name="lattice"),
        paddle.static.InputSpec([None], dtype="int64", name="edge_src"),
        paddle.static.InputSpec([None], dtype="int64", name="edge_dst"),
        paddle.static.InputSpec([None, 3], dtype="float32", name="pbc_offset"),
        paddle.static.InputSpec([None], dtype="int64", name="node_graph_id"),
        paddle.static.InputSpec([None], dtype="int64", name="edge_graph_id"),
        paddle.static.InputSpec([None], dtype="int64", name="idx_kj"),
        paddle.static.InputSpec([None], dtype="int64", name="idx_ji"),
    ]


def compile_dimenetpp(
    model_or_core: DimeNetPlusPlus | DimeNetPPTensorCore,
    *,
    backend: Optional[str] = "CINN",
    full_graph: bool = True,
    input_spec: Optional[list[paddle.static.InputSpec]] = None,
) -> paddle.nn.Layer:
    """Compile a DimeNet++ Tensor core for the selected static backend."""

    core = (
        model_or_core
        if isinstance(model_or_core, DimeNetPPTensorCore)
        else DimeNetPPTensorCore(model_or_core)
    )
    if input_spec is None:
        input_spec = make_dimenetpp_input_spec()
    return paddle.jit.to_static(
        core,
        input_spec=input_spec,
        backend=backend,
        full_graph=full_graph,
    )


def compile_dimenetpp_cinn(
    model_or_core: DimeNetPlusPlus | DimeNetPPTensorCore,
    *,
    full_graph: bool = True,
    input_spec: Optional[list[paddle.static.InputSpec]] = None,
) -> paddle.nn.Layer:
    """Create a lazy CINN wrapper without compiling during module import."""

    return compile_dimenetpp(
        model_or_core,
        backend="CINN",
        full_graph=full_graph,
        input_spec=input_spec,
    )
