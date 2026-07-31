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

"""Tensor-only MEGNet execution path for static-graph/CINN experiments.

The regular :class:`MEGNetPlus` API intentionally remains unchanged.  PGL graph
construction and batching happen outside this module; ``graph_to_tensor_batch``
extracts the small tensor contract consumed by ``MEGNetTensorCore``.  Keeping
that boundary explicit makes it possible to compare the tensor path with the
existing eager implementation before enabling CINN.
"""

from __future__ import annotations

from typing import NamedTuple
from typing import Optional
from typing import Sequence

import numpy as np
import paddle
import pgl

from ppmat.models.megnet.megnet import MEGNetPlus

__all__ = [
    "GraphTensorBatch",
    "MEGNetTensorCore",
    "pack_pgl_graph",
    "graph_to_tensor_batch",
    "make_megnet_input_spec",
    "compile_megnet",
    "compile_megnet_cinn",
]


class GraphTensorBatch(NamedTuple):
    """Flat tensor representation of a (possibly batched) PGL graph.

    All index tensors use global indices, matching the disjoint graph produced
    by ``pgl.Graph.batch``.  ``node_sorted_*`` preserves the destination
    ordering used by the original MEGNet implementation for node aggregation.
    """

    atom_types: paddle.Tensor
    bond_dist: paddle.Tensor
    edge_src: paddle.Tensor
    edge_dst: paddle.Tensor
    node_graph_id: paddle.Tensor
    edge_graph_id: paddle.Tensor
    node_sorted_eid: paddle.Tensor
    node_sorted_dst: paddle.Tensor
    state_attr: paddle.Tensor

    @property
    def node_count(self) -> paddle.Tensor:
        """Total node count as a symbolic scalar tensor."""

        return paddle.shape(self.atom_types)[0]

    @property
    def edge_count(self) -> paddle.Tensor:
        """Total edge count as a symbolic scalar tensor."""

        return paddle.shape(self.bond_dist)[0]

    @property
    def graph_count(self) -> paddle.Tensor:
        """Graph count as a symbolic scalar tensor."""

        return paddle.shape(self.state_attr)[0]


def _to_tensor(value, dtype: str) -> paddle.Tensor:
    if isinstance(value, paddle.Tensor):
        if value.dtype != dtype:
            return paddle.cast(value, dtype)
        return value
    return paddle.to_tensor(np.asarray(value), dtype=dtype)


def _scalar_int(value) -> int:
    if isinstance(value, paddle.Tensor):
        value = value.numpy()
    return int(np.asarray(value).reshape(-1)[0])


def graph_to_tensor_batch(
    graph: pgl.Graph | Sequence[pgl.Graph],
    state_attr: Optional[paddle.Tensor] = None,
    state_dim: int = 2,
) -> GraphTensorBatch:
    """Extract the tensor inputs required by :class:`MEGNetTensorCore`.

    Args:
        graph: A PGL graph or a sequence of graphs.  A sequence is batched in
            Python before conversion.
        state_attr: Optional graph-level state with shape ``[B, state_dim]``.
            When omitted, the MEGNet default zero state is created.
        state_dim: Graph-level state feature dimension used when
            ``state_attr`` is omitted.  The default matches MEGNet's standard
            two-feature state.

    Returns:
        A ``GraphTensorBatch`` whose first dimensions may vary between calls,
        but whose feature dimensions are fixed by the model configuration.

    Raises:
        TypeError: If ``graph`` is not a PGL graph.
        ValueError: If required features are missing or the graph batch has no
            edges.
    """

    if isinstance(graph, (list, tuple)):
        if not graph:
            raise ValueError("graph must contain at least one PGL graph.")
        graph = pgl.Graph.batch(list(graph))
    if not isinstance(graph, pgl.Graph):
        raise TypeError(f"graph must be a pgl.Graph, got {type(graph)!r}.")
    if not isinstance(state_dim, int) or state_dim <= 0:
        raise ValueError(f"state_dim must be a positive integer, got {state_dim!r}.")

    # Do not mutate the caller's graph.  This is also the point where NumPy PGL
    # graphs are converted to tensors, outside the compiled model boundary.
    tensor_graph = graph.tensor(inplace=False)
    edges = tensor_graph.edges
    if len(edges.shape) != 2 or edges.shape[1] != 2:
        raise ValueError(f"graph.edges must have shape [E, 2], got {edges.shape}.")
    if edges.shape[0] == 0:
        raise ValueError("MEGNet requires at least one edge in the graph batch.")
    if "atom_types" not in tensor_graph.node_feat:
        raise ValueError("graph.node_feat['atom_types'] is required.")
    if "bond_dist" not in tensor_graph.edge_feat:
        raise ValueError("graph.edge_feat['bond_dist'] is required.")

    atom_types = _to_tensor(tensor_graph.node_feat["atom_types"], "int64")
    bond_dist = _to_tensor(tensor_graph.edge_feat["bond_dist"], "float32")
    edge_src = _to_tensor(edges[:, 0], "int64")
    edge_dst = _to_tensor(edges[:, 1], "int64")
    node_graph_id = _to_tensor(tensor_graph.graph_node_id, "int64")
    edge_graph_id = _to_tensor(tensor_graph.graph_edge_id, "int64")

    _, sorted_dst, sorted_eid = tensor_graph.sorted_edges(sort_by="dst")
    node_sorted_dst = _to_tensor(sorted_dst, "int64")
    node_sorted_eid = _to_tensor(sorted_eid, "int64")

    num_graphs = _scalar_int(tensor_graph.num_graph)
    if state_attr is None:
        state_attr = paddle.zeros([num_graphs, state_dim], dtype="float32")
    else:
        state_attr = _to_tensor(state_attr, "float32")
        if len(state_attr.shape) != 2 or state_attr.shape[0] != num_graphs:
            raise ValueError(
                f"state_attr must have shape [num_graphs, {state_dim}], "
                f"got {state_attr.shape} for {num_graphs} graphs."
            )
        if state_attr.shape[1] != state_dim:
            raise ValueError(
                f"MEGNet state_attr must have feature dimension {state_dim}, "
                f"got {state_attr.shape}."
            )

    return GraphTensorBatch(
        atom_types=atom_types,
        bond_dist=bond_dist,
        edge_src=edge_src,
        edge_dst=edge_dst,
        node_graph_id=node_graph_id,
        edge_graph_id=edge_graph_id,
        node_sorted_eid=node_sorted_eid,
        node_sorted_dst=node_sorted_dst,
        state_attr=state_attr,
    )


def pack_pgl_graph(
    graph: pgl.Graph | Sequence[pgl.Graph],
    state_attr: Optional[paddle.Tensor] = None,
    state_dim: int = 2,
) -> GraphTensorBatch:
    """Alias with an explicit packing name for the PGL-to-tensor boundary."""

    return graph_to_tensor_batch(
        graph,
        state_attr=state_attr,
        state_dim=state_dim,
    )


def _mean_by_id(
    values: paddle.Tensor,
    segment_ids: paddle.Tensor,
    out_size: paddle.Tensor,
) -> paddle.Tensor:
    """Aggregate rows by an arbitrary graph/node id.

    The implementation deliberately uses ``scatter_nd_add`` instead of PGL's
    ``send_u_recv``.  The latter has no symbolic-shape inference interface in
    the Paddle 3.3 CINN pass, while scatter-add lowers successfully for dynamic
    row counts.  Dividing by a clipped count preserves the original
    ``sorted_segment_mean`` behavior for empty segments (zero output).
    """

    feature_size = values.shape[1]
    indices = paddle.unsqueeze(segment_ids, axis=1)
    sums = paddle.scatter_nd_add(
        paddle.zeros([out_size, feature_size], dtype=values.dtype),
        indices,
        values,
    )
    counts = paddle.scatter_nd_add(
        paddle.zeros([out_size, 1], dtype=values.dtype),
        indices,
        paddle.ones_like(values[:, :1]),
    )
    return sums / paddle.clip(counts, min=1.0)


def _sum_by_id(
    values: paddle.Tensor,
    segment_ids: paddle.Tensor,
    out_size: paddle.Tensor,
) -> paddle.Tensor:
    """Aggregate rows by id with a CINN-compatible scatter sum."""

    feature_size = values.shape[1]
    indices = paddle.unsqueeze(segment_ids, axis=1)
    return paddle.scatter_nd_add(
        paddle.zeros([out_size, feature_size], dtype=values.dtype),
        indices,
        values,
    )


def _segment_softmax(
    values: paddle.Tensor,
    segment_ids: paddle.Tensor,
    out_size: paddle.Tensor,
) -> paddle.Tensor:
    """Softmax independently for each graph segment.

    ``put_along_axis(..., reduce="amax")`` preserves per-segment numerical
    stabilization while avoiding ``segment_max``, whose symbolic shape
    interface is currently missing in the Paddle 3.3 CINN pass.
    """

    feature_size = values.shape[1]
    values_max = paddle.put_along_axis(
        paddle.full(
            [out_size, feature_size],
            paddle.finfo(values.dtype).min,
            dtype=values.dtype,
        ),
        paddle.unsqueeze(segment_ids, axis=1),
        values,
        axis=0,
        reduce="amax",
    )
    values_max = paddle.gather(values_max, segment_ids, axis=0).detach()
    values_exp = paddle.exp(values - values_max)
    values_sum = _sum_by_id(values_exp, segment_ids, out_size)
    values_sum = paddle.gather(values_sum, segment_ids, axis=0)
    return values_exp / values_sum


def _tensor_set2set(
    pooling_layer: paddle.nn.Layer,
    x: paddle.Tensor,
    graph_id: paddle.Tensor,
    batch_size: paddle.Tensor,
) -> paddle.Tensor:
    """Run the existing Set2Set LSTM without a PGL graph object."""

    input_dim = pooling_layer.input_dim
    h = (
        paddle.zeros([pooling_layer.n_layers, batch_size, input_dim], dtype=x.dtype),
        paddle.zeros([pooling_layer.n_layers, batch_size, input_dim], dtype=x.dtype),
    )
    q_star = paddle.zeros([batch_size, pooling_layer.output_dim], dtype=x.dtype)
    for _ in range(pooling_layer.n_iters):
        q, h = pooling_layer.lstm(q_star.unsqueeze(0), h)
        q = paddle.reshape(q, [batch_size, input_dim])
        q_for_items = paddle.gather(q, graph_id, axis=0)
        scores = paddle.sum(x * q_for_items, axis=-1, keepdim=True)
        attention = _segment_softmax(scores, graph_id, batch_size)
        readout = _sum_by_id(attention * x, graph_id, batch_size)
        q_star = paddle.concat([q, readout], axis=-1)
    return q_star


class MEGNetTensorCore(paddle.nn.Layer):
    """MEGNet numerical core with a Tensor-only input boundary.

    The supplied ``model`` owns all trainable parameters.  This layer only
    changes graph unpacking and aggregation, so a checkpoint loaded into
    ``MEGNetPlus`` can be used without parameter-name conversion.
    """

    def __init__(self, model: MEGNetPlus):
        super().__init__()
        if not isinstance(model, MEGNetPlus):
            raise TypeError(f"model must be MEGNetPlus, got {type(model)!r}.")
        if not model.include_state_embedding:
            raise ValueError("MEGNetTensorCore currently requires include_state=True.")
        self.model = model

    def _block(
        self,
        block: paddle.nn.Layer,
        edge_feat: paddle.Tensor,
        node_feat: paddle.Tensor,
        state_feat: paddle.Tensor,
        edge_src: paddle.Tensor,
        edge_dst: paddle.Tensor,
        node_graph_id: paddle.Tensor,
        edge_graph_id: paddle.Tensor,
        node_sorted_eid: paddle.Tensor,
        node_sorted_dst: paddle.Tensor,
    ) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        inputs = edge_feat, node_feat, state_feat
        edge_feat = block.edge_func(edge_feat)
        node_feat = block.node_func(node_feat)
        state_feat = block.state_func(state_feat)

        edge_state = paddle.gather(state_feat, edge_graph_id, axis=0)
        edge_input = paddle.concat(
            [
                paddle.gather(node_feat, edge_src, axis=0),
                paddle.gather(node_feat, edge_dst, axis=0),
                edge_feat,
                edge_state,
            ],
            axis=1,
        )
        edge_feat = block.conv.edge_func(edge_input)

        sorted_edge_feat = paddle.gather(edge_feat, node_sorted_eid, axis=0)
        node_feat_e = _mean_by_id(
            sorted_edge_feat,
            node_sorted_dst,
            out_size=paddle.shape(node_feat)[0],
        )
        node_state = paddle.gather(state_feat, node_graph_id, axis=0)
        node_input = paddle.concat([node_feat, node_feat_e, node_state], axis=1)
        node_feat = block.conv.node_func(node_input)

        edge_mean = _mean_by_id(
            edge_feat,
            edge_graph_id,
            out_size=paddle.shape(state_feat)[0],
        )
        node_mean = _mean_by_id(
            node_feat,
            node_graph_id,
            out_size=paddle.shape(state_feat)[0],
        )
        state_input = paddle.concat([state_feat, edge_mean, node_mean], axis=1)
        state_feat = block.conv.state_func(state_input)

        if block.dropout is not None:
            edge_feat = block.dropout(edge_feat)
            node_feat = block.dropout(node_feat)
            state_feat = block.dropout(state_feat)
        if block.skip:
            edge_feat = edge_feat + inputs[0]
            node_feat = node_feat + inputs[1]
            state_feat = state_feat + inputs[2]
        return edge_feat, node_feat, state_feat

    def forward(
        self,
        atom_types: paddle.Tensor,
        bond_dist: paddle.Tensor,
        edge_src: paddle.Tensor,
        edge_dst: paddle.Tensor,
        node_graph_id: paddle.Tensor,
        edge_graph_id: paddle.Tensor,
        node_sorted_eid: paddle.Tensor,
        node_sorted_dst: paddle.Tensor,
        state_attr: paddle.Tensor,
    ) -> paddle.Tensor:
        model = self.model
        edge_basis = model.bond_expansion(bond_dist)
        node_feat, edge_feat, state_feat = model.embedding(
            atom_types, edge_basis, state_attr
        )
        edge_feat = model.edge_encoder(edge_feat)
        node_feat = model.node_encoder(node_feat)
        state_feat = model.state_encoder(state_feat)

        for block in model.blocks:
            edge_feat, node_feat, state_feat = self._block(
                block,
                edge_feat,
                node_feat,
                state_feat,
                edge_src,
                edge_dst,
                node_graph_id,
                edge_graph_id,
                node_sorted_eid,
                node_sorted_dst,
            )

        batch_size = paddle.shape(state_feat)[0]
        node_vec = _tensor_set2set(model.node_s2s, node_feat, node_graph_id, batch_size)
        edge_vec = _tensor_set2set(model.edge_s2s, edge_feat, edge_graph_id, batch_size)
        vec = paddle.concat([node_vec, edge_vec, state_feat], axis=1)
        if model.dropout is not None:
            vec = model.dropout(vec)
        return model.fc_out(vec)

    def forward_graph(
        self,
        graph: pgl.Graph | Sequence[pgl.Graph],
        state_attr: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """Convenience eager call for a PGL graph."""

        packed = graph_to_tensor_batch(
            graph,
            state_attr=state_attr,
            state_dim=self.model.embedding.dim_state_embedding,
        )
        return self.forward_tensor(packed)

    def forward_tensor(
        self,
        batch: GraphTensorBatch,
        state_attr: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """Run a packed graph and return normalized predictions of shape ``[B, 1]``."""

        if not isinstance(batch, GraphTensorBatch):
            raise TypeError(
                "batch must be a GraphTensorBatch produced by pack_pgl_graph."
            )
        expected_state_dim = self.model.embedding.dim_state_embedding
        if batch.state_attr.shape[1] != expected_state_dim:
            raise ValueError(
                "batch.state_attr has feature dimension "
                f"{batch.state_attr.shape[1]}, expected {expected_state_dim}."
            )
        if state_attr is not None:
            state_attr = _to_tensor(state_attr, "float32")
            if len(state_attr.shape) != 2:
                raise ValueError(
                    f"state_attr must have shape [B, D], got {state_attr.shape}."
                )
            if state_attr.shape[0] != batch.state_attr.shape[0]:
                raise ValueError(
                    "state_attr batch dimension "
                    f"{state_attr.shape[0]} does not match "
                    f"{batch.state_attr.shape[0]}."
                )
            if state_attr.shape[1] != expected_state_dim:
                raise ValueError(
                    "state_attr has feature dimension "
                    f"{state_attr.shape[1]}, expected {expected_state_dim}."
                )
            batch = batch._replace(state_attr=state_attr)
        return self(*batch)

    @paddle.no_grad()
    def predict_tensor(
        self,
        batch: GraphTensorBatch,
        state_attr: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """Return unnormalized predictions as a tensor of shape ``[B, 1]``."""

        return self.model.unnormalize(
            self.forward_tensor(batch, state_attr=state_attr)
        )


def make_megnet_input_spec(
    state_input_dim: int = 2,
) -> list[paddle.static.InputSpec]:
    """Return the dynamic-shape input contract used by ``to_static``."""

    return [
        paddle.static.InputSpec([None], dtype="int64", name="atom_types"),
        paddle.static.InputSpec([None], dtype="float32", name="bond_dist"),
        paddle.static.InputSpec([None], dtype="int64", name="edge_src"),
        paddle.static.InputSpec([None], dtype="int64", name="edge_dst"),
        paddle.static.InputSpec([None], dtype="int64", name="node_graph_id"),
        paddle.static.InputSpec([None], dtype="int64", name="edge_graph_id"),
        paddle.static.InputSpec([None], dtype="int64", name="node_sorted_eid"),
        paddle.static.InputSpec([None], dtype="int64", name="node_sorted_dst"),
        paddle.static.InputSpec(
            [None, state_input_dim],
            dtype="float32",
            name="state_attr",
        ),
    ]


def compile_megnet(
    model_or_core: MEGNetPlus | MEGNetTensorCore,
    *,
    backend: Optional[str] = "CINN",
    full_graph: bool = True,
    input_spec: Optional[list[paddle.static.InputSpec]] = None,
) -> paddle.nn.Layer:
    """Compile a Tensor core for a static backend.

    ``backend=None`` is useful as a dy2static/PIR diagnostic before enabling
    CINN.  The caller owns export and checkpoint paths.
    """

    core = (
        model_or_core
        if isinstance(model_or_core, MEGNetTensorCore)
        else MEGNetTensorCore(model_or_core)
    )
    if input_spec is None:
        input_spec = make_megnet_input_spec(
            core.model.embedding.dim_state_embedding
        )
    return paddle.jit.to_static(
        core,
        input_spec=input_spec,
        backend=backend,
        full_graph=full_graph,
    )


def compile_megnet_cinn(
    model_or_core: MEGNetPlus | MEGNetTensorCore,
    *,
    full_graph: bool = True,
    input_spec: Optional[list[paddle.static.InputSpec]] = None,
) -> paddle.nn.Layer:
    """Create a lazy CINN wrapper without compiling during module import."""

    return compile_megnet(
        model_or_core,
        backend="CINN",
        full_graph=full_graph,
        input_spec=input_spec,
    )
