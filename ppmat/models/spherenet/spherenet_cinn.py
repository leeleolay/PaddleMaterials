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

"""Tensor-only SphereNet execution path for Paddle CINN.

PGL graph batching and the discrete torsion-neighbor selection stay outside the
compiled program. The Tensor core retains differentiable distance, angle, and
torsion calculations and reuses the public ``SphereNet`` parameter layers.
"""

from __future__ import annotations

import math
from typing import NamedTuple
from typing import Optional
from typing import Sequence

import numpy as np
import paddle
import pgl

from ppmat.models.spherenet.spherenet import SphereNet
from ppmat.utils.scatter import scatter_argmin

__all__ = [
    "SphereNetTensorBatch",
    "SphereNetTensorCore",
    "compile_spherenet",
    "compile_spherenet_cinn",
    "graph_to_tensor_batch",
    "make_spherenet_input_spec",
    "pack_pgl_graph",
]


class SphereNetTensorBatch(NamedTuple):
    """Flat dynamic-shape input contract for SphereNet's Tensor core."""

    atomic_number: paddle.Tensor
    pos: paddle.Tensor
    edge_src: paddle.Tensor
    edge_dst: paddle.Tensor
    node_graph_id: paddle.Tensor
    idx_kj: paddle.Tensor
    idx_ji: paddle.Tensor
    idx_qj: paddle.Tensor
    triplet_mask: paddle.Tensor
    node_feature: paddle.Tensor
    graph_template: paddle.Tensor

    @property
    def node_count(self) -> paddle.Tensor:
        return paddle.shape(self.atomic_number)[0]

    @property
    def edge_count(self) -> paddle.Tensor:
        return paddle.shape(self.edge_src)[0]

    @property
    def triplet_count(self) -> paddle.Tensor:
        return paddle.shape(self.idx_kj)[0]

    @property
    def graph_count(self) -> paddle.Tensor:
        return paddle.shape(self.graph_template)[0]


def _to_tensor(value, dtype: str) -> paddle.Tensor:
    if isinstance(value, paddle.Tensor):
        return value if value.dtype == dtype else paddle.cast(value, dtype)
    return paddle.to_tensor(np.asarray(value), dtype=dtype)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, paddle.Tensor):
        value = value.numpy()
    return np.asarray(value)


def _scalar_int(value) -> int:
    return int(_to_numpy(value).reshape(-1)[0])


def _batch_graphs(graphs: Sequence[pgl.Graph]) -> pgl.Graph:
    graphs = list(graphs)
    if not graphs:
        raise ValueError("graph must contain at least one PGL graph.")
    if not all(isinstance(graph, pgl.Graph) for graph in graphs):
        raise TypeError("all graph sequence elements must be pgl.Graph instances.")

    edge_counts = [_to_numpy(graph.edges).shape[0] for graph in graphs]
    edge_offsets = np.cumsum([0, *edge_counts[:-1]], dtype=np.int64)
    triplet_fields = {}
    for key in ("ti_idx_kj", "ti_idx_ji"):
        values = []
        for graph, edge_offset in zip(graphs, edge_offsets):
            if key not in graph.edge_feat:
                raise ValueError(f"graph.edge_feat[{key!r}] is required.")
            values.append(
                _to_numpy(graph.edge_feat[key]).astype(np.int64) + edge_offset
            )
        triplet_fields[key] = np.concatenate(values)

    graph = pgl.Graph.batch(graphs)
    graph.edge_feat.update(triplet_fields)
    return graph


def _cross_product(left: paddle.Tensor, right: paddle.Tensor) -> paddle.Tensor:
    return paddle.stack(
        (
            left[:, 1] * right[:, 2] - left[:, 2] * right[:, 1],
            left[:, 2] * right[:, 0] - left[:, 0] * right[:, 2],
            left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0],
        ),
        axis=-1,
    )


def _select_torsion_edges(
    pos: paddle.Tensor,
    edge_src: paddle.Tensor,
    edge_dst: paddle.Tensor,
    idx_kj: paddle.Tensor,
    idx_ji: paddle.Tensor,
) -> paddle.Tensor:
    """Select the minimum-torsion q->j edge for each cached triplet.

    This selection is discrete and intentionally runs before dy2static/CINN.
    The selected edge ids are then a regular dynamic Tensor input to the core.
    """

    if idx_kj.shape[0] == 0:
        return paddle.empty([0], dtype="int64")

    edge_vector = pos[edge_dst] - pos[edge_src]
    axis = edge_vector[idx_ji].detach()
    reference = -edge_vector[idx_kj].detach()
    incoming_order = paddle.argsort(edge_dst, stable=True)
    incoming_counts = paddle.bincount(edge_dst, minlength=pos.shape[0])
    centers = edge_src[idx_ji]
    i_atoms = edge_dst[idx_ji]
    candidate_counts = incoming_counts[centers]
    candidate_triplet = paddle.repeat_interleave(
        paddle.arange(idx_kj.shape[0], dtype="int64"), candidate_counts
    )
    candidate_starts = paddle.cumsum(candidate_counts) - candidate_counts
    candidate_offsets = paddle.arange(
        candidate_triplet.shape[0], dtype="int64"
    ) - paddle.repeat_interleave(candidate_starts, candidate_counts)
    incoming_starts = paddle.cumsum(incoming_counts) - incoming_counts
    candidate_positions = (
        paddle.repeat_interleave(incoming_starts[centers], candidate_counts)
        + candidate_offsets
    )
    idx_qj = incoming_order[candidate_positions]
    valid_candidates = edge_src[idx_qj] != i_atoms[candidate_triplet]
    idx_qj = idx_qj[valid_candidates]
    candidate_triplet = candidate_triplet[valid_candidates]

    candidate_axis = axis[candidate_triplet]
    candidate_reference = reference[candidate_triplet]
    candidate = -edge_vector.detach()[idx_qj]
    reference_plane = paddle.linalg.cross(candidate_axis, candidate_reference)
    candidate_plane = paddle.linalg.cross(candidate_axis, candidate)
    torsion_x = paddle.sum(reference_plane * candidate_plane, axis=-1)
    torsion_y = paddle.sum(
        paddle.linalg.cross(reference_plane, candidate_plane) * candidate_axis,
        axis=-1,
    ) / paddle.sqrt(paddle.sum(candidate_axis * candidate_axis, axis=-1))
    candidate_torsion = paddle.atan2(torsion_y, torsion_x)
    candidate_torsion = paddle.where(
        candidate_torsion <= 0,
        candidate_torsion + 2 * math.pi,
        candidate_torsion,
    )
    selected = scatter_argmin(candidate_torsion, candidate_triplet, idx_kj.shape[0])
    return idx_qj[selected]


def graph_to_tensor_batch(
    graph: pgl.Graph | Sequence[pgl.Graph],
    *,
    use_extra_node_feature: bool = False,
    extra_node_feature_dim: int = 1,
) -> SphereNetTensorBatch:
    """Pack a PGL graph batch into SphereNet's compiled Tensor contract."""

    if isinstance(graph, (list, tuple)):
        graph = _batch_graphs(graph)
    if not isinstance(graph, pgl.Graph):
        raise TypeError(f"graph must be a pgl.Graph, got {type(graph)!r}.")
    if not isinstance(extra_node_feature_dim, int) or extra_node_feature_dim <= 0:
        raise ValueError(
            "extra_node_feature_dim must be a positive integer, "
            f"got {extra_node_feature_dim!r}."
        )

    tensor_graph = graph.tensor(inplace=False)
    edges = tensor_graph.edges
    if len(edges.shape) != 2 or edges.shape[1] != 2:
        raise ValueError(f"graph.edges must have shape [E, 2], got {edges.shape}.")
    if edges.shape[0] == 0:
        raise ValueError("SphereNet requires at least one edge in the graph batch.")
    for key in ("atomic_number", "pos"):
        if key not in tensor_graph.node_feat:
            raise ValueError(f"graph.node_feat[{key!r}] is required.")
    for key in ("ti_idx_kj", "ti_idx_ji"):
        if key not in tensor_graph.edge_feat:
            raise ValueError(f"graph.edge_feat[{key!r}] is required.")

    atomic_number = _to_tensor(
        tensor_graph.node_feat["atomic_number"], "int64"
    ).reshape([-1])
    pos = _to_tensor(tensor_graph.node_feat["pos"], "float32")
    if len(pos.shape) != 2 or pos.shape[1] != 3:
        raise ValueError(
            f"graph.node_feat['pos'] must have shape [N, 3], got {pos.shape}."
        )
    if atomic_number.shape[0] != pos.shape[0]:
        raise ValueError("atomic_number and pos must contain the same number of nodes.")

    edge_src = _to_tensor(edges[:, 0], "int64")
    edge_dst = _to_tensor(edges[:, 1], "int64")
    node_graph_id = _to_tensor(tensor_graph.graph_node_id, "int64")
    idx_kj = _to_tensor(tensor_graph.edge_feat["ti_idx_kj"], "int64").reshape([-1])
    idx_ji = _to_tensor(tensor_graph.edge_feat["ti_idx_ji"], "int64").reshape([-1])
    if idx_kj.shape[0] != idx_ji.shape[0]:
        raise ValueError("ti_idx_kj and ti_idx_ji must have equal length.")

    if idx_kj.shape[0] == 0:
        # Paddle 3.3 scatter_nd_add cannot consume zero-row updates. A masked
        # sentinel keeps the compiled shape path valid and contributes exactly
        # the zero message used by SphereNet's eager empty-triplet branch.
        idx_kj = paddle.zeros([1], dtype="int64")
        idx_ji = paddle.zeros([1], dtype="int64")
        idx_qj = paddle.zeros([1], dtype="int64")
        triplet_mask = paddle.zeros([1, 1], dtype="float32")
    else:
        with paddle.no_grad():
            idx_qj = _select_torsion_edges(pos, edge_src, edge_dst, idx_kj, idx_ji)
        triplet_mask = paddle.ones([idx_kj.shape[0], 1], dtype="float32")

    if use_extra_node_feature:
        if "node_feature" not in tensor_graph.node_feat:
            raise ValueError(
                "SphereNet use_extra_node_feature=True requires "
                "graph.node_feat['node_feature']."
            )
        node_feature = _to_tensor(tensor_graph.node_feat["node_feature"], "float32")
        expected_shape = [pos.shape[0], extra_node_feature_dim]
        if list(node_feature.shape) != expected_shape:
            raise ValueError(
                "graph.node_feat['node_feature'] must have shape "
                f"{expected_shape}, got {list(node_feature.shape)}."
            )
    else:
        node_feature = paddle.zeros(
            [pos.shape[0], extra_node_feature_dim], dtype="float32"
        )

    graph_template = paddle.zeros(
        [_scalar_int(tensor_graph.num_graph), 1], dtype="float32"
    )
    return SphereNetTensorBatch(
        atomic_number=atomic_number,
        pos=pos,
        edge_src=edge_src,
        edge_dst=edge_dst,
        node_graph_id=node_graph_id,
        idx_kj=idx_kj,
        idx_ji=idx_ji,
        idx_qj=idx_qj,
        triplet_mask=triplet_mask,
        node_feature=node_feature,
        graph_template=graph_template,
    )


def pack_pgl_graph(
    graph: pgl.Graph | Sequence[pgl.Graph],
    *,
    use_extra_node_feature: bool = False,
    extra_node_feature_dim: int = 1,
) -> SphereNetTensorBatch:
    """Alias with an explicit packing name for the PGL-to-Tensor boundary."""

    return graph_to_tensor_batch(
        graph,
        use_extra_node_feature=use_extra_node_feature,
        extra_node_feature_dim=extra_node_feature_dim,
    )


def _sum_by_id(
    values: paddle.Tensor,
    segment_ids: paddle.Tensor,
    out_size: paddle.Tensor,
) -> paddle.Tensor:
    indices = paddle.unsqueeze(segment_ids, axis=1)
    return paddle.scatter_nd_add(
        paddle.zeros([out_size, values.shape[1]], dtype=values.dtype),
        indices,
        values,
    )


def _compute_tensor_geometry(
    pos: paddle.Tensor,
    edge_src: paddle.Tensor,
    edge_dst: paddle.Tensor,
    idx_kj: paddle.Tensor,
    idx_ji: paddle.Tensor,
    idx_qj: paddle.Tensor,
):
    edge_vector = pos[edge_dst] - pos[edge_src]
    dist = paddle.sqrt(paddle.sum(edge_vector * edge_vector, axis=-1))

    axis = edge_vector[idx_ji]
    reference = -edge_vector[idx_kj]
    angle_cross = _cross_product(axis, reference)
    angle_sin = paddle.sqrt(paddle.sum(angle_cross * angle_cross, axis=-1) + 1e-8)
    angle_cos = paddle.sum(axis * reference, axis=-1)
    angle_norm = paddle.sqrt(angle_cos * angle_cos + angle_sin * angle_sin)
    angle_cos = angle_cos / angle_norm
    angle_sin = angle_sin / angle_norm

    candidate = -edge_vector[idx_qj]
    reference_plane = _cross_product(axis, reference)
    candidate_plane = _cross_product(axis, candidate)
    torsion_x = paddle.sum(reference_plane * candidate_plane, axis=-1)
    torsion_y = paddle.sum(
        _cross_product(reference_plane, candidate_plane) * axis, axis=-1
    ) / paddle.sqrt(paddle.sum(axis * axis, axis=-1))
    torsion_squared_norm = torsion_x * torsion_x + torsion_y * torsion_y
    valid_torsion = torsion_squared_norm > 1e-12
    torsion_norm = paddle.sqrt(
        paddle.where(
            valid_torsion,
            torsion_squared_norm,
            paddle.ones_like(torsion_squared_norm),
        )
    )
    torsion_cos = paddle.where(
        valid_torsion, torsion_x / torsion_norm, paddle.ones_like(torsion_x)
    )
    torsion_sin = paddle.where(
        valid_torsion, torsion_y / torsion_norm, paddle.zeros_like(torsion_y)
    )
    return dist, (angle_cos, angle_sin), (torsion_cos, torsion_sin)


def _edge_update_tensor(
    layer: paddle.nn.Layer,
    edge_state,
    embedding,
    idx_kj: paddle.Tensor,
    idx_ji: paddle.Tensor,
    triplet_mask: paddle.Tensor,
):
    """Run EdgeUpdate without its dynamic empty-triplet Python branch."""

    rbf0, sbf, torsion = embedding
    edge_hidden, _ = edge_state
    x_ji = layer.act(layer.lin_ji(edge_hidden))
    x_kj = layer.act(layer.lin_kj(edge_hidden))
    rbf = layer.lin_rbf2(layer.lin_rbf1(rbf0))
    x_kj = layer.act(layer.lin_down(x_kj * rbf))
    sbf = layer.lin_sbf2(layer.lin_sbf1(sbf))
    x_kj = x_kj[idx_kj] * sbf
    torsion = layer.lin_t2(layer.lin_t1(torsion))
    x_kj = x_kj * torsion * triplet_mask
    x_kj = _sum_by_id(x_kj, idx_ji, paddle.shape(edge_hidden)[0])
    x_kj = layer.act(layer.lin_up(x_kj))

    updated = x_ji + x_kj
    for residual in layer.layers_before_skip:
        updated = residual(updated)
    updated = layer.act(layer.lin(updated)) + edge_hidden
    for residual in layer.layers_after_skip:
        updated = residual(updated)
    filtered = layer.lin_rbf(rbf0) * updated
    return updated, filtered


class SphereNetTensorCore(paddle.nn.Layer):
    """SphereNet numerical core with a Tensor-only input boundary."""

    def __init__(self, model: SphereNet):
        super().__init__()
        if not isinstance(model, SphereNet):
            raise TypeError(f"model must be SphereNet, got {type(model)!r}.")
        if model.energy_and_force:
            raise ValueError(
                "SphereNet CINN currently supports property prediction only; "
                "energy_and_force=True requires second-order eager autograd."
            )
        self.model = model

    def forward(
        self,
        atomic_number: paddle.Tensor,
        pos: paddle.Tensor,
        edge_src: paddle.Tensor,
        edge_dst: paddle.Tensor,
        node_graph_id: paddle.Tensor,
        idx_kj: paddle.Tensor,
        idx_ji: paddle.Tensor,
        idx_qj: paddle.Tensor,
        triplet_mask: paddle.Tensor,
        node_feature: paddle.Tensor,
        graph_template: paddle.Tensor,
    ) -> paddle.Tensor:
        model = self.model
        dist, angle, torsion = _compute_tensor_geometry(
            pos, edge_src, edge_dst, idx_kj, idx_ji, idx_qj
        )
        embedding = model.emb_layer(dist, angle, torsion, idx_kj)
        extra_node_feature = (
            model.extra_emb(node_feature) if model.use_extra_node_feature else None
        )
        edge_state = model.init_e(
            atomic_number,
            extra_node_feature,
            embedding,
            edge_dst,
            edge_src,
        )
        num_nodes = paddle.shape(atomic_number)[0]
        num_graphs = paddle.shape(graph_template)[0]
        node_output = model.init_v(edge_state, edge_dst, dim_size=num_nodes)
        graph_output = _sum_by_id(node_output, node_graph_id, num_graphs)

        for update_e, update_v in zip(model.update_es, model.update_vs):
            edge_state = _edge_update_tensor(
                update_e,
                edge_state,
                embedding,
                idx_kj,
                idx_ji,
                triplet_mask,
            )
            node_output = update_v(edge_state, edge_dst, dim_size=num_nodes)
            graph_output = graph_output + _sum_by_id(
                node_output, node_graph_id, num_graphs
            )
        return graph_output

    def forward_graph(self, graph: pgl.Graph | Sequence[pgl.Graph]) -> paddle.Tensor:
        packed = graph_to_tensor_batch(
            graph,
            use_extra_node_feature=self.model.use_extra_node_feature,
            extra_node_feature_dim=self.model.extra_node_feature_dim,
        )
        return self.forward_tensor(packed)

    def forward_tensor(self, batch: SphereNetTensorBatch) -> paddle.Tensor:
        if not isinstance(batch, SphereNetTensorBatch):
            raise TypeError(
                "batch must be a SphereNetTensorBatch produced by pack_pgl_graph."
            )
        if batch.node_feature.shape[1] != self.model.extra_node_feature_dim:
            raise ValueError(
                "batch.node_feature has feature dimension "
                f"{batch.node_feature.shape[1]}, expected "
                f"{self.model.extra_node_feature_dim}."
            )
        return self(*batch)

    @paddle.no_grad()
    def predict_tensor(self, batch: SphereNetTensorBatch) -> paddle.Tensor:
        return self.model.unnormalize(self.forward_tensor(batch))


def make_spherenet_input_spec(
    extra_node_feature_dim: int = 1,
) -> list[paddle.static.InputSpec]:
    """Return SphereNet's dynamic ``N/E/T/B`` static input contract."""

    return [
        paddle.static.InputSpec([None], "int64", name="atomic_number"),
        paddle.static.InputSpec([None, 3], "float32", name="pos"),
        paddle.static.InputSpec([None], "int64", name="edge_src"),
        paddle.static.InputSpec([None], "int64", name="edge_dst"),
        paddle.static.InputSpec([None], "int64", name="node_graph_id"),
        paddle.static.InputSpec([None], "int64", name="idx_kj"),
        paddle.static.InputSpec([None], "int64", name="idx_ji"),
        paddle.static.InputSpec([None], "int64", name="idx_qj"),
        paddle.static.InputSpec([None, 1], "float32", name="triplet_mask"),
        paddle.static.InputSpec(
            [None, extra_node_feature_dim], "float32", name="node_feature"
        ),
        paddle.static.InputSpec([None, 1], "float32", name="graph_template"),
    ]


def compile_spherenet(
    model_or_core: SphereNet | SphereNetTensorCore,
    *,
    backend: Optional[str] = "CINN",
    full_graph: bool = True,
    input_spec: Optional[list[paddle.static.InputSpec]] = None,
) -> paddle.nn.Layer:
    core = (
        model_or_core
        if isinstance(model_or_core, SphereNetTensorCore)
        else SphereNetTensorCore(model_or_core)
    )
    if input_spec is None:
        input_spec = make_spherenet_input_spec(core.model.extra_node_feature_dim)
    return paddle.jit.to_static(
        core,
        input_spec=input_spec,
        backend=backend,
        full_graph=full_graph,
    )


def compile_spherenet_cinn(
    model_or_core: SphereNet | SphereNetTensorCore,
    *,
    full_graph: bool = True,
    input_spec: Optional[list[paddle.static.InputSpec]] = None,
) -> paddle.nn.Layer:
    return compile_spherenet(
        model_or_core,
        backend="CINN",
        full_graph=full_graph,
        input_spec=input_spec,
    )
