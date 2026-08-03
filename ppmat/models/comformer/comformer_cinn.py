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

"""Tensor-only iComformer execution path for Paddle CINN.

PGL graph construction and batching stay outside the compiled boundary. The
public model remains the parameter and checkpoint owner; this module only
adapts its numerical layers to a flat Tensor contract.
"""

from __future__ import annotations

import math
from typing import NamedTuple
from typing import Optional
from typing import Sequence

import numpy as np
import paddle
import pgl

from ppmat.models.comformer.comformer import iComformer

__all__ = [
    "ComformerGraphTensorBatch",
    "ComformerTensorCore",
    "compile_comformer",
    "compile_comformer_cinn",
    "graph_to_tensor_batch",
    "make_comformer_input_spec",
]


class ComformerGraphTensorBatch(NamedTuple):
    """Flat Tensor representation of one or more disjoint Comformer graphs."""

    node_feat: paddle.Tensor
    edge_r: paddle.Tensor
    edge_nei: paddle.Tensor
    edge_src: paddle.Tensor
    edge_dst: paddle.Tensor
    node_graph_id: paddle.Tensor
    graph_node_count: paddle.Tensor

    @property
    def node_count(self) -> paddle.Tensor:
        return paddle.shape(self.node_feat)[0]

    @property
    def edge_count(self) -> paddle.Tensor:
        return paddle.shape(self.edge_r)[0]

    @property
    def graph_count(self) -> paddle.Tensor:
        return paddle.shape(self.graph_node_count)[0]


def _to_tensor(value, dtype: str) -> paddle.Tensor:
    if isinstance(value, paddle.Tensor):
        if value.dtype != getattr(paddle, dtype):
            return paddle.cast(value, dtype)
        return value
    return paddle.to_tensor(np.asarray(value), dtype=dtype)


def graph_to_tensor_batch(
    graph: pgl.Graph | Sequence[pgl.Graph],
) -> ComformerGraphTensorBatch:
    """Extract iComformer's dynamic Tensor inputs from a PGL graph batch."""

    if isinstance(graph, (list, tuple)):
        if not graph:
            raise ValueError("graph must contain at least one PGL graph.")
        graph = pgl.Graph.batch(list(graph))
    if not isinstance(graph, pgl.Graph):
        raise TypeError(f"graph must be a pgl.Graph, got {type(graph)!r}.")

    edges = _to_tensor(graph.edges, "int64")
    if len(edges.shape) != 2 or edges.shape[1] != 2:
        raise ValueError(f"graph.edges must have shape [E, 2], got {edges.shape}.")
    if edges.shape[0] == 0:
        raise ValueError("iComformer requires at least one edge in the graph batch.")
    if "node_feat" not in graph.node_feat:
        raise ValueError("graph.node_feat['node_feat'] is required.")
    if "r" not in graph.edge_feat:
        raise ValueError("graph.edge_feat['r'] is required.")
    if "nei" not in graph.edge_feat:
        raise ValueError("graph.edge_feat['nei'] is required.")

    node_feat = _to_tensor(graph.node_feat["node_feat"], "float32")
    edge_r = _to_tensor(graph.edge_feat["r"], "float32")
    edge_nei = _to_tensor(graph.edge_feat["nei"], "float32")
    edge_src = _to_tensor(edges[:, 0], "int64")
    edge_dst = _to_tensor(edges[:, 1], "int64")
    node_graph_id = _to_tensor(graph.graph_node_id, "int64")

    if len(node_feat.shape) != 2:
        raise ValueError(
            "graph.node_feat['node_feat'] must have shape [N, F], "
            f"got {node_feat.shape}."
        )
    if len(edge_r.shape) != 2 or edge_r.shape[1] != 3:
        raise ValueError(
            f"graph.edge_feat['r'] must have shape [E, 3], got {edge_r.shape}."
        )
    if len(edge_nei.shape) != 3 or list(edge_nei.shape[1:]) != [3, 3]:
        raise ValueError(
            "graph.edge_feat['nei'] must have shape [E, 3, 3], "
            f"got {edge_nei.shape}."
        )
    edge_count = edges.shape[0]
    if edge_r.shape[0] != edge_count or edge_nei.shape[0] != edge_count:
        raise ValueError("Comformer edge feature rows must match graph.edges.")
    if node_graph_id.shape[0] != node_feat.shape[0]:
        raise ValueError("graph_node_id rows must match graph node features.")

    graph_node_count = paddle.bincount(node_graph_id)
    return ComformerGraphTensorBatch(
        node_feat=node_feat,
        edge_r=edge_r,
        edge_nei=edge_nei,
        edge_src=edge_src,
        edge_dst=edge_dst,
        node_graph_id=node_graph_id,
        graph_node_count=graph_node_count,
    )


def _sum_by_id(
    values: paddle.Tensor,
    segment_ids: paddle.Tensor,
    out_size: paddle.Tensor,
) -> paddle.Tensor:
    feature_size = values.shape[1]
    return paddle.scatter_nd_add(
        paddle.zeros([out_size, feature_size], dtype=values.dtype),
        paddle.unsqueeze(segment_ids, axis=1),
        values,
    )


def _mean_by_id(
    values: paddle.Tensor,
    segment_ids: paddle.Tensor,
    out_size: paddle.Tensor,
) -> paddle.Tensor:
    sums = _sum_by_id(values, segment_ids, out_size)
    counts = paddle.scatter_nd_add(
        paddle.zeros([out_size, 1], dtype=values.dtype),
        paddle.unsqueeze(segment_ids, axis=1),
        paddle.ones_like(values[:, :1]),
    )
    return sums / paddle.clip(counts, min=1.0)


def _batch_norm_1d(
    layer: paddle.nn.BatchNorm1D,
    values: paddle.Tensor,
) -> paddle.Tensor:
    """Match Paddle GPU eager BatchNorm statistics inside a static graph.

    Paddle 3.3.1's GPU eager kernel updates the running variance with the
    unbiased batch estimate, while its PIR/static BatchNorm op uses the biased
    estimate. The latter silently changes checkpoint evaluation after CINN
    training. Keeping the update explicit preserves eager training semantics;
    normalization itself uses the biased variance in both kernels.
    """

    if layer.training:
        mean = paddle.mean(values, axis=0)
        centered = values - mean
        variance = paddle.mean(centered * centered, axis=0)
        sample_count = paddle.cast(paddle.shape(values)[0], values.dtype)
        has_multiple_samples = sample_count > 1.0
        correction = sample_count / paddle.clip(sample_count - 1.0, min=1.0)
        unbiased_variance = variance * correction
        momentum = layer._momentum
        updated_mean = layer._mean * momentum + mean.detach() * (1.0 - momentum)
        updated_variance = layer._variance * momentum + unbiased_variance.detach() * (
            1.0 - momentum
        )
        paddle.assign(
            paddle.where(has_multiple_samples, updated_mean, layer._mean),
            output=layer._mean,
        )
        paddle.assign(
            paddle.where(has_multiple_samples, updated_variance, layer._variance),
            output=layer._variance,
        )
    else:
        mean = layer._mean
        variance = layer._variance

    output = (values - mean) * paddle.rsqrt(variance + layer._epsilon)
    if layer.weight is not None:
        output = output * layer.weight
    if layer.bias is not None:
        output = output + layer.bias
    if layer.training:
        output = paddle.where(has_multiple_samples, output, values)
    return output


def _linear_before_batch_norm(
    linear: paddle.nn.Linear,
    values: paddle.Tensor,
    batch_norm: paddle.nn.BatchNorm1D,
) -> paddle.Tensor:
    """Avoid optimizer noise for a bias canceled exactly by training BatchNorm."""

    if batch_norm.training and linear.bias is not None:
        output = paddle.nn.functional.linear(values, linear.weight, bias=None)
        detached_bias = output + linear.bias.detach()
        trainable_bias = output + linear.bias
        return paddle.where(
            paddle.shape(values)[0] > 1,
            detached_bias,
            trainable_bias,
        )
    return linear(values)


def _node_attention(
    layer: paddle.nn.Layer,
    node_feat: paddle.Tensor,
    edge_src: paddle.Tensor,
    edge_dst: paddle.Tensor,
    edge_attr: paddle.Tensor,
) -> paddle.Tensor:
    """Tensor equivalent of ``ComformerConv.propagate``."""

    heads, channels = layer.heads, layer.out_channels
    query = layer.lin_query(node_feat).reshape([-1, heads, channels])
    key = layer.lin_key(node_feat).reshape([-1, heads, channels])
    value = layer.lin_value(node_feat).reshape([-1, heads, channels])

    query_i = paddle.gather(query, edge_dst, axis=0)
    key_i = paddle.gather(key, edge_dst, axis=0)
    key_j = paddle.gather(key, edge_src, axis=0)
    value_i = paddle.gather(value, edge_dst, axis=0)
    value_j = paddle.gather(value, edge_src, axis=0)
    edge_attr = layer.lin_edge(edge_attr).reshape([-1, heads, channels])

    key_j = layer.key_update(paddle.concat((key_i, key_j, edge_attr), axis=-1))
    alpha = query_i * key_j / math.sqrt(channels)
    messages = layer.lin_msg_update(
        paddle.concat((value_i, value_j, edge_attr), axis=-1)
    )
    messages = messages * layer.sigmoid(
        _batch_norm_1d(layer.bn_att, alpha.reshape([-1, channels])).reshape(
            [-1, heads, channels]
        )
    )
    messages = messages.reshape([-1, heads * channels])
    aggregated = _sum_by_id(
        messages,
        edge_dst,
        out_size=paddle.shape(node_feat)[0],
    )
    aggregated = _linear_before_batch_norm(layer.lin_concate, aggregated, layer.bn)
    return layer.softplus(node_feat + _batch_norm_1d(layer.bn, aggregated))


def _edge_attention(
    layer: paddle.nn.Layer,
    edge_feat: paddle.Tensor,
    edge_nei_len: paddle.Tensor,
    edge_nei_angle: paddle.Tensor,
) -> paddle.Tensor:
    """Tensor equivalent of ``ComformerConv_edge.forward``."""

    heads, channels = layer.heads, layer.out_channels
    query_x = (
        layer.lin_query(edge_feat)
        .reshape([-1, heads, channels])
        .unsqueeze(axis=1)
        .tile(repeat_times=[1, 3, 1, 1])
    )
    key_x = (
        layer.lin_key(edge_feat)
        .reshape([-1, heads, channels])
        .unsqueeze(axis=1)
        .tile(repeat_times=[1, 3, 1, 1])
    )
    value_x = (
        layer.lin_value(edge_feat)
        .reshape([-1, heads, channels])
        .unsqueeze(axis=1)
        .tile(repeat_times=[1, 3, 1, 1])
    )
    key_y = paddle.concat(
        (
            layer.lin_key_e1(edge_nei_len[:, 0, :]).reshape([-1, 1, heads, channels]),
            layer.lin_key_e2(edge_nei_len[:, 1, :]).reshape([-1, 1, heads, channels]),
            layer.lin_key_e3(edge_nei_len[:, 2, :]).reshape([-1, 1, heads, channels]),
        ),
        axis=1,
    )
    value_y = paddle.concat(
        (
            layer.lin_value_e1(edge_nei_len[:, 0, :]).reshape([-1, 1, heads, channels]),
            layer.lin_value_e2(edge_nei_len[:, 1, :]).reshape([-1, 1, heads, channels]),
            layer.lin_value_e3(edge_nei_len[:, 2, :]).reshape([-1, 1, heads, channels]),
        ),
        axis=1,
    )
    edge_xy = layer.lin_edge(edge_nei_angle).reshape([-1, 3, heads, channels])
    key = layer.key_update(paddle.concat((key_x, key_y, edge_xy), axis=-1))
    alpha = query_x * key / math.sqrt(channels)
    out = layer.lin_msg_update(paddle.concat((value_x, value_y, edge_xy), axis=-1))
    out = out * layer.sigmoid(
        _batch_norm_1d(layer.bn_att, alpha.reshape([-1, channels])).reshape(
            [-1, 3, heads, channels]
        )
    )
    out = out.reshape([-1, 3, heads * channels])
    out = _linear_before_batch_norm(layer.lin_concate, out, layer.bn).sum(axis=1)
    return layer.softplus(edge_feat + _batch_norm_1d(layer.bn, out))


class ComformerTensorCore(paddle.nn.Layer):
    """iComformer numerical core with a Tensor-only dynamic-shape boundary."""

    def __init__(self, model: iComformer):
        super().__init__()
        if not isinstance(model, iComformer):
            raise TypeError(f"model must be iComformer, got {type(model)!r}.")
        if len(model.att_layers) == 0:
            raise ValueError("ComformerTensorCore requires at least one conv layer.")
        self.model = model

    def forward(
        self,
        node_feat: paddle.Tensor,
        edge_r: paddle.Tensor,
        edge_nei: paddle.Tensor,
        edge_src: paddle.Tensor,
        edge_dst: paddle.Tensor,
        node_graph_id: paddle.Tensor,
        graph_node_count: paddle.Tensor,
    ) -> paddle.Tensor:
        model = self.model
        node_features = model.atom_embedding(node_feat)

        edge_norm = paddle.sqrt(paddle.sum(edge_r * edge_r, axis=1))
        edge_nei_norm = paddle.sqrt(paddle.sum(edge_nei * edge_nei, axis=-1))
        edge_feat = -0.75 / edge_norm
        edge_nei_len = -0.75 / edge_nei_norm
        edge_r_tiled = edge_r.unsqueeze(axis=1).tile(repeat_times=[1, 3, 1])
        edge_nei_angle = paddle.sum(edge_nei * edge_r_tiled, axis=-1) / (
            edge_nei_norm
            * paddle.sqrt(paddle.sum(edge_r_tiled * edge_r_tiled, axis=-1))
        )
        edge_nei_angle = paddle.clip(edge_nei_angle, min=-1.0, max=1.0)

        edge_features = model.rbf(edge_feat)
        edge_nei_len = model.rbf(edge_nei_len.reshape([-1])).reshape(
            [-1, 3, model.node_features]
        )
        edge_nei_angle = model.rbf_angle(edge_nei_angle.reshape([-1])).reshape(
            [-1, 3, model.node_features]
        )
        node_features = _node_attention(
            model.att_layers[0],
            node_features,
            edge_src,
            edge_dst,
            edge_features,
        )
        edge_features = _edge_attention(
            model.edge_update_layer,
            edge_features,
            edge_nei_len,
            edge_nei_angle,
        )
        for layer in model.att_layers[1:]:
            node_features = _node_attention(
                layer,
                node_features,
                edge_src,
                edge_dst,
                edge_features,
            )

        graph_count = paddle.shape(graph_node_count)[0]
        features = _mean_by_id(node_features, node_graph_id, graph_count)
        return model.fc_out(model.fc(features))

    def forward_graph(self, graph: pgl.Graph | Sequence[pgl.Graph]) -> paddle.Tensor:
        return self.forward_tensor(graph_to_tensor_batch(graph))

    def forward_tensor(self, batch: ComformerGraphTensorBatch) -> paddle.Tensor:
        if not isinstance(batch, ComformerGraphTensorBatch):
            raise TypeError(
                "batch must be a ComformerGraphTensorBatch produced by "
                "graph_to_tensor_batch."
            )
        if batch.node_feat.shape[1] != self.model.atom_input_features:
            raise ValueError(
                "batch.node_feat has feature dimension "
                f"{batch.node_feat.shape[1]}, expected "
                f"{self.model.atom_input_features}."
            )
        return self(*batch)

    @paddle.no_grad()
    def predict_tensor(self, batch: ComformerGraphTensorBatch) -> paddle.Tensor:
        return self.model.unnormalize(self.forward_tensor(batch))


def make_comformer_input_spec(
    atom_input_features: int = 92,
) -> list[paddle.static.InputSpec]:
    """Return the dynamic node, edge, and graph input contract for CINN."""

    return [
        paddle.static.InputSpec(
            [None, atom_input_features], dtype="float32", name="node_feat"
        ),
        paddle.static.InputSpec([None, 3], dtype="float32", name="edge_r"),
        paddle.static.InputSpec([None, 3, 3], dtype="float32", name="edge_nei"),
        paddle.static.InputSpec([None], dtype="int64", name="edge_src"),
        paddle.static.InputSpec([None], dtype="int64", name="edge_dst"),
        paddle.static.InputSpec([None], dtype="int64", name="node_graph_id"),
        paddle.static.InputSpec([None], dtype="int64", name="graph_node_count"),
    ]


def compile_comformer(
    model_or_core: iComformer | ComformerTensorCore,
    *,
    backend: Optional[str] = "CINN",
    full_graph: bool = True,
    input_spec: Optional[list[paddle.static.InputSpec]] = None,
) -> paddle.nn.Layer:
    """Compile the Tensor core with CINN or plain PIR static execution."""

    core = (
        model_or_core
        if isinstance(model_or_core, ComformerTensorCore)
        else ComformerTensorCore(model_or_core)
    )
    if input_spec is None:
        input_spec = make_comformer_input_spec(core.model.atom_input_features)
    return paddle.jit.to_static(
        core,
        input_spec=input_spec,
        backend=backend,
        full_graph=full_graph,
    )


def compile_comformer_cinn(
    model_or_core: iComformer | ComformerTensorCore,
    *,
    full_graph: bool = True,
    input_spec: Optional[list[paddle.static.InputSpec]] = None,
) -> paddle.nn.Layer:
    """Create a lazily compiled Paddle CINN iComformer runtime."""

    return compile_comformer(
        model_or_core,
        backend="CINN",
        full_graph=full_graph,
        input_spec=input_spec,
    )
