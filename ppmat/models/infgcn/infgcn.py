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
from ppmat.models.common.activation import NormActivation
from ppmat.models.common.activation import ScalarActivation
from ppmat.models.common.e3nn import o3
from ppmat.models.common.e3nn.math import soft_one_hot_linspace
from ppmat.models.common.e3nn.nn import FullyConnectedNet
from ppmat.models.common.orbital import GaussianOrbital
from ppmat.utils.scatter import scatter


def _to_tensor(value, dtype, device):
    if isinstance(value, paddle.Tensor):
        return value.astype(dtype).to(device)
    return paddle.to_tensor(value, dtype=dtype).to(device)


def _randomly_truncate_atom_edges(edge_index, max_num_neighbors=32):
    if edge_index.shape[1] == 0:
        return edge_index

    num_nodes = int(paddle.max(edge_index).item()) + 1
    order = paddle.argsort(edge_index[0] * num_nodes + edge_index[1])
    edge_index = edge_index[:, order]
    source = edge_index[0]
    unique_sources, counts = paddle.unique(source, return_counts=True)
    keep_mask = paddle.ones_like(source, dtype="bool")
    for node, count in zip(unique_sources, counts):
        count = int(count.item())
        if count > max_num_neighbors:
            edge_ids = paddle.nonzero(source == node, as_tuple=True)[0]
            permutation = paddle.randperm(count)
            keep_mask[edge_ids[permutation[max_num_neighbors:]]] = False
    return edge_index[:, keep_mask]


class GCNLayer(paddle.nn.Layer):
    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_edge,
        radial_embed_size,
        num_radial_layer,
        radial_hidden_size,
        is_fc=True,
        use_sc=True,
        irrep_normalization="component",
        path_normalization="element",
    ):
        """
        A single InfGCN layer for Tensor Product-based message passing.
        If the tensor product is fully connected, we have (for every path)

        .. math::
            z_w=\\sum_{uv}w_{uvw}x_u\\otimes y_v=\\sum_{u}w_{uw}x_u \\otimes y

        Else, we have

        .. math::
            z_u=x_u\\otimes \\sum_v w_{uv}y_v=w_u (x_u\\otimes y)

        Here, uvw are radial (channel) indices of the first input, second input,
        and output, respectively. Notice that in our model, the second input is
        always the spherical harmonics of the edge vector,
        so the index v can be safely ignored.

        :param irreps_in: irreducible representations of input node features
        :param irreps_out: irreducible representations of output node features
        :param irreps_edge: irreducible representations of edge features
        :param radial_embed_size: embedding size of the edge length
        :param num_radial_layer: number of hidden layers in the radial network
        :param radial_hidden_size: hidden size of the radial network
        :param is_fc: whether to use fully connected tensor product
        :param use_sc: whether to use self-connection
        :param irrep_normalization: representation normalization passed to the
            `o3.FullyConnectedTensorProduct`
        :param path_normalization: path normalization passed to the
            `o3.FullyConnectedTensorProduct`
        """
        super(GCNLayer, self).__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.radial_embed_size = radial_embed_size
        self.num_radial_layer = num_radial_layer
        self.radial_hidden_size = radial_hidden_size
        self.is_fc = is_fc
        self.use_sc = use_sc

        if self.is_fc:
            self.tp = o3.FullyConnectedTensorProduct(
                self.irreps_in,
                self.irreps_edge,
                self.irreps_out,
                internal_weights=False,
                shared_weights=False,
                irrep_normalization=irrep_normalization,
                path_normalization=path_normalization,
            )
        else:
            instr = [
                (i_1, i_2, i_out, "uvu", True)
                for i_1, (_, ir_1) in enumerate(self.irreps_in)
                for i_2, (_, ir_edge) in enumerate(self.irreps_edge)
                for i_out, (_, ir_out) in enumerate(self.irreps_out)
                if ir_out in ir_1 * ir_edge
            ]
            self.tp = o3.TensorProduct(
                self.irreps_in,
                self.irreps_edge,
                self.irreps_out,
                instr,
                internal_weights=False,
                shared_weights=False,
                irrep_normalization=irrep_normalization,
                path_normalization=path_normalization,
            )
        # The activation is automatically normalized by a scaling factor.
        self.fc = FullyConnectedNet(
            [radial_embed_size]
            + num_radial_layer * [radial_hidden_size]
            + [self.tp.weight_numel],
            paddle.nn.functional.silu,
        )
        self.sc = None
        if self.use_sc:
            self.sc = o3.Linear(self.irreps_in, self.irreps_out)

    def forward(self, edge_index, node_feat, edge_feat, edge_embed, dim_size=None):
        src, dst = edge_index
        weight = self.fc(edge_embed)  # FFN
        out = self.tp(
            node_feat[src], edge_feat, weight=weight
        )  # Tensor Product [num_edges, tp.irreps_out.dim]

        out = scatter(
            out, dst, dim=0, dim_size=dim_size, reduce="sum"
        )  # message aggregation

        if self.use_sc:
            out = out + self.sc(node_feat)
        return out


def pbc_vec(vec, cell):
    """
    Apply periodic boundary condition to the vector
    :param vec: original vector of (N, K, 3)
    :param cell: cell frame of (N, 3, 3)
    :return: shortest vector of (N, K, 3)
    """
    coord = vec @ paddle.linalg.inv(x=cell)
    coord = coord - paddle.round(coord)
    pbc_vec = coord @ cell
    return pbc_vec.detach()


class InfGCN(paddle.nn.Layer):
    def __init__(
        self,
        vocab,
        num_radial,
        num_spherical,
        radial_embed_size,
        radial_hidden_size,
        num_radial_layer=2,
        num_gcn_layer=3,
        cutoff=3.0,
        grid_cutoff=3.0,
        is_fc=True,
        gauss_start=0.5,
        gauss_end=5.0,
        activation="norm",
        residual=True,
        pbc=False,
        target_name="density",
        loss_eps=1e-8,
        **kwargs,
    ):
        """
        Implement the InfGCN model for electron density estimation
        :param vocab: vocabularies used by the model
        :param num_radial: number of radial basis
        :param num_spherical: maximum number of spherical harmonics for each
            radial basis,
                number of spherical basis will be (num_spherical + 1)^2
        :param radial_embed_size: embedding size of the edge length
        :param radial_hidden_size: hidden size of the radial network
        :param num_radial_layer: number of hidden layers in the radial network
        :param num_gcn_layer: number of InfGCN layers
        :param cutoff: cutoff distance for building the molecular graph
        :param grid_cutoff: cutoff distance for building the grid-atom graph
        :param is_fc: whether the InfGCN layer should use fully connected
            tensor product
        :param gauss_start: start coefficient of the Gaussian radial basis
        :param gauss_end: end coefficient of the Gaussian radial basis
        :param activation: activation type for the InfGCN layer, can be
            ['scalar', 'norm']
        :param residual: whether to use the residue prediction layer
        :param pbc: whether the data satisfy the periodic boundary condition
        """
        super(InfGCN, self).__init__()
        self.vocab = vocab
        n_atom_type = vocab["atom"]["num_embeddings"]
        self.n_atom_type = n_atom_type
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.radial_embed_size = radial_embed_size
        self.radial_hidden_size = radial_hidden_size
        self.num_radial_layer = num_radial_layer
        self.num_gcn_layer = num_gcn_layer
        self.cutoff = cutoff
        self.grid_cutoff = grid_cutoff
        self.is_fc = is_fc
        self.gauss_start = gauss_start
        self.gauss_end = gauss_end
        self.activation = activation
        self.residual = residual
        self.pbc = pbc
        self.target_name = target_name
        self.loss_eps = loss_eps
        assert activation in ["scalar", "norm"]
        self.embedding = paddle.nn.Embedding(
            num_embeddings=n_atom_type, embedding_dim=num_radial
        )
        self.irreps_sh = o3.Irreps.spherical_harmonics(num_spherical, p=1)
        self.irreps_feat = (self.irreps_sh * num_radial).sort().irreps.simplify()

        self.gcns = paddle.nn.LayerList(
            sublayers=[
                GCNLayer(
                    f"{num_radial}x0e" if i == 0 else self.irreps_feat,
                    self.irreps_feat,
                    self.irreps_sh,
                    radial_embed_size,
                    num_radial_layer,
                    radial_hidden_size,
                    is_fc=is_fc,
                    **kwargs,
                )
                for i in range(num_gcn_layer)
            ]
        )
        if self.activation == "scalar":
            self.act = ScalarActivation(
                self.irreps_feat,
                paddle.nn.functional.silu,
                paddle.nn.functional.sigmoid,
            )
        else:
            self.act = NormActivation(self.irreps_feat)
        self.residue = None
        if self.residual:
            self.residue = GCNLayer(
                self.irreps_feat,
                "0e",
                self.irreps_sh,
                radial_embed_size,
                num_radial_layer,
                radial_hidden_size,
                is_fc=True,
                use_sc=False,
                **kwargs,
            )
        self.orbital = GaussianOrbital(
            gauss_start, gauss_end, num_radial, num_spherical
        )
        self._criterion = paddle.nn.MSELoss(reduction="mean")

    def forward(self, data, return_loss=True, return_prediction=True):
        """Run field prediction through the PaddleMaterials model protocol.

        Args:
            data: Collated field sample containing ``graph``, ``grid_coord``,
                optional ``density_mask`` and ``info``, and the supervised
                target under :attr:`target_name` when loss is requested.
            return_loss: Whether to compute and return the supervised loss.
            return_prediction: Whether to expose the predicted field.
        """

        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."

        mask = data.get("density_mask")
        grid = data["grid_coord"]
        graph = data["graph"]
        info = data.get("info")

        # Move model inputs to the active device.
        device = paddle.get_device()
        atom_types = _to_tensor(graph.node_feat["x"], "int64", device)
        atom_coord = _to_tensor(graph.node_feat["pos"], "float32", device)
        atom_edges = _to_tensor(graph.edges, "int64", device).transpose([1, 0])
        graph_batch = _to_tensor(graph.graph_node_id, "int64", device)
        grid = _to_tensor(grid, "float32", device)
        if mask is not None:
            mask = _to_tensor(mask, "float32", device)
        cell = self._prepare_cell(info, device)

        # Predict the field independently of target availability.
        pred = self._forward_density(
            atom_types,
            atom_coord,
            atom_edges,
            grid,
            graph_batch,
            cell,
        )

        # Mask padded grid positions consistently for loss and prediction.
        masked_pred = pred
        if mask is not None:
            mask = mask.astype(pred.dtype)
            masked_pred = pred * mask

        # Calculate loss and NMAE only when requested.
        loss_dict = {}
        if return_loss:
            density = data[self.target_name]
            if density is None:
                raise ValueError(
                    f"data[{self.target_name!r}] must not be None when "
                    "return_loss is True."
                )
            density = _to_tensor(density, "float32", device)
            if mask is not None:
                label_masked = density * mask
                denom = paddle.sum(mask) + self.loss_eps
                loss = paddle.sum((masked_pred - label_masked) ** 2) / denom
                # Normalized MAE (original InfGCN):
                #   mae = sum(|pred - density|) / sum(density)
                mae = paddle.sum(paddle.abs(masked_pred - label_masked)) / (
                    paddle.sum(label_masked) + self.loss_eps
                )
            else:
                label_masked = density
                loss = self._criterion(pred, label_masked)
                mae = paddle.sum(paddle.abs(pred - label_masked)) / (
                    paddle.sum(label_masked) + self.loss_eps
                )
            loss_dict["loss"] = loss
            loss_dict["mae"] = mae

        pred_dict = {}
        if return_prediction:
            pred_dict[self.target_name] = masked_pred
        return {"loss_dict": loss_dict, "pred_dict": pred_dict}

    def _prepare_cell(self, info, device):
        if not self.pbc:
            return None
        if info is None or "cell" not in info:
            raise KeyError("Periodic InfGCN input requires info['cell'].")
        cell = _to_tensor(info["cell"], "float32", device)
        if len(cell.shape) == 2:
            cell = cell.unsqueeze(0)
        if len(cell.shape) != 3 or list(cell.shape[-2:]) != [3, 3]:
            raise ValueError(
                "info['cell'] must have shape [batch_size, 3, 3], but got "
                f"{list(cell.shape)}."
            )
        return cell

    def _forward_density(self, atom_types, atom_coord, atom_edges, grid, batch, cell):
        """
        Network forward with memory optimization
        :param atom_types: atom types of (N,)
        :param atom_coord: atom coordinates of (N, 3)
        :param atom_edges: candidate atom edges of (2, E)
        :param grid: coordinates at grid points of (G, K, 3)
        :param batch: batch index for each node of (N,)
        :param cell: optional batched cell vectors of (G, 3, 3)
        :return: predicted value at each grid point of (G, K)
        """
        feat = self.embedding(atom_types)

        # Preserve original random max-32 atom-neighbor truncation on every forward.
        edge_index = _randomly_truncate_atom_edges(atom_edges)
        src, dst = edge_index
        edge_vec = atom_coord[src] - atom_coord[dst]  # coord vector
        edge_len = edge_vec.norm(axis=-1) + 1e-08  # L2 norm, equal to distance

        # Angular features have shape [num_edges, 2 * degree + 1].
        edge_feat = o3.spherical_harmonics(
            list(range(self.num_spherical + 1)),  # degree of the spherical harmonics
            edge_vec / edge_len[..., None],  # e.g. edge vector
            normalize=False,  # Input vectors are already normalized.
            normalization="integral",  # normalization of the output tensors
        )

        edge_embed = (
            soft_one_hot_linspace(  # radial features, [D_edge_index, radial_embed_size]
                edge_len,
                start=0.0,
                end=self.cutoff,
                number=self.radial_embed_size,  # The number of radial basis functions.
                basis="gaussian",  # Uses Gaussian functions as the radial basis.
                cutoff=False,  # Disables the cutoff/smoothing function at the boundary.
            )
            * (self.radial_embed_size**0.5)
        )  # enhance signal feature due to normalization of output

        for i, gcn in enumerate(self.gcns):
            feat = gcn(
                edge_index, feat, edge_feat, edge_embed, dim_size=atom_types.shape[0]
            )
            if i != self.num_gcn_layer - 1:
                feat = self.act(feat)

        n_graph, n_sample = grid.shape[0], grid.shape[1]
        if self.residual:
            grid_flat = grid.view(-1, 3)
            grid_batch = paddle.arange(end=n_graph).repeat_interleave(repeats=n_sample)
            grid_dst, node_src = radius(
                atom_coord, grid_flat, self.grid_cutoff, batch, grid_batch
            )
            grid_edge = grid_flat[grid_dst] - atom_coord[node_src]
            if grid_edge.shape[0] != 0:
                grid_len = paddle.linalg.norm(x=grid_edge, axis=-1) + 1e-08
                grid_edge_feat = o3.spherical_harmonics(
                    list(range(self.num_spherical + 1)),
                    grid_edge / (grid_len[..., None] + 1e-08),
                    normalize=False,
                    normalization="integral",
                )
                grid_edge_embed = soft_one_hot_linspace(
                    grid_len,
                    start=0.0,
                    end=self.grid_cutoff,
                    number=self.radial_embed_size,
                    basis="gaussian",
                    cutoff=False,
                ) * (self.radial_embed_size**0.5)

                residue = self.residue(
                    (node_src, grid_dst),
                    feat,
                    grid_edge_feat,
                    grid_edge_embed,
                    dim_size=grid_flat.shape[0],
                )
            else:
                residue = paddle.zeros([grid_flat.shape[0], 1], dtype=feat.dtype)
        else:
            residue = 0.0

        # Displacement vectors from each atom to each sampled grid point have
        # shape [num_atoms, num_grid_points, 3].
        sample_vec = grid[batch] - atom_coord.unsqueeze(axis=-2)
        if cell is not None:
            sample_vec = pbc_vec(sample_vec, cell[batch])

        # Expand displacement vectors in the Gaussian-type orbital basis:
        # [num_atoms, num_grid_points, (lmax + 1)^2 * num_gaussians].
        orbital = self.orbital(sample_vec)
        density = (orbital * feat.unsqueeze(axis=1)).sum(
            axis=-1
        )  # linear combination [n_atom, n_grid]
        density = scatter(density, batch, dim=0, reduce="sum")  # molecular/cell density

        if self.residual:
            density = density + residue.view(*tuple(density.shape))

        return density
