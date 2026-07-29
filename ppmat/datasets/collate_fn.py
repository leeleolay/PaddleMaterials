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


from __future__ import annotations

import hashlib
import numbers
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import List

import numpy as np
import paddle
import pgl

from ppmat.datasets.custom_data_type import ConcatData
from ppmat.datasets.custom_data_type import ConcatNumpyWarper
from ppmat.datasets.geometric_data_type.batch import Batch
from ppmat.datasets.geometric_data_type.data import Data
from ppmat.utils.pgl_compat import patch_pgl_empty_edge_batch

patch_pgl_empty_edge_batch()


class DefaultCollator(object):
    def __call__(self, batch: List[Any]) -> Any:
        """Default_collate_fn for paddle dataloader.

        NOTE: This `default_collate_fn` is different from official `default_collate_fn`
        which specially adapt case where sample is `None` and `pgl.Graph`.

        ref: https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/io/dataloader/collate.py#L25

        Args:
            batch (List[Any]): Batch of samples to be collated.

        Returns:
            Any: Collated batch data.
        """
        sample = batch[0]
        if sample is None:
            return None
        elif isinstance(sample, ConcatNumpyWarper):
            batch = np.concatenate(batch, axis=0)
            return batch
        elif isinstance(sample, np.ndarray):
            batch = np.stack(batch, axis=0)
            return batch
        elif isinstance(sample, (paddle.Tensor, paddle.framework.core.eager.Tensor)):
            return paddle.stack(batch, axis=0)
        elif isinstance(sample, numbers.Number):
            batch = np.array(batch)
            return batch
        elif isinstance(sample, Data):
            # Geometric `Data` objects: batch them into a single `Batch`
            return Batch.from_data_list(batch)
        elif isinstance(sample, (str, bytes)):
            return batch
        elif isinstance(sample, Mapping):
            return {key: self([d[key] for d in batch]) for key in sample}
        elif isinstance(sample, Sequence):
            sample_fields_num = len(sample)
            if not all(len(sample) == sample_fields_num for sample in iter(batch)):
                raise RuntimeError("Fields number not same among samples in a batch")
            return [self(fields) for fields in zip(*batch)]
        elif str(type(sample)) == "<class 'pgl.graph.Graph'>":
            # use str(type()) instead of isinstance() in case of pgl is not installed.
            graphs = pgl.Graph.batch(batch)
            # NOTE: when num_works >1, graphs.tensor() will convert numpy.ndarray to
            # CPU Tensor, which will cause error in model training.
            # graphs.tensor()
            return graphs
        elif isinstance(sample, ConcatData):
            return ConcatData.batch(batch)
        raise TypeError(
            "batch data can only contains: paddle.Tensor, numpy.ndarray, "
            f"dict, list, number, None, pgl.Graph, but got {type(sample)}"
        )


class RadiusGraphCollator:
    """Collate PGL radius graphs and offset their edge-based triplet indices."""

    def __call__(self, batch):
        graphs = [sample["graph"] for sample in batch]
        num_edges = [np.asarray(graph.edges).shape[0] for graph in graphs]
        edge_offsets = np.cumsum([0] + num_edges[:-1])

        triplet_fields = {key: [] for key in ("idx_kj", "idx_ji")}
        for index, graph in enumerate(graphs):
            for key in triplet_fields:
                triplet_fields[key].append(
                    np.asarray(graph.edge_feat[f"ti_{key}"], dtype=np.int64)
                    + edge_offsets[index]
                )

        graph = pgl.Graph.batch(graphs)
        graph.edge_feat.update(
            {
                f"ti_{key}": np.concatenate(values)
                for key, values in triplet_fields.items()
            }
        )

        result = DefaultCollator()(
            [
                {key: value for key, value in sample.items() if key != "graph"}
                for sample in batch
            ]
        )
        result["graph"] = graph
        return result


class DensityCollator:
    def __init__(
        self,
        n_samples=None,
        sampling_mode: str = "uniform",
        uniform_random_offset: bool = False,
        sampling_seed: int | None = None,
        resample_each_call: bool = True,
        importance_sampling: bool = False,
        importance_threshold: float = 1e-5,
        importance_ratio: float = 0.8,
        extreme_threshold: float | None = None,
        extreme_ratio: float = 0.05,
        clip_max: float | None = None,
    ):
        if n_samples is not None:
            if isinstance(n_samples, bool) or not isinstance(
                n_samples, numbers.Integral
            ):
                raise TypeError("n_samples must be a positive integer or None.")
            if n_samples <= 0:
                raise ValueError("n_samples must be a positive integer or None.")

        self.n_samples = None if n_samples is None else int(n_samples)

        sampling_mode = sampling_mode.lower()
        if sampling_mode not in {"uniform", "random"}:
            raise ValueError(
                f"Unsupported sampling_mode '{sampling_mode}'. "
                "Use 'uniform' or 'random'."
            )
        self.sampling_mode = sampling_mode
        self.uniform_random_offset = uniform_random_offset

        self.importance_sampling = importance_sampling
        self.importance_threshold = importance_threshold
        self.importance_ratio = float(importance_ratio)
        self.extreme_threshold = extreme_threshold
        self.extreme_ratio = float(extreme_ratio)
        if self.importance_sampling and not 0.0 <= self.importance_ratio <= 1.0:
            raise ValueError("importance_ratio must be between 0 and 1.")
        if self.importance_sampling and not 0.0 <= self.extreme_ratio <= 1.0:
            raise ValueError("extreme_ratio must be between 0 and 1.")

        if sampling_seed is not None and (
            isinstance(sampling_seed, bool)
            or not isinstance(sampling_seed, numbers.Integral)
        ):
            raise TypeError("sampling_seed must be an integer or None.")
        if not isinstance(resample_each_call, bool):
            raise TypeError("resample_each_call must be a boolean.")
        self.sampling_seed = None if sampling_seed is None else int(sampling_seed)
        self.resample_each_call = resample_each_call
        self._uses_random_sampling = self.n_samples is not None and (
            self.importance_sampling
            or self.sampling_mode == "random"
            or self.uniform_random_offset
        )
        if (
            self._uses_random_sampling
            and not self.resample_each_call
            and self.sampling_seed is None
        ):
            raise ValueError(
                "sampling_seed is required when resample_each_call is False."
            )
        self._rng = (
            np.random.default_rng(self.sampling_seed)
            if self._uses_random_sampling and self.resample_each_call
            else None
        )

        self.clip_max = clip_max

    def __call__(self, batch):
        densities = [sample["density"] for sample in batch]
        grid_coord = [sample["grid_coord"] for sample in batch]
        for density, coordinates in zip(densities, grid_coord):
            density_length = int(density.shape[0])
            coordinate_length = int(coordinates.shape[0])
            if density_length != coordinate_length:
                raise ValueError(
                    f"Density length ({density_length}) and grid length "
                    f"({coordinate_length}) must match."
                )
            if density_length == 0:
                raise ValueError("Empty density/grid pair encountered in batch.")

        prepared_density, prepared_grid, masks = [], [], []
        if self.n_samples is None:
            max_length = max(int(density.shape[0]) for density in densities)
            for density, coordinates in zip(densities, grid_coord):
                length = int(density.shape[0])
                pad_width = [(0, max_length - length)] + [
                    (0, 0) for _ in range(density.ndim - 1)
                ]
                prepared_density.append(np.pad(density, pad_width, constant_values=0.0))
                prepared_grid.append(
                    np.pad(
                        coordinates,
                        ((0, max_length - length), (0, 0)),
                        constant_values=0.0,
                    )
                )
                mask = np.zeros_like(prepared_density[-1], dtype=np.float32)
                mask[:length] = 1.0
                masks.append(mask)
        else:
            target_samples = self.n_samples
            for sample, d, coord in zip(batch, densities, grid_coord):
                total = int(d.shape[0])
                rng = (
                    self._rng_for_sample(sample) if self._uses_random_sampling else None
                )
                if self.importance_sampling:
                    total_idx = np.arange(total)
                    dense_vals = np.abs(d.reshape(-1))
                    threshold = float(self.importance_threshold)
                    high_mask = dense_vals >= threshold
                    high_idx = total_idx[high_mask]

                    extreme_idx = np.array([], dtype=int)
                    if self.extreme_threshold is not None:
                        extreme_mask = dense_vals >= self.extreme_threshold
                        extreme_idx = total_idx[extreme_mask]
                        extreme_idx = np.intersect1d(
                            extreme_idx, high_idx, assume_unique=True
                        )
                    mid_idx = np.setdiff1d(high_idx, extreme_idx, assume_unique=True)

                    high_quota = min(
                        target_samples,
                        max(0, int(target_samples * self.importance_ratio)),
                    )
                    extreme_quota = min(
                        target_samples,
                        high_quota,
                        max(0, int(target_samples * self.extreme_ratio)),
                    )

                    extreme_take = min(len(extreme_idx), extreme_quota)
                    indices_extreme = (
                        rng.choice(extreme_idx, extreme_take, replace=False)
                        if extreme_take > 0
                        else np.array([], dtype=int)
                    )

                    remaining_high = high_quota - len(indices_extreme)
                    mid_take = min(len(mid_idx), remaining_high)
                    indices_mid = (
                        rng.choice(mid_idx, mid_take, replace=False)
                        if mid_take > 0
                        else np.array([], dtype=int)
                    )

                    selected = np.concatenate([indices_extreme, indices_mid])
                    remaining = target_samples - len(selected)
                    if remaining > 0:
                        low_candidates = np.setdiff1d(
                            total_idx, selected, assume_unique=False
                        )
                        if len(low_candidates) == 0:
                            low_candidates = total_idx
                        replace_low = remaining > len(low_candidates)
                        indices_low = rng.choice(
                            low_candidates, remaining, replace=replace_low
                        )
                        indices = np.concatenate([selected, indices_low])
                    else:
                        indices = selected
                else:
                    if self.sampling_mode == "uniform":
                        if self.uniform_random_offset:
                            if target_samples <= total:
                                bin_edges = np.linspace(
                                    0,
                                    total,
                                    num=target_samples + 1,
                                    dtype=int,
                                )
                                indices = np.asarray(
                                    [
                                        rng.integers(start, stop)
                                        for start, stop in zip(
                                            bin_edges[:-1],
                                            bin_edges[1:],
                                        )
                                    ],
                                    dtype=int,
                                )
                            else:
                                offset = int(rng.integers(0, total))
                                indices = (
                                    np.linspace(
                                        0,
                                        total,
                                        num=target_samples,
                                        endpoint=False,
                                        dtype=int,
                                    )
                                    + offset
                                ) % total
                        else:
                            indices = np.linspace(
                                0, total - 1, num=target_samples, dtype=int
                            )
                    else:
                        replace = target_samples > total
                        indices = rng.choice(total, target_samples, replace=replace)
                indices.sort()
                prepared_density.append(d[indices])
                prepared_grid.append(coord[indices])
                masks.append(np.ones_like(prepared_density[-1], dtype=np.float32))

        prepared_batch = []
        for sample, density, coordinates, mask in zip(
            batch,
            prepared_density,
            prepared_grid,
            masks,
        ):
            prepared_sample = dict(sample)
            prepared_sample["density"] = density
            prepared_sample["density_mask"] = mask
            prepared_sample["grid_coord"] = coordinates
            prepared_sample["info"] = {
                "cell": sample["info"]["cell"],
            }
            prepared_batch.append(prepared_sample)

        result = DefaultCollator()(prepared_batch)
        if self.clip_max is not None:
            result["density"] = np.minimum(result["density"], self.clip_max)
        return result

    def _rng_for_sample(self, sample):
        if self.resample_each_call:
            return self._rng
        info = sample["info"]
        identity = (
            info.get("source_split"),
            sample["id"],
            info.get("file_name"),
        )
        seed_material = repr((self.sampling_seed, identity)).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
        return np.random.default_rng(seed)
