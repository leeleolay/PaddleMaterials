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

"""Public behavior tests for MSD-NMR training statistics."""

import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pgl
import pytest

_threebody = types.ModuleType("ppmat.models.mattersim.threebody_indices")
_threebody.compute_threebody = lambda *args, **kwargs: None
sys.modules[_threebody.__name__] = _threebody

from ppmat.datasets.msd_nmr_dataset import DataLoaderCollection  # noqa: E402
from ppmat.datasets.msd_nmr_dataset import MSDnmrDataset  # noqa: E402
from ppmat.datasets.msd_nmr_dataset import MSDnmrinfos  # noqa: E402
from ppmat.datasets.msd_nmr_dataset import _get_msd_nmr_subdataset_name  # noqa: E402


class GraphDataset:
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class Loader:
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        raise AssertionError("statistics must not iterate sampler batches")


def graph(node_types, edges=(), bond_types=(), num_atom_types=2, num_bonds=5):
    node_types = np.asarray(node_types, dtype=np.int64)
    edges = np.asarray(edges, dtype=np.int64).reshape((-1, 2))
    edge_feat = np.zeros((len(edges), num_bonds), dtype=np.float32)
    if len(edges):
        edge_feat[np.arange(len(edges)), np.asarray(bond_types)] = 1
    return pgl.Graph(
        num_nodes=len(node_types),
        edges=edges,
        node_feat={"feat": np.eye(num_atom_types, dtype=np.float32)[node_types]},
        edge_feat={"feat": edge_feat},
    )


def sample(*args, **kwargs):
    return {"graph": graph(*args, **kwargs)}


def vocab():
    atom_tokens = ["C", "N"]
    bond_tokens = ["NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
    return {
        "atom": {
            "num_embeddings": len(atom_tokens),
            "token_to_id": {token: index for index, token in enumerate(atom_tokens)},
            "id_to_token": dict(enumerate(atom_tokens)),
        },
        "bond": {
            "num_embeddings": len(bond_tokens),
            "token_to_id": {token: index for index, token in enumerate(bond_tokens)},
            "id_to_token": dict(enumerate(bond_tokens)),
        },
    }


def infos_config(cache_path=None):
    config = {
        "data_flag": "n<15",
        "max_atoms": 3,
        "build_graph_cfg": {
            "__init_params__": {
                "remove_h": True,
                "edge_mode": "bidirectional",
            }
        },
        "build_spectrum_cfg": {"seq_len_H1": 20, "seq_len_C13": 75},
    }
    if cache_path is not None:
        config["statistics_cache_path"] = str(cache_path)
    return config


def test_statistics_iterates_each_train_sample_once():
    dataset = GraphDataset(
        [
            sample([0]),
            sample([0, 1]),
            sample([0, 0, 1]),
        ]
    )
    statistics = DataLoaderCollection(Loader(dataset)).statistics(
        num_node_types=2,
        num_edge_types=5,
        max_nodes_hint=3,
        bond_orders=np.asarray([0, 1, 2, 3, 1.5]),
    )

    np.testing.assert_allclose(
        statistics["n_nodes"].numpy(), [0.0, 1 / 3, 1 / 3, 1 / 3]
    )
    np.testing.assert_allclose(statistics["node_types"].numpy(), [4 / 6, 2 / 6])
    np.testing.assert_allclose(statistics["edge_types"].numpy(), [1, 0, 0, 0, 0])


def test_statistics_use_bond_order_for_valency():
    dataset = GraphDataset(
        [
            sample(
                [0, 1],
                edges=[[0, 1], [1, 0]],
                bond_types=[2, 2],
            )
        ]
    )
    statistics = DataLoaderCollection(Loader(dataset)).statistics(
        num_node_types=2,
        num_edge_types=5,
        max_nodes_hint=2,
        bond_orders=np.asarray([0, 1, 2, 3, 1.5]),
    )

    np.testing.assert_allclose(statistics["edge_types"].numpy(), [0, 0, 1, 0, 0])
    assert statistics["valency_distribution"][2].item() == 1


def test_infos_cache_training_statistics(tmp_path):
    cache_path = tmp_path / "statistics.pdparams"
    dataset = GraphDataset([sample([0, 1])])
    loaders = {"train": Loader(dataset), "val": Loader(GraphDataset([sample([1])]))}

    first = MSDnmrinfos(loaders, infos_config(cache_path), vocab())
    assert cache_path.is_file()

    dataset.samples[:] = [sample([0, 0, 1])]
    second = MSDnmrinfos(loaders, infos_config(cache_path), vocab())
    np.testing.assert_array_equal(first.n_nodes.numpy(), second.n_nodes.numpy())
    assert second.n_nodes[2].item() == 1


def test_loader_free_sampling_requires_cached_statistics():
    atom_tokens = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    release_vocab = vocab()
    release_vocab["atom"] = {
        "num_embeddings": len(atom_tokens),
        "token_to_id": {token: index for index, token in enumerate(atom_tokens)},
        "id_to_token": dict(enumerate(atom_tokens)),
    }

    with pytest.raises(FileNotFoundError, match="provide the training loader"):
        MSDnmrinfos(
            SimpleNamespace(train_dataloader=None),
            infos_config(),
            release_vocab,
        )


def test_nless25_uses_matching_download_directory():
    assert _get_msd_nmr_subdataset_name("n<25") == "msd_nmr_nless25"


def test_dataset_reads_standard_csv_schema(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "smiles": ["C"],
            "tokenized_input": [json.dumps({"1HNMR": [], "13CNMR": []})],
            "atom_count": [1],
        }
    ).to_csv(csv_path, index=False)

    dataset = object.__new__(MSDnmrDataset)
    raw_data, num_samples = dataset.read_data(str(csv_path))

    assert num_samples == 1
    assert raw_data == {
        "smiles": ["C"],
        "tokenized_nmr": [{"1HNMR": [], "13CNMR": []}],
        "atom_count": [1],
    }
