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

"""DimeNet++ edge roles and periodic-image triplets (``triplet_convention``)."""

from pathlib import Path

import numpy as np
import paddle
import pytest
from pymatgen.core import Structure

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.models.common.graph_converter import CrystalNN
from ppmat.models.dimenetpp.dimenetpp import DimeNetPlusPlus

CIF_PATH = (
    Path(__file__).resolve().parents[1]
    / "property_prediction/example_data/cifs/mp-18767-LiMnO2.cif"
)
PROPERTY = "formation_energy_per_atom"
PARAMS = {
    "out_channels": 1,
    "hidden_channels": 16,
    "num_blocks": 2,
    "int_emb_size": 8,
    "basis_emb_size": 4,
    "out_emb_channels": 16,
    "num_spherical": 3,
    "num_radial": 4,
    "cutoff": 7.0,
    "num_before_skip": 1,
    "num_after_skip": 1,
    "num_output_layers": 2,
    "property_names": PROPERTY,
}
PRIMITIVE = (1, 1, 1)
SUPERCELL = (2, 2, 2)
# Predictions of _model() for the 8-atom LiMnO2 cell (mp-18767) and its 2x2x2
# supercell, recorded with DimeNetPlusPlus before triplet_convention existed.
LEGACY_PREDICTIONS = {PRIMITIVE: 0.6343210935592651, SUPERCELL: 0.5156241655349731}
# OCP v0.1.0 DimeNetPlusPlusWrap (use_pbc=True, otf_graph=False) with the same
# weights and graphs, divided by the number of atoms because OCP sums over atoms.
OCP_PREDICTIONS = {PRIMITIVE: 0.5291295647621155, SUPERCELL: 0.5291295051574707}


def _model(**kwargs):
    model = DimeNetPlusPlus(**PARAMS, **kwargs)
    # Fixed weights, large enough to keep the network out of its nearly linear
    # regime, where swapping the edge roles hardly changes the prediction.
    rng = np.random.default_rng(2026)
    for _, param in model.named_parameters():
        bound = 1.4 / np.sqrt(param.shape[0]) if len(param.shape) == 2 else 0.5
        param.set_value(rng.uniform(-bound, bound, param.shape).astype("float32"))
    model.rbf.freq.set_value(
        paddle.arange(1, PARAMS["num_radial"] + 1, dtype="float32") * np.pi
    )
    model.eval()
    return model


def _predict(model, graph):
    return float(model.predict(graph.tensor(inplace=False))[PROPERTY])


@pytest.fixture(scope="module")
def graphs():
    graph_converter = CrystalNN()
    result = {}
    for supercell in (PRIMITIVE, SUPERCELL):
        structure = Structure.from_file(CIF_PATH)
        structure.make_supercell(supercell)
        result[supercell] = graph_converter(structure)
    return result


def test_triplets_keep_other_periodic_images_of_the_target_atom():
    # One atom bonded to its six nearest periodic images: k = i in every
    # triplet, but only the six back-tracking triplets, whose two image
    # offsets cancel, return to the same image of atom i.
    cell_offsets = np.concatenate([np.eye(3), -np.eye(3)]).astype("float32")
    edge_index = paddle.zeros([2, 6], dtype="int64")
    model = _model()

    legacy = model.triplets(edge_index, num_nodes=1)
    periodic = model.triplets(
        edge_index, num_nodes=1, cell_offsets=paddle.to_tensor(cell_offsets)
    )

    assert legacy[5].shape[0] == 0
    idx_kj, idx_ji = periodic[5].numpy(), periodic[6].numpy()
    assert len(idx_kj) == 6 * 5
    assert np.all(np.any(cell_offsets[idx_kj] + cell_offsets[idx_ji] != 0, axis=-1))


def test_ocp_convention_gives_the_same_prediction_for_a_supercell(graphs):
    model = _model(triplet_convention="ocp")
    batch = DefaultCollator()([graphs[PRIMITIVE], graphs[SUPERCELL]])
    with paddle.no_grad():
        output = model({"graph": batch}, return_loss=False)
    primitive, supercell = output["pred_dict"][PROPERTY].numpy().reshape(-1)

    np.testing.assert_allclose(supercell, primitive, rtol=1e-5)


def test_ocp_convention_matches_ocp_dimenetpp(graphs):
    model = _model(triplet_convention="ocp")
    for supercell, expected in OCP_PREDICTIONS.items():
        np.testing.assert_allclose(
            _predict(model, graphs[supercell]), expected, rtol=1e-5
        )


@pytest.mark.parametrize("kwargs", [{}, {"triplet_convention": "legacy"}])
def test_legacy_convention_reproduces_previous_predictions(graphs, kwargs):
    model = _model(**kwargs)
    for supercell, expected in LEGACY_PREDICTIONS.items():
        np.testing.assert_allclose(
            _predict(model, graphs[supercell]), expected, rtol=1e-5
        )
