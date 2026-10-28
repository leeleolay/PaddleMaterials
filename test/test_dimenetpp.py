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
# Predictions of _model() for the 8-atom LiMnO2 cell (mp-18767) and its 2x2x2
# supercell, computed with OCP v0.1.0 DimeNetPlusPlusWrap (use_pbc=True,
# otf_graph=False) from the same weights and graphs and divided by the number of
# atoms, because OCP sums over atoms.
OCP_PREDICTIONS = [0.5291295647621155, 0.5291295051574707]


def _model():
    model = DimeNetPlusPlus(
        out_channels=1,
        hidden_channels=16,
        num_blocks=2,
        int_emb_size=8,
        basis_emb_size=4,
        out_emb_channels=16,
        num_spherical=3,
        num_radial=4,
        cutoff=7.0,
        num_after_skip=1,
        num_output_layers=2,
        property_names=PROPERTY,
    )
    # Fixed random weights, large enough that swapping the edge roles changes the
    # prediction. rbf.freq keeps its initial values n * pi, which OCP used too; its
    # draw is discarded so that the other weights stay those of the OCP run.
    rng = np.random.default_rng(2026)
    for name, param in model.named_parameters():
        bound = 1.4 / np.sqrt(param.shape[0]) if len(param.shape) == 2 else 0.5
        value = rng.uniform(-bound, bound, param.shape).astype("float32")
        if name != "rbf.freq":
            param.set_value(value)
    model.eval()
    return model


@pytest.fixture(scope="module")
def graphs():
    result = []
    for scale in (1, 2):
        structure = Structure.from_file(CIF_PATH)
        structure.make_supercell(scale)
        result.append(CrystalNN()(structure))
    return result


def test_supercell_gives_the_same_prediction(graphs):
    batch = DefaultCollator()(graphs)
    with paddle.no_grad():
        output = _model()({"graph": batch}, return_loss=False)
    primitive, supercell = output["pred_dict"][PROPERTY].numpy().reshape(-1)

    np.testing.assert_allclose(supercell, primitive, rtol=1e-5)


def test_matches_ocp_dimenetpp(graphs):
    model = _model()
    predictions = [
        float(model.predict(graph.tensor(inplace=False))[PROPERTY]) for graph in graphs
    ]

    np.testing.assert_allclose(predictions, OCP_PREDICTIONS, rtol=1e-5)
