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

from ppmat.models.common.graph_converter import CrystalNN
from ppmat.models.dimenetpp.dimenetpp import DimeNetPlusPlus

CIF_PATH = (
    Path(__file__).resolve().parents[1]
    / "property_prediction/example_data/cifs/mp-18767-LiMnO2.cif"
)


# Per-atom predictions of a small model with fixed weights for the 8-atom LiMnO2
# cell (mp-18767) and its 2x2x2 supercell.
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        # Default ("legacy"): recorded on develop before triplet_convention existed.
        ({}, [0.6343210935592651, 0.5156241655349731]),
        # OCP v0.1.0 DimeNetPlusPlusWrap with the same weights and graphs, which
        # gives the same value for both cells.
        ({"triplet_convention": "ocp"}, [0.5291295647621155, 0.5291295051574707]),
    ],
    ids=["default", "ocp"],
)
def test_triplet_convention(kwargs, expected):
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
        **kwargs,
    )
    # Weights large enough that swapping the edge roles changes the prediction.
    rng = np.random.default_rng(2026)
    for param in model.parameters():
        bound = 1.4 / np.sqrt(param.shape[0]) if len(param.shape) == 2 else 0.5
        param.set_value(rng.uniform(-bound, bound, param.shape).astype("float32"))
    model.rbf.freq.set_value(paddle.arange(1, 5, dtype="float32") * np.pi)
    model.eval()

    predictions = []
    for scale in (1, 2):
        structure = Structure.from_file(CIF_PATH)
        structure.make_supercell(scale)
        prediction = model.predict(CrystalNN()(structure))
        predictions.append(float(prediction["formation_energy_per_atom"]))
    np.testing.assert_allclose(predictions, expected, rtol=1e-5)
