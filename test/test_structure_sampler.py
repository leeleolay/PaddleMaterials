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

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_structure_sampler_cli_accepts_config_overrides():
    from structure_generation.sample import parse_args

    args, overrides = parse_args(
        [
            "--model_name",
            "mattergen_mp20",
            "--output_dir",
            "generated",
            "Sample.foo=bar",
        ]
    )
    assert args.output_dir == "generated"
    assert overrides == ["Sample.foo=bar"]


def test_structure_sampler_is_exported_from_ppmat():
    from ppmat.sampler import StructureSampler
    from ppmat.sampler.structure_sampler import StructureSampler as Implementation

    assert StructureSampler is Implementation


def test_structure_sampler_rejects_num_atoms_for_composition_only_model():
    from ppmat.sampler import StructureSampler

    sampler = StructureSampler.__new__(StructureSampler)
    sampler.model = type(
        "CompositionOnlyModel", (), {"supports_num_atoms_sampling": False}
    )()

    with pytest.raises(NotImplementedError, match="sample_by_chemical_formula"):
        sampler.sample_by_num_atoms(4)


def test_structure_sampler_uses_configured_sample_parameters():
    from ppmat.sampler import StructureSampler

    class Model:
        def sample(self, data, **kwargs):
            return {"result": data, "sample_params": kwargs}

    sampler = StructureSampler.__new__(StructureSampler)
    sampler.model = Model()
    sampler.sample_config = {"model_sample_params": {"num_inference_steps": 7}}
    sampler.post_transforms = None

    result = sampler.sample({"value": 1})

    assert result["sample_params"] == {"num_inference_steps": 7}


@pytest.mark.parametrize(
    "document_name",
    [
        "README.md",
        "README_PYPI.md",
        "README_zh.md",
        "README_ja.md",
        "Install.md",
        "Install_cn.md",
    ],
)
def test_user_facing_docs_use_current_one_click_inference_contract(document_name):
    document = (ROOT / document_name).read_text()

    assert "--config=" not in document
    assert "--checkpoint=" not in document
    assert "--save_path='result_diffnmr" not in document
    assert "--model_name='infgcn_qm9'" in document
    assert "--mol_file_path=" in document
    assert "electronic_structure/configs/infgcn/example/methane.mol" in document
    assert "--model_name='diffnmr_msdnmr_nless15'" in document
    assert "--model_name='sfin_haadf_enhance'" in document
    assert "--input_path='path/to/noisy_image.png'" in document


def test_homepage_language_badges_follow_project_badges():
    readme = (ROOT / "README.md").read_text()

    project_badges_end = readme.index("</p>", readme.index("PyPI version"))
    language_badges_start = readme.index('<a href="README.md">', project_badges_end)
    assert language_badges_start > project_badges_end
    assert '<a href="README_zh.md">' in readme
    assert '<a href="README_ja.md">' in readme
