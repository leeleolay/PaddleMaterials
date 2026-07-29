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

import pytest


def test_concrete_samplers_inherit_base_sampler():
    from ppmat.sampler import BaseSampler
    from ppmat.sampler import StructureSampler
    from ppmat.sampler.diffnmr import DiffNMRSampler

    assert issubclass(StructureSampler, BaseSampler)
    assert issubclass(DiffNMRSampler, BaseSampler)


def test_molecular_sampler_cli_accepts_config_overrides():
    from spectrum_elucidation.sample import parse_args

    args, overrides = parse_args(
        [
            "--model_name",
            "diffnmr_msdnmr_nless15",
            "--output_dir",
            "generated",
            "Sampler.foo=bar",
        ]
    )

    assert args.output_dir == "generated"
    assert overrides == ["Sampler.foo=bar"]


def test_base_sampler_initializes_and_runs_shared_sampling(monkeypatch):
    import ppmat.sampler.base as base_sampler

    class Model:
        def __init__(self):
            self.eval_called = False

        def eval(self):
            self.eval_called = True

        def sample(self, data, **sample_params):
            return {"data": data, "sample_params": sample_params}

    monkeypatch.setattr(
        base_sampler,
        "build_post_transforms",
        lambda cfg: lambda data: {**data, "post_transform_cfg": cfg},
    )

    model = Model()
    config = {"Model": {}}
    sample_config = {"post_transforms": {"name": "test"}}
    sampler = base_sampler.BaseSampler(model, config, sample_config)

    result = sampler.sample({"value": 1}, sample_params={"steps": 2})

    assert model.eval_called
    assert sampler.model is model
    assert sampler.config is config
    assert sampler.sample_config is sample_config
    assert result["sample_params"] == {"steps": 2}
    assert result["post_transform_cfg"] == {"name": "test"}


def test_base_sampler_rejects_invalid_sample_params():
    from ppmat.sampler import BaseSampler

    model = type("Model", (), {"eval": lambda self: None})()
    sampler = BaseSampler(model=model, config={}, sample_config={})

    with pytest.raises(TypeError, match="dict or None"):
        sampler.sample({}, sample_params=[])
