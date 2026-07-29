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

from typing import Any
from typing import Dict
from typing import Optional

from ppmat.datasets.transform import build_post_transforms


class BaseSampler:
    """Common runtime behavior for concrete samplers.

    Model and configuration loading remain the responsibility of each concrete
    sampler. This class only initializes an already-built model and provides the
    shared sampling and post-processing flow.

    Args:
        model (Any): Model exposing ``eval`` and ``sample`` methods.
        config (Dict[str, Any]): Resolved experiment configuration.
        sample_config (Dict[str, Any]): Task-specific sampling configuration.
    """

    def __init__(
        self,
        model: Any,
        config: Dict[str, Any],
        sample_config: Dict[str, Any],
    ):
        if sample_config is None:
            raise ValueError("Sampling config must be provided.")

        self.model = model
        self.config = config
        self.sample_config = sample_config

        self.model.eval()

        self.post_transforms_cfg = self.sample_config.get("post_transforms", None)
        if self.post_transforms_cfg is not None:
            self.post_transforms = build_post_transforms(self.post_transforms_cfg)
        else:
            self.post_transforms = None

    def _get_default_sample_params(self) -> Dict[str, Any]:
        return {}

    def post_process(self, data: Any) -> Any:
        """Apply configured post-transforms to sampled data."""
        if self.post_transforms is None:
            return data
        return self.post_transforms(data)

    def sample(
        self,
        data: Any,
        sample_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Run model sampling followed by configured post-processing."""
        if sample_params is None:
            sample_params = self._get_default_sample_params()
        if not isinstance(sample_params, dict):
            raise TypeError("sample_params must be a dict or None.")
        sampled_data = self.model.sample(data, **sample_params)
        return self.post_process(sampled_data)
