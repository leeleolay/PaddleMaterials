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

"""Resolve model package resources."""

import os.path as osp


def resolve_model_config_path(model_name: str, extracted_path: str) -> str:
    """Resolve a model config from a package containing config and weights.

    Args:
        model_name (str): Registry model name.
        extracted_path (str): Path after downloaded model zip.

    Returns:
        str: Path to ``<model_name>.yaml`` or ``<model_name>.yml``.

    """

    for package_dir in (extracted_path, osp.join(extracted_path, model_name)):
        for suffix in (".yaml", ".yml"):
            config_path = osp.join(package_dir, f"{model_name}{suffix}")
            if osp.isfile(config_path):
                return config_path
    return osp.join(extracted_path, model_name, f"{model_name}.yaml")
