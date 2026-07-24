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

import argparse
from typing import Sequence


def add_model_loading_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the common registered/local model loading arguments."""
    parser.add_argument(
        "--model_name",
        default=None,
        help="Registered model name.",
    )
    parser.add_argument(
        "--weights_name",
        default=None,
        help="Optional weight filename in a model package or checkpoint directory.",
    )
    parser.add_argument(
        "--config_path",
        default=None,
        help="Path to a local configuration file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Path to a local checkpoint file or directory.",
    )


def validate_model_loading_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Require exactly one complete registered or local loading mode."""
    uses_registered_model = args.model_name is not None
    uses_local_model = args.config_path is not None or args.checkpoint_path is not None

    if uses_registered_model and uses_local_model:
        parser.error(
            "--model_name cannot be combined with --config_path or --checkpoint_path"
        )
    if not uses_registered_model and not uses_local_model:
        parser.error(
            "provide --model_name, or both --config_path and --checkpoint_path"
        )
    if uses_local_model and (args.config_path is None or args.checkpoint_path is None):
        parser.error("--config_path and --checkpoint_path must be provided together")


def validate_config_overrides(
    parser: argparse.ArgumentParser, config_overrides: Sequence[str]
) -> None:
    """Reject unknown options while allowing OmegaConf ``key=value`` overrides."""
    invalid_overrides = [
        value for value in config_overrides if value.startswith("-") or "=" not in value
    ]
    if invalid_overrides:
        parser.error("unrecognized arguments: " + " ".join(invalid_overrides))
