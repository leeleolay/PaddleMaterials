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

import argparse

from ppmat.sampler import MolecularSampler
from ppmat.utils import logger
from ppmat.utils.inference_cli import add_model_loading_arguments
from ppmat.utils.inference_cli import validate_config_overrides
from ppmat.utils.inference_cli import validate_model_loading_arguments


def build_parser():
    parser = argparse.ArgumentParser(description="Molecular structure sampling")
    add_model_loading_arguments(parser)
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Directory in which sampling results are saved.",
    )
    parser.add_argument(
        "--mode",
        choices=["by_dataloader", "compute_metric"],
        default="by_dataloader",
    )
    return parser


def main():
    parser = build_parser()
    args, config_overrides = parser.parse_known_args()
    validate_model_loading_arguments(parser, args)
    validate_config_overrides(parser, config_overrides)

    sampler = MolecularSampler(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        config_overrides=config_overrides,
    )
    if args.mode == "compute_metric":
        metric_result = sampler.compute_metric(save_path=args.output_dir)
        for metric_name, metric_value in metric_result.items():
            logger.info(f"{metric_name}: {metric_value}")
    else:
        sampler.sample_by_dataloader(save_path=args.output_dir)


if __name__ == "__main__":
    main()
