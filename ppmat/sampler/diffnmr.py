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

import copy
import os
import os.path as osp
import time
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import paddle
import pandas as pd
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.datasets import build_dataset_infos
from ppmat.datasets import set_signal_handlers
from ppmat.metrics import DiffNMRStreamingAdapter
from ppmat.models import MODEL_REGISTRY
from ppmat.models import build_model
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils import save_load
from ppmat.utils.model_package import get_model_config_path
from ppmat.utils.model_package import resolve_model_package_dir
from ppmat.utils.visualization import MolecularVisualization
from ppmat.vocab import build_vocab


class DiffNMRSampler:
    """DiffNMR Sampler.

    This class provides an interface for sampling structures using pre-trained deep
    learning models. Supports two initialization modes:

    1. **Automatic Model Loading**
       Specify `model_name` and `weights_name` to automatically download
       and load pre-trained weights from the `MODEL_REGISTRY`.

    2. **Custom Model Loading**
       Provide explicit `config_path` and `checkpoint_path` to load
       custom-trained models from local files.

    Args:
        model_name (Optional[str], optional): Name of the pre-defined model architecture
            from the `MODEL_REGISTRY` registry. When specified, associated weights
            will be automatically downloaded. Defaults to None.

        weights_name (Optional[str], optional): Specific pre-trained weight identifier.
            Used only when `model_name` is provided. Valid options include:
            - 'best.pdparams' (highest validation performance)
            - 'latest.pdparams' (most recent training checkpoint)
            - Custom weight files ending with '.pdparams'
            Defaults to None.

        config_path (Optional[str], optional): Path to model configuration file (YAML)
            for custom models. Required when not using predefined `model_name`.
            Defaults to None.
        checkpoint_path (Optional[str], optional): Path to model checkpoint file
            (.pdparams) for custom models. Required when not using predefined
            `model_name`. Defaults to None.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        weights_name: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        config_overrides: Optional[List[str]] = None,
    ):
        package_config_dir = None
        if model_name is None:
            assert config_path is not None and checkpoint_path is not None, (
                "config_path and checkpoint_path must be provided when model_name is "
                "None."
            )

            logger.info(f"Loading model from {config_path} and {checkpoint_path}.")

            config_base_dir = os.path.dirname(os.path.abspath(config_path))
            checkpoint_dir = (
                checkpoint_path
                if checkpoint_path and os.path.isdir(checkpoint_path)
                else None
            )
            config = OmegaConf.load(config_path)
            if config_overrides:
                cli_config = OmegaConf.from_dotlist(config_overrides)
                config = OmegaConf.merge(config, cli_config)
            config = OmegaConf.to_container(config, resolve=True)
            self._resolve_package_paths(
                config,
                config_base_dir=config_base_dir,
                checkpoint_dir=checkpoint_dir,
            )
        else:
            logger.info(f"Loading registered model: {model_name}")
            extracted_path = download.get_weights_path_from_url(
                MODEL_REGISTRY[model_name]
            )
            package_config_dir = resolve_model_package_dir(model_name, extracted_path)
            config_path = get_model_config_path(model_name, package_config_dir)
            checkpoint_path = package_config_dir
            config = OmegaConf.load(config_path)
            if config_overrides:
                cli_config = OmegaConf.from_dotlist(config_overrides)
                config = OmegaConf.merge(config, cli_config)
            config = OmegaConf.to_container(config, resolve=True)
            self._resolve_package_paths(
                config,
                config_base_dir=package_config_dir,
                checkpoint_dir=None,
            )

        model_config = config.get("Model", None)
        assert model_config is not None, "Model config must be provided."
        self.vocab = build_vocab(config.get("Vocabulary"))

        set_signal_handlers()
        sample_loader = build_dataloader(config["Sampler"]["data"], vocab=self.vocab)

        # Build dataset infos without constructing full train/val/test dataloaders.
        dataset_info_config = copy.deepcopy(config)
        dataset_infos = build_dataset_infos(
            dataloaders=None,
            cfg=dataset_info_config,
            vocab=self.vocab,
            recompute_statistics=False,
        )
        # CLIP for sample metric
        model_cfg = config["CLIP"]
        self.clip = build_model(
            model_cfg,
            vocab=self.vocab,
            dataset_infos=dataset_infos,
        )

        # visualization tools
        self.visualization_tools = MolecularVisualization(
            dataset_infos=dataset_infos,
            output_dir=config["Trainer"]["output_dir"],
        )

        model_cfg = config["Model"]
        model = build_model(
            model_cfg,
            vocab=self.vocab,
            dataset_infos=dataset_infos,
            visualization_tools=self.visualization_tools,
            clip=self.clip,
        )

        self.pretrained_model_path = (
            checkpoint_path
            if checkpoint_path is not None
            else config.get("pretrained_model_path", None)
        )
        self.pretrained_weight_name = weights_name
        if self.pretrained_weight_name is None:
            self.pretrained_weight_name = config.get("pretrained_weight_name", None)
        if (
            self.pretrained_weight_name is None
            and self.pretrained_model_path is not None
            and os.path.isdir(self.pretrained_model_path)
        ):
            sampler_pretrained_path = config.get("Sampler", {}).get(
                "pretrained_model_path", None
            )
            if sampler_pretrained_path is not None:
                self.pretrained_weight_name = os.path.basename(sampler_pretrained_path)
        if self.pretrained_model_path is not None:
            save_load.load_pretrain(
                model, self.pretrained_model_path, self.pretrained_weight_name
            )

        self.model = model
        self.model.eval()
        self.config = config
        self.sample_config = config["Sampler"]

        sample_config = self.sample_config
        self._sample_loader = sample_loader
        self.samp_per_val = sample_config["sample_every_val"]
        self.visual_num = sample_config["visual_num"]
        self.chains_left_to_save = sample_config["chains_to_save"]
        self.number_chain_steps = sample_config["number_chain_steps"]
        self.sample_batch_iters = sample_config["sample_batch_iters"]
        self.metric_dict_sample = sample_config.get("out_dict", None)
        self.flag_retrieval_sampling = sample_config.get(
            "flag_retrieval_sampling", False
        )
        self.flag_use_formula = sample_config.get("flag_use_formula", False)
        self.flag_retrieval_initialization = sample_config.get(
            "flag_retrieval_initialization", False
        )
        self.num_candidates = sample_config.get("num_candidates", 1)

        # runtime info
        self.rank = (
            int(paddle.distributed.get_rank())
            if paddle.distributed.is_initialized()
            else 0
        )
        self.output_dir = self.config.get("Sampler", {}).get("output_dir", "./outputs")
        self._set_output_dir(self.output_dir)

        if self.flag_retrieval_sampling or self.flag_retrieval_initialization:
            self.molecular_vectors, self.smiles_list = self._init_retrieval_bank(
                self.sample_config,
            )
        else:
            self.molecular_vectors, self.smiles_list = None, None

        self.streaming = DiffNMRStreamingAdapter(
            t_scale=float(self.sample_config.get("t_scale", 1.0)),
            dataset_infos=dataset_infos,
            sample_metrics=self.metric_dict_sample,
        )
        self.streaming.bind(
            model=self.model,
            dataset_infos=dataset_infos,
            clip=self.clip,
            num_candidate=self.num_candidates,
        )

    @staticmethod
    def _resolve_package_paths(
        config: Dict,
        config_base_dir: Optional[str],
        checkpoint_dir: Optional[str] = None,
    ):
        def resolve_path(path):
            if path is None or osp.isabs(path) or path.startswith("http"):
                return path

            if checkpoint_dir is not None:
                candidate = osp.join(checkpoint_dir, osp.basename(path))
                if osp.exists(candidate):
                    return candidate

            if config_base_dir is not None:
                if path.startswith("./checkpoints/") or path.startswith("checkpoints/"):
                    candidate = osp.join(
                        config_base_dir, "checkpoints", osp.basename(path)
                    )
                    if osp.exists(candidate):
                        return candidate
                candidate = osp.normpath(osp.join(config_base_dir, path))
                if osp.exists(candidate):
                    return candidate

            return path

        def visit(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in {
                        "pretrained_path",
                        "pretrained_model_path",
                        "retrieval_database_path",
                        "path",
                        "datadir",
                    } and isinstance(value, str):
                        resolved_path = resolve_path(value)
                        obj[key] = resolved_path
                    else:
                        visit(value)
            elif isinstance(obj, list):
                for item in obj:
                    visit(item)

        visit(config)

        sampler_config = config.get("Sampler", {})
        retrieval_enabled = sampler_config.get(
            "flag_retrieval_sampling", False
        ) or sampler_config.get("flag_retrieval_initialization", False)
        retrieval_path = sampler_config.get("retrieval_database_path")
        if retrieval_enabled and (
            retrieval_path is None or not osp.isfile(retrieval_path)
        ):
            raise FileNotFoundError(
                "Retrieval sampling requires an existing "
                "Sampler.retrieval_database_path."
            )

    def _set_output_dir(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        if self.visualization_tools is not None:
            self.visualization_tools.result_path = osp.join(self.output_dir, "graph")

    def compute_metric(
        self,
        save_path=None,
    ):
        if save_path is not None:
            self._set_output_dir(save_path)
        return self.sample_by_dataloader(
            self.output_dir,
        )

    def sample_by_dataloader(
        self,
        save_path=None,
        data_loader=None,
    ):
        if save_path is not None:
            self._set_output_dir(save_path)
        if data_loader is None:
            data_loader = getattr(self, "_sample_loader", None)
        if data_loader is None:
            data_loader = build_dataloader(self.sample_config["data"], vocab=self.vocab)

        logger.info(f"Total iterations: {len(data_loader)}")
        logger.info("Start sampling process...")

        self.model.eval()
        epoch_id = 0

        data_length = len(data_loader)
        logger.message(f"Start to sample ... | Total Batches: {data_length}")
        start = time.time()

        # sample epoch
        metric_dict = self.sample_epoch(
            data_loader,
            epoch_id,
            keep_onehot=self.flag_retrieval_sampling,
            num_candidates=self.num_candidates,
        )

        # log eval sample metric info
        if paddle.distributed.get_rank() == 0:
            msg = "Sample:"
            msg += f" | sample_metric cost: {time.time() - start:.5f}s"
            for k, v in metric_dict.items():
                if isinstance(v, paddle.Tensor):
                    v = v.item() if v.numel() == 1 else v.tolist()
                if self.metric_dict_sample is None or k in self.metric_dict_sample:
                    msg += (
                        f" | {k}(metric): {', '.join(f'{x:.5f}' for x in v)}"
                        if isinstance(v, (list, tuple))
                        else f" | {k}(metric): {v:.5f}"
                    )
            logger.info(msg)
        return metric_dict

    @paddle.no_grad()
    def sample_epoch(
        self,
        dataloader: paddle.io.DataLoader,
        epoch_id: int,
        num_candidates: int = 1,
        keep_onehot: bool = False,
    ):
        """Sample one epoch through the model-owned batch interface."""

        self.model.eval()
        self.streaming.reset()
        core_model = (
            self.model._layers
            if isinstance(self.model, paddle.DataParallel)
            else self.model
        )
        samples: Dict[str, object] = {
            "pred": [],
            "true": [],
            "n_all": 0,
            "node_mask_meta": [],
            "batch_condition": [],
            "dict": core_model.dataset_info.atom_decoder,
        }
        if keep_onehot:
            samples["candidates"] = [[] for _ in range(num_candidates)]
            samples["candidates_X"] = [[] for _ in range(num_candidates)]
            samples["candidates_E"] = [[] for _ in range(num_candidates)]

        for iter_id, batch_data in enumerate(dataloader):
            atom_counts = paddle.to_tensor(
                batch_data["property"]["atom_count"], dtype="int64"
            ).reshape([-1])
            batch_size = int(atom_counts.shape[0])

            if core_model.conditioning_mode == "spectrum":
                spectrum = batch_data["spectrum"]
                batch_condition = [
                    paddle.to_tensor(spectrum["H_nmr"]),
                    paddle.to_tensor(spectrum["num_H_peak"]),
                    paddle.to_tensor(spectrum["C_nmr"]),
                    paddle.to_tensor(spectrum["num_C_peak"]),
                ]
            else:
                batch_condition = None

            for candidate_id in range(num_candidates):
                sample_kwargs = {
                    "batch_id": iter_id,
                    "visual_num": self.visual_num,
                    "keep_chain": self.chains_left_to_save,
                    "number_chain_steps": self.number_chain_steps,
                    "return_onehot": keep_onehot,
                    "flag_useformula": self.flag_use_formula,
                    "iter_idx": candidate_id,
                }
                if self.flag_retrieval_initialization:
                    sample_kwargs.update(
                        retrieval_initialization=True,
                        clip=self.clip,
                        molecular_vectors=self.molecular_vectors,
                        smiles_list=self.smiles_list,
                    )
                result = core_model.sample(batch_data, **sample_kwargs)

                if keep_onehot:
                    mol_pred, mol_true, X_hot, E_hot = result
                    samples["candidates"][candidate_id].extend(mol_pred)
                    samples["candidates_X"][candidate_id].extend(X_hot)
                    samples["candidates_E"][candidate_id].extend(E_hot)
                else:
                    mol_pred, mol_true = result

                if candidate_id == 0:
                    samples["pred"].extend(mol_pred)
                    samples["true"].extend(mol_true)

            if batch_condition is not None:
                if samples["batch_condition"]:
                    samples["batch_condition"] = [
                        paddle.concat([previous, current], axis=0)
                        for previous, current in zip(
                            samples["batch_condition"], batch_condition
                        )
                    ]
                else:
                    samples["batch_condition"] = batch_condition

            samples["node_mask_meta"].extend(atom_counts)
            samples["n_all"] += batch_size
            if iter_id + 1 >= self.sample_batch_iters:
                break

        self.streaming.update_step(
            result={
                "samples": samples,
                "epoch_id": epoch_id,
                "local_rank": self.rank,
                "output_dir": self.output_dir,
            },
            batch=None,
            stage="sample",
        )
        return self.streaming.compute_epoch(stage="sample")

    def _init_retrieval_bank(self, cfg):
        """
        load the molecular vector library for retrieval initialization/evaluation
        from configuration
        """
        if not cfg:
            return None, None
        path = cfg.get("retrieval_database_path", None)
        if path is None or not os.path.exists(path):
            logger.warning(f"[retrieval_bank] path missing or not found: {path}")
            return None, None

        ext = os.path.splitext(path)[1].lower()
        embs, smiles = None, None
        try:
            if ext == ".csv":
                data = pd.read_csv(path)
                data["molecularRep"] = data["molecularRep"].apply(
                    lambda x: np.fromstring(x.strip("[]"), sep=" ")
                )
                # to paddle tensor
                embs = paddle.to_tensor(
                    np.stack(data["molecularRep"].values), dtype="float32"
                )
                smiles = data["smiles"].tolist()
            else:
                raise ValueError(f"Unsupported retrieval bank ext: {ext}")
        except Exception as e:
            logger.warning(f"[retrieval_bank] load failed: {e}")
            return None, None

        return embs, smiles
