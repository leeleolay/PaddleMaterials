"""Opt-in tests for the large model packages published in MODEL_REGISTRY.

Run after uploading release archives:

    RUN_MODEL_PACKAGE_INTEGRATION=1 \
      pytest test/test_registered_model_packages_integration.py -q
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import paddle
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_PACKAGE_INTEGRATION") != "1",
    reason="set RUN_MODEL_PACKAGE_INTEGRATION=1 to download published packages",
)

INFGCN_MODEL_NAMES = (
    "infgcn_md17_benzene",
    "infgcn_md17_ethane",
    "infgcn_md17_ethanol",
    "infgcn_md17_malonaldehyde",
    "infgcn_md17_phenol",
    "infgcn_md17_resorcinol",
    "infgcn_mp",
    "infgcn_omol25_mc_5k_trimmed",
    "infgcn_qm9",
)


def _assert_raw_state_dict(checkpoint_path, model):
    checkpoint = paddle.load(str(checkpoint_path))
    assert isinstance(checkpoint, dict)
    assert "model" not in checkpoint
    assert "state_dict" not in checkpoint

    model_state = model.state_dict()
    assert set(checkpoint) == set(model_state)
    for name, expected in checkpoint.items():
        np.testing.assert_allclose(
            model_state[name].numpy(),
            expected.numpy(),
            rtol=0,
            atol=0,
            err_msg=name,
        )


@pytest.mark.parametrize("model_name", INFGCN_MODEL_NAMES)
def test_published_infgcn_packages_load_exact_weights(model_name):
    from ppmat.models import MODEL_REGISTRY
    from ppmat.models import build_model_from_name
    from ppmat.utils import download
    from ppmat.utils.resource import resolve_model_config_path

    extracted_path = download.get_weights_path_from_url(MODEL_REGISTRY[model_name])
    config_path = Path(resolve_model_config_path(model_name, extracted_path))
    package_dir = config_path.parent
    checkpoint_path = package_dir / "checkpoints" / "best.pdparams"
    assert checkpoint_path.is_file()

    model, config = build_model_from_name(model_name, "best.pdparams")

    assert config["Model"]["__class_name__"] == "InfGCN"
    _assert_raw_state_dict(checkpoint_path, model)


def test_published_infgcn_qm9_runs_documented_inference(tmp_path):
    model_name = "infgcn_qm9"
    output_dir = tmp_path / "infgcn"
    command = [
        sys.executable,
        "electronic_structure/predict.py",
        "--model_name",
        model_name,
        "--weights_name",
        "best.pdparams",
        "--mol_file_path",
        "electronic_structure/configs/infgcn/example/methane.mol",
        "--grid_shape",
        "8",
        "--grid_batch_size",
        "128",
        "--save_path",
        str(output_dir),
    ]
    subprocess.run(command, check=True)
    assert (output_dir / "methane_pred.cube").is_file()


def test_published_diffnmr_package_loads_and_samples(tmp_path):
    from ppmat.models import MODEL_REGISTRY
    from ppmat.sampler import MolecularSampler
    from ppmat.utils import download
    from ppmat.utils.resource import resolve_model_config_path

    model_name = "diffnmr_msdnmr_nless15"
    extracted_path = download.get_weights_path_from_url(MODEL_REGISTRY[model_name])
    config_path = Path(resolve_model_config_path(model_name, extracted_path))
    package_dir = config_path.parent
    checkpoint_path = package_dir / "checkpoints" / "best.pdparams"
    assert checkpoint_path.is_file()

    sampler = MolecularSampler(
        model_name=model_name,
        weights_name="best.pdparams",
    )
    _assert_raw_state_dict(checkpoint_path, sampler.model)

    result = sampler.sample_by_dataloader(save_path=str(tmp_path / "sample"))
    assert result["Total Number"] == 1
