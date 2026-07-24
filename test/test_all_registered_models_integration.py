"""Opt-in integration tests for every package in ``MODEL_REGISTRY``.

Run with:

    RUN_ALL_MODEL_PACKAGE_INTEGRATION=1 \
      pytest test/test_all_registered_models_integration.py -q
"""

import gc
import os
import re
from pathlib import Path

import paddle
import pytest

from ppmat.models import MODEL_REGISTRY
from ppmat.utils import download
from ppmat.utils.model_package import resolve_model_package_dir

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ALL_MODEL_PACKAGE_INTEGRATION") != "1",
    reason="set RUN_ALL_MODEL_PACKAGE_INTEGRATION=1 to test every model package",
)


@pytest.mark.parametrize("model_name", sorted(MODEL_REGISTRY))
def test_all_registered_model_packages_load(model_name):
    """Download the published package and load its configured checkpoint."""
    if model_name == "diffnmr_msdnmr_nless15":
        from ppmat.sampler import MolecularSampler

        model_or_sampler = MolecularSampler(
            model_name=model_name,
            weights_name="best.pdparams",
        )
        model = model_or_sampler.model
    else:
        from ppmat.models import build_model_from_name

        model_or_sampler, config = build_model_from_name(model_name)
        assert config.get("Model") is not None
        model = model_or_sampler

    extracted_path = download.get_weights_path_from_url(MODEL_REGISTRY[model_name])
    package_dir = Path(resolve_model_package_dir(model_name, extracted_path))
    checkpoint_path = _select_checkpoint(package_dir)
    checkpoint = paddle.load(str(checkpoint_path))
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    assert set(checkpoint) == set(model.state_dict()), (
        f"{model_name} checkpoint keys differ: "
        f"missing={sorted(set(model.state_dict()) - set(checkpoint))}, "
        f"unexpected={sorted(set(checkpoint) - set(model.state_dict()))}"
    )

    del model_or_sampler
    gc.collect()


def _select_checkpoint(package_dir: Path) -> Path:
    checkpoints = list(package_dir.rglob("*.pdparams"))
    assert checkpoints, f"No checkpoint found under {package_dir}"
    for preferred_name in ("best.pdparams", "latest.pdparams"):
        preferred = [path for path in checkpoints if path.name == preferred_name]
        if preferred:
            return sorted(preferred)[0]
    epochs = [
        (int(match.group(1)), path)
        for path in checkpoints
        if (match := re.fullmatch(r"epoch_(\d+)\.pdparams", path.name))
    ]
    if epochs:
        return max(epochs)[1]
    return sorted(checkpoints)[0]
