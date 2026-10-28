import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import paddle
import pytest

from ppmat import models
from ppmat.models.dimenetpp.dimenetpp import DimeNetPlusPlus
from ppmat.utils import download
from ppmat.utils.download import _uncompress_file_zip


def test_uncompress_single_directory_preserves_archive_root(tmp_path):
    archive_path = tmp_path / "model.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("model/", b"")
        archive.writestr("model/checkpoints/best.pdparams", b"weights")
        archive.writestr("model/model.yaml", b"Model: {}")

    extracted_path = _uncompress_file_zip(str(archive_path))

    assert Path(extracted_path) == tmp_path / "model"
    assert (tmp_path / "model" / "model" / "model.yaml").exists()


NAME = "dimenetpp_tiny"
PARAMS = {
    "out_channels": 1,
    "hidden_channels": 4,
    "num_blocks": 1,
    "int_emb_size": 2,
    "basis_emb_size": 2,
    "out_emb_channels": 4,
    "num_spherical": 2,
    "num_radial": 2,
}


def _package(freq):
    """Registered package whose weights are identified by rbf.freq."""
    model = DimeNetPlusPlus(**PARAMS)
    model.rbf.freq.set_value(paddle.full([2], freq))
    weights = io.BytesIO()
    paddle.save(model.state_dict(), weights)
    config = "Model:\n  __class_name__: DimeNetPlusPlus\n  __init_params__:\n"
    config += "".join(f"    {key}: {value}\n" for key, value in PARAMS.items())
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as files:
        files.writestr(f"{NAME}/", b"")
        files.writestr(f"{NAME}/{NAME}.yaml", config)
        files.writestr(f"{NAME}/checkpoints/best.pdparams", weights.getvalue())
    return archive.getvalue()


class _Response:
    status_code = 200

    def __init__(self, content):
        self.content = content
        self.headers = {"content-length": str(len(content))}

    def iter_content(self, chunk_size):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]


@pytest.mark.parametrize(
    "cache, downloads",
    [(None, 1), ("old", 1), ("new", 0)],
    ids=["clean_cache", "stale_cache", "current_cache"],
)
def test_registered_archive_is_downloaded_again_when_its_md5_changed(
    tmp_path, monkeypatch, cache, downloads
):
    package = _package(2.0)
    if cache is not None:
        cached = package if cache == "new" else _package(1.0)
        (tmp_path / f"{NAME}.zip").write_bytes(cached)
        _uncompress_file_zip(str(tmp_path / f"{NAME}.zip"))
    requests = []
    monkeypatch.setattr(
        download.requests,
        "get",
        lambda *args, **kwargs: requests.append(args[0]) or _Response(package),
    )
    monkeypatch.setattr(download, "WEIGHTS_HOME", str(tmp_path))
    monkeypatch.setitem(models.MODEL_REGISTRY, NAME, f"https://example.com/{NAME}.zip")
    monkeypatch.setitem(models.MODEL_MD5, NAME, hashlib.md5(package).hexdigest())

    model, _ = models.build_model_from_name(NAME, "best.pdparams")

    assert len(requests) == downloads
    np.testing.assert_array_equal(model.rbf.freq.numpy(), [2.0, 2.0])
