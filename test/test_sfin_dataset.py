from __future__ import annotations

import copy

import numpy as np
from PIL import Image

from ppmat.datasets.build_image import BuildImage


def _write_image(path, values, *, image_format=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values, dtype=np.uint8)).save(path, format=image_format)


def _make_sfin_fixture(root):
    values = {
        "noisy": [[1, 2], [3, 4]],
        "gt_enhance": [[5, 6], [7, 8]],
        "gt_detect": [[9, 10], [11, 12]],
    }
    for subdir, image in values.items():
        _write_image(root / "train" / subdir / "0001.png", image)
    return values


def _sfin_params(root, cache_path):
    return {
        "path": str(root),
        "split": "train",
        "target_subdir": "gt_enhance",
        "build_image_cfg": {
            "format": "image_file",
            "mode": "L",
            "dtype": "float32",
            "num_cpus": 1,
        },
        "cache_path": str(cache_path),
    }


def test_build_image_reads_png_as_channel_first_float(tmp_path):
    image_path = tmp_path / "sample.png"
    expected = np.asarray([[0, 64], [128, 255]], dtype=np.float32)
    _write_image(image_path, expected)

    image = BuildImage(
        format="image_file",
        mode="L",
        dtype="float32",
        num_cpus=1,
    )(image_path)

    assert image.shape == (1, 2, 2)
    assert image.dtype == np.dtype("float32")
    np.testing.assert_array_equal(image[0], expected)


def test_build_image_reads_rgb_tiff_as_channel_first(tmp_path):
    image_path = tmp_path / "sample.tiff"
    expected = np.asarray(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=np.uint8,
    )
    _write_image(image_path, expected, image_format="TIFF")

    image = BuildImage(
        format="image_file",
        mode="RGB",
        dtype="float32",
        num_cpus=1,
    )(image_path)

    assert image.shape == (3, 2, 2)
    assert image.dtype == np.dtype("float32")
    np.testing.assert_array_equal(image, expected.transpose(2, 0, 1))


def test_sfin_dataset_builds_and_reloads_image_cache(tmp_path, monkeypatch):
    from ppmat.datasets import SFINDataset

    root = tmp_path / "sfin_fixture"
    values = _make_sfin_fixture(root)
    params = _sfin_params(root, tmp_path / "cache")

    first = SFINDataset(**params)
    sample = first[0]

    assert set(sample) == {"noisy", "gt_enhance", "gt_detect", "name", "id"}
    assert tuple(sample["noisy"].shape) == (1, 2, 2)
    np.testing.assert_array_equal(sample["noisy"].numpy()[0], values["noisy"])
    np.testing.assert_array_equal(sample["gt_enhance"].numpy()[0], values["gt_enhance"])

    calls = []
    original_build_one = BuildImage.build_one

    def fail_if_rebuilt(image_data, format, mode, dtype):
        calls.append(str(image_data))
        return original_build_one(image_data, format, mode, dtype)

    monkeypatch.setattr(BuildImage, "build_one", staticmethod(fail_if_rebuilt))
    reloaded = SFINDataset(**params)

    assert calls == []
    np.testing.assert_array_equal(reloaded[0]["noisy"].numpy(), sample["noisy"].numpy())


def test_sfin_dataset_rebuilds_when_image_config_changes(tmp_path, monkeypatch):
    from ppmat.datasets import SFINDataset

    root = tmp_path / "sfin_fixture"
    _make_sfin_fixture(root)
    params = _sfin_params(root, tmp_path / "cache")
    SFINDataset(**params)

    calls = []
    original_build_one = BuildImage.build_one

    def track_rebuild(image_data, format, mode, dtype):
        calls.append(str(image_data))
        return original_build_one(image_data, format, mode, dtype)

    monkeypatch.setattr(BuildImage, "build_one", staticmethod(track_rebuild))
    changed_params = copy.deepcopy(params)
    changed_params["build_image_cfg"]["dtype"] = "float64"
    rebuilt = SFINDataset(**changed_params)

    assert len(calls) == 3
    assert rebuilt[0]["noisy"].numpy().dtype == np.dtype("float64")


def test_sfin_dataset_invalidates_cache_when_source_image_changes(tmp_path):
    from ppmat.datasets import SFINDataset

    root = tmp_path / "sfin_fixture"
    _make_sfin_fixture(root)
    params = _sfin_params(root, tmp_path / "cache")
    SFINDataset(**params)

    changed = [[101, 102], [103, 104]]
    _write_image(root / "train" / "noisy" / "0001.png", changed)

    rebuilt = SFINDataset(**params)
    np.testing.assert_array_equal(rebuilt[0]["noisy"].numpy()[0], changed)
