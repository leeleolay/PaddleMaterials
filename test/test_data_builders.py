from __future__ import annotations

import pytest

from ppmat.datasets.build_field import BuildField
from ppmat.datasets.build_image import BuildImage
from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.datasets.build_spectrum import BuildSpectrumNMR
from ppmat.datasets.build_structure import BuildStructure


@pytest.mark.parametrize(
    ("builder_type", "config"),
    [
        (
            BuildStructure,
            {
                "format": "array",
                "primitive": False,
                "niggli": False,
                "canocial": False,
                "num_cpus": 1,
            },
        ),
        (
            BuildMolecule,
            {
                "format": "smiles",
                "sanitize": False,
                "add_hs": False,
                "remove_hs": False,
                "kekulize": False,
                "num_cpus": 1,
            },
        ),
        (
            BuildImage,
            {
                "format": "image_file",
                "mode": "L",
                "dtype": "float32",
                "num_cpus": 1,
            },
        ),
        (
            BuildField,
            {
                "format": "array",
                "name": "density",
                "value_unit": "unknown",
                "coordinate_unit": "angstrom",
                "num_cpus": 1,
            },
        ),
    ],
)
def test_data_builders_accept_flat_configs(builder_type, config):
    builder = builder_type(**config)

    assert isinstance(builder, builder_type)
    assert builder.num_cpus == 1


def test_structure_builder_builds_one_array_sample():
    builder = BuildStructure(
        format="array",
        primitive=False,
        niggli=False,
        canocial=False,
        num_cpus=1,
    )

    structure = builder(
        {
            "frac_coords": [[0.0, 0.0, 0.0]],
            "atom_types": ["Si"],
            "lengths": [3.0, 3.0, 3.0],
            "angles": [90.0, 90.0, 90.0],
        }
    )

    assert len(structure) == 1
    assert structure[0].specie.symbol == "Si"


def test_spectrum_builder_accepts_flat_config_and_runtime_vocabulary():
    vocab = {
        "peakwidth": {"token_to_id": {"<unk>": 0}},
        "split": {"token_to_id": {"<unk>": 0}},
        "integral": {"token_to_id": {"<unk>": 0}},
    }
    config = {
        "seq_len_H1": 2,
        "seq_len_C13": 3,
        "j_len": 6,
        "dtype": "float32",
        "num_cpus": 1,
    }

    builder = BuildSpectrumNMR(vocab=vocab, **config)

    assert builder.seq_len_H1 == 2
    assert builder.seq_len_C13 == 3
    assert builder.num_cpus == 1


@pytest.mark.parametrize(
    ("builder_type", "config"),
    [
        (
            BuildStructure,
            {
                "__class_name__": "BuildStructure",
                "__init_params__": {"format": "array"},
            },
        ),
        (
            BuildMolecule,
            {
                "__class_name__": "BuildMolecule",
                "__init_params__": {"format": "smiles"},
            },
        ),
        (
            BuildImage,
            {
                "__class_name__": "BuildImage",
                "__init_params__": {"format": "image_file"},
            },
        ),
        (
            BuildField,
            {
                "__class_name__": "BuildField",
                "__init_params__": {
                    "format": "array",
                    "name": "density",
                    "value_unit": "unknown",
                    "coordinate_unit": "angstrom",
                },
            },
        ),
    ],
)
def test_data_builders_do_not_accept_registry_configs(builder_type, config):
    with pytest.raises(TypeError, match="__class_name__"):
        builder_type(**config)


def _write_image(path, values):
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values, dtype=np.uint8)).save(path)


def test_image_builder_returns_channel_first_float_array(tmp_path):
    import numpy as np

    image_path = tmp_path / "sample.png"
    _write_image(image_path, [[0, 64], [128, 255]])

    image = BuildImage(
        format="image_file",
        mode="L",
        dtype="float32",
        num_cpus=1,
    )(image_path)

    assert image.shape == (1, 2, 2)
    assert image.dtype == np.float32
    np.testing.assert_array_equal(image[0], [[0.0, 64.0], [128.0, 255.0]])


def test_sfin_dataset_builds_and_reloads_image_cache(tmp_path, monkeypatch):
    import numpy as np
    import paddle

    from ppmat.datasets import SFINDataset

    root = tmp_path / "sfin_fixture"
    values = {
        "noisy": [[1, 2], [3, 4]],
        "gt_enhance": [[5, 6], [7, 8]],
        "gt_detect": [[9, 10], [11, 12]],
    }
    for subdir, image in values.items():
        _write_image(root / "train" / subdir / "0001.png", image)

    cache_path = tmp_path / "cache"
    params = {
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
    dataset = SFINDataset(**params)
    sample = dataset[0]

    assert set(sample) == {
        "noisy",
        "gt_enhance",
        "gt_detect",
        "name",
        "id",
    }
    assert isinstance(sample["noisy"], paddle.Tensor)
    assert list(sample["noisy"].shape) == [1, 2, 2]
    np.testing.assert_array_equal(sample["gt_enhance"].numpy()[0], values["gt_enhance"])

    def fail_if_rebuilt(*args, **kwargs):
        raise AssertionError("a valid SFIN image cache must not be rebuilt")

    monkeypatch.setattr(BuildImage, "build_one", staticmethod(fail_if_rebuilt))
    reloaded = SFINDataset(**params)
    np.testing.assert_array_equal(reloaded[0]["noisy"].numpy(), sample["noisy"].numpy())
