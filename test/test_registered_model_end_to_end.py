"""Opt-in end-to-end tests for every registered model variant.

These tests complement package-loading checks by exercising the public task
interfaces and asserting reader-visible output artifacts.

Run with:

    RUN_REGISTERED_MODEL_E2E=1 \
      pytest test/test_registered_model_end_to_end.py -q
"""

import json
import os
from pathlib import Path

import numpy as np
import paddle
import pytest
from PIL import Image

from ppmat.models import MODEL_REGISTRY

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REGISTERED_MODEL_E2E") != "1",
    reason="set RUN_REGISTERED_MODEL_E2E=1 to run published-model workflows",
)

ROOT = Path(__file__).resolve().parents[1]

CRYSTAL_PROPERTY_MODELS = tuple(
    sorted(
        name
        for name in MODEL_REGISTRY
        if name.startswith(("comformer_", "dimenetpp_", "megnet_"))
    )
)
MOLECULAR_PROPERTY_MODELS = tuple(
    sorted(name for name in MODEL_REGISTRY if name.startswith("spherenet_qm9_"))
)
POTENTIAL_MODELS = (
    "chgnet_mptrj",
    "mattersim_1M",
    "mattersim_5M",
    *sorted(name for name in MODEL_REGISTRY if name.startswith("spherenet_md17_")),
)
STRUCTURE_MODELS = tuple(
    sorted(
        name for name in MODEL_REGISTRY if name.startswith(("diffcsp_", "mattergen_"))
    )
)
SPECTRUM_MODELS = tuple(
    sorted(name for name in MODEL_REGISTRY if name.startswith("sfin_"))
)
FIELD_MODELS = tuple(
    sorted(name for name in MODEL_REGISTRY if name.startswith("infgcn_"))
)

CONDITION_VALUES = {
    "chemical_system": "Mo-Si",
    "dft_band_gap": 0.897,
    "dft_bulk_modulus": 100.0,
    "dft_mag_density": 0.1,
    "energy_above_hull": 0.05,
    "hhi_score": 0.5,
    "ml_bulk_modulus": 100.0,
    "space_group": 225,
    "spacegroup": 225,
}


@pytest.mark.parametrize("model_name", CRYSTAL_PROPERTY_MODELS)
def test_crystal_property_prediction(model_name, tmp_path):
    from property_prediction.predict import PropertyPredictor

    output_path = tmp_path / f"{model_name}.csv"
    predictor = PropertyPredictor(model_name=model_name)
    result = predictor.from_cif_file(
        str(ROOT / "property_prediction/example_data/cifs/mp-18767-LiMnO2.cif"),
        str(output_path),
    )

    assert result
    assert output_path.is_file()


@pytest.mark.parametrize("model_name", MOLECULAR_PROPERTY_MODELS)
def test_molecular_property_prediction(model_name, tmp_path):
    from property_prediction.predict import PropertyPredictor

    output_path = tmp_path / f"{model_name}.csv"
    predictor = PropertyPredictor(model_name=model_name)
    result = predictor.from_xyz_file(
        str(ROOT / "property_prediction/example_data/molecules/isoguvacine.xyz"),
        str(output_path),
    )

    assert result
    assert output_path.is_file()


@pytest.mark.parametrize("model_name", POTENTIAL_MODELS)
def test_interatomic_potential_prediction(model_name, tmp_path):
    from interatomic_potentials.predict import PotentialPredictor

    output_path = tmp_path / f"{model_name}.csv"
    predictor = PotentialPredictor(model_name=model_name)
    if model_name.startswith("spherenet_md17_"):
        result = predictor.from_xyz_file(
            str(ROOT / "interatomic_potentials/example_data/xyz/md17_aspirin.xyz"),
            str(output_path),
        )
    else:
        result = predictor.from_cif_file(
            str(ROOT / "interatomic_potentials/example_data/cifs/mp-18767-LiMnO2.cif"),
            str(output_path),
        )

    assert result
    assert output_path.is_file()


@pytest.mark.parametrize("model_name", STRUCTURE_MODELS)
def test_structure_generation(model_name, tmp_path):
    from ppmat.sampler import StructureSampler

    paddle.seed(42)
    output_dir = tmp_path / model_name
    sampler = StructureSampler(model_name=model_name)
    sample_params = {"num_inference_steps": 2}

    if model_name == "diffcsp_mp20":
        result = sampler.sample_by_chemical_formula(
            "LiMnO2",
            save_path=str(output_dir),
            sample_params=sample_params,
        )
    elif getattr(sampler.model, "condition_names", None):
        conditions = {
            name: CONDITION_VALUES[name] for name in sampler.model.condition_names
        }
        result = sampler.sample_by_condition(
            4,
            conditions,
            save_path=str(output_dir),
            sample_params=sample_params,
        )
    else:
        result = sampler.sample_by_num_atoms(
            4,
            save_path=str(output_dir),
            sample_params=sample_params,
        )

    assert result["result"]
    assert list(output_dir.glob("*.cif"))


@pytest.mark.parametrize("model_name", SPECTRUM_MODELS)
def test_spectrum_enhancement_prediction(model_name, tmp_path):
    from spectrum_enhancement.predict import SpectrumPredictor

    input_path = tmp_path / "input.png"
    Image.fromarray(np.arange(4096, dtype=np.uint8).reshape(64, 64)).save(input_path)
    output_dir = tmp_path / model_name

    predictor = SpectrumPredictor(model_name=model_name)
    saved_paths = predictor.from_image_path(
        str(input_path),
        str(output_dir),
    )

    assert len(saved_paths) == 1
    assert Path(saved_paths[0]).is_file()


@pytest.mark.parametrize("model_name", FIELD_MODELS)
def test_electronic_structure_prediction(model_name, tmp_path):
    from electronic_structure.predict import build_parser
    from ppmat.predictor import FieldPredictor

    if model_name.startswith("infgcn_md17_"):
        atom_file = tmp_path / "md17_atoms.json"
        atom_file.write_text(
            json.dumps(
                [
                    {"name": "C", "atom_num": 6},
                    {"name": "H", "atom_num": 1},
                    {"name": "O", "atom_num": 8},
                ]
            )
        )
    elif model_name == "infgcn_qm9":
        atom_file = ROOT / "electronic_structure/configs/qm9.json"
    else:
        atom_file = ROOT / "electronic_structure/configs/crystal.json"

    output_dir = tmp_path / model_name
    args = build_parser().parse_args(
        [
            "--model_name",
            model_name,
            "--weights_name",
            "best.pdparams",
            "--mol_input",
            str(ROOT / "electronic_structure/configs/infgcn/example/methane.mol"),
            "--atom_file",
            str(atom_file),
            "--mol_grid_shape",
            "8",
            "--grid_batch_size",
            "128",
            "--output_dir",
            str(output_dir),
            "--cube_dir",
            str(output_dir),
            "--save_pred_cube",
            "--skip_vis",
        ]
    )
    predictor = FieldPredictor(
        model_name=args.model_name,
        weights_name=args.weights_name,
    )
    predictor.predict(args)

    assert (output_dir / "methane_pred.cube").is_file()
