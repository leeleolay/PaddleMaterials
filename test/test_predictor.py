import importlib
from pathlib import Path

import paddle
import pytest

from ppmat.predictor import BasePredictor


class _Model:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True


@pytest.mark.parametrize(
    ("module_name", "argv"),
    [
        ("electronic_structure.predict", ["--model_name", "infgcn_qm9"]),
        (
            "interatomic_potentials.predict",
            [
                "--model_name",
                "chgnet_mptrj",
                "--cif_file_path",
                "sample.cif",
            ],
        ),
        (
            "property_prediction.predict",
            [
                "--model_name",
                "megnet_mp2018_train_60k_e_form",
                "--cif_file_path",
                "sample.cif",
            ],
        ),
        (
            "spectrum_enhancement.predict",
            ["--model_name", "sfin_haadf_enhance"],
        ),
    ],
)
def test_task_predictor_cli_accepts_injected_argv(module_name, argv):
    parse_args = importlib.import_module(module_name).parse_args

    args, overrides = parse_args([*argv, "Predict.foo=bar"])

    assert args.model_name == argv[1]
    assert overrides == ["Predict.foo=bar"]


def test_base_predictor_uses_checkpoint_from_predict_config(tmp_path, monkeypatch):
    from ppmat.predictor import base

    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        """
Model:
  __class_name__: Dummy
Predict:
  checkpoint_path: https://example.com/model.zip
""".strip()
    )
    model = _Model()
    loaded = {}
    monkeypatch.setattr(base, "build_model", lambda config, **kwargs: model)
    monkeypatch.setattr(
        base.save_load,
        "load_pretrain",
        lambda target, path: loaded.update(model=target, path=path),
    )

    predictor = BasePredictor(config_path=str(config_path))
    predictor.load_inference_model()

    assert predictor.model is model
    assert model.eval_called
    assert predictor.device == paddle.get_device()
    assert loaded == {
        "model": model,
        "path": "https://example.com/model.zip",
    }


def test_base_predictor_accepts_pathlike_model_paths():
    predictor = BasePredictor(
        config_path=Path("configs/model.yaml"),
        checkpoint_path=Path("checkpoints/best.pdparams"),
        work_dir="workspace",
    )

    assert predictor.config_path == "workspace/configs/model.yaml"
    assert predictor.checkpoint_path == "workspace/checkpoints/best.pdparams"


def test_registered_predictor_rejects_model_overrides_before_loading(monkeypatch):
    from ppmat.predictor import base

    monkeypatch.setattr(
        base,
        "build_model_from_name",
        lambda *args, **kwargs: pytest.fail("model should not be loaded"),
    )
    predictor = BasePredictor(
        model_name="dummy",
        config_overrides=["Model.__init_params__.cutoff=3.0"],
    )

    with pytest.raises(ValueError, match="Registered-model overrides"):
        predictor.load_inference_model()


def test_registered_predictor_rejects_silent_global_overrides(monkeypatch):
    from ppmat.predictor import base

    monkeypatch.setattr(
        base,
        "build_model_from_name",
        lambda *args, **kwargs: pytest.fail("model should not be loaded"),
    )
    predictor = BasePredictor(
        model_name="dummy",
        config_overrides=["Global.do_test=True"],
    )

    with pytest.raises(ValueError, match="Global.do_test"):
        predictor.load_inference_model()


def test_registered_interface_modifies_config_before_model_construction(monkeypatch):
    from ppmat.predictor import base

    captured = {}

    def build_registered_model(
        model_name,
        weights_name,
        model_config_modifier=None,
    ):
        model_config = {
            "__class_name__": "CHGNet",
            "__init_params__": {"is_intensive": True},
        }
        model_config = model_config_modifier(model_config)
        captured["is_intensive_at_construction"] = model_config["__init_params__"][
            "is_intensive"
        ]
        return _Model(), {"Model": model_config, "Predict": {}}

    monkeypatch.setattr(base, "build_model_from_name", build_registered_model)
    predictor = BasePredictor(model_name="chgnet_mptrj")

    predictor.load_inference_model(interface_type="ase")

    assert captured["is_intensive_at_construction"] is False


@pytest.mark.parametrize("eval_with_no_grad", [True, False])
def test_base_predictor_run_model_respects_gradient_setting(eval_with_no_grad):
    class Model:
        def predict(self, data):
            return {"grad_enabled": paddle.is_grad_enabled(), "data": data}

    predictor = BasePredictor()
    predictor.model = Model()
    predictor.eval_with_no_grad = eval_with_no_grad
    predictor.post_transforms = None

    result = predictor._run_model("sample")

    assert result["grad_enabled"] is not eval_with_no_grad
    assert result["data"] == "sample"


def test_task_predictors_are_public_and_task_scripts_remain_thin():
    from ppmat.predictor import FieldPredictor
    from ppmat.predictor import PotentialPredictor
    from ppmat.predictor import PropertyPredictor
    from ppmat.predictor import SpectrumPredictor

    for predictor_class in (
        FieldPredictor,
        PotentialPredictor,
        PropertyPredictor,
        SpectrumPredictor,
    ):
        assert issubclass(predictor_class, BasePredictor)

    root = Path(__file__).resolve().parents[1]
    for script_path in (
        "electronic_structure/predict.py",
        "interatomic_potentials/predict.py",
        "property_prediction/predict.py",
        "spectrum_enhancement/predict.py",
    ):
        source = (root / script_path).read_text()
        assert "class " not in source
        assert "from ppmat.predictor import " in source
        assert '"--device"' in source
        assert '"--save_path"' in source

    for script_path in (
        "electronic_structure/predict.py",
        "interatomic_potentials/predict.py",
        "property_prediction/predict.py",
        "spectrum_enhancement/predict.py",
        "spectrum_elucidation/sample.py",
    ):
        source = (root / script_path).read_text()
        assert "def build_parser" not in source
        assert "def parse_args(argv=None):" in source


def test_spectrum_predictor_stages_images_with_dataset_suffix(tmp_path):
    from PIL import Image

    from ppmat.predictor import SpectrumPredictor

    input_path = tmp_path / "input.jpg"
    Image.new("L", (4, 4), color=128).save(input_path)
    predictor = SpectrumPredictor.__new__(SpectrumPredictor)
    predictor.config = {
        "Dataset": {
            "test": {
                "dataset": {
                    "__init_params__": {
                        "path": "unused",
                        "split": "test",
                        "noisy_subdir": "noisy",
                        "file_suffix": ".png",
                    }
                }
            }
        }
    }

    with predictor._dataset_cfg_from_input_path(str(input_path)) as dataset_config:
        params = dataset_config["dataset"]["__init_params__"]
        staged_path = Path(params["path"]) / "test" / "noisy" / "input.png"
        assert staged_path.is_file()
