import datetime
import re
from pathlib import Path

from omegaconf import OmegaConf

from ppmat.trainer import utils
from ppmat.trainer.utils import append_timestamp_to_output_dir


def test_append_timestamp_to_output_dir_uses_seed_and_timestamp():
    config = OmegaConf.create({"Trainer": {"output_dir": "./output/demo", "seed": 7}})

    append_timestamp_to_output_dir(
        config,
        now=datetime.datetime(2026, 6, 29, 12, 34, 56),
    )

    assert config["Trainer"]["output_dir"] == "./output/demo_t_20260629_123456_s_7"


def test_append_timestamp_to_output_dir_uses_rank_zero_timestamp(monkeypatch):
    config = OmegaConf.create({"Trainer": {"output_dir": "./output/demo", "seed": 7}})

    monkeypatch.setattr(utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(utils.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(utils.dist, "get_rank", lambda: 1)

    def broadcast_timestamp(timestamp_list, src):
        assert timestamp_list == [None]
        assert src == 0
        timestamp_list[0] = "20260629_123456"

    monkeypatch.setattr(utils.dist, "broadcast_object_list", broadcast_timestamp)

    append_timestamp_to_output_dir(
        config,
        now=datetime.datetime(2026, 6, 30, 1, 2, 3),
    )

    assert config["Trainer"]["output_dir"] == "./output/demo_t_20260629_123456_s_7"


def test_output_dir_helper_lives_in_trainer_utils():
    assert "append_timestamp_to_output_dir" not in Path("ppmat/utils/io.py").read_text()


def test_train_entrypoints_append_timestamp_by_default():
    train_scripts = [
        "property_prediction/train.py",
        "electronic_structure/train.py",
        "structure_generation/train.py",
        "spectrum_elucidation/train.py",
        "interatomic_potentials/train.py",
        "spectrum_enhancement/train.py",
    ]

    for script in train_scripts:
        source = Path(script).read_text()
        assert "def read_independent_dataloader_config(config" in source
        assert "def parse_args():" in source
        assert "return parser.parse_known_args()" in source
        assert re.search(r"append_timestamp_to_output_dir\((config|cfg)\)", source)
        assert "--append_timestamp" not in source
