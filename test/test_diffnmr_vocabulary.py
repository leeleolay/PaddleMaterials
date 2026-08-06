import copy
import csv
import inspect
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import paddle
from omegaconf import OmegaConf

from ppmat.datasets.build_spectrum import BuildSpectrumNMR
from ppmat.datasets.msd_nmr_dataset import MSDnmrDataset
from ppmat.models.diffnmr.nmr_encoder import H1nmr_embedding
from ppmat.vocab import build_vocab

ROOT = Path(__file__).resolve().parents[1]
DIFFNMR_CONFIG_DIR = ROOT / "spectrum_elucidation" / "configs" / "diffnmr"
DIFFNMR_CONFIG_NAMES = (
    "DiffNMR.yaml",
    "DiffNMR_DiffGraphFormer.yaml",
    "DiffNMR_NMRNet.yaml",
    "PP-DiffNMR.yaml",
    "PP-DiffNMR_DiffPrior.yaml",
)


def _token_vocab(tokens):
    return {
        "type": "token",
        "tokens": list(tokens),
        "num_embeddings": len(tokens),
        "token_to_id": {token: index for index, token in enumerate(tokens)},
        "id_to_token": dict(enumerate(tokens)),
    }


def _diffnmr_vocab():
    atom_tokens = ["C", "N"]
    atom_vocab = _token_vocab(atom_tokens)
    atom_vocab["type"] = "element"
    return {
        "atom": atom_vocab,
        "bond": _token_vocab(["NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]),
        "peakwidth": _token_vocab(["<pad>", "<unk>", "0.1"]),
        "split": _token_vocab(["<pad>", "<unk>", "s"]),
        "integral": _token_vocab(
            ["<pad>", "<unk>", *[f"{value}H" for value in range(24)]]
        ),
    }


def _write_sample_csv(path):
    spectrum = {
        "1HNMR": [[1.0, "0.1", "s", "0H", []]],
        "13CNMR": [20.0],
    }
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=["smiles", "tokenized_input", "atom_count"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "smiles": "CN",
                "tokenized_input": json.dumps(spectrum),
                "atom_count": 2,
            }
        )


def _dataset(path, cache_path, vocab):
    return MSDnmrDataset(
        path=str(path),
        data_flag="n<15",
        max_atoms=15,
        vocab=vocab,
        build_molecule_cfg={
            "format": "smiles",
            "sanitize": True,
            "add_hs": False,
            "remove_hs": False,
            "kekulize": False,
            "num_cpus": 1,
        },
        build_graph_cfg={
            "__class_name__": "MolecularGraphConverter",
            "__init_params__": {
                "remove_h": True,
                "add_self_loops": False,
                "edge_mode": "bidirectional",
                "num_cpus": 1,
            },
        },
        build_spectrum_cfg={
            "seq_len_H1": 2,
            "seq_len_C13": 2,
            "j_len": 6,
            "dtype": "float32",
            "num_cpus": 1,
        },
        cache_path=str(cache_path),
        filter_unvalid=False,
    )


def _count_key(value, expected_key):
    if isinstance(value, dict):
        return sum(
            int(key == expected_key) + _count_key(item, expected_key)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return sum(_count_key(item, expected_key) for item in value)
    return 0


def test_registered_diffnmr_integral_ids_preserve_checkpoint_lookup():
    vocab = build_vocab("diffnmr_msdnmr_nless15")
    integral = vocab["integral"]

    assert integral["num_embeddings"] == 26
    assert integral["token_to_id"]["<pad>"] == 0
    assert integral["token_to_id"]["<unk>"] == 1
    assert integral["token_to_id"]["0H"] == 2
    assert integral["token_to_id"]["23H"] == 25
    expected_lookup_ids = [(value + 1) + 1 for value in range(24)]
    assert [
        integral["token_to_id"][f"{value}H"] for value in range(24)
    ] == expected_lookup_ids

    builder = BuildSpectrumNMR(
        vocab=vocab,
        seq_len_H1=25,
        seq_len_C13=2,
        num_cpus=1,
    )
    converted = builder(
        {
            "1HNMR": [
                *[[float(value), "0.0", "s", f"{value}H", []] for value in range(24)],
                [24.0, "0.0", "s", "invalid", []],
            ],
            "13CNMR": [],
        }
    )

    np.testing.assert_array_equal(
        converted["H_nmr"][:, 3],
        [*expected_lookup_ids, 1],
    )


def test_h1_embedding_consumes_integral_vocabulary_id_directly():
    signature = inspect.signature(H1nmr_embedding.__init__)
    for parameter_name in (
        "peakwidthemb_num",
        "splitemb_num",
        "integralemb_num",
    ):
        assert signature.parameters[parameter_name].default is inspect.Parameter.empty

    embedding = H1nmr_embedding(
        split_dim=2,
        peakwidth_dim=2,
        integral_dim=2,
        H_shift_min=-1,
        H_shift_max=1,
        H_shift_bin=3,
        j_bins1=3,
        j_bins2=3,
        dim=4,
        drop_prob=0.0,
        peakwidthemb_num=2,
        splitemb_num=2,
        integralemb_num=26,
    )
    h1nmr = np.zeros((1, 1, 10), dtype=np.float32)
    h1nmr[:, :, 3] = 25

    output = embedding(
        paddle.to_tensor(h1nmr),
        paddle.ones([1, 1], dtype="float32"),
    )

    assert list(output.shape) == [1, 1, 4]


def test_diffnmr_configs_keep_sequence_shape_in_builder_only():
    for config_name in DIFFNMR_CONFIG_NAMES:
        config = OmegaConf.to_container(
            OmegaConf.load(DIFFNMR_CONFIG_DIR / config_name),
            resolve=False,
        )
        builder_cfg = config["Global"]["build_spectrum_cfg"]

        assert builder_cfg["seq_len_H1"] == 20
        assert builder_cfg["seq_len_C13"] == 75
        assert "integral_offset" not in builder_cfg
        assert "unk_token" not in builder_cfg
        assert _count_key(config, "seq_len_H1") == 1
        assert _count_key(config, "seq_len_C13") == 1
        assert _count_key(config, "integralemb_num") == 0


def test_diffnmr_models_derive_spectrum_runtime_values(monkeypatch):
    from ppmat.models.diffnmr import diffnmr as diffnmr_module

    class DummyLayer(paddle.nn.Layer):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.args = args
            self.kwargs = kwargs

    class DummyMetric:
        pass

    input_dims = {"X": 2, "E": 2, "y": 1}
    output_dims = {"X": 2, "E": 2, "y": 1}
    monkeypatch.setattr(
        diffnmr_module,
        "_build_graph_features",
        lambda dataset_infos, diffmodel_cfg: (
            None,
            None,
            input_dims,
            output_dims,
        ),
    )
    monkeypatch.setattr(diffnmr_module, "MolecularEncoder", DummyLayer)
    monkeypatch.setattr(diffnmr_module, "NMR_encoder", DummyLayer)
    monkeypatch.setattr(diffnmr_module, "GraphTransformer", DummyLayer)
    monkeypatch.setattr(diffnmr_module, "PredefinedNoiseScheduleDiscrete", DummyLayer)
    monkeypatch.setattr(diffnmr_module, "DiscreteUniformTransition", DummyLayer)
    monkeypatch.setattr(diffnmr_module, "TrainLossDiscrete", DummyLayer)
    for metric_name in (
        "NLL",
        "SumExceptBatchKL",
        "SumExceptBatchMetric",
    ):
        monkeypatch.setattr(diffnmr_module, metric_name, DummyMetric)
    monkeypatch.setattr(diffnmr_module.paddle, "load", lambda path: {})

    vocab = _diffnmr_vocab()
    dataset_infos = SimpleNamespace(
        seq_len_H1=7,
        seq_len_C13=11,
        nodes_dist=None,
    )
    spectrum_encoder = {
        "pretrained_model_path": None,
        "dim_enc_H": 8,
        "dimff_enc_H": 16,
        "dim_enc_C": 4,
        "dimff_enc_C": 8,
        "ffn_hidden": 4,
        "n_head": 1,
        "n_layers": 1,
        "drop_prob": 0.0,
    }
    graph_encoder = {
        "pretrained_model_path": None,
        "n_layers_GT": 1,
        "hidden_mlp_dims": {},
        "hidden_dims": {},
    }

    clip = diffnmr_module.NMRNetCLIP(
        graph_encoder=graph_encoder,
        spectrum_encoder=spectrum_encoder,
        vocab=vocab,
        dataset_infos=dataset_infos,
    )
    assert clip.spectrum_encoder.kwargs["peakwidthemb_num"] == 3
    assert clip.spectrum_encoder.kwargs["splitemb_num"] == 3
    assert clip.spectrum_encoder.kwargs["integralemb_num"] == 26
    assert clip.seq_len_H1 == 7
    assert clip.seq_len_C13 == 11

    model = diffnmr_module.DiffNMR(
        encoder_cfg={
            **spectrum_encoder,
            "pretrained_path": "encoder.pdparams",
            "onlyH": False,
        },
        decoder_cfg={
            "pretrained_path": "decoder.pdparams",
            "num_layers": 1,
            "hidden_mlp_dims": {},
            "hidden_dims": {},
        },
        diffmodel_cfg={
            "diffusion_steps": 1,
            "diffusion_noise_schedule": "cosine",
            "transition": "uniform",
            "lambda_train": [1, 1],
        },
        dataset_infos=dataset_infos,
        vocab=vocab,
    )
    assert model.encoder.kwargs["peakwidthemb_num"] == 3
    assert model.encoder.kwargs["splitemb_num"] == 3
    assert model.encoder.kwargs["integralemb_num"] == 26
    assert model.seq_len_H1 == 7
    assert model.seq_len_C13 == 11


def test_msdnmr_graph_cache_invalidates_when_atom_vocab_changes(tmp_path):
    csv_path = tmp_path / "sample.csv"
    cache_path = tmp_path / "cache"
    _write_sample_csv(csv_path)
    vocab = _diffnmr_vocab()

    first = _dataset(csv_path, cache_path, vocab)
    first_atom_ids = np.argmax(first[0]["graph"].node_feat["feat"], axis=1)
    np.testing.assert_array_equal(first_atom_ids, [0, 1])

    changed_vocab = copy.deepcopy(vocab)
    changed_vocab["atom"]["token_to_id"] = {"C": 1, "N": 0}
    changed_vocab["atom"]["id_to_token"] = {0: "N", 1: "C"}
    second = _dataset(csv_path, cache_path, changed_vocab)
    second_atom_ids = np.argmax(second[0]["graph"].node_feat["feat"], axis=1)

    np.testing.assert_array_equal(second_atom_ids, [1, 0])
    with (cache_path / "graph_vocab.pkl").open("rb") as file_obj:
        cached_vocab = pickle.load(file_obj)
    assert set(cached_vocab) == {"atom", "bond"}
    assert cached_vocab == {
        "atom": changed_vocab["atom"],
        "bond": changed_vocab["bond"],
    }


def test_msdnmr_spectrum_cache_invalidates_when_integral_vocab_changes(tmp_path):
    csv_path = tmp_path / "sample.csv"
    cache_path = tmp_path / "cache"
    _write_sample_csv(csv_path)
    vocab = _diffnmr_vocab()

    first = _dataset(csv_path, cache_path, vocab)
    assert first[0]["spectrum"]["H_nmr"][0, 3] == 2

    changed_vocab = copy.deepcopy(vocab)
    changed_vocab["integral"]["token_to_id"]["0H"] = 3
    changed_vocab["integral"]["token_to_id"]["1H"] = 2
    changed_vocab["integral"]["id_to_token"][2] = "1H"
    changed_vocab["integral"]["id_to_token"][3] = "0H"
    second = _dataset(csv_path, cache_path, changed_vocab)

    assert second[0]["spectrum"]["H_nmr"][0, 3] == 3
    with (cache_path / "spectrum_vocab.pkl").open("rb") as file_obj:
        cached_vocab = pickle.load(file_obj)
    assert set(cached_vocab) == {"peakwidth", "split", "integral"}
    assert cached_vocab == {
        "peakwidth": changed_vocab["peakwidth"],
        "split": changed_vocab["split"],
        "integral": changed_vocab["integral"],
    }
