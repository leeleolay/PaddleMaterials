# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public-behavior regression tests for DiffNMR sampling."""

from types import SimpleNamespace

import numpy as np
import pytest

paddle = pytest.importorskip("paddle")

from ppmat.metrics.diffnmr_metric import SamplingMolecularMetrics  # noqa: E402
from ppmat.models.diffnmr.diffnmr import DiffPrior  # noqa: E402
from ppmat.models.diffnmr.diffnmr import MolecularGraphFormer  # noqa: E402
from ppmat.models.diffnmr.utils import diffgraphformer_utils  # noqa: E402
from ppmat.sampler.diffnmr import DiffNMRSampler  # noqa: E402
from ppmat.schedulers import scheduling_diffnmr  # noqa: E402
from ppmat.utils.ext_rdkit import BasicMolecularMetrics  # noqa: E402
from ppmat.utils.ext_rdkit import mol_from_graphs  # noqa: E402

BOND_VOCAB = {
    "token_to_id": {
        "NO_BOND": 0,
        "SINGLE": 1,
        "DOUBLE": 2,
        "TRIPLE": 3,
        "AROMATIC": 4,
    },
    "num_embeddings": 5,
}


class _CapturePrior(DiffPrior):
    def __init__(self, condition_on_spectrum_encodings=True):
        paddle.nn.Layer.__init__(self)
        self.graph_embed_dim = 3
        self.sample_timesteps = 2
        self.condition_on_spectrum_encodings = condition_on_spectrum_encodings
        self.noise_scheduler = SimpleNamespace(num_timesteps=4)
        self.captured = None

    def p_sample_loop(self, shape, spectrum_cond, cond_scale=1.0, timesteps=None):
        self.captured = (shape, spectrum_cond, cond_scale, timesteps)
        return spectrum_cond["spectrum_embed"]


def test_diffprior_repeats_encoder_tokens_for_each_candidate():
    prior = _CapturePrior()
    tokens = paddle.arange(24, dtype="float32").reshape([2, 4, 3])
    mask = paddle.ones([2, 4], dtype="bool")

    result = prior.sample(
        paddle.ones([2, 3]),
        spectrum_encodings=(tokens, mask),
        num_samples_per_batch=2,
        timesteps=1,
    )

    repeated = prior.captured[1]["spectrum_encodings"]
    assert list(result.shape) == [2, 3]
    assert list(repeated.shape) == [4, 4, 3]
    assert repeated[:, 0, 0].numpy().tolist() == [0.0, 0.0, 12.0, 12.0]


def test_diffprior_accepts_missing_optional_encoder_tokens():
    prior = _CapturePrior()

    prior.sample(paddle.ones([1, 3]), spectrum_encodings=None, timesteps=1)

    assert prior.captured[1]["spectrum_encodings"] is None


class _DDIMScheduler:
    num_timesteps = 4
    alphas_cumprod = paddle.to_tensor([1.0, 0.95, 0.85, 0.7])

    @staticmethod
    def predict_noise_from_start(x_t, t, x0):
        return paddle.zeros_like(x_t)


class _DDIMNetwork:
    self_cond = False

    def __init__(self):
        self.times = []

    def forward_with_cond_scale(self, graph_embed, times, **kwargs):
        self.times.extend(times.numpy().tolist())
        return paddle.full_like(graph_embed, 2.0)


class _DDIMPrior(DiffPrior):
    def __init__(self, scheduler=None):
        paddle.nn.Layer.__init__(self)
        self.noise_scheduler = scheduler or _DDIMScheduler()
        self.net = _DDIMNetwork()
        self.predict_v = False
        self.predict_x_start = True
        self.sampling_clamp_l2norm = False
        self.sampling_final_clamp_l2norm = False
        self.init_graph_embed_l2norm = False
        self.graph_embed_scale = 1.0


def _run_ddim(prior, timesteps):
    return prior.p_sample_loop_ddim(
        (1, 3),
        {"spectrum_embed": paddle.ones([1, 3]), "spectrum_encodings": None},
        timesteps=timesteps,
        eta=0.0,
    )


def test_diffprior_ddim_one_requested_step_runs_the_network_once():
    prior = _DDIMPrior()

    result = _run_ddim(prior, timesteps=1)

    assert prior.net.times == [3]
    np.testing.assert_allclose(result.numpy(), [[2.0, 2.0, 2.0]])


def test_diffprior_ddim_uses_requested_steps_and_current_alpha_bar():
    class _CurrentAlphaOnlyScheduler(_DDIMScheduler):
        @property
        def alphas_cumprod_prev(self):
            raise AssertionError("DDIM must use alphas_cumprod")

    prior = _DDIMPrior(_CurrentAlphaOnlyScheduler())

    _run_ddim(prior, timesteps=3)

    assert len(prior.net.times) == 3


def test_h_only_reverse_step_keeps_connector_batch_aligned(monkeypatch):
    class _Q:
        def __init__(self):
            self.X = paddle.eye(2).unsqueeze(0)
            self.E = paddle.eye(2).unsqueeze(0)

    class _NoiseSchedule:
        def __call__(self, t_normalized):
            return paddle.zeros_like(t_normalized)

        def get_alpha_bar(self, t_normalized):
            return paddle.ones_like(t_normalized)

    class _Transitions:
        def get_Qt_bar(self, _value):
            return _Q()

        def get_Qt(self, _value):
            return _Q()

    class _Connector:
        def sample(self, embedding, **kwargs):
            self.kwargs = kwargs
            return embedding

    model = SimpleNamespace(
        conditioning_mode="spectrum",
        flag_onlyH=True,
        connector_flag=True,
        Xdim_output=2,
        Edim_output=2,
        noise_schedule=_NoiseSchedule(),
        transition_model=_Transitions(),
        encoder=lambda _condition: (paddle.ones([1, 1]), None),
        connector=_Connector(),
        decoder=lambda _X, _E, _y, _mask: diffgraphformer_utils.PlaceHolder(
            X=paddle.zeros([1, 1, 2]),
            E=paddle.zeros([1, 1, 1, 2]),
            y=paddle.zeros([1, 0]),
        ),
    )
    monkeypatch.setattr(
        scheduling_diffnmr,
        "compute_extra_data",
        lambda *_args, **_kwargs: diffgraphformer_utils.PlaceHolder(
            X=paddle.zeros([1, 1, 0]),
            E=paddle.zeros([1, 1, 1, 0]),
            y=paddle.zeros([1, 0]),
        ),
    )

    scheduling_diffnmr.step(
        model,
        s=paddle.to_tensor([[0.0]]),
        t=paddle.to_tensor([[1.0]]),
        X_t=paddle.to_tensor([[[1.0, 0.0]]]),
        E_t=paddle.to_tensor([[[[1.0, 0.0]]]]),
        y_t=paddle.zeros([1, 0]),
        node_mask=paddle.ones([1, 1], dtype="bool"),
        conditionVec=[paddle.zeros([1, 1]) for _ in range(4)],
        batch_X=paddle.to_tensor([[[1.0, 0.0]]]),
        batch_E=paddle.to_tensor([[[[1.0, 0.0]]]]),
        batch_y=paddle.zeros([1, 0]),
    )

    assert model.connector.kwargs["spectrum_encodings"] is None
    assert model.connector.kwargs["num_samples_per_batch"] == 1


def test_graphformer_forward_accepts_a_valid_empty_edge_graph(monkeypatch):
    graph = SimpleNamespace(
        node_feat={"feat": np.asarray([[1.0, 0.0]], dtype=np.float32)},
        edges=np.empty([0, 2], dtype=np.int64),
        edge_feat={"feat": np.empty([0, 3], dtype=np.float32)},
        graph_node_id=np.zeros([1], dtype=np.int64),
    )
    batch = {
        "graph": graph,
        "property": {"y": np.zeros([1, 0], dtype=np.float32)},
    }
    monkeypatch.setattr(
        scheduling_diffnmr,
        "apply_noise",
        lambda _model, X, E, y, node_mask, _flag: {
            "X_t": X,
            "E_t": E,
            "y_t": y,
            "node_mask": node_mask,
        },
    )
    monkeypatch.setattr(
        scheduling_diffnmr,
        "compute_extra_data",
        lambda _model, noisy_data, **_kwargs: diffgraphformer_utils.PlaceHolder(
            X=paddle.zeros([*noisy_data["X_t"].shape[:2], 0]),
            E=paddle.zeros([*noisy_data["E_t"].shape[:3], 0]),
            y=paddle.zeros([noisy_data["X_t"].shape[0], 0]),
        ),
    )
    model = SimpleNamespace(
        flag_use_formula=False,
        encoder=lambda _X, _E, _y, _mask: paddle.ones([1, 1]),
        decoder=lambda X, E, _y, _mask: diffgraphformer_utils.PlaceHolder(
            X=X, E=E, y=paddle.zeros([1, 0])
        ),
        train_loss=lambda **_kwargs: {"loss": paddle.to_tensor(0.0)},
    )

    result = MolecularGraphFormer.forward(model, batch)

    assert result is not None
    assert list(result["pred_dict"]["masked_pred_X"].shape) == [1, 1, 2]
    assert list(result["pred_dict"]["masked_pred_E"].shape) == [1, 1, 1, 3]


def test_sampler_resets_metrics_and_calls_model_sample_with_batch():
    events = []

    class _Model:
        conditioning_mode = "spectrum"
        dataset_info = SimpleNamespace(atom_decoder=["C"])

        def eval(self):
            events.append("eval")

        def sample(self, batch, **kwargs):
            events.append(("sample", batch, kwargs))
            molecule = [np.asarray([0]), np.zeros([1, 1], dtype=np.int64)]
            return [molecule], [molecule]

    class _Streaming:
        def reset(self):
            events.append("reset")

        def update_step(self, result, **_kwargs):
            events.append(("update", result["samples"]))

        def compute_epoch(self, **_kwargs):
            events.append("compute")
            return {"Accuracy": 1.0}

    batch = {
        "graph": object(),
        "property": {"atom_count": np.asarray([1], dtype=np.int64)},
        "spectrum": {
            "H_nmr": np.zeros([1, 2], dtype=np.float32),
            "num_H_peak": np.ones([1], dtype=np.int64),
            "C_nmr": np.zeros([1, 2], dtype=np.float32),
            "num_C_peak": np.ones([1], dtype=np.int64),
        },
    }
    sampler = object.__new__(DiffNMRSampler)
    sampler.model = _Model()
    sampler.streaming = _Streaming()
    sampler.sample_batch_iters = 1
    sampler.visual_num = 0
    sampler.chains_left_to_save = 0
    sampler.number_chain_steps = 1
    sampler.flag_use_formula = False
    sampler.flag_retrieval_initialization = False
    sampler.rank = 0
    sampler.output_dir = "."

    result = sampler.sample_epoch([batch], epoch_id=0)

    sample_event = next(event for event in events if isinstance(event, tuple))
    assert result == {"Accuracy": 1.0}
    assert events[:2] == ["eval", "reset"]
    assert sample_event[0] == "sample"
    assert sample_event[1] is batch
    assert not {
        "batch_size",
        "batch_condition",
        "batch_X",
        "batch_E",
        "batch_y",
        "num_nodes",
    }.intersection(sample_event[2])


def test_retrieval_initialization_overwrites_only_the_retrieved_prefix(monkeypatch):
    vocab = {
        "atom": {"token_to_id": {"C": 0, "O": 1}, "num_embeddings": 2},
        "bond": BOND_VOCAB,
    }
    initial_X = paddle.zeros([1, 3, 2], dtype="float32")
    initial_X[:, :, 1] = 1.0
    initial_E = paddle.zeros([1, 3, 3, 5], dtype="float32")
    initial_E[:, :, :, 4] = 1.0
    captured = {}

    monkeypatch.setattr(
        scheduling_diffnmr,
        "sample_discrete_feature_noise",
        lambda **_kwargs: diffgraphformer_utils.PlaceHolder(
            X=initial_X.clone(),
            E=initial_E.clone(),
            y=paddle.zeros([1, 0]),
        ),
    )

    def reverse_step(_model, **kwargs):
        captured["X"] = kwargs["X_t"].clone()
        captured["E"] = kwargs["E_t"].clone()
        sampled = diffgraphformer_utils.PlaceHolder(
            X=kwargs["X_t"], E=kwargs["E_t"], y=kwargs["y_t"]
        )
        return sampled, sampled

    monkeypatch.setattr(scheduling_diffnmr, "step", reverse_step)
    graph = SimpleNamespace(
        node_feat={"feat": np.tile([[0.0, 1.0]], [3, 1]).astype(np.float32)},
        edges=np.empty([0, 2], dtype=np.int64),
        edge_feat={"feat": np.empty([0, 5], dtype=np.float32)},
        graph_node_id=np.zeros([3], dtype=np.int64),
    )
    model = SimpleNamespace(
        conditioning_mode="graph",
        limit_dist=None,
        T=1,
        vocab=vocab,
        visualization_tools=None,
    )
    clip = SimpleNamespace(
        spectrum_encoder=lambda _condition: paddle.to_tensor([[1.0, 0.0]])
    )

    MolecularGraphFormer.sample(
        model,
        {
            "graph": graph,
            "property": {
                "y": np.zeros([1, 0], dtype=np.float32),
                "atom_count": np.asarray([3], dtype=np.int64),
            },
        },
        retrieval_initialization=True,
        clip=clip,
        molecular_vectors=paddle.to_tensor([[1.0, 0.0]]),
        smiles_list=["C"],
    )

    np.testing.assert_allclose(captured["X"][0, 0].numpy(), [1.0, 0.0])
    np.testing.assert_allclose(captured["X"][0, 1:].numpy(), initial_X[0, 1:].numpy())
    np.testing.assert_allclose(captured["E"][0, 1:].numpy(), initial_E[0, 1:].numpy())


def _dataset_info():
    return SimpleNamespace(
        atom_decoder=["C", "O"],
        max_n_nodes=2,
        num_atom_types=2,
        n_nodes=[0, 1, 0],
        node_types=[1, 0],
        edge_types=paddle.to_tensor([1, 0, 0, 0, 0], dtype="float32"),
        valency_distribution=[1, 0, 0, 0],
        remove_h=True,
        vocab={"bond": BOND_VOCAB},
    )


def test_sampling_metric_reset_starts_the_next_epoch_clean(tmp_path):
    metric = SamplingMolecularMetrics(_dataset_info(), train_smiles=[])
    two_carbons = [
        np.asarray([0, 0]),
        np.asarray([[0, 1], [1, 0]], dtype=np.int64),
    ]
    two_oxygens = [
        np.asarray([1, 1]),
        np.asarray([[0, 1], [1, 0]], dtype=np.int64),
    ]

    metric(
        {"pred": [two_carbons], "true": [two_carbons], "n_all": 1},
        current_epoch=0,
        local_rank=0,
        output_dir=str(tmp_path),
    )
    metric.reset()
    result = metric(
        {"pred": [two_oxygens], "true": [two_oxygens], "n_all": 1},
        current_epoch=1,
        local_rank=0,
        output_dir=str(tmp_path),
    )

    np.testing.assert_allclose(result["Gen node distribution"].numpy(), [0.0, 1.0])


def test_standard_graph_conversion_and_rdkit_metrics():
    graph = [
        np.asarray([0, 0]),
        np.asarray([[0, 1], [1, 0]], dtype=np.int64),
    ]
    molecule = mol_from_graphs(["C"], *graph)
    metrics = BasicMolecularMetrics(_dataset_info(), train_smiles=[])

    valid, validity, components, smiles = metrics.compute_validity([graph])

    assert molecule.GetNumAtoms() == 2
    assert molecule.GetBondWithIdx(0).GetBondType().name == "SINGLE"
    assert valid == ["CC"]
    assert validity == 1.0
    assert components.tolist() == [1]
    assert smiles == ["CC"]
