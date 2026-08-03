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

from __future__ import annotations

import paddle
import pytest

from ppmat.predictor import BasePredictor
from ppmat.trainer.base_trainer import BaseTrainer
from ppmat.utils.execution import configure_execution_backend
from ppmat.utils.execution import ensure_execution_backend
from ppmat.utils.execution import validate_execution_backend


@pytest.fixture(autouse=True)
def _restore_device():
    original_device = paddle.get_device()
    paddle.set_device("cpu")
    yield
    paddle.set_device(original_device)


class _HookedModel(paddle.nn.Layer):
    """Small model that implements the shared compiled-runtime protocol."""

    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(1, 1)
        self.execution_backend = "eager"
        self.prepare_calls = []
        self.predict_calls = []

    def set_execution_backend(self, backend):
        if backend not in {"eager", "cinn"}:
            raise ValueError(backend)
        self.execution_backend = backend

    def validate_execution_backend(self):
        if self.execution_backend != "cinn":
            raise AssertionError("validation should only be used for CINN")

    def prepare_execution(self, sample):
        self.prepare_calls.append((sample, paddle.is_grad_enabled(), self.training))

    def forward(self, batch, return_loss=True, return_prediction=True):
        prediction = self.linear(batch["x"])
        output = {"loss_dict": {}, "pred_dict": {"value": prediction}}
        if return_loss:
            output["loss_dict"]["loss"] = paddle.nn.functional.mse_loss(
                prediction, batch["y"]
            )
        if not return_prediction:
            output["pred_dict"] = {}
        return output

    def predict(self, data):
        self.predict_calls.append(data)
        return {"value": self.linear(data["x"]).numpy()[0, 0]}


def _trainer_config(output_dir):
    return {
        "max_epochs": 1,
        "output_dir": str(output_dir),
        "save_freq": 1,
        "log_freq": 100,
        "start_eval_epoch": 1,
        "eval_freq": 1,
        "seed": 2026,
        "compute_metric_during_train": False,
        "use_amp": False,
        "eval_with_no_grad": True,
        "gradient_accumulation_steps": 1,
        "execution_backend": "cinn",
    }


def _loader():
    samples = [
        {"x": paddle.to_tensor([[1.0]]), "y": paddle.to_tensor([[2.0]])},
        {"x": paddle.to_tensor([[2.0]]), "y": paddle.to_tensor([[4.0]])},
    ]
    return paddle.io.DataLoader(samples, batch_size=2, shuffle=False, return_list=True)


def test_execution_backend_protocol_is_generic_for_trainer_and_predictor(tmp_path):
    model = _HookedModel()
    optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
    trainer = BaseTrainer(
        _trainer_config(tmp_path / "trainer"),
        model,
        train_dataloader=_loader(),
        val_dataloader=_loader(),
        optimizer=optimizer,
    )

    trainer.train()

    assert model.execution_backend == "cinn"
    assert [call[1:] for call in model.prepare_calls] == [
        (True, True),
        (False, False),
    ]
    assert trainer._execution_prepared_modes == {"train", "eval"}

    predictor = BasePredictor()
    predictor.model = model
    predictor.model.eval()
    predictor.execution_backend = "cinn"
    predictor._execution_prepared_modes = set()
    predictor.eval_with_no_grad = True
    predictor.post_transforms = None
    sample = {"x": paddle.to_tensor([[3.0]])}
    predictor._run_model(sample)
    predictor._run_model(sample)

    # The predictor receives the exact public input and warms only once.
    assert len(model.prepare_calls) == 3
    assert model.prepare_calls[-1][0] is sample
    assert model.prepare_calls[-1][1:] == (False, False)
    assert model.predict_calls == [sample, sample]


def test_eager_override_is_compatible_with_legacy_models():
    class LegacyModel:
        pass

    model = LegacyModel()
    assert configure_execution_backend(model, "eager", owner="Trainer") == "eager"
    validate_execution_backend(model, "eager", owner="Trainer")

    class FixedCompiledModel:
        execution_backend = "cinn"

    with pytest.raises(ValueError, match="cannot switch"):
        configure_execution_backend(
            FixedCompiledModel(), "eager", owner="Trainer"
        )


def test_backend_setter_must_honor_requested_backend():
    class MisreportingModel:
        execution_backend = "eager"

        def set_execution_backend(self, backend):
            del backend

    with pytest.raises(ValueError, match="model selected"):
        configure_execution_backend(MisreportingModel(), "cinn", owner="Predict")


def test_workflow_rejects_backend_drift():
    model = _HookedModel()
    model.set_execution_backend("cinn")
    ensure_execution_backend(model, "cinn", owner="Trainer")

    model.set_execution_backend("eager")
    with pytest.raises(RuntimeError, match="Create a new Trainer"):
        ensure_execution_backend(model, "cinn", owner="Trainer")


def test_compiled_backend_requires_model_hooks():
    with pytest.raises(ValueError, match="set_execution_backend"):
        configure_execution_backend(object(), "cinn", owner="Predict")

    with pytest.raises(ValueError, match="validated execution runtime"):
        validate_execution_backend(object(), "cinn", owner="Predictor")
