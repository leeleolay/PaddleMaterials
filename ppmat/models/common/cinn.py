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

"""Shared lifecycle for model-owned CINN execution runtimes."""

from __future__ import annotations

import contextlib
from typing import Any

import paddle


class CINNExecutionMixin:
    """Implement the execution hooks consumed by Trainer and Predictor.

    A model using this mixin remains the parameter and checkpoint owner. The
    compiled callables are deliberately stored as unregistered Python objects,
    so enabling CINN cannot change state-dict keys.
    """

    def _init_cinn_execution(
        self,
        execution_backend: str = "eager",
        cinn_full_graph: bool = True,
    ) -> None:
        if execution_backend is None:
            execution_backend = "eager"
        if execution_backend not in {"eager", "cinn"}:
            raise ValueError(
                "execution_backend must be either 'eager' or 'cinn', "
                f"got {execution_backend!r}."
            )
        if not isinstance(cinn_full_graph, bool):
            raise TypeError("cinn_full_graph must be a bool.")

        self.execution_backend = execution_backend
        self.cinn_full_graph = cinn_full_graph
        object.__setattr__(self, "_cinn_runtimes", {})
        object.__setattr__(self, "_cinn_warmed_modes", set())

    @property
    def _cinn_model_name(self) -> str:
        return type(self).__name__

    def _validate_cinn_model(self) -> None:
        """Validate model-specific constraints before compiling."""

    def _validate_cinn_environment(self) -> None:
        self._validate_cinn_model()
        model_name = self._cinn_model_name
        if not paddle.is_compiled_with_cuda():
            raise RuntimeError(
                f"{model_name} execution_backend='cinn' requires a CUDA-enabled "
                "Paddle installation; use execution_backend='eager' on CPU."
            )
        if not paddle.base.is_compiled_with_cinn():
            raise RuntimeError(
                f"{model_name} execution_backend='cinn' requires a Paddle build "
                "with CINN support."
            )
        device = paddle.get_device()
        if not device.startswith("gpu"):
            raise RuntimeError(
                f"{model_name} execution_backend='cinn' requires a GPU device, "
                f"but the current device is {device!r}."
            )

    def validate_execution_backend(self) -> None:
        if self.execution_backend == "cinn":
            self._validate_cinn_environment()

    def set_execution_backend(self, backend: str) -> None:
        if backend is None:
            backend = "eager"
        if backend not in {"eager", "cinn"}:
            raise ValueError(
                "execution_backend must be either 'eager' or 'cinn', "
                f"got {backend!r}."
            )
        if backend != self.execution_backend:
            self.execution_backend = backend
            self.invalidate_cinn_runtime()

    def invalidate_cinn_runtime(self) -> None:
        runtimes = getattr(self, "_cinn_runtimes", None)
        if runtimes is not None:
            runtimes.clear()
        warmed_modes = getattr(self, "_cinn_warmed_modes", None)
        if warmed_modes is not None:
            warmed_modes.clear()

    def _compile_cinn_runtime(self):
        raise NotImplementedError

    def _get_cinn_runtime(self):
        mode = "train" if self.training else "eval"
        runtime = self._cinn_runtimes.get(mode)
        if runtime is None:
            self._validate_cinn_environment()
            runtime = self._compile_cinn_runtime()
            runtime.training = self.training
            self._cinn_runtimes[mode] = runtime
        return runtime

    def _prepare_execution_batch(self, sample_input: Any):
        return (
            sample_input if isinstance(sample_input, dict) else {"graph": sample_input}
        )

    def _snapshot_warmup_state(self):
        state = []
        seen = set()
        tensors = [
            tensor for _, tensor in self.named_parameters() if tensor.stop_gradient
        ]
        tensors.extend(tensor for _, tensor in self.named_buffers())
        for tensor in tensors:
            is_initialized = getattr(tensor, "_is_initialized", None)
            if callable(is_initialized) and not is_initialized():
                continue
            if id(tensor) not in seen:
                state.append((tensor, tensor.clone()))
                seen.add(id(tensor))
        return state

    @staticmethod
    def _restore_warmup_state(state) -> None:
        with paddle.no_grad():
            for tensor, value in state:
                tensor.set_value(value)

    def prepare_execution(self, sample_input: Any = None) -> None:
        if self.execution_backend != "cinn":
            return

        mode = "train" if self.training else "eval"
        if mode in self._cinn_warmed_modes:
            return
        if sample_input is None:
            raise ValueError(
                "prepare_execution requires one real collated batch or predictor "
                "input."
            )

        execution_batch = self._prepare_execution_batch(sample_input)
        model_state = self._snapshot_warmup_state()
        rng_state = paddle.get_rng_state()
        device = paddle.get_device()
        cuda_rng_state = (
            paddle.get_cuda_rng_state()
            if paddle.is_compiled_with_cuda() and device.startswith("gpu")
            else None
        )
        try:
            self._get_cinn_runtime()
            context = contextlib.nullcontext() if self.training else paddle.no_grad()
            with context:
                self(
                    execution_batch,
                    return_loss=False,
                    return_prediction=True,
                )
        finally:
            self._restore_warmup_state(model_state)
            paddle.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                paddle.set_cuda_rng_state(cuda_rng_state)
        self._cinn_warmed_modes.add(mode)


__all__ = ["CINNExecutionMixin"]
