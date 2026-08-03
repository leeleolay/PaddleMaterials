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

"""Shared execution-backend hooks used by training and prediction.

The trainer and predictor own orchestration, while a model owns the numerical
runtime and its parameters.  A model that implements a compiled backend may
provide these optional hooks::

    set_execution_backend(name)
    validate_execution_backend()
    prepare_execution(sample_input)

Keeping the protocol here prevents public workflows from depending on a
MEGNet-specific adapter and leaves eager models completely unchanged.
"""

from __future__ import annotations

from typing import Any
from typing import Optional


DEFAULT_EXECUTION_BACKEND = "eager"


def configure_execution_backend(
    model: Any,
    configured_backend: Optional[str],
    *,
    owner: str,
) -> str:
    """Apply a workflow-level backend override and return the active backend.

    ``eager`` is a valid no-op for legacy models.  Any compiled (or otherwise
    non-eager) backend must be implemented by the model so that the model,
    trainer, predictor, and checkpoint loader share one numerical contract.
    """

    if configured_backend is None:
        return getattr(model, "execution_backend", DEFAULT_EXECUTION_BACKEND)
    if not isinstance(configured_backend, str) or not configured_backend:
        raise ValueError(
            f"{owner}.execution_backend must be a non-empty string, "
            f"got {configured_backend!r}."
        )

    setter = getattr(model, "set_execution_backend", None)
    if setter is None:
        if configured_backend != DEFAULT_EXECUTION_BACKEND:
            raise ValueError(
                f"{owner}.execution_backend={configured_backend!r} was requested, "
                "but the model does not implement set_execution_backend()."
            )
        current_backend = getattr(
            model, "execution_backend", DEFAULT_EXECUTION_BACKEND
        )
        if current_backend != DEFAULT_EXECUTION_BACKEND:
            raise ValueError(
                f"{owner}.execution_backend='eager' was requested, but the model "
                f"reports execution_backend={current_backend!r} and cannot switch "
                "without set_execution_backend()."
            )
        return DEFAULT_EXECUTION_BACKEND

    setter(configured_backend)
    active_backend = getattr(model, "execution_backend", configured_backend)
    if active_backend != configured_backend:
        raise ValueError(
            f"{owner}.execution_backend requested {configured_backend!r}, but the "
            f"model selected {active_backend!r}."
        )
    return active_backend


def validate_execution_backend(
    model: Any,
    backend: str,
    *,
    use_amp: bool = False,
    world_size: int = 1,
    owner: str,
) -> None:
    """Validate a model/runtime combination before entering a workflow."""

    if backend == DEFAULT_EXECUTION_BACKEND:
        return

    if backend == "cinn" and use_amp:
        raise ValueError(
            f"{owner} with execution_backend='cinn' does not support AMP; "
            "set use_amp=False until the CINN AMP path is validated."
        )
    if backend == "cinn" and world_size > 1:
        raise ValueError(
            f"{owner} with execution_backend='cinn' supports world_size=1 only; "
            "distributed CINN execution is not enabled yet."
        )

    validator = getattr(model, "validate_execution_backend", None)
    if validator is None:
        raise ValueError(
            f"{owner}.execution_backend={backend!r} requires a model with a "
            "validated execution runtime (missing "
            "validate_execution_backend())."
        )
    validator()


def ensure_execution_backend(model: Any, backend: str, *, owner: str) -> None:
    """Reject backend changes after a workflow has initialized its runtime."""

    active_backend = getattr(model, "execution_backend", DEFAULT_EXECUTION_BACKEND)
    if active_backend != backend:
        raise RuntimeError(
            f"{owner} was initialized with execution_backend={backend!r}, but the "
            f"model now reports {active_backend!r}. Create a new {owner} after "
            "changing the model backend."
        )


def prepare_execution(
    model: Any,
    backend: str,
    sample_input: Any,
    *,
    owner: str,
) -> None:
    """Warm a compiled backend with an input from the real workflow.

    The sample is passed through unchanged.  This matters for models whose
    public input is a dict, tensor tuple, image batch, or graph object rather
    than MEGNet's ``{"graph": ...}`` representation.
    """

    if backend == DEFAULT_EXECUTION_BACKEND:
        return
    if sample_input is None:
        raise ValueError(f"{owner} requires a real sample input for backend warmup.")

    prepare = getattr(model, "prepare_execution", None)
    if prepare is None:
        raise ValueError(
            f"{owner}.execution_backend={backend!r} requires model.prepare_execution()."
        )
    prepare(sample_input)


def execution_mode(model: Any) -> str:
    """Return the mode key used for train/eval compiled-runtime caches."""

    return "train" if getattr(model, "training", False) else "eval"


__all__ = [
    "DEFAULT_EXECUTION_BACKEND",
    "configure_execution_backend",
    "ensure_execution_backend",
    "execution_mode",
    "prepare_execution",
    "validate_execution_backend",
]
