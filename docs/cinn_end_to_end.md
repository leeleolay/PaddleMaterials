# End-to-End CINN Runtime Contract

## Priority

The primary deliverable is a shared CINN execution contract for the public
`BaseTrainer` and `BasePredictor` workflows. A model-specific compiler adapter
is an implementation detail and can be replaced without changing data loading,
metrics, logging, checkpoint ownership, or user-facing prediction APIs.

The runtime boundary is:

```text
dataset / converter -> real collated model input
                    -> BaseTrainer or BasePredictor orchestration
                    -> model execution backend
                    -> standard forward/predict result
                    -> optimizer, metrics, checkpoint, or output file
```

The current implementation uses the following optional model hooks:

```python
model.set_execution_backend("eager" or "cinn")
model.validate_execution_backend()
model.prepare_execution(sample_input)
```

The model remains the sole owner of trainable parameters and compiled runtime
objects. Compiled objects must not be registered as child layers when doing so
would alter checkpoint keys. Checkpoints contain model parameters and optimizer
state only; a restored process warms a fresh runtime from its first real input.

## Workflow Semantics

`Trainer.execution_backend` and `Predict.execution_backend` are optional
workflow-level overrides. If omitted, the model's backend is used. Legacy models
without backend hooks continue to work with `eager`; selecting a compiled
backend fails before the workflow starts with an actionable error.
The selected backend is immutable for the lifetime of a Trainer or Predictor;
changing it requires constructing a new workflow object so its warmup cache and
the model runtime cannot diverge.

The trainer warms once per train/eval mode using the first actual DataLoader
batch. Evaluation warmup runs under `no_grad`; training warmup keeps gradients
enabled so a backend can compile its backward path. The predictor passes its
public input unchanged and warms once in eval mode. This supports dicts,
tensors, image batches, and graph objects without a MEGNet-specific wrapper.

Backend selection does not move graph construction, collation, metric
reduction, logging, or checkpoint serialization into CINN. Those operations
remain dynamic Python orchestration, which keeps the contract usable by models
with different input representations.

## Current Boundary

PaddlePaddle `3.3.1` CUDA builds are the validated target. CINN currently runs
in single-process FP32 mode only. AMP and `world_size > 1` fail fast instead of
silently falling back to eager execution. Persistent compiled-program export
and a cross-process runtime cache are follow-up work.

## MEGNet Adapter

`MEGNetPlus` implements the hooks with a Tensor-only CINN numerical core. PGL
graph construction and conversion stay outside the compiled program, and the
public `loss_dict`/`pred_dict` and `predict` contracts are preserved. The
adapter is covered by [the MEGNet validation record](megnet_cinn_phase1.md), but
its scatter/set2set implementation is intentionally lower priority than the
shared Trainer/Predictor contract.

## Validation

The generic protocol is exercised by `test/test_execution_backend.py`. MEGNet
workflow and numerical parity tests remain under
`ppmat/models/megnet/tests/`. A minimal focused run is:

```bash
python -m pytest \
  test/test_execution_backend.py \
  ppmat/models/megnet/tests/test_megnet_cinn.py \
  ppmat/models/megnet/tests/test_megnet_cinn_workflows.py \
  test/test_predictor.py -q
```
