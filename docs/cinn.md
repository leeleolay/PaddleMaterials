# CINN execution backend

[中文版](cinn_ch.md)

PaddleMaterials uses eager execution by default and exposes CINN as an optional
model-owned runtime. The same configuration is consumed by training,
prediction, and sampling workflows; model parameters and public input/output
contracts do not change.

## Configuration

Add an `Execution` block to a task configuration:

```yaml
Execution:
  backend: cinn
  __init_params__:
    full_graph: false
```

`Execution.__init_params__` is optional. CINN defaults to `full_graph=False`.
Use `full_graph=True` only for a model whose strict-AST path is listed below.
The same values can be supplied as command-line overrides:

```bash
python property_prediction/predict.py \
  --model_name megnet_mp2018_train_60k_e_form \
  --weights_name best.pdparams \
  --device gpu:0 \
  --input_path property_prediction/example_data/cifs \
  --input_format cif \
  Execution.backend=cinn
```

Omitting `Execution`, or setting `Execution.backend=eager`, preserves the
existing eager workflow. CINN currently requires a CUDA device and a Paddle
build with CINN support. AMP and distributed execution are not qualified and
are rejected before the workflow starts.

## Runtime contract

The runtime has three responsibilities:

- `RuntimeMixin` owns backend selection, normalized backend options, and lazy
  train/eval runtime caches.
- `runtime_boundary` marks one complete numerical operation for compilation.
- `CinnBackend` validates the environment and calls `paddle.jit.to_static`.

The model owns the numerical boundary. Trainer, Predictor, and Sampler only
select and validate the requested backend, so checkpoints do not gain an extra
wrapper layer and eager and CINN keep identical parameter keys.

Boundary cache names follow a small vocabulary:

- `forward` for a model's standard numerical forward;
- `denoise_step` for one diffusion denoising step;
- a stable component role, currently `graph_encoder` or `spectrum_encoder`,
  when a workflow compiles independent components.

`full_graph=False` uses SOT capture and permits graph breaks around unsupported
Python regions. `full_graph=True` uses strict AST capture: the complete boundary
must become one static program. A strict program can still contain several
CINN kernels; `full_graph` describes capture, not kernel fusion.

Compilation is lazy. The first real model call creates and caches a runtime by
backend, train/eval mode, and boundary name. Changing the backend or runtime
options invalidates the cache.

## Supported models

The table describes the model-owned boundary rather than claiming that file
parsing, graph construction, schedulers, or post-processing are compiled.
"Qualified" refers to the current Paddle 3.3 registered-checkpoint workflow.
Detailed measurements and dated qualification evidence live only in the
[full performance matrix](cinn_performance.md).

| Model | Boundary | Capture | Qualified workflow |
| --- | --- | --- | --- |
| `MEGNetPlus` | complete `_forward` | SOT | prediction and Trainer integration |
| `iComformer` | complete `_forward` | SOT | registered-checkpoint prediction |
| `DimeNetPlusPlus` | complete `_forward` | SOT | prediction and Trainer integration |
| `SphereNet` | numerical `_runtime_forward` | SOT for properties; strict AST for potentials | property prediction and energy/force inference |
| `SFIN` | complete `_forward` | strict AST | image inference |
| `M3GNet` | differentiable `_runtime_forward` | strict AST | energy/force inference |
| `CHGNet` | differentiable `_runtime_forward` | strict AST | energy/force/stress/magnetic-moment inference |
| `DiffCSP` | `_runtime_decode` denoising step | SOT | structure sampling |
| `MatterGen`, `MatterGenWithCondition` | `_runtime_denoise` step | SOT | unconditional and conditional sampling |
| `MolecularGraphFormer` | `graph_encoder` and `denoise_step` | SOT | implemented; no standalone registered workflow |
| `NMRNetCLIP` | graph and spectrum encoders | SOT | implemented; no standalone registered workflow |
| `DiffPrior` | prior denoising step | SOT | implemented; not exercised by the registered DiffNMR workflow |
| `DiffNMR` | `spectrum_encoder`, `denoise_step`, and optional prior denoising | SOT | complete reverse diffusion |
| `InfGCN` | none | eager only | CINN lowering did not finish within the qualification limit |

InfGCN remains a supported eager model. It has no runtime boundary or CINN
constructor options; a future Paddle release must first complete isolated
lowering before the model is integrated again.

## Boundary placement

Keep file parsing, PGL/Python-object unpacking, dynamic topology, schedulers,
sampling loops, metrics, and visualization in eager execution. Pass tensors and
integer topology into the compiled numerical core. Current examples include:

- DiffCSP builds variable-size fully connected edges before `_runtime_decode`.
- MatterGen rebuilds its dynamic periodic neighbor graph before every compiled
  denoising step.
- SphereNet selects discrete torsion edges eagerly, then recomputes continuous
  geometry inside `_runtime_forward`.
- CHGNet batches discrete graph indices eagerly and keeps geometry, message
  passing, energy, force, and stress in one strict boundary.

This `eager topology -> tensor indices -> compiled numerics` split is part of
the model contract, not a fallback adapter.

Derivative models must create differentiable coordinate or strain leaves before
entering strict capture. Force/stress training additionally requires
second-order gradients. Current registered-checkpoint qualification is focused
on inference and sampling; production force/stress training needs an isolated
eager/CINN gradient-parity run for the target model and shape.

## Compatibility guarantees

- `forward`, `predict`, and `sample` keep their existing inputs and outputs.
- Eager remains the default and legacy models accept an explicit eager
  workflow override without implementing runtime hooks.
- Compiled callables are stored outside the model's registered sublayers, so
  eager and CINN checkpoints have identical parameter keys.
- Train and eval runtimes are cached separately.
- The runtime validates backend environment and workflow capabilities; it does
  not maintain model-specific allowlists or rejection branches.

## Integrating a model

Expose `execution_backend` and `runtime_options`, initialize the mixin, and mark
one complete numerical operation. Keep the repository's standard
`forward -> _forward -> predict` protocol:

```python
from ppmat.models.common.runtime import RuntimeMixin, runtime_boundary


class NewModel(RuntimeMixin, paddle.nn.Layer):
    def __init__(
        self,
        execution_backend="eager",
        runtime_options=None,
    ):
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)
        self.encoder = paddle.nn.Linear(16, 32)

    def forward(self, data, return_loss=True, return_prediction=True):
        prediction = self._forward(data)
        return build_output(prediction, data, return_loss, return_prediction)

    @runtime_boundary("forward")
    def _forward(self, data):
        return self.encoder(data["x"])
```

SOT boundaries can accept Python/PGL inputs and graph-break where necessary. A
strict model should prepare those inputs outside the boundary while reusing one
numerical implementation:

```python
def _forward(self, data):
    graph = pack_graph(data["graph"])
    return self._runtime_forward(
        graph["node"], graph["edge"], graph["edge_index"]
    )

@runtime_boundary("forward")
def _runtime_forward(self, node, edge, edge_index):
    return self.network(node, edge, edge_index)
```

Do not add a model-specific CINN wrapper, duplicate `_tensor_forward`, or expose
runtime options on a class that is never compiled. A new boundary needs eager
numerical regression, runtime-cache tests, public workflow tests, and isolated
real-GPU eager/CINN parity before it is listed as qualified.

## Qualification and isolation

The dated environment, methodology, family summary, all per-weight timings,
break-even estimates, numerical tolerances, and InfGCN lowering results are
maintained in one place: [CINN full performance matrix](cinn_performance.md).

CINN lowering can terminate the current process. Qualify a new model in a
disposable subprocess before enabling it in a long-running service:

```bash
timeout --signal=KILL 600 \
  python property_prediction/predict.py \
  --model_name megnet_mp2018_train_60k_e_form \
  --weights_name best.pdparams \
  --device gpu:0 \
  --input_path property_prediction/example_data/cifs \
  --input_format cif \
  Execution.backend=cinn
echo "exit=$?"
```

GNU `timeout` returns 124 when the limit is reached. A process terminated by
signal *n* is reported by the shell as 128+*n*; for example, `SIGABRT` is 134.
