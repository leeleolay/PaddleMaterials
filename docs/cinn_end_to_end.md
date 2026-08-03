# End-to-End CINN Runtime Contract

## Scope and Priority

The public `BaseTrainer` and `BasePredictor` workflows are the primary CINN
interface. Model-specific Tensor cores are compiler adapters behind that
interface; dataset loading, graph conversion, collation, losses, metrics,
logging, checkpointing, and prediction output remain owned by the existing
PaddleMaterials workflows.

```text
dataset / converter -> real collated model input
                    -> BaseTrainer or BasePredictor
                    -> model execution backend
                    -> standard loss_dict / pred_dict or predict result
                    -> optimizer, metric, checkpoint, or output file
```

The shared runtime uses these model hooks:

```python
model.set_execution_backend("eager" or "cinn")
model.validate_execution_backend()
model.prepare_execution(sample_input)
```

Models remain the sole owners of trainable parameters. Compiled callables are
kept outside the registered layer tree, so enabling CINN does not add prefixes
or other entries to checkpoint state keys. Loading a checkpoint invalidates the
process-local runtime and lazily compiles a new one from the restored model.

## Supported Property-Prediction Models

The four model families used by `property_prediction` implement the same
Trainer/Predictor contract. The configuration count below is repository
coverage, not a claim that every configuration has completed its full training
schedule or reproduced a paper metric.

| Model family | Repository YAML coverage | CINN Trainer | CINN Predictor | Dynamic boundary outside CINN |
|---|---:|---|---|---|
| `MEGNetPlus` | 36 / 36 | Single-GPU FP32 | Single-GPU FP32 | PGL batching, graph-state packing |
| `iComformer` | 8 / 8 | Single-GPU FP32 | Single-GPU FP32 | PGL batching and feature packing |
| `DimeNetPlusPlus` | 4 / 4 | Single-GPU FP32 | Single-GPU FP32 | PGL batching and discrete triplet construction |
| `SphereNet` | 12 / 12 | Single-GPU FP32 | Single-GPU FP32 | PGL batching and discrete torsion index selection |

All 12 SphereNet property-prediction YAML files set
`Model.__init_params__.energy_and_force=false`. That scalar-property path is
covered. `energy_and_force=true` requires force construction followed by a
second derivative during loss backward and is not supported by the SphereNet
CINN adapter; backend validation rejects it before compilation. Use the eager
backend for energy/force work.

`MEGNetPlus` requires its default graph-state embedding (`include_state=true`)
for CINN. This is the setting used by the repository property-prediction
configurations.

## Validated Runtime

The validated target is the stable PaddlePaddle GPU `3.3.1` release, using its
CUDA 12.6 build, one CUDA device, and FP32. Before selecting CINN, both of these
checks must be true:

```bash
python -m pip install paddlepaddle-gpu==3.3.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

```python
import paddle

assert paddle.is_compiled_with_cuda()
assert paddle.base.is_compiled_with_cinn()
```

The active Paddle device must be a GPU. CPU execution remains available with
`execution_backend=eager`; selecting CINN on CPU fails before workflow warmup.

The following modes are not supported by this runtime:

- automatic mixed precision (AMP);
- distributed training or inference (`world_size > 1`);
- persistent export of a compiled CINN runtime;
- cross-process reuse of a compiled runtime cache;
- SphereNet `energy_and_force=true` and second-order force training.

Trainer rejects CINN with `use_amp=true` or more than one process rather than
falling back to eager. The first call in each process incurs compilation cost,
and train and eval modes keep separate compiled runtimes.

## Selecting CINN

Workflow-level selection is preferred. It works with the ordinary repository
YAMLs; no model-specific wrapper or model-level backend edit is required.

### Training

Run the normal entry point as a single process and add the Trainer override:

```bash
CUDA_VISIBLE_DEVICES=0 python property_prediction/train.py \
  -c property_prediction/configs/comformer/comformer_mp2018_train_60k_e_form.yaml \
  Trainer.execution_backend=cinn \
  Trainer.use_amp=false
```

The same override applies to configurations under `configs/megnet`,
`configs/dimenet++`, and `configs/spherenet`. Do not launch the CINN workflow
through a distributed launcher. Normal CLI overrides such as data paths,
worker counts, output directory, or a short canary update count can be supplied
alongside the backend override.

### Prediction from a Local Config and Checkpoint

For crystal models, select CINN in `Predict` and provide a CIF input:

```bash
python property_prediction/predict.py \
  --config_path property_prediction/configs/megnet/megnet_mp2018_train_60k_e_form.yaml \
  --checkpoint_path /path/to/checkpoints/best.pdparams \
  --device gpu:0 \
  --cif_file_path property_prediction/example_data/cifs/mp-18767-LiMnO2.cif \
  --save_path output/megnet_cinn_prediction.csv \
  Predict.execution_backend=cinn
```

SphereNet uses the same Predictor contract with an XYZ input:

```bash
python property_prediction/predict.py \
  --config_path property_prediction/configs/spherenet/spherenet_qm9_U0.yaml \
  --checkpoint_path /path/to/checkpoints/best.pdparams \
  --device gpu:0 \
  --xyz_file_path property_prediction/example_data/molecules/isoguvacine.xyz \
  --save_path output/spherenet_cinn_prediction.csv \
  Predict.execution_backend=cinn
```

For a registered model package, keep `--model_name` and `--weights_name` and
append `Predict.execution_backend=cinn`. Registered-model overrides permit
`Predict.*` and `Dataset.*`; the workflow hook switches the loaded model, so a
`Model.__init_params__.execution_backend` override is unnecessary.

The model-level `execution_backend` constructor argument remains available for
direct model calls. A Trainer or Predictor backend is immutable after workflow
construction; create a new workflow object to change it.

## Lifecycle Semantics

Trainer warms the first real collated batch once in train mode and once in eval
mode. Training warmup keeps the runtime differentiable; the first loss backward
then exercises the compiled training path. Eval and Predictor warmup run under
`no_grad`. Warmup restores RNG state, non-trainable parameters, and buffers, so
compilation does not consume dropout randomness or commit BatchNorm
running-stat updates.

Predictor passes the converted public input unchanged to warmup. PGL graph
construction and model-specific discrete indexing remain in Python, while
distance/angle/torsion calculations, basis functions, message passing,
readout, losses, gradients, and optimizer updates stay in the Tensor/CINN
portion where supported.

Checkpoint files contain the ordinary model and optimizer state only. Resume
and Predictor loading therefore remain compatible with eager checkpoints, but
each restored process compiles fresh train/eval callables from its first real
input.

## Validation Levels

Support is established in layers; a passing import or `to_static` conversion
alone is not end-to-end evidence.

| Level | Evidence |
|---|---|
| Shared workflow contract | Backend selection, fail-fast checks, real-batch warmup, train/eval mode separation, state/RNG preservation, and checkpoint-key stability |
| Adapter parity | PGL-to-Tensor packing, single and batched forward, dynamic shapes, loss, parameter gradients, and an Adam update against the eager model |
| GPU CINN canary | Actual Paddle 3.3.1 CINN forward, backward, and optimizer step on one GPU |
| Trainer workflow | Training step, evaluation, `latest`/`best` checkpoint creation, load/resume, and optimizer-state restoration |
| Predictor workflow | Checkpoint load, graph conversion, first-input eval warmup, single/batched prediction, and output serialization |
| Full reproduction | Full dataset schedule, final metric, and paper/checkpoint comparison; not claimed by this adapter work |

The PaddlePaddle 3.3.1 GPU validation run included these end-to-end canaries:

| Model | Real-data Trainer evidence | Registered-checkpoint Predictor evidence |
|---|---|---|
| `MEGNetPlus` | MP2018 train/eval, checkpoint, and cross-process resume | Real CIF eager/CINN parity; see the detailed adapter record |
| `iComformer` | Three MP2018 training samples with dynamic batch 2 then 1, four validation samples, and best/latest checkpoints | Real CIF: eager and CINN both `-2.1615875 eV/atom` |
| `DimeNetPlusPlus` | Three MP2018 training samples with dynamic batch 2 then 1, four validation samples, and best/latest checkpoints | Real CIF: `-2.1553285` eager versus `-2.1553288 eV/atom` CINN |
| `SphereNet` | A real QM9 batch, checkpoint save, cross-process optimizer resume, and a second update | Real XYZ: eager and CINN both `U0=-77.14919` |

The Comformer, DimeNet++, and SphereNet Trainer canaries use reduced channel
counts to keep compiler validation bounded. Their registered Predictor checks
use the released, full-size model configurations and checkpoints. These values
are execution/parity evidence, not task-quality measurements.

A focused non-GPU/PIR regression run is:

```bash
python -m pytest \
  test/test_execution_backend.py \
  ppmat/models/megnet/tests/test_megnet_cinn.py \
  ppmat/models/megnet/tests/test_megnet_cinn_workflows.py \
  ppmat/models/comformer/tests/test_comformer_cinn.py \
  ppmat/models/dimenetpp/tests/test_dimenetpp_cinn.py \
  ppmat/models/dimenetpp/tests/test_dimenetpp_cinn_workflows.py \
  ppmat/models/spherenet/tests/test_spherenet_cinn.py \
  ppmat/models/spherenet/tests/test_spherenet_cinn_workflows.py \
  test/test_predictor.py -q
```

GPU tests are opt-in because the first compilation is expensive. The numerical
and workflow test modules document their respective environment switches. Run
the applicable canaries with PaddlePaddle 3.3.1 on a single GPU before changing
a Tensor core or compiler boundary.

## Known Numerical Boundary

A two-node DimeNet++ graph has directed edges but no valid non-backtracking
triplet. On PaddlePaddle 3.3.1, eager backward through this zero-triplet path
has a framework limitation and is not a reliable eager-versus-CINN backward
oracle. Forward behavior is covered for the zero-triplet case; gradient and
Adam parity are covered on graphs containing valid triplets. Treat two-node
zero-triplet training as unsupported on this validated version rather than
inferring support from the forward test.

Short deterministic and real-data canaries establish workflow execution,
checkpointing, and numerical sanity. They do not establish full-dataset
convergence, final task quality, throughput, or paper-level metric parity.
