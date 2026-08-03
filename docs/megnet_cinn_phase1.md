# MEGNet CINN Adapter

## Scope

The shared end-to-end contract is documented in
[`docs/cinn_end_to_end.md`](cinn_end_to_end.md). This document records the
MEGNet adapter that implements that contract; it is deliberately lower priority
than the public Trainer/Predictor runtime. `MEGNetPlus` remains the public model
and the sole checkpoint/optimizer owner; its `forward(batch_dict)` and
`predict(graphs)` contracts select either the existing eager path or the
validated CINN Tensor core through `execution_backend`. PGL graph construction,
batching, metrics, logging, and checkpoint management remain in the dynamic
Python boundary.

The Tensor-only APIs are retained as low-level diagnostics and are not the
primary Trainer or Predictor interface. The public workflow has priority:

```text
DataLoader/Predictor -> PGL graph or dict batch -> MEGNetPlus public API
                     -> dynamic PGL-to-Tensor packing -> CINN numerical core
                     -> loss/prediction dict -> Trainer/Predictor/checkpoint
```

`execution_backend="cinn"` is supported on PaddlePaddle 3.3.1 CUDA builds
with world size 1 and AMP disabled. AMP, distributed CINN, and persistent
compiled-program export are intentionally not claimed yet.

Baseline and validation environment:

- PaddleMaterials commit: `c6f4705e61afe887b2cacd0325843b8f27dda7d7`
- development branch: `feat/megnet-cinn-phase1`
- PaddlePaddle GPU stable: `3.3.1` (CUDA 12.6 build)
- PGL: `2.2.6`
- CINN: `0.3.0`
- PIR: enabled
- reference: the repository's existing `MEGNetPlus` eager path

The isolated validation environment is `/tmp/ppmat-paddle331`. Because Paddle
was installed into a target directory, its bundled libraries must be visible:

```bash
export PADDLE331=/tmp/ppmat-paddle331
export LD_LIBRARY_PATH="$PADDLE331/lib/python3.10/site-packages/paddle/libs:${LD_LIBRARY_PATH:-}"
```

For a regular virtual environment, install the same stable CUDA 12.6 wheel:

```bash
python -m pip install paddlepaddle-gpu==3.3.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

## Design

Pymatgen/PGL graph construction and batching stay outside the static program.
`graph_to_tensor_batch` converts a PGL graph to this contract:

```text
atom_types[N]       int64
bond_dist[E]        float32
edge_src[E]         int64
edge_dst[E]         int64
node_graph_id[N]    int64
edge_graph_id[E]    int64
node_sorted_eid[E]  int64
node_sorted_dst[E]  int64
state_attr[B, D]    float32 (D=2 by default)
```

`MEGNetTensorCore` is an internal numerical layer. A `MEGNetPlus` instance owns
the core's parameters and keeps compiled runtimes in non-registered runtime
storage, so its state keys remain compatible with ordinary eager checkpoints
(there is no `model.` prefix). The public model packs PGL graphs immediately
before calling the core and adapts its normalized output into the standard
`loss_dict`/`pred_dict` contract.

Training and inference use the same backend selection:

```yaml
Trainer:
  execution_backend: cinn
Predict:
  execution_backend: cinn
```

The workflow-level setting is preferred. A model-level default remains
available for direct `MEGNetPlus` calls, but is not required by Trainer or
Predictor.

`BaseTrainer` warms the first real collated batch after pretrained/resume
weights are loaded, and `BasePredictor` warms the first converted graph. The
compiled object is deliberately not serialized. A new process builds a fresh
runtime, while checkpoint values loaded into an existing process are visible
through the same parameter objects without recompilation. Train/eval runtimes
are kept separate because dropout and static graph mode are mode-dependent.

Paddle 3.3.1 still cannot infer CINN symbolic shapes for these graph reduction
operators:

```text
pd_op.send_u_recv
pd_op.segment_pool (SUM and MAX)
```

The Tensor path therefore uses:

- `scatter_nd_add(sum) / scatter_nd_add(count)` for segment mean;
- `scatter_nd_add(sum)` for Set2Set readout;
- `put_along_axis(reduce="amax")` plus scatter sum for segmented softmax.

This preserves the original per-segment maximum shift without lowering to the
unsupported `segment_pool(MAX)` operation.

## Usage

The normal public workflow is unchanged apart from the optional backend flag:

```bash
PYTHONPATH=/path/to/PaddleMaterials \
python property_prediction/train.py \
  -c property_prediction/configs/megnet/megnet_mp2018_train_60k_e_form.yaml \
  Trainer.execution_backend=cinn

PYTHONPATH=/path/to/PaddleMaterials \
python property_prediction/predict.py \
  --config_path /path/to/resolved.yaml \
  --checkpoint_path /path/to/checkpoints/best.pdparams \
  --device gpu:0 \
  --cif_file_path property_prediction/example_data/cifs/mp-18767-LiMnO2.cif \
  Predict.execution_backend=cinn
```

When developing from an additional git worktree, set `PYTHONPATH` (or install
that worktree editable) so the script does not import a different checkout.
The low-level Tensor API remains useful for compiler diagnostics:

```python
from ppmat.models.megnet.megnet_cinn import compile_megnet_cinn
from ppmat.models.megnet.megnet_cinn import graph_to_tensor_batch

model.eval()
compiled = compile_megnet_cinn(model)
compiled.eval()

batch = graph_to_tensor_batch(
    pgl_graph,
    state_dim=model.embedding.dim_state_embedding,
)
normalized_prediction = compiled(*batch)
```

The first invocation performs lazy CINN compilation. The dynamic InputSpec
allows `N`, `E`, and `B` to vary between calls.

## Verification

The default focused suite checks:

- non-mutating PGL-to-Tensor conversion;
- single-graph and heterogeneous two-graph forward parity;
- parameter-gradient and one-step SGD parity in eager mode;
- numerically stable per-graph softmax across widely separated score ranges;
- packing/count/convenience inference APIs;
- direct compilation from an original `MEGNetPlus` instance;
- explicit non-default state feature dimensions;
- dynamic `N`, `E`, and `B` through PIR with `backend=None`;
- `paddle.jit.save` and `paddle.jit.load` parity.

The GPU-only CINN tests are opt-in because first compilation is expensive.

Run:

```bash
"$PADDLE331/bin/python" -m pytest \
  test/test_megnet_cinn.py -q
```

Result on Paddle 3.3.1:

```text
15 passed, 2 skipped
```

The opt-in GPU smoke test compiles one dynamic CINN program and executes both a
single graph and a heterogeneous two-graph batch:

```bash
PPMAT_RUN_CINN_TESTS=1 "$PADDLE331/bin/python" -m pytest \
  test/test_megnet_cinn.py \
  -k dynamic_shape_cinn_inference -q
```

Result on Paddle 3.3.1:

```text
1 passed, 16 deselected, 5 warnings in 94.83s
```

Both graph shapes match the original PGL eager execution within `atol=1e-6`
and `rtol=1e-6`. The test-sized model took approximately 85 seconds for its
first lazy compilation. Paddle emitted non-fatal RNN shape-cache and
pattern-rewrite warnings. No steady-state performance claim is made in phase 1.

## Training Canary

The opt-in training canary compares an independent original PGL eager model and
CINN from identical weights. It executes three batches with graph batch sizes
`2 -> 1 -> 2`, using MSE loss and the production Adam settings (`beta1=0.9`,
`beta2=0.999`, learning rate `1e-3`). Every step compares predictions, loss,
all parameter gradients, and post-update parameters.

```bash
PPMAT_RUN_CINN_TRAINING_TESTS=1 "$PADDLE331/bin/python" -m pytest \
  test/test_megnet_cinn.py \
  -k dynamic_shape_cinn_training -q
```

Result on Paddle 3.3.1:

```text
1 passed, 16 deselected, 6 warnings in 180.54s
```

All three steps pass with `atol=2e-6` and `rtol=2e-6`. A direct first-step
probe measured forward and loss error `0.0`, maximum parameter-gradient error
`9.31e-10`, and post-SGD parameter error `0.0`. That direct SGD probe is kept as
an independent optimizer sanity check; the formal three-step canary uses Adam.

## MP2018 Training Smoke

The repository downloader was exercised against the official MP2018 archive.
It verified MD5 `216202f16a5081358798e15c060facee` and reused the local archive;
the archive SHA-256 is
`97be46b691d7cae77ee1da5416425a33bba1bf364fecd799bdbc7de0af02c9fe`.

A deterministic smoke subset selects the smallest valid structures among the
first 128 records of each official split, with source index as the tie breaker:

- train: 8 samples, SHA-256
  `934c50dc971e6fe16344afe94fd7c3f2e8aa8e06a65edf6923066a13e2841d7c`;
- validation: 4 samples, SHA-256
  `1332dae3d91d724a7c9bd68c78e9889bc374cca8b2e6fac0827d5caf9d56cdfe`.

Structures and radius graphs were built with one CPU worker and a 4.0 Angstrom
cutoff. Fresh-cache and cache-reload samples have identical labels and graph
hashes. The repository `property_prediction/train.py` entry point then ran the
production MEGNet configuration for two Adam updates, saved `epoch_1`, `latest`,
and `best` checkpoints, and evaluated the four-sample validation subset. The
observed per-step losses were `1.529762` and `3.021938`; validation MAE was
`0.841166 eV/atom`. Resuming `latest` advanced TrainerState from global step 2
to step 3 and produced a readable second checkpoint. Loading `best.pdparams`
back into `MEGNetPlus` produced eager/Tensor-Core predictions within
`2.24e-8` normalized absolute error.

A separate real-data CINN smoke canary used the production MEGNet configuration
and Adam settings for batches of two graphs (`N=9`, `E=140`) and one graph (`N=5`,
`E=76`). Across both steps, the maximum errors were:

- normalized prediction: `7.45e-9`;
- loss: `1.49e-8`;
- parameter gradient: `5.96e-8`;
- post-Adam parameter: `2.61e-7`.

The first production-configuration CINN invocation, including compilation, took
`186.76s`; the second dynamic-shape invocation took `0.0107s`. These timings are
diagnostic single-run measurements, not a benchmark. The manifest, resolved
configs, logs, checkpoints, and detailed reports are local validation artifacts
under `output/reproduction/megnet/mp2018/`; `output/` is ignored by Git by
design, so these files are not distributed by the source commit. The exact
resolved CLI configurations are preserved beside each run as YAML, while a
fresh checkout must regenerate the two subsets using the selection rule and
source/archive hashes above before repeating the smoke.

## End-to-End Workflow Verification

The focused public-contract suite covers backend selection, eager/CINN output
adaptation, CPU fail-fast behavior, non-default state dimensions, checkpoint
key compatibility, Trainer mode switching and Adam resume, plus single- and
batch-Predictor calls. The generic hook protocol is covered separately by
`test/test_execution_backend.py`:

```bash
PADDLE331=/tmp/ppmat-paddle331
"$PADDLE331/bin/python" -m pytest \
  test/test_execution_backend.py \
  test/test_megnet_cinn.py \
  test/test_megnet_cinn_workflows.py \
  test/test_predictor.py -q
```

Result on Paddle 3.3.1: `38 passed, 3 skipped`.

The opt-in GPU workflow test trains independent eager and CINN BaseTrainers from
identical state, compares their post-update model and Adam state, and creates
`epoch_1/latest/best`. Both backends then resume the same CINN `latest`
checkpoint for a second update before eager/CINN Predictors load the resumed
checkpoint and compare a converted CIF:

```bash
PPMAT_RUN_CINN_WORKFLOW_TESTS=1 \
  "$PADDLE331/bin/python" -m pytest \
  test/test_megnet_cinn_workflows.py \
  -k gpu_cinn_trainer_checkpoint_and_property_predictor -q
```

It passed on PaddlePaddle 3.3.1/CUDA 12.6 in `181.78s`. The small
production-shaped model's steady-state train/eval steps were below one second;
first compilation accounts for most of the workflow time.

The real MP2018 entry point was also run from this worktree with the official
downloaded/cache-backed smoke subset (8 train and 4 validation structures):

```bash
PYTHONPATH=/path/to/PaddleMaterials \
python property_prediction/train.py \
  -c output/reproduction/megnet/mp2018/trainer_eager_t_20260803_024600_s_42/megnet_mp2018_train_60k_e_form.yaml \
  Trainer.execution_backend=cinn Model.__init_params__.execution_backend=cinn \
  Trainer.max_iter=2 Trainer.output_dir=output/reproduction/megnet/mp2018/trainer_cinn_entry \
  Dataset.train.num_workers=0 Dataset.val.num_workers=0
```

The run completed two CINN optimizer steps (`1.529762`, `3.021938`), produced
`epoch_1/latest/best`, and reached validation MAE `0.841166 eV/atom`. Resuming
`latest` in a fresh process completed the third global step and validation
(`global_step 2 -> 3`); the restored `.pdopt` contains Adam moments and beta
powers. A subsequent `property_prediction/predict.py` invocation loaded the
same `best.pdparams` and predicted `-1.7361407` for
`mp-18767-LiMnO2.cif`, identical to the eager prediction to float32 precision.

The first production configuration compile was roughly three minutes and is
startup cost, not steady-state throughput. CINN emitted non-fatal RNN dynamic
shape and pattern-rewrite warnings during compilation.

After extracting the shared execution-backend contract, a fresh one-step
MP2018 revalidation completed with training loss `2.024862`, validation MAE
`0.830173 eV/atom`, and `epoch_1/latest/best` checkpoints. A fresh public
Predictor process loaded `best.pdparams` and produced `-1.7477959` for the same
LiMnO2 CIF. These values are smoke evidence for the runtime workflow, not a
full-schedule accuracy claim.

## Adapter Follow-up

The public Trainer and Predictor integration is covered by the shared runtime
contract. Remaining adapter work is deliberately separate:

1. Validate AMP and distributed execution in the shared runtime instead of
   enabling them implicitly.
2. Add an explicit compiled-runtime cache/export format if startup latency
   justifies the persistence and invalidation complexity.
3. Measure representative graph-size distributions and throughput after
   numerical parity is maintained.
4. Run longer fixed-seed curves and full production schedules.
