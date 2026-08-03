# MEGNet CINN Phase 1

## Scope

Phase 1 establishes an inference-oriented, Tensor-only CINN path for the
existing `MEGNetPlus` model. It does not change the public model, dataset,
collator, trainer, predictor, checkpoint format, or PGL graph-construction
semantics. A synthetic training canary validates CINN backward and optimizer
parity without claiming end-to-end Trainer support.

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

`MEGNetTensorCore` wraps the original `MEGNetPlus` layer, so a checkpoint can be
loaded into `MEGNetPlus` before wrapping without parameter conversion. It
replaces graph-object methods with Tensor gather/scatter operations and returns
the same normalized prediction shape `[B, 1]`.

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
  ppmat/models/megnet/tests/test_megnet_cinn.py -q
```

Result on Paddle 3.3.1:

```text
11 passed, 2 skipped
```

The opt-in GPU smoke test compiles one dynamic CINN program and executes both a
single graph and a heterogeneous two-graph batch:

```bash
PPMAT_RUN_CINN_TESTS=1 "$PADDLE331/bin/python" -m pytest \
  ppmat/models/megnet/tests/test_megnet_cinn.py \
  -k dynamic_shape_cinn_inference -q
```

Result on Paddle 3.3.1:

```text
1 passed, 12 deselected, 5 warnings in 94.83s
```

Both graph shapes match Tensor eager execution within `atol=1e-6` and
`rtol=1e-6`. The test-sized model took approximately 85 seconds for its first
lazy compilation. Paddle emitted non-fatal RNN shape-cache and pattern-rewrite
warnings. No steady-state performance claim is made in phase 1.

## Training Canary

The opt-in training canary compares Tensor eager and CINN from identical
weights. It executes three batches with graph batch sizes `2 -> 1 -> 2`, using
MSE loss and the production Adam settings (`beta1=0.9`, `beta2=0.999`, learning
rate `1e-3`). Every step compares predictions, loss, all parameter gradients,
and post-update parameters.

```bash
PPMAT_RUN_CINN_TRAINING_TESTS=1 "$PADDLE331/bin/python" -m pytest \
  ppmat/models/megnet/tests/test_megnet_cinn.py \
  -k dynamic_shape_cinn_training -q
```

Result on Paddle 3.3.1:

```text
1 passed, 12 deselected, 6 warnings in 180.54s
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

## Follow-up Phases

Phase 2 should add an opt-in predictor/config integration, cache or export the
compiled inference program, measure warm and steady-state latency, and validate
representative MP-2018 graph-size distributions.

Phase 3 should integrate the Tensor core with the public Trainer contract and
validate a longer fixed-seed curve and final metrics. The MP2018 smoke above
validates the existing eager Trainer and a separate Tensor-only CINN training
harness; it is not end-to-end CINN Trainer support. AMP, distributed training,
and production schedules remain untested.
