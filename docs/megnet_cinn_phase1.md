# MEGNet CINN Phase 1

## Scope

Phase 1 establishes an inference-oriented, Tensor-only CINN path for the
existing `MEGNetPlus` model. It does not change the public model, dataset,
collator, trainer, predictor, checkpoint format, or PGL graph-construction
semantics.

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

Run:

```bash
"$PADDLE331/bin/python" -m pytest \
  ppmat/models/megnet/tests/test_megnet_cinn.py -q
```

Result on Paddle 3.3.1:

```text
11 passed, 1 skipped
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
1 passed, 11 deselected, 5 warnings in 94.83s
```

Both graph shapes match Tensor eager execution within `atol=1e-6` and
`rtol=1e-6`. The test-sized model took approximately 85 seconds for its first
lazy compilation. Paddle emitted non-fatal RNN shape-cache and pattern-rewrite
warnings. No steady-state performance claim is made in phase 1.

## Follow-up Phases

Phase 2 should add an opt-in predictor/config integration, cache or export the
compiled inference program, measure warm and steady-state latency, and validate
representative MP-2018 graph-size distributions.

Phase 3 should validate CINN backward/optimizer parity before enabling compiled
training. Eager backward parity alone is not evidence that CINN training is
supported.
