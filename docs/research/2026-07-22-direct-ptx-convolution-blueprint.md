# Direct PTX convolution assembly-line blueprint (issue #841)

Status: experimental golden slice. No specialization in this document is
promoted for production. GPU correctness, performance, resource, occupancy,
and spill evidence are intentionally pending.

## Scope and inventory

`DirectPtxConvolutionCoverageManifest` is the executable inventory for the
complete issue: Conv1D/2D/3D, input/weight/bias gradients, transposed,
depthwise, deformable and grouped-deformable, locally connected, unfold and
im2col, mixed precision, and bias/normalization/activation epilogues. Each
entry is assigned either to the first experiment or an explicit future PTX
family. Adding one specialization must never silently mark the family done.

This PR implements one representative inference cell:

- operation: `relu(conv2d(input, weights) + bias)`;
- dtype: FP32 input, weights, bias, accumulator, and output;
- logical/physical input: exact contiguous NCHW `[1,64,16,16]`;
- logical/physical weights: exact contiguous OIHW `[64,64,1,1]`;
- bias: exact contiguous `[64]`;
- output: exact contiguous NCHW `[1,64,16,16]`;
- stride 1, padding 0, dilation 1, groups 1;
- architecture: a distinct experimental SM86 emitter only;
- activation: ReLU, not a runtime selector.

Everything else uses the existing cuDNN/Winograd/tiled/NVRTC route. Ada,
other Ampere variants, Hopper, and Blackwell are separate code-generation and
evidence domains; they are not inferred from SM86.

## Why this is the first cell

A 1x1 convolution is a real and common projection primitive while allowing the
first ABI to prove the parts every later family needs without hiding layout
conversion, padding, or boundary predicates inside the result. Its fixed
geometry is large enough to exercise output-channel and spatial mapping, weight
reuse, a bias epilogue, and activation fusion. It is not assumed to beat cuDNN;
the evidence gate decides whether this emitter should be optimized, replaced,
or abandoned.

## Host admission and physical capability

`DirectPtxConvolutionEligibility.Validate` runs exactly once before module
lookup. Admission requires:

1. the dedicated opt-in gate or master gate;
2. an available CUDA backend;
3. exact SM86 identity;
4. the exact baked geometry and convolution semantics;
5. four non-null device allocations and nonzero pointers;
6. exact allocation byte extents (views and oversized buffers are rejected);
7. 16-byte pointer alignment; and
8. no overlap between output and input, weights, or bias.

Stable rejection codes identify the first failed contract. Rejection returns
to the existing `CudaBackend.Conv2D` + `Conv2DBiasAdd` + `Relu` composition.
The admitted kernel receives only four pointers. There are no shape, stride,
layout, extent, padding, dilation, group, dtype, bias, or activation arguments.

The feature is disabled by default. Enable only for evidence collection with:

```text
AIDOTNET_DIRECT_PTX_CONVOLUTION=1
```

## Emitted PTX and dataflow

The host string-emits `aidotnet_conv2d_n1_c64_h16_w16_k64_k1_bias_relu` for
`sm_86`. A 128-thread block covers 128 exact output elements; 128 blocks cover
all 16,384 outputs, so no device bounds branch is necessary. Each thread:

1. derives baked output-channel and spatial indices from its linear ID;
2. performs exactly 64 unrolled FP32 global input/weight loads and FMAs;
3. reads its channel bias once;
4. applies ReLU in the accumulator register; and
5. writes its final output once.

There is no im2col tensor, convolution temporary, bias output, or activation
output in global memory. The accumulator and fused epilogue remain in
registers. The v1 1x1 cell does not use shared memory or `cp.async`: those add
no reuse for this direct per-output mapping unless a measured output tile is
introduced. Later 3x3/Tensor-Core emitters must document their shared-memory
tile, `cp.async` stages, banking proof, and semantic boundary handling as
separate blueprints.

The checked-in resource budget rejects more than 32 registers/thread, any
static shared memory, any local bytes/thread, or fewer than four active
blocks/SM. These are JIT admission limits, not evidence: the actual values are
unknown until the module is loaded on the target GPU.

## Module lifetime and CUDA graphs

The module uses the shared bounded `DirectPtxKernelCache`. Prewarm performs JIT
and module load outside capture. A capture-time cold miss fails closed without
JIT, allocation, tuning, or file I/O. A resident module used during capture is
pinned because a graph executable retains its `CUfunction`; pinned entries are
not LRU victims and are released only when the owning CUDA backend disposes.
The convolution cache is disposed before the shared `DirectPtxRuntime`.

The first cache contains one shape key but intentionally follows the same
bounded pattern as multi-shape families. Later cells add independent keys for
geometry, dtype, numerical mode, architecture, and fusion semantics; none may
turn these into device-side branches.

## Evidence harness

`--direct-ptx-convolution` prepares resident allocations and compares:

- direct emitted PTX;
- the current AiDotNet established CUDA route (cuDNN when selected), including
  its separate bias and ReLU launches;
- PyTorch cuDNN eager; and
- PyTorch cuDNN CUDA Graph replay.

Each timing cell uses at least 30 warmups and 101 samples. The harness reports
median, P95, P99, mean, GFLOPS/TFLOPS, managed bytes/call, temporary device
bytes, maximum absolute error, registers, static shared bytes, local bytes, and
active blocks/SM. Convolution FLOPs use `2*N*K*H*W*C`; bias and ReLU are not
used to inflate throughput. Inputs, weights, bias, output, and framework
workspaces remain resident, and transfers/setup are outside timing.

The release run must use three independent clean processes and preserve the
complete environment fingerprint (commit, OS, .NET SDK/runtime, Python,
PyTorch, CUDA driver/runtime, cuDNN, GPU UUID/name, SM, clocks/power mode, and
the feature configuration). No CPU row belongs in this table.

### Pending GPU evidence

| Method | median R1/R2/R3 | P95 | P99 | mean | GFLOPS | TFLOPS | managed B/call | temporary device B | max abs error | regs | static/dynamic shared B | local B | blocks/SM | zero executed spill/local proof | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Direct PTX fused | pending | pending | pending | pending | pending | pending | pending | 0 expected; measure | pending | pending | pending | must be 0 | pending | pending Nsight | **not promoted** |
| AiDotNet established CUDA/cuDNN | pending | pending | pending | pending | pending | pending | pending | pending | pending | n/a | n/a | n/a | n/a | n/a | competitor |
| PyTorch cuDNN CUDA Graph | pending | pending | pending | pending | pending | pending | n/a | pending | pending | n/a | n/a | n/a | n/a | n/a | competitor |
| PyTorch cuDNN eager | pending | pending | pending | pending | pending | pending | n/a | pending | pending | n/a | n/a | n/a | n/a | n/a | competitor |

No row may be bolded as a winner until all cells are populated from the same
controlled run set.

## Nsight target and fail-closed verifier

The deterministic target emits one launch and prints its JIT audit:

```powershell
powershell -ExecutionPolicy Bypass -File `
  tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 `
  -Target convolution -OutputCsv <path>
```

The script filters the exact entry point and requires one complete raw row for
every requested metric. The shared verifier rejects missing metric groups and
any executed local, local-load, or local-store instruction. The PR must not
claim zero spills merely because the source declares no `.local` object.

## Promotion gate

Promotion remains blocked until the exact commit demonstrates:

- agreement with a high-precision oracle and current AiDotNet on deterministic
  random, zero, negative-bias, positive-bias, NaN/Infinity-policy, and alias/
  rejection cases;
- prewarm, CUDA graph capture/replay, pinned-lifetime churn, disposal, and zero
  hot managed allocation on hardware;
- zero JIT local bytes and zero executed spill/local counters;
- three clean runs with at least 30 warmups and 101 samples each;
- at least 1.10x median speedup over the fastest eligible resident competitor;
- candidate P95 no worse than competitor P95 +10%; and
- zero avoidable temporary VRAM with the documented error tolerance.

If the v1 emitter fails, it remains disabled and serves as the reproducible
starting point for output tiling, vectorized loads, weight staging, or a
Tensor-Core redesign. Evidence—not the existence of PTX—chooses the next step.

## Assembly-line increments

1. Run and archive v1 correctness/resource/performance evidence.
2. Introduce a measured output tile for 1x1 projection and compare global-load
   reuse against cuDNN without changing the ABI.
3. Add FP16/BF16 Tensor-Core 1x1 families with exact packing and numerical
   contracts.
4. Add 3x3 no-padding tiles, then separately model semantic padding and edge
   tiles without hidden layout conversion.
5. Add deterministic backward-input, backward-weight, and fused bias-gradient
   cells.
6. Apply the same contract/cache/evidence template independently to depthwise,
   transposed, deformable, locally connected, and 3D families.

## ResNet-class evidence campaign (issue #841 promotion track)

The golden-slice cells in `DirectPtxConvolutionCoverageManifest` establish the
ABI, contracts, correctness, and fallback for every scoped family. They are
GPU-correctness verified against a high-precision CPU oracle
(`DirectPtxConvolutionTests` `DriverOnly*` facts) but **none is promoted**: they
are registered `Deferred` in `PtxParityRegistry` pending competitive evidence.
Promotion is decided only by the resident, apples-to-apples gate below.

### Target regime and matrix

"Bigger realistic shapes where cuDNN is strongest" (not dynamic-shape PTX —
#841 forbids runtime shape branches — but realistic-size *exact* SM86
specializations). The evidence matrix is the ResNet workhorse:

| scope | N | C | H | W | K | kernel | stride | pad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet_3x3_c64_56 | 32 | 64 | 56 | 56 | 64 | 3 | 1 | 1 |
| resnet_3x3_c128_28 | 16 | 128 | 28 | 28 | 128 | 3 | 1 | 1 |
| resnet_3x3_c256_14 | 8 | 256 | 14 | 14 | 256 | 3 | 1 | 1 |
| resnet_1x1_c64_56 | 32 | 64 | 56 | 56 | 64 | 1 | 1 | 0 |
| resnet_1x1_c128_28 | 16 | 128 | 28 | 28 | 128 | 1 | 1 | 0 |
| resnet_1x1_c256_14 | 8 | 256 | 14 | 14 | 256 | 1 | 1 | 0 |

### Competitor bar (cuDNN at its best)

`BaselineRunners/py/run_direct_ptx_convolution_resnet_competitors.py` measures
the strongest compiled PyTorch/cuDNN lane per shape: per-shape algorithm
autotuning (`cudnn.benchmark=True`), the autotuned kernel captured in a CUDA
graph (strongest latency-bound lane), and a `torch.compile` lane. FP32/NCHW,
TF32 off, for an apples-to-apples fp32-accumulation comparison. It emits per-shape
JSON with the full environment fingerprint.

### Harness and gate

`--direct-ptx-convolution-resnet` (`DirectPtxConvolutionResnetExperiment`)
benchmarks the established AiDotNet CUDA/cuDNN path and each verified direct-PTX
specialization (median/p95/p99/mean, GFLOPS, managed bytes, max abs error vs the
fp64 oracle), parses the competitor JSON, and prints a per-shape verdict:

- **PASS** = candidate median <= strongest-competitor median / 1.10 **and**
  candidate p95 <= competitor p95 * 1.10;
- **HOLD** otherwise. A HOLD cell is never promoted; it stays experimental with
  the measured numbers recorded, exactly as PR #863 handles losing cells.

### Optimization strategy per op

- **1x1 projection** = a `K x C` by `C x (N*H*W)` GEMM. v1 is a thread-per-output
  channel dot; the promotion attempt stages the `K x C` weight block in shared
  memory, tiles the `M = N*H*W` columns, uses `v4.f32` coalesced loads, and keeps
  the tile accumulators in registers. It competes against cuBLASLt/implicit-GEMM.
- **3x3 s1 p1** — cuDNN uses Winograd F(2x2,3x3)/F(4x4,3x3) or implicit-precomp
  GEMM on Tensor Cores. The promotion attempt is shared-memory input-halo tiling
  with register accumulation over the 9 taps and `C` channels, then a Winograd
  F(2,3) transform stage if the direct tile does not clear the gate.

### Honest expectation

At these compute-bound shapes cuDNN's Winograd + Tensor-Core lanes are extremely
strong; several cells are expected to HOLD, precisely as #863's normalization
backward/reduction cells did. Wins are most likely in the resident, fused,
latency-bound sub-regime where cuDNN's fixed launch/workspace overhead dominates.
The deliverable is the strong-attempt kernels **plus the honest measured table**,
never an overclaim; rejected optimization attempts are retained as evidence.

### Reproduction (idle GPU required)

```text
# competitor bar
python tests/AiDotNet.Tensors.Benchmarks/BaselineRunners/py/run_direct_ptx_convolution_resnet_competitors.py

# candidate + gate (screening; promotion needs 3 clean processes)
dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks -- --direct-ptx-convolution-resnet
```
