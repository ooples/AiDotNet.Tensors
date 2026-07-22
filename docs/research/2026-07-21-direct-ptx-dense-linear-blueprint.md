# Direct-PTX dense linear and GEMM blueprint

Date: 2026-07-21

Tracking issue: #836

Parent blueprint: `2026-07-20-fused-attention-championship-blueprint.md`

## Status and promotion rule

This issue now has an executable 44-entry inventory and a direct Driver-API PTX
implementation for every assigned dense-product, linear, epilogue, backward,
LoRA, and vector-product family. Implementation is not the same as promotion.
Only an exact specialization that has passed the timing gate is reachable from
the disabled-by-default public preview flag; the remaining correct
implementations require the internal experiment scope and fail closed in normal
dispatch. Release promotion is stricter than preview admission and additionally
requires executed-spill evidence from the committed head.

The original screening candidate was the Ampere FP32 decode specialization:

```text
input [1,512] + output-major weights [2048,512] + bias [2048]
    -> gelu_tanh(input @ transpose(weights) + bias) [1,2048]
```

The final clean release matrix did not reproduce a gate-qualified win for this
cell or any other issue cell. All general-M, decode, FP16/BF16, backward, LoRA,
CE, dot, outer, and batched cells therefore remain experimental until a later
clean three-run matrix proves at least a 1.10x conservative median win,
acceptable P95, zero hot managed bytes, zero avoidable temporary VRAM,
numerical tolerance, and zero spill evidence.
Correctness-only scalar FP16 PTX is deliberately not described as a Tensor Core
kernel.

The public rollout switch is disabled by default:

```text
AIDOTNET_DIRECT_PTX_FUSED_LINEAR=1
```

Unqualified shapes, semantics, layouts, and architectures continue through the
established backend. The implementation never silently widens its admission
domain because a related shape happened to work.

## The assembly-line pattern

Every future PTX operation should follow the same sequence:

1. Inventory every public API and existing CUDA/NVRTC/vendor route in scope.
2. Define one semantic cell: formula, forward/backward mode, dtype, numerical
   mode, exact logical shape, exact physical layout, alignment, alias rules,
   and target architecture.
3. Choose the intended fused boundary and write a global-memory ledger before
   emitting instructions. Any intermediate global write needs a justification.
4. Bake shape, layout, activation, transpose, batch, dtype, and numerical mode
   into a versioned blueprint. Keep the launch ABI pointer-only.
5. Validate the tensor contracts once on the host. Admitted PTX contains no
   stride checks, layout switches, generic shape branches, or leading-dimension
   parameters.
6. Stage reusable panels in shared memory, retain accumulators and epilogues in
   registers, and write each final output once. Use warp shuffles, vector loads,
   `cp.async`, or `mma.sync` only when the exact cell benefits measurably.
7. JIT outside capture, reject nonzero local bytes before caching, and record
   PTX hash, GPU UUID, SM, driver, resources, occupancy, and JIT log.
8. Prewarm and pin modules before graph capture. The captured path may not JIT,
   tune, allocate, perform file I/O, or evict a referenced module.
9. Compare resident NVIDIA paths only: current AiDotNet, the strongest eligible
   NVIDIA library, PyTorch eager/Graph, and direct PTX. Packing, uploads, and
   downloads remain outside device timing.
10. Promote only exact cells that pass all gates. Keep losing cells useful as
    direct-PTX correctness baselines, but fail them closed in production.

This separation is essential: a broad generic kernel is useful for semantic
coverage, while a narrow winning specialization is what earns production
dispatch.

## Inventory and implementation ownership

`DirectPtxDenseLinearCoverageManifest` is the executable source of truth. Its
44 unique entries cover:

- FP32 `Gemm`, `GemmAsync`, `MatMul`, `MatMulTransposed`, batched GEMM, and
  fanout;
- FP16/BF16 GEMM, `Hgemm`, FP16 output conversion, mixed-precision fanout, and
  transpose-free backward;
- `GemmBias*`, fused bias/activation async, and all forward fused-linear
  activations;
- ReLU, sigmoid, tanh, tanh-GELU, and swish fused-linear backward;
- indexed and dense-target fused linear cross entropy;
- fused LoRA forward;
- dot, strided dot, outer, batched dot, and batched outer products; and
- the corresponding rank-aware `DirectGpuTensorEngine` entry points.

No manifest entry has `PlannedDirectPtx` status. `ExperimentalDirectPtx` means
real emitted PTX plus backend routing and tests, not a placeholder or a CUDA C
wrapper.

## Kernel families and fused dataflow

### Decode linear

One warp owns two output rows. Lanes stream adjacent K values and output-major
weights, hold two dot products in registers, reduce with shuffles, add bias,
evaluate tanh-GELU, and perform the only output stores. There is no shared or
global intermediate.

### General FP32 GEMM and fused linear

The current Ampere experiment tile is `BM=16, BN=16, BK=32, TM=2, TN=2`:
64 threads cooperatively stage one A tile and one weight tile in 4 KiB of
shared memory. Each thread holds a 2-by-2 FP32 accumulator tile. Input-major
`[K,N]` and prepacked output-major `[N,K]` are separate PTX modules; the layout
choice is absent from the launch ABI and the instruction path. Bias and the
selected activation are applied to accumulator registers before the one final
store. Batched GEMM bakes contiguous batch strides into address arithmetic.

### FP16 and BF16

The correctness baseline bakes input dtype, output dtype, transposition, batch,
and FP16-versus-FP32 accumulation into separate modules. It reads 16-bit
operands, accumulates without a global conversion buffer, converts only at the
final store, and supports the two transpose-free backward products. It is a
scalar baseline and remains fail-closed pending an architecture-specific
Tensor Core replacement.

### Fused-linear backward

Three pointer-only entry points write `dInput`, `dWeight`, and `dBias` directly.
Each recomputes the activation derivative in registers, so no masked-gradient
tensor is materialized. ReLU and tanh-GELU consume saved preactivation;
sigmoid and tanh consume saved output, matching the established contracts.
Gradient outputs must be disjoint from each other and from saved forward
tensors.

### LoRA

One block owns one batch row. It computes `input @ A` into a rank-sized shared
projection, reuses that projection for `@ B`, fuses scaling and the base
residual in registers, and writes the final output once. Exact output aliasing
with `baseOutput` is allowed; partial overlap is rejected.

### Linear cross entropy

One block owns one row. Projection logits are recomputed as needed but never
materialized. Stable max and exponential-sum reductions use shared memory; the
selected/dense loss remains in registers and contributes to the resident mean
scalar. The block width is specialized to the baked vocabulary bucket
(32/64/128/256 threads), avoiding idle lanes for small vocabularies. The final
row accumulation is atomic, so this version fails closed in deterministic
mode.

### Dot and outer products

Dense and batched dot use register partials, warp shuffles, eight shared warp
partials (32 bytes), one block barrier, and one final global store per result.
Strided dot bakes its valid interval, offset, and step into the module rather
than loading dynamic stride arguments. Outer and batched outer map one output
element per thread and write it once.

## Global-memory ledger

| Family | Global inputs | On-chip lifetime | Global outputs |
|---|---|---|---|
| decode linear | input, output-major weights, bias | accumulators and GELU in registers | final output once |
| tiled GEMM/linear | A and weight panels, optional bias | panels in shared; accumulator/epilogue in registers | final C once |
| FP16/BF16 GEMM | 16-bit A/B | FP32 or FP16-rounded accumulator in registers | converted C once |
| backward linear | grad output, input, weights, saved activation | derivative and reduction accumulators in registers | dInput/dWeight/dBias once |
| LoRA | input, base, A, B | rank projection in shared; residual epilogue in registers | final output once |
| linear CE | hidden, weight, bias, target | logits in registers; reductions in shared | one atomic scalar contribution/row |
| dot | two vectors | register partials, then 32-byte shared warp totals | one scalar/batch |
| outer | two vectors | product in a register | each output element once |

There are no shape-dependent temporary device allocations in admitted direct
paths. Caller-owned outputs and the backend stream are reused.

## Contiguous-layout and ABI contract

Direct PTX accepts capability-style `DirectPtxTensorView` values constructed
only after the host proves:

- non-null device pointer and exact physical byte extent;
- declared FP32, FP16, or BF16 physical dtype;
- canonical vector, row-major 2D/3D, input-major weight, or output-major weight
  identity;
- 16-byte pointer alignment and element-compatible offsets;
- declared read/write access and non-overlap rules; and
- no silent padding beyond the blueprint's physical extent.

The emitter then receives no strides. Contiguity is therefore an admission
fact, not a branch executed for every element. Views that cannot prove the
required identity must be materialized or packed by the tensor/weight registry
before dispatch, or use the established fallback. Immutable output-major decode
weights are a model-load concern and packing time is never included in kernel
timing.

## Architecture, cache, and capture

The checked-in dense-linear specializations are admitted only on measured
GA10x/SM86. Ada, Hopper, and Blackwell do not inherit Ampere admission; each
needs its own emitter/tile catalog, resource limits, and evidence. Unsupported
architectures fail closed before module creation.

Every module key includes all baked semantics. Caches are bounded, disposed
with `CudaBackend`, and pin graph-referenced modules. Fast cache hits avoid
capturing delegates, preserving zero managed allocation. A capture miss returns
an explicit prewarm-required reason rather than JITing inside capture.

The loader rejects any JIT result with nonzero local bytes or a blueprint
resource-budget violation. Audit records include blueprint id, SHA-256, device
fingerprint, PTX/binary version, registers, shared/local bytes, maximum threads,
active blocks per SM, occupancy inputs, and the JIT log.

## Benchmark contract

The per-process runner uses 30 warmups, 101 samples, ten launches per CUDA-event
sample, and a 64-launch tiered-JIT settling window before its exact allocation
measurement. It reports device and E2E
mean/median/P95/P99, GFLOPS/TFLOPS, managed bytes/call, temporary device bytes,
maximum error, registers, shared/local bytes, active blocks/SM, blueprint id,
fingerprint, and PTX hash. Missing established kernels are emitted as explicit
`unsupported` JSON rows; they do not abort the matrix or masquerade as wins.

Final evidence uses `run-direct-ptx-release-evidence.ps1 -Issue836Only`. The
orchestrator starts three separate .NET processes and three separate Python
processes, requires three consecutive clean `nvidia-smi` samples before each
suite, rejects foreign compute and unsafe temperature, records start/end GPU
snapshots and exact commands, retries contaminated attempts, and hashes every
accepted/rejected log in a manifest. Its machine-readable release-gate file
retains every per-run verdict; a losing cell is never averaged into a win.

PyTorch 2.12.1+cu130 eager and CUDA Graph routes are measured by the companion
Python runner. CUDA Graph rows use preallocated resident outputs and report
zero replay-time device allocation. The strongest eligible competitor for a
cell determines its gate.

## Current release evidence

The final issue matrix ran from clean implementation commit
`814d5460d672ecb55a8c04afa3c66e883bc79843` on 2026-07-22. The environment was
Windows 10.0.26200, .NET 10.0.10 / SDK 10.0.302, NVIDIA GeForce RTX 3080,
SM86, driver 610.47, and PyTorch 2.12.1+cu130. Three independent .NET and
Python process pairs produced 144 AiDotNet/NVIDIA rows and 72 PyTorch rows.
Every accepted process used the benchmark contract above; the evidence
orchestrator rejected no attempt.

The result is an honest `timing-fail`, so all 44 issue cells remain
experimental and fail closed. No operation cleared the full conservative gate
in all three runs. Direct PTX had the best average device median for dot,
outer, and strided dot, but those cells still missed either the 1.10x
cross-run median requirement or the E2E requirement. The other nine groups
were won by an established AiDotNet, NVIDIA, or PyTorch route. This evidence
does not support release promotion.

All 72 Direct PTX rows reported zero managed bytes/call, zero temporary device
bytes, zero JIT local bytes/thread, and error within the declared tolerance.
The complete per-operation tables group every competitor with mean, median,
P95, P99, TFLOPS/GFLOPS, allocation, temporary memory, error, and resource
metrics in the [checked-in evidence report](2026-07-22-direct-ptx-dense-linear-results.md).
Its source report SHA-256 is
`644de70469a7c04e79b0889468f1380c6e9869670d3dcde247491383d35d3fe4`;
the machine-readable release-gate SHA-256 is
`a229d8abaf100ccd47ae594c8625755fc79426bdc15112f59fd9410edae16590`.

Nsight Compute CLI 2025.4.1 was extracted from the official package and
attached to the committed kernel target on 2026-07-22. Counter collection still
returned `ERR_NVGPUCTRPERM`; the resulting diagnostic CSV is not spill proof.
Driver-JIT local size zero is enforced now, but no specialization is claimed as
release-promoted until the checked-in CSV verifier observes zero executed local
loads, local stores, and local/spill instructions.

## Reproduction

```powershell
dotnet build tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release --no-restore

dotnet tests/AiDotNet.Tensors.Benchmarks/bin/Release/net10.0/AiDotNet.Tensors.Benchmarks.dll `
  --direct-ptx-dense-linear-full 1

powershell -ExecutionPolicy Bypass -File `
  tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-release-evidence.ps1 `
  -Issue836Only -Runs 3 -SkipBuild `
  -OutputDirectory artifacts/perf/direct-ptx-dense-linear

powershell -ExecutionPolicy Bypass -File `
  tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 `
  -Target fused-linear
```

Use `--no-python` only for focused C# screening and `--only=<operation>` for a
single family. Neither shortcut is sufficient for final acceptance.
