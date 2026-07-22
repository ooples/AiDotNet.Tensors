# Direct PTX stochastic-kernel blueprint: issue #849

Status: full experimental implementation, disabled by default. Static proof is
checked in; GPU correctness, benchmark, and Nsight promotion evidence remains
pending and no performance win is claimed.

This document is the reusable assembly-line contract for RNG, dropout,
sampling, stochastic activations, and stochastic-noise kernels. The executable
inventory is `DirectPtxRngCoverageManifest`; a public or CUDA entry point is not
considered assigned unless that manifest names its direct specialization or an
explicit fail-closed reason.

## Implemented operation inventory

| Operation family | Direct PTX implementation | Exact FP32 shapes | Fused boundary and required global traffic |
|---|---|---|---|
| Uniform and random initialization | `PtxPhiloxFillF32Kernel.Uniform` | N=4,096 / 65,536 / 1,048,576 | no reads; one float4 output write |
| Normal and Gaussian noise | `PtxPhiloxFillF32Kernel.Normal` | same N | Philox plus two Box-Muller pairs in registers; one float4 write |
| Bernoulli/dropout masks | `PtxPhiloxFillF32Kernel.BernoulliMask` and `DropThresholdMask` | same N | predicate and scale remain in registers; one required public-mask write |
| Dropout forward | `PtxFusedPhiloxDropoutF32Kernel` | same N | input read plus required output and saved-mask writes; no random-mask intermediate or second launch |
| Dropout backward | `PtxDropoutBackwardF32Kernel` | same N | gradient and saved-mask reads plus gradient write |
| Bias + dropout | `PtxFusedBiasPhiloxDropout256F32Kernel` | [16/256/4,096, 256] | bias read, input read, required output and saved-mask writes; no biased-tensor temporary |
| Gumbel softmax | `PtxFusedGumbelSoftmax32F32Kernel` | [128/2,048/32,768, 32] | Philox Gumbel perturbation and warp max/sum remain in registers; only logits read and probabilities written |
| Gumbel backward | `PtxGumbelSoftmaxBackward32F32Kernel` | same rows x 32 | softmax Jacobian reduction and temperature scale remain in one warp; no reduction temporary |
| Categorical sampling | `PtxPhiloxCategorical32F32Kernel` | same rows x 32 | warp CDF scan, draw, and one-hot selection remain in registers; no CDF or index intermediate |
| NeRF importance sampling | `PtxFusedImportanceSampling64F32Kernel` | 64/1,024/16,384 rays x 64 coarse x 64 fine | each coarse t/weight read once into two bank-aligned shared arrays; unrolled CDF traversal writes only final samples |
| Training RReLU | `PtxFusedPhiloxRreluF32Kernel` | same vector N | slope generation and negative-side multiply are fused; input read plus required output and backward-visible saved-noise writes |
| Saved-noise RReLU forward/backward | `PtxRreluF32Kernel` | same vector N | exact float4 ports of both existing NVRTC kernels; no intermediate |
| Advertised DDIM step | `PtxFusedDdimStepF32Kernel` | same vector N | two float4 reads and one output write after host-side FP64 coefficient collapse |

The tensor random-uniform/range/normal construction and in-place entry points,
`Tensor.CreateRandom`, `RandomUniformGpu`, and `RandomNormalGpu` route through the
same admitted fill specializations. `TensorDropoutMask` routes directly through
the appropriate mask specialization for both explicit and engine-owned seeds.
The host-result `GenerateDropoutMask` and `GenerateGaussianNoise` operations use
the same PTX generation only at admitted FP32 lengths, followed by the public
contract's unavoidable device-to-host result transfer; resident callers should
use the tensor/GPU APIs.

`CuRand.Uniform`, `CuRand.Normal`, and `CuRand.Bernoulli` return host tensors.
They deliberately retain their host/provider implementation because wrapping
PTX would add a private device allocation, synchronization, and D2H copy.
Cryptographic and secure-random APIs are also explicitly excluded: deterministic
Philox cannot preserve unpredictability. `GenerateSecureRandomUniform` bypasses
the direct route even when the feature gate is enabled.

## Versioned RNG ABI

The first ABI is `philox4x32-10-v1`:

| Field | Mapping |
|---|---|
| 64-bit seed | key `(seed.low32, seed.high32)` |
| 64-bit counter offset | low 64 bits of the 128-bit counter |
| 64-bit subsequence | high 64 bits of the 128-bit counter |
| float4 group `g` | counter `(counterOffset + g, subsequence)` |
| vector element `i` | word `i mod 4` of group `floor(i / 4)` |
| keep mask | unsigned word `< floor(keepProbability * 2^32)` |
| drop-threshold mask | unsigned word `< threshold` means dropped |

Counter ownership is scheduling-independent:

- vector fills, dropout, bias-dropout, and RReLU consume one counter per four
  adjacent outputs;
- Gumbel softmax consumes one counter per element and uses word zero;
- categorical consumes one counter per row and uses word zero;
- importance sampling consumes one counter per final sample and uses word zero.

The benchmark CPU oracle uses the same ten-round reference and explicit mapping.
Seed, subsequence, and counter-offset tests include the published all-zero
Philox4x32-10 vector. No variable-length rejection loop owns an implicit stream.

## Physical ABI and fail-closed admission

Every specialization is SM86-only and FP32-only. Ada, other Ampere SMs,
Hopper, and Blackwell remain separate, unmeasured architecture families and
fall back with an exact reason. Every tensor contract fixes:

- logical and physical extents;
- dense vector or row-major layout identity;
- exact allocation byte length and zero view offset;
- 4- or 16-byte alignment according to transaction width;
- read/write access and non-alias rules;
- block geometry, resource budget, and minimum active blocks/SM.

Admission validates those facts once on the host and creates
`DirectPtxTensorView` capability tokens. The admitted PTX accepts pointers,
mathematical scalars, and explicit RNG state only. It has no element-count,
stride, layout, dtype, alignment, padding, or tail argument and no generic
dynamic-shape branch. Non-contiguous public tensors are not silently assigned a
different physical meaning.

## Fused memory dataflow

Random state, thresholds, slopes, masks that are not public outputs, Gumbel
perturbations, warp reductions, CDF state, and interpolation state remain in
registers. Dropout and RReLU write masks/slopes only because their public
backward contract requires them. Bias-dropout eliminates the previous biased
activation temporary. Categorical eliminates both CDF and sampled-index
buffers. Importance sampling is the only family that benefits from shared
memory: each warp owns one ray, lane `i` owns elements `i` and `i+32`, so the
two 64-float arrays are aligned and bank conflict free for each cooperative
load/traversal step.

`cp.async` and Tensor Cores are intentionally absent. These kernels either have
no reusable global tile or perform scalar probability arithmetic; adding an
asynchronous copy pipeline or MMA instruction would not match their dataflow.
Architecture-specific instructions are required only when measurement shows a
benefit, not as a checklist decoration.

## Runtime, cache, and capture contract

`AIDOTNET_DIRECT_PTX_RNG_DROPOUT=1` is snapshotted when `CudaBackend` is
constructed and is off by default. Each kernel family has a bounded exact-shape
module cache. Cache keys include operation semantics and shape; audit identity
also includes PTX SHA-256 and the GPU/SM/driver fingerprint. Disposal unloads
all RNG modules before the CUDA context and stream are released.

The runtime, emitters, admission checks, and CUDA dispatch hooks compile on both
net471 and modern .NET targets. They use the shared `PtxCompat` layer for BCL
features and pointer conversions missing from .NET Framework; no target-specific
guard is allowed to compile the PTX route out of one supported framework.

Warm dispatches use direct cache probes; factory delegates exist only in
no-inline cold creation helpers. Saved-noise RReLU likewise uses explicit
forward/backward calls instead of a capturing launch delegate. This makes the
steady-state managed-allocation requirement structural as well as measurable.

Every family exposes prewarm and audit entry points. A cold CUDA graph capture
rejects rather than JIT-compiling. A warm capture pins its module so LRU
eviction cannot invalidate a captured `CUfunction`. Capture performs no tuning,
allocation, file I/O, or cache eviction. These exact domains have one legal
plan per key, so there is no artificial autotuning search; future operations
with multiple legal plans must use a bounded device/driver/shape/semantics-keyed
plan cache.

Any gate, SM, shape, extent, pointer, alignment, alias, numerical, resource,
JIT, cache, or capture rejection uses the established CUDA/CPU path. The new
categorical API has a seeded, resident GPU fallback based on the Gumbel-max
identity, so disabling direct PTX does not silently move valid FP32 last-axis
sampling to the CPU. Secure semantics and host-only provider contracts bypass
direct PTX explicitly.

## Resource and deterministic static proof

Module loading records function registers, static shared memory, local bytes,
max threads, PTX/binary versions, active blocks/SM, PTX hash, complete JIT log,
and device fingerprint. A specialization is rejected for any nonzero local
bytes or for violating its per-kernel register/shared/occupancy budget.

Focused non-GPU tests prove:

- ten Philox rounds, counter-domain reproducibility, and the published vector;
- fixed instruction/dataflow structure for every emitter;
- exact supported shape buckets and unmeasured-SM rejection;
- no runtime stride/shape/tail ABI or emitted local-memory declaration;
- mask polarity, Box-Muller pairs, warp reductions, shared layout, and fused
  consumer boundaries;
- unique operation-manifest assignment, explicit secure/host exclusions,
  transfer-ledger validity, module pinning, and deterministic cleanup.

Static emitter tests do not prove that the NVIDIA driver accepts the PTX or that
it executes correctly. Those claims remain pending until the device run.

## Resident benchmark

With exclusive access to the admitted GPU and CUDA-enabled Python PyTorch:

```powershell
$env:AIDOTNET_DIRECT_PTX_RNG_DROPOUT='1'
dotnet run -c Release -f net10.0 --project tests/AiDotNet.Tensors.Benchmarks -- --direct-ptx-rng-stochastic 3
```

Use `--no-external` only for harness diagnostics; it is not promotion evidence.
The full run covers every table row and every exact shape with:

- direct PTX launch and direct PTX CUDA-graph replay;
- an isolated AiDotNet backend with the direct gate disabled, preventing the
  established row from silently dispatching the candidate;
- PyTorch eager, CUDA graph, and `torch.compile(mode="max-autotune",
  fullgraph=True)` where supported; unsupported peers are printed as skipped,
  not silently omitted.

Every cell uses 30 warmups, 101 samples, ten resident launches per device-time
sample, and three independent runs. Output is grouped by operation and shape and
reports device and synchronized end-to-end mean/median/P95/P99, effective GB/s,
managed bytes/call, temporary or peak device bytes, maximum oracle/semantic
error, registers, shared/local bytes, blocks/SM, and complete .NET/PyTorch/CUDA/
GPU/driver/audit fingerprints. GFLOPS is intentionally not fabricated for RNG
and probability kernels; GB/s is the meaningful throughput measure.

## Nsight zero-spill proof

```powershell
tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 -Target rng-stochastic
```

The deterministic target emits exactly 45 launches: one launch for every
kernel family and exact shape. The verifier requires all launches plus the
requested metric groups and rejects any executed register-spill instruction,
local load, local store, or local spilling request. Driver-reported local bytes
alone are not sufficient. No Nsight result has been collected in this change.

## Promotion evidence (intentionally pending)

| Operation group | Shapes | Direct PTX | Direct graph | AiDotNet established | PyTorch eager/graph/compile | Oracle | Resources | Spill/local counters | Promotion |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform / normal / Gaussian / masks | three vector N | pending | pending | pending | pending | pending | pending | pending | pending |
| dropout forward / backward | three vector N | pending | pending | pending | pending | pending | pending | pending | pending |
| RReLU fused / saved forward / backward | three vector N | pending | pending | pending | pending | pending | pending | pending | pending |
| DDIM | three vector N | pending | pending | pending | pending | pending | pending | pending | pending |
| Gumbel forward / backward / categorical | three row counts x 32 | pending | pending | pending | pending | pending | pending | pending | pending |
| importance sampling | three ray counts x 64 x 64 | pending | pending | pending | pending | pending | pending | pending | pending |
| bias-dropout | three row counts x 256 | pending | pending | pending | pending | pending | pending | pending | pending |

No cell may be filled from another machine, a CPU competitor, a partial run, or
a screening result. Promotion requires at least 1.10x candidate median speedup
over the strongest eligible peer, candidate P95 no worse than peer P95 +10%,
zero hot managed allocation, zero avoidable temporary VRAM, operation-specific
correctness, zero local bytes, and zero executed spill/local evidence. A losing
or unproven specialization remains disabled and on the established fallback.

## Assembly-line extension order

1. Inventory every public and CUDA entry point before coding.
2. Freeze mathematical, backward, deterministic, and counter-consumption semantics.
3. Freeze logical/physical layouts and the true fusion boundary.
4. Add exact fail-closed admission and a disabled per-family gate.
5. Emit architecture- and exact-shape PTX without dynamic layout paths.
6. Add bounded cache, prewarm, capture pinning, cleanup, and audit.
7. Add high-precision/current-backend oracles and every rejection test.
8. Add full-operation resident competitors and deterministic Nsight coverage.
9. Run three clean suites and populate the grouped evidence table.
10. Promote only cells that pass every correctness, tail-latency, allocation,
    temporary-memory, speed, and zero-spill gate.
