# Direct-PTX dense linear and GEMM blueprint

Date: 2026-07-21

Tracking issue: #836

Parent blueprint: `2026-07-20-fused-attention-championship-blueprint.md`

## Verdict

The first issue-#836 golden slice is an Ampere FP32 decode-token
linear+bias+tanh-GELU specialization:

```text
input [1,K] + output-major weights [N,K] + bias [N]
    -> gelu_tanh(input @ transpose(weights) + bias) [1,N]
```

`K=512,N=2048` passes the production device-time gate against current
AiDotNet, NVIDIA cuBLASLt, and PyTorch CUDA Graphs. It is disabled by default
behind `AIDOTNET_DIRECT_PTX_FUSED_LINEAR=1` (or the master direct-PTX gate).
The measured `256x256` and `1024x4096` kernels remain experiment-only: the
small cell showed an earlier P95 excursion, while the large bandwidth-bound
cell wins but reaches only a conservative 1.06x over PyTorch Graph rather than
the required 1.10x. Production dispatch fails closed for both with
`fused-linear-performance-gate-not-met`.

This is one golden increment, not closure of #836. The executable 39-entry
coverage manifest assigns general GEMM, transposed/batched/fanout products,
FP16 families, all bias/activation epilogues, forward/backward linear kernels,
dot/outer products, cross entropy, and LoRA to later direct-PTX increments.

## Assembly-line decisions

1. Choose an exact semantic cell, dtype, architecture, and physical layout.
2. Prepack immutable weights outside timing and retain a layout identity; do
   not inspect strides in the kernel.
3. Emit a pointer-only PTX ABI with K/N and launch geometry baked into the
   module.
4. Fuse reductions, bias, and activation in registers and write the final
   tensor once.
5. Audit the JIT result before caching; any local bytes reject admission.
6. Prewarm before graph capture and keep the hot dispatch allocation-free.
7. Benchmark current AiDotNet, the strongest NVIDIA library path, and PyTorch
   eager/Graph using resident operands and identical arithmetic.
8. Promote only cells that pass median, tail, allocation, numerical, and spill
   gates; retain exact fallback reasons for every other cell.

The generic `CudaBackend.FusedLinearGELU` contract uses input-major `[K,N]`
weights and is deliberately unchanged. The new
`FusedLinearGELUTransposedM1` route requires output-major `[N,K]` weights, so a
caller cannot silently reinterpret an existing allocation. Packing is a
model-load/weight-registry responsibility and is excluded from hot timing.

## Fused dataflow and global-memory ledger

One warp owns two output rows. Its lanes stream adjacent K values and weights,
accumulate two FP32 dot products in lane-private registers, perform two warp
shuffle reductions, and let lane zero add bias, evaluate tanh-GELU, and commit
the two final scalars.

| Allocation | Global reads | Global writes | Lifetime |
|---|---:|---:|---|
| input | one scalar or FP32x4 per lane/K step/output pair | 0 | read-only cache stream |
| weights | each output-major weight exactly once | 0 | read-only cache stream |
| bias | one scalar per output | 0 | register epilogue |
| output | 0 | one scalar per output | final commit only |

There is no GEMM output intermediate, bias intermediate, activation
intermediate, shared memory, scratch allocation, or second global write. The
dominant vector transactions are naturally aligned and warp-coalesced. With no
shared allocation, shared-bank conflicts are impossible.

Shared staging, 256-thread blocks, tile-major weight repacking, L2 prefetch
hints, four outputs/warp, and full K unrolling were each implemented and
measured during the experiment. None improved the championship cell; shared
staging and unrolling were regressions. They are not retained merely because
they are fashionable. `cp.async` is appropriate for future multi-row FP16/BF16
Tensor Core tiles with reusable panels, not this M=1 FP32 GEMV stream. TF32
`mma.sync` would also change the declared FP32 numerical contract.

## Formal ABI and dispatch

The launch has four pointer parameters and no scalar parameters:

| Tensor | Layout | Exact extent | Access | Alignment |
|---|---|---|---|---:|
| input | `Vector` | `[K]` FP32 | read | 16 B |
| weights | `LinearWeightOutputMajor` | `[N,K]` FP32 | read | 16 B |
| bias | `Vector` | `[N]` FP32 | read | 16 B |
| output | `Vector` | `[N]` FP32 | write | 16 B |

The boundary checks the gate, Ampere architecture, promoted shape, exact byte
extents, layout token, alignment, and output aliasing once. The emitted PTX has
no shape, stride, leading-dimension, batch, activation-mode, or layout
argument. Unsupported shapes, architectures, extents, aliases, and capture
state have stable diagnostic reasons and fall back to the established
MatMulTransposed+BiasAdd+GELU composition.

The specialization key is `(K,N)`. Modules use the bounded direct-PTX cache,
are disposed with `CudaBackend`, and carry the standard blueprint id, PTX hash,
GPU UUID/SM/driver fingerprint, function resources, occupancy, and JIT log.
Capture may only use a prewarmed module and performs no JIT, tuning,
allocation, I/O, or cache eviction. A captured specialization is pinned in the
bounded cache so later specialization churn cannot unload its retained
`CUfunction` while a graph executable still references it.

## NVIDIA benchmark evidence

Environment: Windows 10.0.26200, .NET 10.0.10 / SDK 10.0.302, NVIDIA GeForce
RTX 3080 12 GiB, SM86, driver 610.47, PyTorch 2.12.1+cu130. Every cell uses 30
warmups and 101 samples in each of three runs. CUDA-event samples average ten
resident launches; E2E samples are one launch plus synchronization. Packing,
uploads, downloads, and allocations are outside timing.

`GFLOPS = 2*K*N / device-time`; bias and GELU operations are not inflated into
the rate. The following ranges are the final three-run capture. The executable
TUI also prints E2E mean/median/P95/P99 for every row.

| Shape | Method | device mean us | median us | P95 us | P99 us | GFLOPS | managed B/call | temp device B | max error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256x256 | Direct PTX | 11.61-11.89 | 10.56-11.17 | 16.49-19.15 | 17.82-26.83 | 11.7-12.4 | 0 | 0 | 3.7e-8-5.2e-8 |
| 256x256 | AiDotNet NVRTC | 14.83-16.43 | 13.62-14.23 | 18.02-25.70 | 27.65-31.13 | 9.2-9.6 | 0 | 0 | 2.6e-8-4.5e-8 |
| 256x256 | NVIDIA cuBLASLt | 24.16-26.99 | 23.76-25.91 | 28.87-38.40 | 33.48-51.20 | 5.1-5.5 | 248 | 0 | 3.7e-8-5.2e-8 |
| 256x256 | cuBLASLt Graph | 19.19-21.61 | 18.94-20.89 | 24.06-28.26 | 26.93-58.98 | 6.3-6.9 | 0 | 0 | 3.7e-8-5.2e-8 |
| 256x256 | PyTorch eager | 70.00-90.08 | 58.88-96.36 | 104.65-166.50 | 150.12-322.15 | 1.4-2.2 | n/a | 2,048 | 0.7e-8-1.0e-8 |
| 256x256 | PyTorch Graph | 12.56-23.36 | 12.37-25.09 | 15.87-35.64 | 16.39-36.66 | 5.2-10.6 | n/a | 0 | 0.7e-8-1.0e-8 |
| 512x2048 | Direct PTX | 10.01-11.90 | 9.63-11.67 | 13.62-18.22 | 18.23-21.59 | 179.6-217.9 | 0 | 0 | 1.1e-7-1.3e-7 |
| 512x2048 | AiDotNet NVRTC | 23.89-25.19 | 23.35-23.86 | 24.88-26.42 | 26.73-33.59 | 87.9-89.8 | 0 | 0 | 5.2e-8-6.7e-8 |
| 512x2048 | NVIDIA cuBLASLt | 31.32-38.44 | 30.60-33.16 | 35.01-65.95 | 47.82-70.86 | 63.2-68.5 | 248 | 0 | 1.1e-7-1.3e-7 |
| 512x2048 | cuBLASLt Graph | 22.63-31.35 | 21.30-24.06 | 31.54-38.20 | 33.38-42.39 | 87.2-98.5 | 0 | 0 | 1.1e-7-1.3e-7 |
| 512x2048 | PyTorch eager | 59.69-76.67 | 58.68-67.69 | 64.51-111.41 | 68.81-125.75 | 31.0-35.7 | n/a | 16,384 | 1.9e-8-2.4e-8 |
| 512x2048 | PyTorch Graph | 13.15-21.30 | 13.21-15.24 | 16.90-34.41 | 17.72-81.72 | 137.7-159.0 | n/a | 0 | 1.9e-8-2.4e-8 |
| 1024x4096 | Direct PTX candidate | 23.68-24.04 | 23.14-23.65 | 25.60-26.93 | 28.67-30.00 | 354.6-362.5 | 0 | 0 | 2.7e-7-4.3e-7 |
| 1024x4096 | AiDotNet NVRTC | 87.51-88.29 | 87.35-87.55 | 88.17-91.75 | 89.08-93.70 | 95.8-96.0 | 0 | 0 | 0.9e-7-3.6e-7 |
| 1024x4096 | NVIDIA cuBLASLt | 29.72-33.48 | 29.39-30.62 | 33.59-45.47 | 39.01-96.15 | 274.0-285.4 | 248 | 0 | 3.0e-7-4.3e-7 |
| 1024x4096 | cuBLASLt Graph | 42.28-43.74 | 41.16-42.09 | 44.34-50.79 | 61.24-62.87 | 199.3-203.8 | 0 | 0 | 3.0e-7-4.3e-7 |
| 1024x4096 | PyTorch eager | 56.01-62.54 | 52.84-55.28 | 69.53-102.60 | 91.14-259.89 | 151.7-159.0 | n/a | 32,768 | 3.7e-8-6.7e-8 |
| 1024x4096 | PyTorch Graph | 25.84-27.03 | 25.09-25.19 | 27.65-30.11 | 28.88-36.76 | 333.0-334.4 | n/a | 0 | 3.7e-8-6.7e-8 |

For the promoted 512x2048 cell, the conservative cross-run median comparison
is `13.21 / 11.67 = 1.13x` versus the strongest PyTorch Graph competitor. Each
paired direct P95 is below PyTorch P95, managed allocation is 0 B/call,
temporary VRAM is 0 B, and maximum error is below `1.3e-7`. The 1024x4096
candidate's conservative median win is only `25.09 / 23.65 = 1.06x`, so it is
correctly not admitted despite beating all measured competitors in raw time.

## Resource and spill evidence

The promoted 512x2048 module records:

```text
BlueprintId: fused-linear-bias-gelu-v1-Ampere-decode-fp32-m1-k512-n2048
DeviceFingerprint: gpu-79d5ac8ef419a3bce86d78a8222664cd-sm86-drv13030
PTX SHA-256: efa80c4f5ca72b82106a432807cdfd3b22b64b302765b5b310d020d8abb3866d
registers/thread: 24
static shared: 0
local bytes/thread: 0
active 128-thread blocks/SM: 12
PTX/binary version: 86/86
```

The module loader rejects nonzero local bytes, more than 40 registers/thread,
any shared allocation, or fewer than eight active blocks/SM before caching.
Nsight Compute 2026.2.1 attaches to `aidotnet_fused_linear_gelu_m1`, but this
Windows host returns `ERR_NVGPUCTRPERM` before counter collection. Therefore
the PR does not claim executed SASS spill counters. Driver-JIT local size zero
is enforced; promotion beyond the experimental rollout gate still requires an
administrator to enable counters and the CSV verifier to prove zero executed
spill instructions, local loads, local stores, and local spilling requests.

## Reproduction

```powershell
$env:AIDOTNET_DIRECT_PTX_FUSED_LINEAR = '1'
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release -- --direct-ptx-fused-linear 3

powershell -ExecutionPolicy Bypass -File `
  tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 `
  -Target fused-linear
```

## Next increments for #836

1. FP16/BF16 multi-row GEMM with architecture-specific async panel staging,
   Tensor Core MMA, and fused bias/activation/residual epilogues.
2. General M/N/K and transposition families using a bounded, offline-measured
   tile catalog rather than hot-path dynamic stride branches.
3. Batched/fanout GEMM and forward/backward linear pairs with graph-safe
   prepacked-weight ownership.
4. LoRA, dot/outer, and cross-entropy fusion cells from the executable manifest.
5. Separate Ada, Hopper, and Blackwell emitters and evidence; no architecture
   inherits Ampere promotion by assumption.
