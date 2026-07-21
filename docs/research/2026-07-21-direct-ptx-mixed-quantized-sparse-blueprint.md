# Direct-PTX mixed-precision, quantized, and sparse blueprint

Date: 2026-07-21

Tracking issue: #837

Parent blueprint: `2026-07-20-fused-attention-championship-blueprint.md`

## Verdict

The first two #837 golden slices are FP16 linear kernels with FP32
accumulation, FP32 bias, and tanh-GELU:

```text
input FP16 [K] + output-major weights FP16 [N,K] + bias FP32 [N]
    -> gelu_tanh(fp32(input @ transpose(weights)) + bias) FP32 [N]

input FP16 [16,K] + output-major weights FP16 [N,K] + bias FP32 [N]
    -> gelu_tanh(fp32(input @ transpose(weights)) + bias) FP32 [16,N]
```

The Ampere `K=512,N=2048` and `K=1024,N=4096` cells beat every measured
resident NVIDIA GPU peer by more than the required 1.10x median threshold in
all three paired runs, also win paired P95, allocate 0 managed bytes/call, use
0 temporary device bytes, and report 0 driver-JIT local bytes. They are the
only promoted v2 cells. `256x256` remains fail-closed because one run was a
1.01x near tie with PyTorch CUDA Graph.

The M=16 v3 slice extends that result with bank-conflict-free shared panels,
double-buffered `cp.async`, `mma.sync.m16n8k16`, a register-resident fused
epilogue, and one final output store. The 1024/4096 cell passes the full 1.10x
production gate; 512/2048 remains experiment-only because its conservative
1.057x ratio misses that gate. The executable 15-cell
`DirectPtxQuantizedMixedSparseCoverageManifest` assigns the remaining FP16
GEMM/backward, INT8, INT4, FP8, 2:4, and CSR operations to later direct-PTX
families.

## Why this kernel does not use Tensor Cores

M=1 is GEMV-shaped. Padding it into an `mma.sync.m16n8k16` tile would perform
at least 16 rows of work to retain one row and add shared-memory staging and
fragment movement without cross-row reuse. The measured winner instead lets
one warp own two output rows:

1. each lane loads one contiguous FP16 `half2` from input and each weight row;
2. values convert to FP32 registers and accumulate with FP32 FMA;
3. warp shuffles reduce the two dot products;
4. lane zero adds FP32 bias, evaluates tanh-GELU, and stores each output once.

The M=16 specialization is a separate kernel family using
`cp.async`-staged panels and `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
The blueprint chooses the primitive by shape evidence rather than requiring
Tensor Cores where their tile geometry is counterproductive.

## Exact physical ABI

The launch contains four pointers and no shape, stride, layout, dtype, batch,
or activation parameters.

| Tensor | Physical layout | Exact extent | Type | Access | Alignment |
|---|---|---|---|---|---:|
| input | vector | `[K]` | FP16 | read | 16 B |
| weights | output-major row-major | `[N,K]` | FP16 | read | 16 B |
| bias | vector | `[N]` | FP32 | read | 16 B |
| output | vector | `[N]` | FP32 | write | 16 B |

Packing immutable weights is a model-load responsibility. The hot path never
checks dynamic strides and never materializes a transpose. The dispatch
boundary checks the Ampere family, promoted shape, exact byte extents,
alignment/layout tokens, and output aliasing. Unsupported cases return stable
fallback reasons and execute FP16 cuBLAS plus resident bias/GELU kernels.

The kernel has no shared memory, global intermediate, scratch allocation, or
second output write. FP16 halves input/weight traffic relative to #836 while
retaining FP32 accumulation and output accuracy.

The v3 M=16 ABI keeps the same four-pointer launch and changes only the exact
input/output extents to `[16,K]` and `[16,N]` with canonical `RowMajor2D`
tokens. It has no runtime stride, M, K, N, layout, dtype, or epilogue
parameters. K=512 uses K=64 asynchronous stages (12,288 shared bytes); K=1024
uses K=128 stages (24,576 shared bytes). Each stage is rearranged into compact
K=16 subpanels so MMA fragment loads do not introduce shared-bank conflicts.
Four warps reuse one staged A panel across 32 output columns. Accumulators,
bias, and GELU remain in FP32 registers until one `st.global.v2.f32` per
fragment row.

## GPU-only championship evidence

Environment: Windows 10.0.26200, .NET 10.0.10 / SDK 10.0.302, NVIDIA GeForce
RTX 3080 12 GiB, SM86, driver 610.47, and PyTorch 2.12.1+cu130. Each cell uses
30 warmups and 101 samples in each of three independent runs. CUDA-event
samples average 50 resident launches. Upload, conversion, preferred-layout
packing, allocation, and download are outside timing.

All peers consume the same FP16-rounded input and weights and produce the same
linear+bias+tanh-GELU result. Direct PTX, cuBLAS, and cuBLASLt store FP32.
PyTorch's public FP16 linear stores the intermediate dot product as FP16 before
the FP32 bias/GELU epilogue, so its larger error is reported rather than hidden.
cuBLASLt receives an equivalent resident input-major copy because that is its
preferred column-major interpretation; no repack is timed for any peer.

`GFLOPS = 2*K*N / device median`; epilogue operations are not added to the
rate. Values below are min-max ranges across the final three-run capture.

| Shape | Method | mean us | median us | P95 us | P99 us | GFLOPS | managed B/call | temp device B | max error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256x256 | Direct PTX | 9.99-12.88 | 9.65-12.00 | 12.00-18.43 | 13.80-31.03 | 10.9-13.6 | 0 | 0 | 4.5e-8-5.2e-8 |
| 256x256 | NVIDIA cuBLAS | 34.55-39.97 | 33.79-36.31 | 38.22-53.02 | 49.52-59.15 | 3.6-3.9 | 40 | 0 | 4.1e-8-6.0e-8 |
| 256x256 | cuBLAS Graph | 20.59-21.90 | 19.70-19.93 | 25.76-30.70 | 29.14-50.81 | 6.6-6.7 | 0 | 0 | 4.1e-8-6.0e-8 |
| 256x256 | NVIDIA cuBLASLt | 32.88-39.05 | 31.25-39.90 | 37.29-47.62 | 42.15-57.59 | 3.3-4.2 | 248 | 0 | 4.1e-8-5.6e-8 |
| 256x256 | cuBLASLt Graph | 18.84-19.62 | 18.10-18.86 | 22.34-24.29 | 25.97-27.10 | 6.9-7.2 | 0 | 0 | 4.1e-8-5.6e-8 |
| 256x256 | PyTorch eager | 113.68-118.40 | 113.03-116.67 | 120.48-130.17 | 126.22-140.45 | 1.1-1.2 | n/a | 2,048 | 1.8e-5-2.2e-5 |
| 256x256 | PyTorch Graph | 12.35-13.85 | 12.00-12.33 | 15.46-23.33 | 18.25-24.82 | 10.6-10.9 | n/a | 0 | 1.8e-5-2.2e-5 |
| 512x2048 | Direct PTX | 9.97-11.67 | 9.69-11.12 | 12.41-15.99 | 15.07-22.71 | 188.6-216.5 | 0 | 0 | 1.0e-7-1.3e-7 |
| 512x2048 | NVIDIA cuBLAS | 33.84-38.79 | 33.03-34.90 | 38.30-58.80 | 43.03-60.99 | 60.1-63.5 | 40 | 0 | 1.9e-7-2.4e-7 |
| 512x2048 | cuBLAS Graph | 28.41-29.81 | 28.26-30.09 | 32.46-34.24 | 34.75-39.44 | 69.7-74.2 | 0 | 0 | 1.9e-7-2.4e-7 |
| 512x2048 | NVIDIA cuBLASLt | 22.80-23.76 | 22.24-22.57 | 26.13-30.74 | 28.35-35.86 | 92.9-94.3 | 248 | 0 | 1.0e-7-1.3e-7 |
| 512x2048 | cuBLASLt Graph | 21.58-22.08 | 20.66-21.38 | 26.17-29.57 | 28.06-32.52 | 98.1-101.5 | 0 | 0 | 1.0e-7-1.3e-7 |
| 512x2048 | PyTorch eager | 128.03-130.71 | 127.86-128.88 | 132.79-152.35 | 135.23-193.29 | 16.3-16.4 | n/a | 16,384 | 3.7e-5-3.8e-5 |
| 512x2048 | PyTorch Graph | 14.90-19.80 | 14.50-15.20 | 17.59-29.55 | 18.66-118.87 | 138.0-144.6 | n/a | 0 | 3.7e-5-3.8e-5 |
| 1024x4096 | Direct PTX | 14.48-15.09 | 13.99-14.19 | 16.02-17.06 | 17.86-19.93 | 591.1-599.7 | 0 | 0 | 2.7e-7-3.3e-7 |
| 1024x4096 | NVIDIA cuBLAS | 33.29-34.89 | 32.54-33.44 | 37.72-44.36 | 40.00-48.50 | 250.8-257.8 | 40 | 0 | 5.4e-7-7.0e-7 |
| 1024x4096 | cuBLAS Graph | 31.78-44.99 | 31.11-49.03 | 34.08-57.55 | 39.40-60.33 | 171.1-269.7 | 0 | 0 | 5.4e-7-7.0e-7 |
| 1024x4096 | NVIDIA cuBLASLt | 24.95-29.53 | 24.10-26.56 | 31.27-43.72 | 36.37-52.04 | 315.8-348.0 | 248 | 0 | 5.4e-7-7.0e-7 |
| 1024x4096 | cuBLASLt Graph | 32.67-33.95 | 32.15-32.77 | 35.45-41.41 | 39.02-44.34 | 256.0-260.9 | 0 | 0 | 5.4e-7-7.0e-7 |
| 1024x4096 | PyTorch eager | 113.88-117.05 | 112.57-115.40 | 122.31-128.27 | 124.78-176.42 | 72.7-74.5 | n/a | 32,768 | 7.3e-5-8.2e-5 |
| 1024x4096 | PyTorch Graph | 19.39-19.52 | 19.15-19.29 | 19.95-20.60 | 23.49-24.47 | 434.8-438.1 | n/a | 0 | 7.3e-5-8.2e-5 |

For 512x2048 the conservative cross-run median ratio is
`14.50 / 11.12 = 1.30x`; paired ratios are 1.37x, 1.49x, and 1.54x. For
1024x4096 the conservative ratio is `19.15 / 14.19 = 1.35x`; paired ratios
are 1.35x, 1.37x, and 1.37x. Direct P95 is below the paired PyTorch P95 in
all six promoted-shape runs.

## M=16 Tensor Core championship evidence

The M=16 harness uses the same environment and sampling contract. FLOP rate
is `2*16*K*N / device median`; epilogue operations are excluded. The table is
the min-max range from the final three independent runs. PyTorch's FP16 GEMM
rounding explains its larger reported error. No conversion, transpose,
allocation, or transfer is timed for any peer.

| Shape | Method | median us | P95 us | P99 us | TFLOPS | managed B/call | temp device B | max error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| M16 512x2048 | Direct PTX | 7.76-11.61 | 9.11-18.31 | 13.78-21.71 | 2.890-4.323 | 0 | 0 | 2.5e-7-3.7e-7 |
| M16 512x2048 | NVIDIA cuBLAS | 26.99-39.32 | 32.30-47.19 | 37.31-55.56 | 0.853-1.243 | 40 | 0 | 2.5e-7-3.7e-7 |
| M16 512x2048 | cuBLAS Graph | 15.83-22.87 | 21.59-28.02 | 23.61-33.61 | 1.467-2.120 | 0 | 0 | 2.5e-7-3.7e-7 |
| M16 512x2048 | NVIDIA cuBLASLt | 23.27-29.70 | 27.69-38.60 | 32.07-48.01 | 1.130-1.442 | 248 | 0 | 2.5e-7-3.7e-7 |
| M16 512x2048 | cuBLASLt Graph | 25.50-29.37 | 30.72-35.04 | 34.25-38.79 | 1.143-1.316 | 0 | 0 | 2.5e-7-3.7e-7 |
| M16 512x2048 | PyTorch eager | 116.04-116.98 | 123.62-145.22 | 128.31-157.27 | 0.287-0.289 | n/a | 262,144 | 3.9e-5-4.1e-5 |
| M16 512x2048 | PyTorch Graph | 12.27-12.51 | 15.38-17.76 | 16.63-20.34 | 2.682-2.735 | n/a | 0 | 3.9e-5-4.1e-5 |
| M16 1024x4096 | Direct PTX | 14.25-14.68 | 15.22-16.22 | 15.54-18.15 | 9.140-9.416 | 0 | 0 | 8.0e-7-1.3e-6 |
| M16 1024x4096 | NVIDIA cuBLAS | 33.46-37.09 | 41.31-42.99 | 44.97-47.49 | 3.619-4.011 | 40 | 0 | 8.0e-7-1.3e-6 |
| M16 1024x4096 | cuBLAS Graph | 31.08-35.33 | 35.10-39.71 | 39.67-45.83 | 3.799-4.318 | 0 | 0 | 8.0e-7-1.3e-6 |
| M16 1024x4096 | NVIDIA cuBLASLt | 22.04-27.93 | 25.50-35.16 | 28.75-38.58 | 4.805-6.091 | 248 | 0 | 8.0e-7-1.3e-6 |
| M16 1024x4096 | cuBLASLt Graph | 31.89-36.45 | 35.94-40.47 | 39.57-45.47 | 3.682-4.209 | 0 | 0 | 8.0e-7-1.3e-6 |
| M16 1024x4096 | PyTorch eager | 107.19-112.74 | 113.73-135.27 | 117.43-149.89 | 1.190-1.252 | n/a | 524,288 | 8.6e-5-9.3e-5 |
| M16 1024x4096 | PyTorch Graph | 21.16-21.25 | 21.83-24.00 | 25.29-26.91 | 6.315-6.344 | n/a | 0 | 8.6e-5-9.3e-5 |

Direct PTX wins all six paired medians. Conservative worst-direct versus
best-PyTorch-Graph median ratios are `12.27/11.61 = 1.057x` for 512x2048 and
`21.16/14.68 = 1.441x` for 1024x4096. The large cell also wins all paired
P95/P99 samples. The small cell wins two of three paired P95 samples; this
tail caveat is retained in the evidence. Only 1024x4096 is promoted: the
512x2048 ratio is below the epic's required 1.10x production threshold.

## Resource and spill evidence

The promoted 512x2048 module records:

```text
BlueprintId: fused-linear-bias-gelu-v2-Ampere-decode-fp16-fp32acc-m1-k512-n2048
DeviceFingerprint: gpu-79d5ac8ef419a3bce86d78a8222664cd-sm86-drv13030
PTX SHA-256: a8cbe903c4f66be4b82f4a8d9637d9cf07ab5bb68b2fb1ef83ed96b33d029f2d
registers/thread: 21
static shared: 0
local bytes/thread: 0
active 128-thread blocks/SM: 12
PTX/binary version: 86/86
```

The loader rejects nonzero local bytes, more than 40 registers/thread, any
shared allocation, or fewer than eight active blocks/SM before caching.
Nsight Compute 2026.2.1 attaches to
`aidotnet_fused_linear_gelu_fp16_m1`, but this Windows host returns
`ERR_NVGPUCTRPERM` before collection. Executed-SASS zero-spill proof is
therefore not claimed. The CSV verifier remains the release-evidence gate once
an administrator enables NVIDIA performance counters.

The v3 M=16 K=1024/N=4096 module records:

```text
BlueprintId: fused-linear-bias-gelu-v3-Ampere-tensorcore-async-fp16-fp32acc-m16-k1024-n4096
DeviceFingerprint: gpu-79d5ac8ef419a3bce86d78a8222664cd-sm86-drv13030
PTX SHA-256: 5cd14a7a65fde5f1aa86a3c4633ee03448f373c3bd617b667f39fa3ff2a57cd1
registers/thread: 32
static shared: 24,576
local bytes/thread: 0
active 128-thread blocks/SM: 4
PTX/binary version: 86/86
```

The experiment-only K=512/N=2048 module reports 40 registers/thread, 12,288 static shared
bytes, 0 local bytes/thread, and seven active blocks/SM. These driver-JIT
attributes are enforced before either module enters the cache and prove that
the compiled function has no local-memory frame or spill allocation. The new
`mixed-linear-m16` Nsight target attaches to the correct entry point, but the
host again returns `ERR_NVGPUCTRPERM`; executed-counter proof is not claimed.

## Reproduction

```powershell
$env:AIDOTNET_DIRECT_PTX_MIXED_LINEAR = '1'
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release -- --direct-ptx-mixed-linear 3

dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release -- --direct-ptx-mixed-linear-m16 3

powershell -ExecutionPolicy Bypass -File `
  tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 `
  -Target mixed-linear

powershell -ExecutionPolicy Bypass -File `
  tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 `
  -Target mixed-linear-m16
```

## Next #837 increments

1. Generalize the proven FP16 M=16 async panel family to larger M buckets,
   BF16, and fused residual epilogues without adding dynamic stride checks.
2. W8A8 INT8 DP4A/IMMA and W4A16 register-dequant projection families.
3. FP8 E4M3/E5M2 architecture-specific families with explicit scale ABI.
4. Native 2:4 `mma.sp` GEMM and fused bias/activation without decompression.
5. CSR sparse-linear density buckets, including a deliberate dense fallback
   threshold and graph-safe prepacked ownership.
