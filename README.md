# AiDotNet.Tensors

[![NuGet](https://img.shields.io/nuget/v/AiDotNet.Tensors.svg)](https://www.nuget.org/packages/AiDotNet.Tensors/)
[![Build](https://github.com/ooples/AiDotNet.Tensors/actions/workflows/build.yml/badge.svg)](https://github.com/ooples/AiDotNet.Tensors/actions/workflows/build.yml)
[![License](https://img.shields.io/badge/license-BSL%201.1-blue.svg)](LICENSE)

A high-performance .NET tensor library with hand-written AVX2/AVX-512 SIMD kernels in `SimdKernels.cs` / `SimdGemm.cs` / `SimdConvHelper.cs`. Every hot path runs through our own managed-C# kernels — we do NOT call into `System.Numerics.Tensors`, MKL.NET, or oneDNN through the standard wrappers. In the latest benchmark run on a Threadripper 3990X (see [CPU Benchmarks](#cpu-benchmarks) for every number, the versions compared and the losses), AiDotNet.Tensors was faster than TorchSharp 0.107 (libtorch) in 73 of 82 measurements, than ML.NET 5.0 and TensorFlow.NET 0.150 in all of them, than NumSharp 0.70 in 45 of 48 and than MathNet.Numerics 5.0 in all 33.

> **Note on dependencies.** The .nupkg ships with the following PackageReferences:
> `Microsoft.Extensions.Logging.Abstractions`, `System.Text.Json`,
> `System.Threading.Channels`, `K4os.Compression.LZ4` (LZ4 compression for
> serialized tensor blobs), `AiDotNet.Native.OpenBLAS` (transitive native
> OpenBLAS for fallback paths only — our SimdGemm beats it for d=128
> transformer hot paths), and **MKL via Microsoft.ML.Mkl.Redist (~66 MB on
> win-x64) + `intelmkl.redist.win-x64` (~500 MB on win-x64)** for the FP64
> kernels that haven't yet been ported to pure-managed AVX2 (Phase 0 remediation
> work tracks the port). For air-gapped / federal deployments we ship a custom
> build with MKL/OpenBLAS removed and the entire telemetry namespace compiled
> out — see [aidotnet.dev/enterprise](https://aidotnet.dev/enterprise) for
> the Enterprise tier including air-gapped builds.
>
> **Performance numbers above assume net8.0+.** On net471 the SIMD/intrinsics
> helpers are excluded (System.Runtime.Intrinsics is unavailable pre-net6); a
> custom net471 SIMD path that beats `System.Numerics.Vector<T>` is on the
> roadmap as Phase 5.

## Features

- **Zero Allocations**: In-place operations with `ArrayPool<T>` and `Span<T>` for hot paths
- **Hand-Tuned SIMD**: Custom AVX2/FMA kernels with 4x loop unrolling, not just `Vector<T>` wrappers
- **JIT-Compiled Kernels**: Runtime x86-64 machine code generation for size-specialized operations
- **BLIS-Style GEMM**: Tiled matrix multiply with FMA micro-kernel, cache-aware panel packing
- **GPU Acceleration**: Optional CUDA, HIP/ROCm, OpenCL, Metal, Vulkan, and WebGPU support via separate packages, with CPU-vs-GPU op-parity validated across every backend (#775)
- **Native ANN Index**: Dependency-free approximate-nearest-neighbour search (Flat / IVF / PQ / IVFPQ) via `AnnIndex`, with fused `IAnnBackend` GPU kernels across all seven backends — no FAISS / MKL dependency (#824)
- **Multi-Target**: Supports .NET 10.0 and .NET Framework 4.7.1
- **Generic Math**: Works with any numeric type via `INumericOperations<T>` interface

## Installation

```bash
# Core package (CPU SIMD acceleration)
dotnet add package AiDotNet.Tensors

# Optional: OpenBLAS for optimized CPU BLAS operations
dotnet add package AiDotNet.Native.OpenBLAS

# Optional: CLBlast for OpenCL GPU acceleration (AMD/Intel/NVIDIA)
dotnet add package AiDotNet.Native.CLBlast

# Optional: CUDA for NVIDIA GPU acceleration (requires NVIDIA GPU)
dotnet add package AiDotNet.Native.CUDA
```

## Quick Start

```csharp
using AiDotNet.Tensors.LinearAlgebra;

// Create vectors
var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
var v2 = new Vector<double>(new[] { 5.0, 6.0, 7.0, 8.0 });

// SIMD-accelerated operations
var sum = v1 + v2;
var dot = v1.Dot(v2);

// Create matrices
var m1 = new Matrix<double>(3, 3);
var m2 = Matrix<double>.Identity(3);

// Matrix operations
var product = m1 * m2;
var transpose = m1.Transpose();
```

## CPU Benchmarks

Every number below comes from two BenchmarkDotNet runs on one machine on 2026-10-08: `--vs-all` at commit `dcb0926f` and `--linalg`
at commit `587358ed`, using the suites in this repository. AiDotNet.Tensors runs its own managed C#
SIMD kernels; the comparison libraries run their native backends (libtorch for TorchSharp, the TensorFlow runtime for
TensorFlow.NET).

**Machine:** AMD Ryzen Threadripper 3990X (64 cores / 128 threads, AVX2 + FMA, no AVX-512), Windows 11, .NET 10.0.401,
BenchmarkDotNet 0.15.8. BenchmarkDotNet confines each benchmark process to one 64-thread processor group.

**Versions compared:** TorchSharp 0.107.0 (its bundled libtorch CPU build), ML.NET 5.0.0, TensorFlow.NET 0.150.0
(TensorFlow runtime 2.16.0), NumSharp 0.70.0, MathNet.Numerics 5.0.0.

**Summary of this run** (a cell is a win when AiDotNet's mean time is lower):

| Compared with | Wins | Losses |
|---|--:|--:|
| TorchSharp (41 ops, steady state and cold call) | 73 | 9 |
| ML.NET | 6 | 0 |
| TensorFlow.NET | 11 | 0 |
| NumSharp | 45 | 3 |
| MathNet.Numerics | 33 | 0 |

Reproduce:

```bash
dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks --framework net10.0 -- --vs-all   # TorchSharp, ML.NET, TensorFlow.NET
dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks --framework net10.0 -- --linalg   # NumSharp, MathNet
```

`--vs-all-filter <glob>` and `--linalg-filter <glob>` run a subset.

The BenchmarkDotNet reports behind every table are in [`tests/AiDotNet.Tensors.Benchmarks/Results/2026-10-08-threadripper-3990x/`](tests/AiDotNet.Tensors.Benchmarks/Results/2026-10-08-threadripper-3990x/); `python tests/AiDotNet.Tensors.Benchmarks/make_readme_tables.py <reports> <reports>` regenerates the tables from them.

### How to read these numbers

- **Steady state** is BenchmarkDotNet's normal mode: many calls per iteration, averaged.
- **Cold call** times one call per iteration with a forced garbage collection before it, which is what an op costs when
  it runs occasionally. Both libraries' first call (JIT and library start-up) is excluded.
- **Each arm matches the competitor's allocation behaviour.** Where the other library reuses or frees its output (ML.NET
  writes into a preallocated destination; TorchSharp disposes its result), the AiDotNet arm returns its result to the
  tensor pool. Where the other library allocates a fresh result every call (TensorFlow.NET, NumSharp, MathNet), so does
  the AiDotNet arm. The same op can therefore show very different absolute times in different tables.
- **Speedup** is the other library's time divided by AiDotNet's; above 1× means AiDotNet is faster.

### vs TorchSharp (libtorch CPU)

| Operation | Shape | AiDotNet steady | TorchSharp steady | Speedup | AiDotNet cold call | TorchSharp cold call | Speedup |
|---|---|--:|--:|--:|--:|--:|--:|
| Abs | 1M | 8.2 µs | 18 µs | **2.26×** | 25 µs | 41 µs | **1.63×** |
| Add | 100K | 31 µs | 39 µs | **1.25×** | 34 µs | 39 µs | **1.15×** |
| Add | 1M | 162 µs | 221 µs | **1.37×** | 187 µs | 216 µs | **1.15×** |
| Attention Q·Kᵀ | 512×64 · 64×512 | 77 µs | 115 µs | **1.49×** | 92 µs | 147 µs | **1.59×** |
| BatchNorm | 32×64×32×32 | 62 µs | 187 µs | **3.04×** | 94 µs | 380 µs | **4.03×** |
| Conv2D | 1×16×64×64 → 32, 3×3 | 174 µs | 277 µs | **1.59×** | 280 µs | 327 µs | **1.17×** |
| Conv2D (double) | 1×3×32×32 → 16, 3×3 | 49 µs | 115 µs | **2.36×** | 68 µs | 111 µs | **1.63×** |
| Divide | 1M | 12 µs | 21 µs | **1.86×** | 24 µs | 58 µs | **2.36×** |
| Exp | 1M | 27 µs | 44 µs | **1.65×** | 57 µs | 72 µs | **1.26×** |
| Exp (double) | 1M | 55 µs | 79 µs | **1.43×** | 106 µs | 136 µs | **1.28×** |
| GELU | 1M | 36 µs | 56 µs | **1.54×** | 75 µs | 117 µs | **1.55×** |
| GELU (double) | 1M | 134 µs | 207 µs | **1.54×** | 147 µs | 364 µs | **2.49×** |
| GroupNorm | 32×64×32×32, 32 groups | 54 µs | 67 µs | **1.24×** | 81 µs | 101 µs | **1.24×** |
| LayerNorm | 32768×64 | 72 µs | 81 µs | **1.13×** | 187 µs | 142 µs | 0.76× (slower) |
| LeakyReLU | 1M | 8.5 µs | 21 µs | **2.42×** | 21 µs | 45 µs | **2.13×** |
| Log | 1M | 47 µs | 46 µs | 0.98× (slower) | 96 µs | 77 µs | 0.80× (slower) |
| Log (double) | 1M | 114 µs | 94 µs | 0.83× (slower) | 172 µs | 214 µs | **1.25×** |
| LogSoftmax | 512×1024 | 19 µs | 32 µs | **1.64×** | 59 µs | 81 µs | **1.37×** |
| MatMul | 256×256 | 26 µs | 67 µs | **2.54×** | 86 µs | 106 µs | **1.24×** |
| MatMul | 512×512 | 146 µs | 213 µs | **1.46×** | 566 µs | 508 µs | 0.90× (slower) |
| MatMul (double) | 256×256 | 54 µs | 119 µs | **2.22×** | 143 µs | 199 µs | **1.39×** |
| Max | 1M | 6.4 µs | 12 µs | **1.93×** | 11 µs | 29 µs | **2.70×** |
| MaxPool2D | 1×32×64×64, 3×3 / 2 | 14 µs | 54 µs | **3.82×** | 26 µs | 100 µs | **3.86×** |
| Mean | 1M | 5.9 µs | 32 µs | **5.49×** | 18 µs | 65 µs | **3.56×** |
| Min | 1M | 6.2 µs | 13 µs | **2.02×** | 11 µs | 31 µs | **2.84×** |
| Mish | 1M | 997 µs | 1.02 ms | **1.02×** | 980 µs | 905 µs | 0.92× (slower) |
| Mish (double) | 1M | 347 µs | 2.16 ms | **6.22×** | 357 µs | 1.86 ms | **5.22×** |
| Multiply | 100K | 28 µs | 40 µs | **1.44×** | 32 µs | 37 µs | **1.17×** |
| Multiply | 1M | 179 µs | 228 µs | **1.27×** | 171 µs | 199 µs | **1.16×** |
| ReLU | 1M | 221 µs | 206 µs | 0.93× (slower) | 218 µs | 219 µs | **1.00×** |
| Sigmoid | 1M | 183 µs | 189 µs | **1.03×** | 181 µs | 199 µs | **1.10×** |
| Sigmoid (double) | 1M | 72 µs | 150 µs | **2.08×** | 103 µs | 282 µs | **2.73×** |
| Sigmoid backward | 1M | 12 µs | 64 µs | **5.41×** | 40 µs | 111 µs | **2.76×** |
| Softmax | 512×1024 | 16 µs | 29 µs | **1.80×** | 50 µs | 59 µs | **1.19×** |
| Softmax (double) | 512×1024 | 60 µs | 56 µs | 0.93× (slower) | 76 µs | 138 µs | **1.81×** |
| Sqrt | 1M | 14 µs | 29 µs | **2.04×** | 21 µs | 47 µs | **2.31×** |
| Subtract | 1M | 12 µs | 21 µs | **1.86×** | 33 µs | 51 µs | **1.56×** |
| Sum | 1M | 6.1 µs | 24 µs | **3.96×** | 20 µs | 49 µs | **2.44×** |
| Tanh | 1M | 61 µs | 90 µs | **1.47×** | 134 µs | 124 µs | 0.93× (slower) |
| Tanh (double) | 1M | 86 µs | 163 µs | **1.91×** | 119 µs | 275 µs | **2.31×** |
| Tanh backward | 1M | 12 µs | 64 µs | **5.13×** | 37 µs | 113 µs | **3.08×** |

**Where AiDotNet is behind:** float and double `Log`, the double `Softmax` steady state, `ReLU` steady state, and the cold call
of `LayerNorm`, `Log`, `Tanh`, `Mish` and the 512×512 `MatMul`. Most are within 10%; the largest gap is the `LayerNorm` cold
call at 1.32×.

### vs ML.NET

| Operation | Shape | AiDotNet | ML.NET 5.0.0 | Speedup |
|---|---|--:|--:|--:|
| Add | 100K | 5.9 µs | 50 µs | **8.46×** |
| Add | 1M | 12 µs | 521 µs | **42.20×** |
| Mean | 1M | 5.3 µs | 89 µs | **16.79×** |
| Multiply | 100K | 5.7 µs | 51 µs | **8.89×** |
| Multiply | 1M | 12 µs | 546 µs | **45.61×** |
| Sum | 1M | 8.0 µs | 168 µs | **20.91×** |

### vs TensorFlow.NET

| Operation | Shape | AiDotNet | TensorFlow.NET 0.150.0 | Speedup |
|---|---|--:|--:|--:|
| Add | 100K | 86 µs | 117 µs | **1.36×** |
| Add | 1M | 583 µs | 1.41 ms | **2.42×** |
| Conv2D | 1×16×64×64 → 32, 3×3 | 398 µs | 490 µs | **1.23×** |
| MatMul | 256×256 | 309 µs | 480 µs | **1.55×** |
| MatMul | 512×512 | 444 µs | 1.25 ms | **2.82×** |
| Mean | 1M | 6.0 µs | 106 µs | **17.50×** |
| Multiply | 100K | 86 µs | 118 µs | **1.37×** |
| Multiply | 1M | 577 µs | 1.34 ms | **2.31×** |
| ReLU | 1M | 585 µs | 1.48 ms | **2.54×** |
| Sigmoid | 1M | 642 µs | 1.62 ms | **2.51×** |
| Sum | 1M | 6.8 µs | 136 µs | **19.98×** |

### vs NumSharp and MathNet.Numerics (double precision)

| Operation | N | AiDotNet | NumSharp 0.70.0 | MathNet 5.0.0 | Speedup vs NumSharp | Speedup vs MathNet |
|---|--:|--:|--:|--:|--:|--:|
| Dot Product | 100 | 21 ns | 949 ns | 80 ns | **45.29×** | **3.81×** |
| Dot Product | 500 | 57 ns | 720 ns | 355 ns | **12.64×** | **6.23×** |
| Dot Product | 1000 | 102 ns | 766 ns | 696 ns | **7.47×** | **6.79×** |
| L2 Norm | 100 | 14 ns | 1.2 µs | 962 ns | **86.44×** | **69.04×** |
| L2 Norm | 500 | 44 ns | 1.3 µs | 4.9 µs | **28.30×** | **110.09×** |
| L2 Norm | 1000 | 81 ns | 1.3 µs | 9.8 µs | **15.67×** | **120.48×** |
| Matrix Add | 100 | 3.1 µs | 6.7 µs | 5.5 µs | **2.13×** | **1.74×** |
| Matrix Add | 500 | 248 µs | 298 µs | 504 µs | **1.20×** | **2.03×** |
| Matrix Add | 1000 | 1.80 ms | 2.13 ms | 3.48 ms | **1.19×** | **1.93×** |
| Matrix Multiply | 100 | 52 µs | 126 µs | 199 µs | **2.44×** | **3.85×** |
| Matrix Multiply | 500 | 3.54 ms | 9.74 ms | 6.87 ms | **2.75×** | **1.94×** |
| Matrix Multiply | 1000 | 18.67 ms | 74.84 ms | 35.88 ms | **4.01×** | **1.92×** |
| Matrix Scalar Multiply | 100 | 2.6 µs | 7.9 µs | 5.1 µs | **3.10×** | **2.00×** |
| Matrix Scalar Multiply | 500 | 253 µs | 213 µs | 338 µs | 0.84× (slower) | **1.34×** |
| Matrix Scalar Multiply | 1000 | 1.09 ms | 1.17 ms | 2.02 ms | **1.07×** | **1.85×** |
| Matrix Subtract | 100 | 3.7 µs | 8.7 µs | 7.0 µs | **2.32×** | **1.87×** |
| Matrix Subtract | 500 | 252 µs | 242 µs | 427 µs | 0.96× (slower) | **1.69×** |
| Matrix Subtract | 1000 | 1.52 ms | 1.65 ms | 2.46 ms | **1.08×** | **1.62×** |
| Transpose | 100 | 5.2 µs | 10 µs | 13 µs | **2.00×** | **2.56×** |
| Transpose | 500 | 283 µs | 568 µs | 540 µs | **2.01×** | **1.91×** |
| Transpose | 1000 | 1.68 ms | 1.57 ms | 4.53 ms | 0.94× (slower) | **2.69×** |
| Transpose (view) | 100 | 84 ns | 479 ns | — | **5.70×** | — |
| Transpose (view) | 500 | 104 ns | 599 ns | — | **5.76×** | — |
| Transpose (view) | 1000 | 114 ns | 445 ns | — | **3.90×** | — |
| Vector Add | 100 | 85 ns | 914 ns | 121 ns | **10.81×** | **1.43×** |
| Vector Add | 500 | 322 ns | 1.4 µs | 338 ns | **4.37×** | **1.05×** |
| Vector Add | 1000 | 599 ns | 1.6 µs | 652 ns | **2.68×** | **1.09×** |
| Vector Scalar Multiply | 100 | 52 ns | 1.6 µs | 81 ns | **30.38×** | **1.57×** |
| Vector Scalar Multiply | 500 | 227 ns | 1.9 µs | 274 ns | **8.25×** | **1.20×** |
| Vector Scalar Multiply | 1000 | 322 ns | 2.1 µs | 518 ns | **6.46×** | **1.61×** |
| Vector Subtract | 100 | 65 ns | 837 ns | 104 ns | **12.81×** | **1.59×** |
| Vector Subtract | 500 | 262 ns | 1.1 µs | 292 ns | **4.28×** | **1.11×** |
| Vector Subtract | 1000 | 391 ns | 1.4 µs | 592 ns | **3.63×** | **1.51×** |

The three NumSharp losses (`Matrix Scalar Multiply` and `Matrix Subtract` at N=500, `Transpose` at N=1000) are benchmarks
that allocate a 2–8 MB result and discard it on every call. AiDotNet returns a managed array, so that churn triggers
gen-2 garbage collections (about one every seven calls for the 8 MB transpose); NumSharp allocates unmanaged memory and
does not. Writing into an existing matrix avoids the allocation: `TransposeInPlace` at N=1000 takes 125 µs.

**Element-wise (vs NumSharp, double):**

| Operation | N | AiDotNet | NumSharp 0.70.0 | Speedup vs NumSharp |
|---|--:|--:|--:|--:|
| Exp | 1000 | 2.1 µs | 5.4 µs | **2.64×** |
| Exp | 10000 | 16 µs | 58 µs | **3.60×** |
| Exp | 100000 | 114 µs | 610 µs | **5.35×** |
| Max | 1000 | 123 ns | 1.3 µs | **10.37×** |
| Max | 10000 | 906 ns | 2.3 µs | **2.53×** |
| Max | 100000 | 5.0 µs | 12 µs | **2.45×** |
| Multiply | 1000 | 858 ns | 1.5 µs | **1.76×** |
| Multiply | 10000 | 4.9 µs | 9.5 µs | **1.93×** |
| Multiply | 100000 | 59 µs | 91 µs | **1.53×** |
| Sum | 1000 | 91 ns | 1.2 µs | **13.19×** |
| Sum | 10000 | 684 ns | 1.7 µs | **2.44×** |
| Sum | 100000 | 4.1 µs | 9.8 µs | **2.40×** |

**Small matrix multiply (double):**

| Operation | N | AiDotNet | NumSharp 0.70.0 | MathNet 5.0.0 | Speedup vs NumSharp | Speedup vs MathNet |
|---|--:|--:|--:|--:|--:|--:|
| 16x16 Multiply |  | 654 ns | 1.9 µs | 3.0 µs | **2.90×** | **4.57×** |
| 32x32 Multiply |  | 2.7 µs | 5.8 µs | 25 µs | **2.14×** | **9.39×** |
| 4x4 Multiply |  | 112 ns | 1.0 µs | 138 ns | **9.00×** | **1.23×** |

### SIMD Instruction Support

The library automatically detects and uses the best available SIMD instructions:

| Instruction Set | Vector Width | Supported |
|----------------|--------------|-----------|
| AVX-512 | 512-bit (16 floats) | .NET 8+ |
| AVX2 + FMA | 256-bit (8 floats) | .NET 6+ |
| AVX | 256-bit (8 floats) | .NET 6+ |
| SSE4.2 | 128-bit (4 floats) | .NET 6+ |
| ARM NEON | 128-bit (4 floats) | .NET 6+ |

### Check Available Acceleration

```csharp
using AiDotNet.Tensors.Engines;

var caps = PlatformDetector.Capabilities;

// SIMD capabilities
Console.WriteLine($"AVX2: {caps.HasAVX2}");
Console.WriteLine($"AVX-512: {caps.HasAVX512F}");

// GPU support
Console.WriteLine($"CUDA: {caps.HasCudaSupport}");
Console.WriteLine($"OpenCL: {caps.HasOpenCLSupport}");

// Native library availability
Console.WriteLine($"OpenBLAS: {caps.HasOpenBlas}");
Console.WriteLine($"CLBlast: {caps.HasClBlast}");

// Or get a full status summary
Console.WriteLine(NativeLibraryDetector.GetStatusSummary());
```

## Optional Acceleration Packages

### AiDotNet.Native.OpenBLAS

Provides optimized CPU BLAS operations using OpenBLAS:

```bash
dotnet add package AiDotNet.Native.OpenBLAS
```

**Performance**: Accelerated BLAS operations for matrix multiply and decompositions.

### AiDotNet.Native.CLBlast

Provides GPU acceleration via OpenCL (works on AMD, Intel, and NVIDIA GPUs):

```bash
dotnet add package AiDotNet.Native.CLBlast
```

**Performance**: 10x+ faster for large matrix operations on GPU.

### AiDotNet.Native.CUDA

Provides GPU acceleration via NVIDIA CUDA (NVIDIA GPUs only):

```bash
dotnet add package AiDotNet.Native.CUDA
```

**Performance**: 30,000+ GFLOPS for matrix operations on modern NVIDIA GPUs.

**Requirements**:
- NVIDIA GPU (GeForce, Quadro, or Tesla)
- NVIDIA display driver 525.60+ (includes CUDA driver)

**Usage with helpful error messages**:

```csharp
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// Recommended: throws beginner-friendly exception if CUDA unavailable
using var cuda = CudaBackend.CreateOrThrow();

// Or check availability first
if (CudaBackend.IsCudaAvailable)
{
    using var backend = new CudaBackend();
    // Use CUDA acceleration
}
```

If CUDA is not available, you'll get detailed troubleshooting steps explaining exactly what's missing and how to fix it.

## Requirements

- .NET 10.0 or .NET Framework 4.7.1+
- Windows x64, Linux x64, or macOS x64/arm64

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
