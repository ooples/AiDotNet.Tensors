# AiDotNet.Tensors

[![NuGet](https://img.shields.io/nuget/v/AiDotNet.Tensors.svg)](https://www.nuget.org/packages/AiDotNet.Tensors/)
[![Build](https://github.com/ooples/AiDotNet.Tensors/actions/workflows/build.yml/badge.svg)](https://github.com/ooples/AiDotNet.Tensors/actions/workflows/build.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

The fastest .NET tensor library. Beats MathNet, NumSharp, TensorPrimitives, and matches TorchSharp CPU on pure managed code with hand-tuned AVX2/FMA SIMD kernels and JIT-compiled machine code.

## Features

- **Zero Allocations**: In-place operations with `ArrayPool<T>` and `Span<T>` for hot paths
- **Hand-Tuned SIMD**: Custom AVX2/FMA kernels with 4x loop unrolling, not just `Vector<T>` wrappers
- **JIT-Compiled Kernels**: Runtime x86-64 machine code generation for size-specialized operations
- **BLIS-Style GEMM**: Tiled matrix multiply with FMA micro-kernel, cache-aware panel packing
- **GPU Acceleration**: Optional CUDA, HIP/ROCm, and OpenCL support via separate packages
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

All numbers from the latest BenchmarkDotNet run on AMD Ryzen 9 3950X (16 cores, AVX2/FMA, no AVX-512), .NET 10.0. Reproduce with:

```bash
dotnet run -c Release --project tests/AiDotNet.Tensors.Benchmarks --framework net10.0 -- --vs-all
```

The full per-op result set with error bars lives in [`tests/AiDotNet.Tensors.Benchmarks/BENCHMARK_RESULTS.md`](tests/AiDotNet.Tensors.Benchmarks/BENCHMARK_RESULTS.md). The summary below is a hand-curated subset.

### vs TorchSharp CPU (libtorch C++ backend, float)

Head-to-head on identical data sizes. AiDotNet wins on the majority of ops using pure managed C# with hand-tuned SIMD — no native C++ dependencies required.

| Operation | Size | AiDotNet | TorchSharp | Speedup |
|-----------|------|----------|------------|---------|
| TensorAdd | 100K | 51 us | 4,263 us | **83× faster** |
| TensorMultiply | 100K | 40 us | 1,132 us | **28× faster** |
| TensorMean | 1M | 258 us | 7,811 us | **30× faster** |
| TensorSum | 1M | 250 us | 3,133 us | **13× faster** |
| MaxPool2D | — | 325 us | 10,924 us | **34× faster** |
| Conv2D | — | 525 us | 8,575 us | **16× faster** |
| BatchNorm | — | 3,230 us | 17,180 us | **5.3× faster** |
| ReLU | 1M | 421 us | 4,045 us | **9.6× faster** |
| Tanh | 1M | 1,688 us | 4,950 us | **2.9× faster** |
| GELU | 1M | 1,892 us | 3,822 us | **2.0× faster** |
| Mish | 1M | 2,349 us | 5,195 us | **2.2× faster** |
| LeakyReLU | 1M | 1,833 us | 3,702 us | **2.0× faster** |
| Subtract | 1M | 2,308 us | 5,771 us | **2.5× faster** |
| Divide | 1M | 3,399 us | 4,699 us | **1.4× faster** |
| Log | 1M | 1,961 us | 3,021 us | **1.5× faster** |
| Sqrt | 1M | 1,862 us | 2,099 us | **1.1× faster** |
| TensorMinValue | 1M | 2,280 us | 5,492 us | **2.4× faster** |
| TanhBackward | 1M | 984 us | 6,518 us | **6.6× faster** |

After the [#209 audit fixes](https://github.com/ooples/AiDotNet.Tensors/issues/209), float64 ops also stay competitive — `Exp_Double`, `Log_Double`, and `Softmax_Double` route through `System.Numerics.Tensors.TensorPrimitives` on net8+ instead of falling back to scalar `Math.Exp`, closing a previous 40–70× cliff. Re-run the suite after the next merge for updated numbers.

### vs MathNet.Numerics (Linear Algebra, double, N=1000)

| Operation | AiDotNet | MathNet | Speedup |
|-----------|----------|---------|---------|
| Matrix Multiply 1000×1000 | 8.3 ms | 49.2 ms | **6× faster** |
| Matrix Add | 1.87 ms | 2.50 ms | **1.3× faster** |
| Matrix Subtract | 2.08 ms | 2.47 ms | **1.2× faster** |
| Matrix Scalar Multiply | 1.66 ms | 2.14 ms | **1.3× faster** |
| Transpose | 2.85 ms | 3.68 ms | **1.3× faster** |
| Dot Product | 97 ns | 817 ns | **8.4× faster** |
| L2 Norm | 92 ns | 11,552 ns | **125× faster** |

### vs NumSharp (N=1000)

| Operation | AiDotNet | NumSharp | Speedup |
|-----------|----------|----------|---------|
| Matrix Multiply 1000×1000 | 8.3 ms | 26.5 s | **3,200× faster** |
| Matrix Add | 1.87 ms | 1.98 ms | 1.1× faster |
| Transpose | 2.85 ms | 13.7 ms | **4.8× faster** |
| Vector Add | 1.47 us | 54.5 us | **37× faster** |

### vs System.Numerics.Tensors.TensorPrimitives (N=1000)

In-place operations (zero allocation) compared to raw TensorPrimitives calls.

| Operation | AiDotNet | TensorPrimitives | Speedup |
|-----------|----------|-----------------|---------|
| Dot Product | 97 ns | 185 ns | **1.9× faster** |
| L2 Norm | 92 ns | 187 ns | **2.0× faster** |
| Vector AddInPlace | 154 ns | 117 ns | 0.8× |
| Vector SubtractInPlace | 116 ns | 118 ns | **tied** |
| Vector ScalarMulInPlace | 105 ns | 75 ns | 0.7× |
| Vector Add to Span | 116 ns | 119 ns | **tied** |

### Small Matrix Multiply (double)

| Size | AiDotNet | MathNet | NumSharp |
|------|----------|---------|----------|
| 4×4 | 172 ns | 165 ns | 2,198 ns |
| 16×16 | 2.1 us | 2.9 us | 107.5 us |
| 32×32 | 10.5 us | 36.2 us | 774.8 us |

AiDotNet is **1.4× faster** at 16×16 and **3.4× faster** at 32×32 than MathNet.

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
