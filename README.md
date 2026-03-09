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

All benchmarks run on AMD Ryzen 9 3950X, .NET 10.0, BenchmarkDotNet. No AVX-512.

### vs TorchSharp CPU (Tensor Operations, float)

Head-to-head against TorchSharp's libtorch C++ backend on identical data sizes.

| Operation | AiDotNet | TorchSharp | Speedup | Result |
|-----------|----------|------------|---------|--------|
| MatMul 256x256 | 95 us | 125 us | **1.3x faster** | WIN |
| MatMul 512x512 | 427 us | 533 us | **1.2x faster** | WIN |
| Mean 1M | 194 us | 224 us | **1.2x faster** | WIN |
| Add 100K | 30 us | 30 us | tied | TIED |
| Multiply 100K | 42 us | 42 us | tied | TIED |
| Sum 1M | 200 us | 183 us | 0.9x | Close |
| Sigmoid 1M | 222 us | 196 us | 0.9x | Close |
| Add 1M | 209 us | 182 us | 0.9x | Close |
| ReLU 1M | 196 us | 169 us | 0.9x | Close |

AiDotNet wins or matches TorchSharp CPU on the majority of operations using pure managed C# with hand-tuned SIMD, no native C++ dependencies required.

### vs MathNet.Numerics (Linear Algebra, double, N=1000)

| Operation | AiDotNet | MathNet | Speedup |
|-----------|----------|---------|---------|
| Matrix Multiply 1000x1000 | 8.3 ms | 49.2 ms | **6x faster** |
| Matrix Add | 1.87 ms | 2.50 ms | **1.3x faster** |
| Matrix Subtract | 2.08 ms | 2.47 ms | **1.2x faster** |
| Matrix Scalar Multiply | 1.66 ms | 2.14 ms | **1.3x faster** |
| Transpose | 2.85 ms | 3.68 ms | **1.3x faster** |
| Dot Product | 97 ns | 817 ns | **8.4x faster** |
| L2 Norm | 92 ns | 11,552 ns | **125x faster** |

### vs NumSharp (N=1000)

| Operation | AiDotNet | NumSharp | Speedup |
|-----------|----------|----------|---------|
| Matrix Multiply 1000x1000 | 8.3 ms | 26.5 s | **3,200x faster** |
| Matrix Add | 1.87 ms | 1.98 ms | 1.1x faster |
| Transpose | 2.85 ms | 13.7 ms | **4.8x faster** |
| Vector Add | 1.47 us | 54.5 us | **37x faster** |

### vs System.Numerics.Tensors.TensorPrimitives (N=1000)

In-place operations (zero allocation) compared to raw TensorPrimitives calls.

| Operation | AiDotNet | TensorPrimitives | Speedup |
|-----------|----------|-----------------|---------|
| Dot Product | 97 ns | 185 ns | **1.9x faster** |
| L2 Norm | 92 ns | 187 ns | **2.0x faster** |
| Vector AddInPlace | 154 ns | 117 ns | 0.8x |
| Vector SubtractInPlace | 116 ns | 118 ns | **tied** |
| Vector ScalarMulInPlace | 105 ns | 75 ns | 0.7x |
| Vector Add to Span | 116 ns | 119 ns | **tied** |

### Small Matrix Multiply (double)

| Size | AiDotNet | MathNet | NumSharp |
|------|----------|---------|----------|
| 4x4 | 172 ns | 165 ns | 2,198 ns |
| 16x16 | 2.1 us | 2.9 us | 107.5 us |
| 32x32 | 10.5 us | 36.2 us | 774.8 us |

AiDotNet is **1.4x faster** at 16x16 and **3.4x faster** at 32x32 than MathNet.

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
