# AiDotNet.Tensors

[![NuGet](https://img.shields.io/nuget/v/AiDotNet.Tensors.svg)](https://www.nuget.org/packages/AiDotNet.Tensors/)
[![Build](https://github.com/ooples/AiDotNet.Tensors/actions/workflows/build.yml/badge.svg)](https://github.com/ooples/AiDotNet.Tensors/actions/workflows/build.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

High-performance tensor operations with SIMD and GPU acceleration for .NET.

## Features

- **Zero Allocations**: Uses `ArrayPool<T>` and `Span<T>` for hot path performance
- **SIMD Acceleration**: Automatic vectorization using SSE, AVX, AVX2, AVX-512, and ARM NEON
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

## Performance

The library automatically detects and uses the best available SIMD instructions:

| Instruction Set | Vector Width | Supported |
|----------------|--------------|-----------|
| AVX-512 | 512-bit (16 floats) | .NET 8+ |
| AVX2 | 256-bit (8 floats) | .NET 6+ |
| AVX | 256-bit (8 floats) | .NET 6+ |
| SSE4.2 | 128-bit (4 floats) | .NET 6+ |
| ARM NEON | 128-bit (4 floats) | .NET 6+ |

### Check Available Acceleration

```csharp
using AiDotNet.Tensors.Engines;

var caps = PlatformDetector.Capabilities;
Console.WriteLine($"AVX2: {caps.HasAVX2}");
Console.WriteLine($"AVX-512: {caps.HasAVX512F}");
Console.WriteLine($"CUDA: {caps.HasCudaSupport}");
Console.WriteLine($"OpenCL: {caps.HasOpenCLSupport}");
```

## Optional Acceleration Packages

### AiDotNet.Native.OpenBLAS

Provides optimized CPU BLAS operations using OpenBLAS:

```bash
dotnet add package AiDotNet.Native.OpenBLAS
```

**Performance**: 2x faster matrix operations compared to managed code.

### AiDotNet.Native.CLBlast

Provides GPU acceleration via OpenCL (works on AMD, Intel, and NVIDIA GPUs):

```bash
dotnet add package AiDotNet.Native.CLBlast
```

**Performance**: 10x+ faster for large matrix operations on GPU.

## Requirements

- .NET 10.0 or .NET Framework 4.7.1+
- Windows x64, Linux x64, or macOS x64/arm64

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
