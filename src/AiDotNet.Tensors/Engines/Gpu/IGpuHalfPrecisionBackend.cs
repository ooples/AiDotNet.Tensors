// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Issue #337: optional capability interface for backends that ship
/// half-precision GEMM kernels. Consumers downcast a backend reference
/// to this interface to access FP16 (and optionally BF16 via the
/// CUDA-native data-type constants) matmul without paying the fp32 cost
/// on Tensor-Core-capable hardware.
/// <para>
/// Not a member of <see cref="IDirectGpuBackend"/> directly because
/// most backends (Vulkan, WebGpu, Metal MPSGraph) don't ship a tensor-
/// core HGEMM path — making it part of the base interface would force
/// every backend to throw <c>NotSupportedException</c> on the call.
/// </para>
/// </summary>
public interface IGpuHalfPrecisionBackend
{
    /// <summary>True when this backend ships at least one half-precision
    /// matmul kernel. Consumers check before dispatching.</summary>
    bool SupportsHgemm { get; }

    /// <summary>
    /// FP16 matmul: C = A · B where A, B, C are FP16 buffers
    /// (<c>__half</c> on CUDA). Equivalent to cuBLAS <c>cublasHgemm</c>.
    /// Accumulation precision is FP16 (lower than mixed-precision Gemm32f-with-tf32);
    /// use <see cref="GemmFp16In32fOut"/> when training stability matters.
    /// </summary>
    /// <param name="aFp16">A in FP16 (rows=M, cols=K).</param>
    /// <param name="bFp16">B in FP16 (rows=K, cols=N).</param>
    /// <param name="cFp16">Output C in FP16 (rows=M, cols=N).</param>
    void Hgemm(IGpuBuffer aFp16, IGpuBuffer bFp16, IGpuBuffer cFp16,
        int m, int n, int k);

    /// <summary>
    /// Mixed-precision matmul: A, B are FP16, accumulator + output C are
    /// FP32. Maps to cuBLAS <c>cublasGemmEx</c> with input dtypes
    /// <c>CUDA_R_16F</c> and compute type <c>CUBLAS_COMPUTE_32F</c>.
    /// This is the standard "FP16 forward / FP32 master-grad" pattern
    /// AMP / AutocastScope was designed for — lower memory bandwidth on
    /// activations, no precision loss on gradient accumulation.
    /// </summary>
    void GemmFp16In32fOut(IGpuBuffer aFp16, IGpuBuffer bFp16, IGpuBuffer cFp32,
        int m, int n, int k);
}
