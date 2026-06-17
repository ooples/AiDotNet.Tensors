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

    /// <summary>True when this backend ships the FP16-NATIVE element / activation / normalization kernels
    /// (GELU/ReLU/residual-add and row Softmax/LayerNorm over half-stored activations, FP32 math). Consumers
    /// check before routing to the half-resident fast path so they can fall back to the FP32 path otherwise.</summary>
    bool SupportsFp16NativeOps { get; }

    /// <summary>Row softmax over the last axis of a half buffer: one block/work-group per row, FP32 max/sum
    /// reductions (numerically stable), half in/out.</summary>
    void Fp16Softmax(IGpuBuffer input, IGpuBuffer output, int rows, int cols);

    /// <summary>Row layernorm over the last axis of a half buffer with half gamma/beta: one block/work-group
    /// per row, FP32 mean/var, half in/out. Writes the per-row FP32 mean + variance for the backward (pass a
    /// real buffer; backends supply a temporary internally if a backend-specific null is allowed).</summary>
    void Fp16LayerNorm(IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer beta, IGpuBuffer output,
        IGpuBuffer meanFp32, IGpuBuffer varFp32, int rows, int cols, float eps);
}
