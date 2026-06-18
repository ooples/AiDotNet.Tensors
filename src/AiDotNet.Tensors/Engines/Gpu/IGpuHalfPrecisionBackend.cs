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

    /// <summary>True when this backend ships the fused FP16 matmul BACKWARD (<see cref="MatMulBackwardFp16Fused"/>).
    /// Consumers check before routing the FP16-activation backward through the fused two-GEMM path instead of the
    /// generic transpose+matmul backward. Distinct from <see cref="SupportsHgemm"/> only so a backend that ships a
    /// forward HGEMM but not the backward can advertise the capability honestly; in practice every backend that has
    /// the FP16-in/FP32-out GEMM also has the fused backward (it is two of those GEMMs).</summary>
    bool SupportsFp16FusedBackward { get; }

    /// <summary>
    /// Fused FP16 matmul BACKWARD for the forward <c>C[M,N] = A[M,K] · B[K,N]</c>: computes BOTH gradients with
    /// FP16 inputs and FP32 accumulation in two GEMMs, with NO materialized transpose and NO FP32 up-cast scratch —
    /// <list type="bullet">
    /// <item><c>gradA[M,K] = gradC[M,N] · Bᵀ</c></item>
    /// <item><c>gradB[K,N] = Aᵀ · gradC[M,N]</c></item>
    /// </list>
    /// This is the backend-agnostic contract for the engine's FP16-activation hetero backward
    /// (<see cref="Engines.Compilation.MixedPrecisionGraphBackward"/>): it replaces the eager backward's two
    /// <c>TensorTranspose</c> + two <c>TensorMatMul</c> FP32 dispatches (four FP32 device allocations per matmul)
    /// with two transpose-free Half GEMMs into caller-owned (reused) buffers. The accumulate is ALWAYS FP32 (the
    /// AMP-correct precision); only the STORED output dtype is selected by <paramref name="gradOutHalf"/>. All
    /// buffers are device-resident and caller-owned (capture-safe: no alloc/free inside).
    /// </summary>
    /// <param name="gradCFp16">FP16 upstream gradient, shape [M,N].</param>
    /// <param name="aFp16">FP16 forward input A, shape [M,K].</param>
    /// <param name="bFp16">FP16 forward input B, shape [K,N].</param>
    /// <param name="gradAOut">Output buffer for gradA, shape [M,K] (caller-owned, reused). FP32 or FP16 per <paramref name="gradOutHalf"/>.</param>
    /// <param name="gradBOut">Output buffer for gradB, shape [K,N] (caller-owned, reused). FP32 or FP16 per <paramref name="gradOutHalf"/>.</param>
    /// <param name="gradOutHalf">When true the gradient outputs are stored FP16 (half the down-stream bandwidth, the
    /// dtype the FP16 hetero backward flows on as Half); when false they are FP32. The accumulate is FP32 either way.</param>
    void MatMulBackwardFp16Fused(
        IGpuBuffer gradCFp16, IGpuBuffer aFp16, IGpuBuffer bFp16,
        IGpuBuffer gradAOut, IGpuBuffer gradBOut,
        int m, int n, int k, bool gradOutHalf);

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
