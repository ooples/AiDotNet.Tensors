// Copyright (c) AiDotNet. All rights reserved.
// IGpuHalfPrecisionBackend implementation for the Vulkan backend (issue #560):
// FP16 GEMM on the GPU via a GLSL compute shader, built on the new real Vulkan
// GPU GEMM (the previous Gemm was a CPU download/loop/upload fallback).

using System;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// Half-precision GEMM dispatch for the <see cref="IGpuHalfPrecisionBackend"/>
/// capability interface, so the backend-agnostic
/// <c>DirectGpuTensorEngine.MatrixMultiply</c> FP16 path (issue #560) runs on
/// Vulkan devices. Mirrors the CUDA backend's row-major contract.
/// <para>
/// FP16 operands are packed two halves per 32-bit word (the layout
/// <c>ConvertToFp16</c> produces); the GLSL kernel reads them with the core
/// <c>unpackHalf2x16</c> built-in and accumulates in FP32 — the AMP-standard
/// mixed-precision matmul. Requires libshaderc for runtime GLSL→SPIR-V
/// compilation; <see cref="SupportsHgemm"/> reflects that.
/// </para>
/// </summary>
public sealed partial class VulkanBackend : IGpuHalfPrecisionBackend
{
    /// <inheritdoc/>
    /// <remarks>
    /// Gated on libshaderc availability: the FP16 GEMM is a runtime-compiled
    /// GLSL compute shader, so without the compiler there is no GPU FP16 path
    /// (the engine then falls back to CPU).
    /// </remarks>
    public bool SupportsHgemm => IsGlslCompilerAvailable;

    /// <inheritdoc/>
    /// <remarks>The fused backward is two dispatches of the runtime-compiled <c>GemmFp16BackwardTransposed</c>
    /// GLSL shader, so it shares the libshaderc gate with <see cref="SupportsHgemm"/>.</remarks>
    public bool SupportsFp16FusedBackward => IsGlslCompilerAvailable;

    /// <inheritdoc/>
    /// <remarks>
    /// Two dispatches of the transposed FP16 GLSL GEMM (FP16 in, FP32 accumulate, no materialized transpose):
    /// gradA[M,K] = gradC·Bᵀ (transA=0, transB=1) and gradB[K,N] = Aᵀ·gradC (transA=1, transB=0). The shader
    /// writes FP32; FP16-output grads run into an FP32 scratch then <c>ConvertToFp16</c> (mirrors <see cref="Hgemm"/>).
    /// </remarks>
    public void MatMulBackwardFp16Fused(
        IGpuBuffer gradCFp16, IGpuBuffer aFp16, IGpuBuffer bFp16,
        IGpuBuffer gradAOut, IGpuBuffer gradBOut,
        int m, int n, int k, bool gradOutHalf)
    {
        if (gradCFp16 is null) throw new ArgumentNullException(nameof(gradCFp16));
        if (aFp16 is null) throw new ArgumentNullException(nameof(aFp16));
        if (bFp16 is null) throw new ArgumentNullException(nameof(bFp16));
        if (gradAOut is null) throw new ArgumentNullException(nameof(gradAOut));
        if (gradBOut is null) throw new ArgumentNullException(nameof(gradBOut));
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m), "Dimensions must be positive.");
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Dimensions must be positive.");
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), "Dimensions must be positive.");

        // gradA[M,K] = gradC[M,N] · Bᵀ : Mo=M, No=K, Kc=N; A=gradC (transA=0), B=Bfwd (transB=1).
        DispatchGlslGemmBackward(gradCFp16, bFp16, gradAOut, m, k, n, transA: 0, transB: 1, gradOutHalf);
        // gradB[K,N] = Aᵀ · gradC[M,N] : Mo=K, No=N, Kc=M; A=Afwd (transA=1), B=gradC (transB=0).
        DispatchGlslGemmBackward(aFp16, gradCFp16, gradBOut, k, n, m, transA: 1, transB: 0, gradOutHalf);
    }

    /// <summary>
    /// One dispatch of the transposed FP16 backward GLSL GEMM into <paramref name="cOut"/>. When
    /// <paramref name="gradOutHalf"/> the shader writes an FP32 scratch which is then packed to FP16 by
    /// <c>ConvertToFp16</c>; otherwise it writes <paramref name="cOut"/> (FP32) directly.
    /// </summary>
    private void DispatchGlslGemmBackward(IGpuBuffer a, IGpuBuffer b, IGpuBuffer cOut,
        int mo, int no, int kc, int transA, int transB, bool gradOutHalf)
    {
        var pipeline = GetOrCreateGlslPipeline(VulkanGemmKernels.GemmFp16BackwardTransposed, 3, 5 * sizeof(uint));
        if (pipeline is null)
            throw new NotSupportedException(
                "Vulkan FP16 GEMM requires libshaderc for runtime GLSL compilation, which is unavailable.");

        IGpuBuffer? scratch = gradOutHalf ? AllocateBuffer(mo * no) : null;
        try
        {
            var dest = scratch ?? cOut;
            var vbA = AsVulkan(a);
            var vbB = AsVulkan(b);
            var vbC = AsVulkan(dest);
            var pushConstants = new uint[] { (uint)mo, (uint)no, (uint)kc, (uint)transA, (uint)transB };
            var threadRes = _device.AcquireThreadResources();
            lock (_computeLock)
            {
                pipeline.UpdateDescriptorSet(vbA.Storage, vbB.Storage, vbC.Storage);
                RecordAndExecuteWithPushData(pipeline, mo * no, pushConstants, 5 * sizeof(uint), threadRes);
            }
            if (scratch is not null)
                ConvertToFp16(scratch, cOut, mo * no);
        }
        finally
        {
            scratch?.Dispose();
        }
    }

    /// <inheritdoc/>
    public void GemmFp16In32fOut(IGpuBuffer aFp16, IGpuBuffer bFp16, IGpuBuffer cFp32,
        int m, int n, int k)
    {
        ValidateHalfGemmArgs(aFp16, bFp16, cFp32, m, n, k);
        if (!TryGlslGemmFp16In32fOut(aFp16, bFp16, cFp32, m, n, k))
            throw new NotSupportedException(
                "Vulkan FP16 GEMM requires libshaderc for runtime GLSL compilation, which is unavailable.");
    }

    /// <inheritdoc/>
    public void Hgemm(IGpuBuffer aFp16, IGpuBuffer bFp16, IGpuBuffer cFp16,
        int m, int n, int k)
    {
        ValidateHalfGemmArgs(aFp16, bFp16, cFp16, m, n, k);
        // Accumulate in FP32 into a scratch buffer, then round the result down to
        // FP16 (matches cublasHgemm's FP16 output). Keeping the accumulation in
        // FP32 avoids the worst half-precision drift; the output is then packed
        // to halves by ConvertToFp16. A separate scratch avoids reading and
        // writing the same buffer in one pass.
        using var scratch = AllocateBuffer(m * n);
        if (!TryGlslGemmFp16In32fOut(aFp16, bFp16, scratch, m, n, k))
            throw new NotSupportedException(
                "Vulkan FP16 GEMM requires libshaderc for runtime GLSL compilation, which is unavailable.");
        ConvertToFp16(scratch, cFp16, m * n);
    }

    /// <summary>
    /// Runs C(fp32) = A(fp16)·B(fp16) on the GPU via the GLSL FP16 GEMM kernel.
    /// Returns false when libshaderc is unavailable so callers can surface a
    /// clear error / fall back. One invocation per output element; push
    /// constants carry {M, N, K}.
    /// </summary>
    private bool TryGlslGemmFp16In32fOut(IGpuBuffer aFp16, IGpuBuffer bFp16, IGpuBuffer cFp32,
        int m, int n, int k)
    {
        var pipeline = GetOrCreateGlslPipeline(VulkanGemmKernels.GemmFp16In32fOut, 3, 3 * sizeof(uint));
        if (pipeline is null)
            return false;

        var vbA = AsVulkan(aFp16);
        var vbB = AsVulkan(bFp16);
        var vbC = AsVulkan(cFp32);
        var pushConstants = new uint[] { (uint)m, (uint)n, (uint)k };
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(vbA.Storage, vbB.Storage, vbC.Storage);
            RecordAndExecuteWithPushData(pipeline, m * n, pushConstants, 3 * sizeof(uint), threadRes);
        }

        return true;
    }

    private static void ValidateHalfGemmArgs(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
        int m, int n, int k)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (c is null) throw new ArgumentNullException(nameof(c));
        // Report the offending dimension by name (mirrors the CUDA backend).
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m), "Dimensions must be positive.");
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Dimensions must be positive.");
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), "Dimensions must be positive.");
    }
    // #1650/#638 FP16 conv im2col — STUB pending the Vulkan kernel port (dispatch gates on Fp16Im2colAvailable
    // so Im2colKNFp16 is never invoked while this returns false; the engine keeps the FP32 conv on Vulkan).
    /// <inheritdoc/>
    public bool Fp16Im2colAvailable => false;
    /// <inheritdoc/>
    public void Im2colKNFp16(IGpuBuffer input, IGpuBuffer outputHalf,
        int batch, int channels, int height, int width,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
        => throw new System.NotSupportedException("FP16 im2col (Im2colKNFp16) is not yet ported to the Vulkan backend.");

}
